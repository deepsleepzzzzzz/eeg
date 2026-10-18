"""Multi-seed ablation with one-shot test evaluation (user-authorized).

Phase 1: seed 42 -> test-only eval of saved early-stopped (val) checkpoints; also re-checks val acc.
Phase 2: seeds 43-46 -> train (early stop on val) then evaluate that checkpoint once on test.
Validation results -> outputs/ablation/summary_runs.json (same schema as seed 42).
Test results       -> outputs/ablation/test_results_runs.json (separate; never used for selection).
Resumable: a run is skipped when both its val and test entries exist.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.ablation.run_ablation import (  # noqa: E402
    ARCHS, CONDITIONS, OUT, TASKS, build_all, load_summary, make_cfg, save_summary, settings,
)
from scripts.experiments.train_seu import metrics_from_predictions, train_one  # noqa: E402
from data.seu_dds import class_names  # noqa: E402
from experiments.train_runtime import evaluate, get_device, load_array_pair, make_loader, resolve_path  # noqa: E402
from models.cnn import build_model  # noqa: E402

TEST = OUT / "test_results_runs.json"


def load_test() -> dict:
    return json.loads(TEST.read_text(encoding="utf-8")) if TEST.exists() else {"runs": {}}


def save_test(d: dict) -> None:
    tmp = TEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(TEST)


def base_row(s, task, arch, cond, seed, note):
    return {"ok": True, "ablation": s["ablation"], "tag": s["tag"], "bins": s["bins"], "lags": s["lags"],
            "task": task, "arch": arch, "condition": cond, "seed": seed, "arch_note": note}


def eval_saved(task, arch, cond, s, seed):
    cfg, note = make_cfg(task, arch, s, seed)
    names = class_names(task)
    data_dir = resolve_path(cfg.data.output_root) / task / cond
    run_dir = resolve_path(cfg.train.output_dir) / task / cond / f"seed_{seed}"
    ck = torch.load(run_dir / cfg.train.checkpoint_name, map_location="cpu", weights_only=False)
    Xv, yv = load_array_pair(data_dir, "val")
    Xt, yt = load_array_pair(data_dir, "test")
    device = get_device(cfg)
    model = build_model(len(names), cfg, in_channels=int(Xv.shape[1]), image_size=int(Xv.shape[-1])).to(device)
    model.load_state_dict(ck["model"])
    crit = nn.CrossEntropyLoss()
    bs = int(cfg.train.batch_size)
    _, vacc, _, _ = evaluate(model, make_loader(Xv, yv, bs, 0, False), device, crit, predictions=True)
    tloss, tacc, tt, tp = evaluate(model, make_loader(Xt, yt, bs, 0, False), device, crit, predictions=True)
    m = metrics_from_predictions(tt, tp, "test", names)
    return note, vacc, float(ck["val_acc"]), int(ck["epoch"]), tloss, tacc, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="43,44,45,46")
    p.add_argument("--skip-seed42", action="store_true")
    a = p.parse_args()
    val = load_summary()
    test = load_test()
    sets = settings()
    t0 = time.perf_counter()
    build_all(val, sets)  # no-op when all caches exist
    if not a.skip_seed42:
        for s in sets:
            for task in TASKS:
                for arch in ARCHS:
                    for cond in CONDITIONS:
                        key = f"{s['tag']}|{task}|{arch}|{cond}|42"
                        if test["runs"].get(key, {}).get("ok") or not val["runs"].get(key, {}).get("ok"):
                            continue
                        try:
                            note, vacc, ck_vacc, ep, tloss, tacc, m = eval_saved(task, arch, cond, s, 42)
                        except Exception as exc:
                            traceback.print_exc()
                            test["runs"][key] = {"ok": False, "error": repr(exc)}
                            save_test(test)
                            continue
                        row = base_row(s, task, arch, cond, 42, note)
                        row.update({"source": "saved_checkpoint_eval", "ckpt_epoch": ep,
                                    "val_acc_recheck": vacc, "val_acc_in_ckpt": ck_vacc,
                                    "val_acc_summary": val["runs"][key]["best_val_acc"],
                                    "test_loss": tloss, "test_acc": tacc,
                                    "test_confusion_matrix": m["test_confusion_matrix"]})
                        test["runs"][key] = row
                        save_test(test)
                        print(f"TEST42 {key} val_recheck={vacc:.4f} (summary {row['val_acc_summary']:.4f}) test={tacc:.4f}", flush=True)
    for seed in [int(x) for x in a.seeds.split(",") if x.strip()]:
        for s in sets:
            for task in TASKS:
                for arch in ARCHS:
                    for cond in CONDITIONS:
                        key = f"{s['tag']}|{task}|{arch}|{cond}|{seed}"
                        if val["runs"].get(key, {}).get("ok") and test["runs"].get(key, {}).get("ok"):
                            continue
                        cfg, note = make_cfg(task, arch, s, seed)
                        print(f"RUN {key} {note}", flush=True)
                        t = time.perf_counter()
                        try:
                            r = train_one(task, cond, seed, cfg, evaluate_test=True, confirm_test=True,
                                          write_paper_figures=False)
                        except Exception as exc:
                            traceback.print_exc()
                            val["runs"][key] = {"ok": False, "error": repr(exc)}
                            save_summary(val)
                            continue
                        dt = time.perf_counter() - t
                        vrow = base_row(s, task, arch, cond, seed, note)
                        vrow.update({"lr": float(cfg.train.lr), "dropout": float(cfg.train.dropout), "train_s": dt,
                                     "epochs_ran": r["epochs_ran"], "best_epoch": r["best_epoch"],
                                     "best_val_acc": r["best_val_acc"], "val_macro_f1": r["val_macro_f1"],
                                     "input_shape": r["input_shape"], "val_confusion_matrix": r["val_confusion_matrix"],
                                     "class_names": r["class_names"]})
                        trow = base_row(s, task, arch, cond, seed, note)
                        trow.update({"source": "train_then_test_once", "ckpt_epoch": r["best_epoch"],
                                     "test_loss": r["test_loss"], "test_acc": r["test_acc_once"],
                                     "test_confusion_matrix": r["test_confusion_matrix"]})
                        val["runs"][key] = vrow
                        test["runs"][key] = trow
                        save_summary(val)
                        save_test(test)
                        print(f"DONE {key} val={r['best_val_acc']:.4f} test={r['test_acc_once']:.4f} "
                              f"epochs={r['epochs_ran']} best={r['best_epoch']} {dt:.1f}s", flush=True)
    print(f"ALL DONE session_wall_s={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
