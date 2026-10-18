"""SEU bins / lag ablation runner (seed 42, validation only; test stays sealed).

Writes only under outputs/ablation/. Resumable via outputs/ablation/summary_runs.json.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import traceback
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.experiments.build_seu_dataset as bsd  # noqa: E402
from scripts.experiments.train_seu import train_one  # noqa: E402

OUT = ROOT / "outputs" / "ablation"
SUMMARY = OUT / "summary_runs.json"
TASKS = ("bearing", "gear")
CONDITIONS = ("20_0", "30_2")
ARCHS = ("cnn", "gcn")
# best (lr, dropout) per task/model from commit 19265bf (val sweep)
BEST = {
    ("bearing", "cnn"): (5e-4, 0.1),
    ("bearing", "gcn"): (5e-4, 0.3),
    ("gear", "cnn"): (5e-4, 0.1),
    ("gear", "gcn"): (1e-3, 0.3),
}


def settings() -> list[dict]:
    out = []
    for b in (3, 4, 5, 6, 7, 8, 9, 10, 12, 16):
        out.append({"ablation": "bins", "bins": b, "lags": [1, 2, 3]})
    for lags in ([1], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]):
        out.append({"ablation": "lags", "bins": 6, "lags": lags})
    for s in out:
        s["tag"] = f"b{s['bins']}_L{''.join(map(str, s['lags']))}"
    return out


def make_cfg(task: str, arch: str, s: dict, seed: int):
    cfg = OmegaConf.load(ROOT / "config" / f"seu_{task}_{arch}.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.method.symbol_bins = int(s["bins"])
    cfg.method.transition_steps = list(s["lags"])
    arch_note = ""
    if arch == "gcn":
        cfg.model.gcn_steps = len(s["lags"])
    if arch == "cnn" and int(s["bins"]) < 4:
        cfg.model.pool_size = 1
        arch_note = "pool_size=1 (2x MaxPool2d(2) would collapse 3x3 input)"
    cfg.data.output_root = str(OUT / "representations" / s["tag"])
    lr, dr = BEST[(task, arch)]
    cfg.train.lr = float(lr)
    cfg.train.dropout = float(dr)
    cfg.train.output_dir = str(OUT / "runs" / s["tag"] / arch)
    cfg.train.checkpoint_name = f"best_{task}_{arch}.pt"
    cfg.train.evaluate_test = False
    cfg.train.seed = int(seed)
    cfg.train.seeds = [int(seed)]
    return cfg, arch_note


def load_summary() -> dict:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {"runs": {}, "builds": {}}


def save_summary(data: dict) -> None:
    tmp = SUMMARY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(SUMMARY)


def build_all(data: dict, sets: list[dict]) -> None:
    orig = bsd.load_seu_csv
    for task in TASKS:
        for cond in CONDITIONS:
            todo = [s for s in sets if not (OUT / "representations" / s["tag"] / task / cond / "meta.json").exists()]
            if not todo:
                continue
            cached = functools.lru_cache(maxsize=None)(lambda p: orig(p))
            bsd.load_seu_csv = cached  # parse each CSV once per task/condition
            for s in todo:
                cfg, _ = make_cfg(task, "cnn", s, 42)
                t = time.perf_counter()
                bsd.build_condition(task, cond, cfg, force=False)
                dt = time.perf_counter() - t
                data["builds"][f"{s['tag']}|{task}|{cond}"] = {"build_s": dt}
                save_summary(data)
                print(f"BUILT {s['tag']} {task} {cond} {dt:.1f}s", flush=True)
            cached.cache_clear()
            bsd.load_seu_csv = orig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_summary()
    sets = settings()
    t0 = time.perf_counter()
    build_all(data, sets)
    for s in sets:
        for task in TASKS:
            for arch in ARCHS:
                for cond in CONDITIONS:
                    key = f"{s['tag']}|{task}|{arch}|{cond}|{a.seed}"
                    if data["runs"].get(key, {}).get("ok"):
                        continue
                    cfg, note = make_cfg(task, arch, s, a.seed)
                    print(f"RUN {key} lr={cfg.train.lr} dropout={cfg.train.dropout} {note}", flush=True)
                    t = time.perf_counter()
                    try:
                        r = train_one(task, cond, a.seed, cfg, evaluate_test=False,
                                      confirm_test=False, write_paper_figures=False)
                    except Exception as exc:  # record and continue
                        traceback.print_exc()
                        data["runs"][key] = {"ok": False, "error": repr(exc)}
                        save_summary(data)
                        continue
                    dt = time.perf_counter() - t
                    data["runs"][key] = {
                        "ok": True, "ablation": s["ablation"], "tag": s["tag"], "bins": s["bins"],
                        "lags": s["lags"], "task": task, "arch": arch, "condition": cond, "seed": a.seed,
                        "lr": float(cfg.train.lr), "dropout": float(cfg.train.dropout),
                        "arch_note": note, "train_s": dt, "epochs_ran": r["epochs_ran"],
                        "best_epoch": r["best_epoch"], "best_val_acc": r["best_val_acc"],
                        "val_macro_f1": r["val_macro_f1"], "input_shape": r["input_shape"],
                        "val_confusion_matrix": r["val_confusion_matrix"], "class_names": r["class_names"],
                    }
                    save_summary(data)
                    print(f"DONE {key} val_acc={r['best_val_acc']:.4f} epochs={r['epochs_ran']} {dt:.1f}s", flush=True)
    print(f"ALL DONE session_wall_s={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
