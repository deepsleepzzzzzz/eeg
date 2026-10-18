"""SEU tuning driver: search method/train knobs until every task-condition hits the target.

Val only. Test stays sealed. Results append to a JSON log so reruns resume.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_runtime import resolve_path
from scripts.experiments.build_seu_dataset import build_condition
from scripts.experiments.train_seu import train_one
from utils.io_utils import ensure_dir, load_json, save_json

TUNE_ROOT = "./datas/experiments_seu/tune"

SEARCH_SPACE = {
    "bearing": {
        "20_0": [
            {"w": "zq_s7_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_over", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_s7", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq", "lr": 5e-4, "dropout": 0.1},
            {"w": "zq", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_b8", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_b10", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_b4", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_noself", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_row", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s5", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_self", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_attn", "lr": 5e-4, "dropout": 0.2},
        ],
        "30_2": [
            {"w": "zq", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_over", "lr": 5e-4, "dropout": 0.2},
        ],
    },
    "gear": {
        "20_0": [
            {"w": "zq_s7_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_over", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_s7", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_b4", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s9", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s5", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_b4", "lr": 5e-4, "dropout": 0.2},
        ],
        "30_2": [
            {"w": "zq_s7_w1600_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_w1600_over", "lr": 3e-4, "dropout": 0.2},
            {"w": "zq_s9_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s12_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_row_s7_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_b8_s7_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_b4_s9_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_w800_stride200", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_s7", "lr": 5e-4, "dropout": 0.3},
            {"w": "zq_s7", "lr": 5e-4, "dropout": 0.1},
            {"w": "zq_s7_b4", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_over", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s7_w1600", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s9", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s9", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_s12", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s10_b4", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s5_w1600", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_s5", "lr": 1e-3, "dropout": 0.2},
            {"w": "zq_self", "lr": 5e-4, "dropout": 0.2},
            {"w": "zq_attn", "lr": 5e-4, "dropout": 0.2},
        ],
    },
}


def _variant(normalize="none", weight="probability", bins=6, steps=(1, 2, 3),
             include_self=True, stride=800, length=800, attention="none",
             gcn_attention=False, edges="quantile", state_stats="none",
             occupancy_scale=1.0):
    return {
        "normalize": normalize,
        "transition_weight": weight,
        "symbol_bins": bins,
        "transition_steps": list(steps),
        "include_self_transition": include_self,
        "window_stride": stride,
        "epoch_samples": length,
        "attention": attention,
        "gcn_attention": gcn_attention,
        "bin_edges_mode": edges,
        "state_stats": state_stats,
        "occupancy_scale": occupancy_scale,
    }


VARIANTS = {
    "baseline": _variant(),
    "row": _variant(weight="row_probability"),
    "count": _variant(weight="count"),
    "row_noself": _variant(weight="row_probability", include_self=False),
    # z-score per window: removes per-class amplitude scale differences that
    # otherwise collapse every class into a single symbol state.
    "z": _variant(normalize="zscore"),
    "z_row": _variant(normalize="zscore", weight="row_probability"),
    "z_noself": _variant(normalize="zscore", include_self=False),
    "z_b8": _variant(normalize="zscore", bins=8),
    "z_b10": _variant(normalize="zscore", bins=10),
    "z_over": _variant(normalize="zscore", stride=400),
    "z_over_b10": _variant(normalize="zscore", bins=10, stride=400),
    "z_self": _variant(normalize="zscore", attention="self"),
    "z_gcn_attn": _variant(normalize="zscore", gcn_attention=True),
    # z-score per window + quantile (equal-frequency) bin edges: the combination
    # that restores a fully used symbol state space on the hard conditions.
    "zq": _variant(normalize="zscore", edges="quantile"),
    "zq_row": _variant(normalize="zscore", weight="row_probability", edges="quantile"),
    "zq_noself": _variant(normalize="zscore", include_self=False, edges="quantile"),
    "zq_b4": _variant(normalize="zscore", bins=4, edges="quantile"),
    "zq_b8": _variant(normalize="zscore", bins=8, edges="quantile"),
    "zq_b10": _variant(normalize="zscore", bins=10, edges="quantile"),
    "zq_over": _variant(normalize="zscore", stride=400, edges="quantile"),
    "zq_s5": _variant(normalize="zscore", steps=(1, 2, 3, 4, 5), edges="quantile"),
    "zq_self": _variant(normalize="zscore", edges="quantile", attention="self"),
    "zq_attn": _variant(normalize="zscore", edges="quantile", gcn_attention=True),
    # longer strides / longer windows / more time-scale steps
    "zq_s7": _variant(normalize="zscore", steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_s7_b4": _variant(normalize="zscore", bins=4, steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_s7_over": _variant(normalize="zscore", stride=400, steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_s7_w1600": _variant(normalize="zscore", length=1600, stride=1600,
                            steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_s9": _variant(normalize="zscore", steps=(1, 2, 3, 4, 5, 6, 7, 8, 9), edges="quantile"),
    "zq_s12": _variant(normalize="zscore", steps=tuple(range(1, 13)), edges="quantile"),
    "zq_s10_b4": _variant(normalize="zscore", bins=4, steps=tuple(range(1, 11)), edges="quantile"),
    "zq_s5_w1600": _variant(normalize="zscore", length=1600, stride=1600,
                            steps=(1, 2, 3, 4, 5), edges="quantile"),
    "zq_w1600": _variant(normalize="zscore", length=1600, stride=1600, edges="quantile"),
    "zq_w1600_s5": _variant(normalize="zscore", length=1600, stride=1600, steps=(1, 2, 3, 4, 5), edges="quantile"),
    "zq_w1600_over": _variant(normalize="zscore", length=1600, stride=800, edges="quantile"),
    "zq_w2000_over": _variant(normalize="zscore", length=2000, stride=1000, edges="quantile"),
    "zq_b4_w1600_over": _variant(normalize="zscore", bins=4, length=1600, stride=800, edges="quantile"),
    "zq_b4_w1600": _variant(normalize="zscore", bins=4, length=1600, stride=1600, edges="quantile"),
    "zq_b4_s5": _variant(normalize="zscore", bins=4, steps=(1, 2, 3, 4, 5), edges="quantile"),
    # occupancy channel: keeps the amplitude/energy information that global
    # probability normalization throws away, which is exactly what the hard
    # gear pair (Miss vs Surface) needs.
    # occupancy channel (proved unhelpful: z-score already equalizes energy)
    "zq_occ": _variant(normalize="zscore", edges="quantile", state_stats="occupancy"),
    # raw amplitude + quantile edges: keep the between-class amplitude scale, which
    # z-score removes. Equal-frequency edges still prevent state collapse.
    "q": _variant(normalize="none", edges="quantile"),
    "q_b4": _variant(normalize="none", bins=4, edges="quantile"),
    "q_b8": _variant(normalize="none", bins=8, edges="quantile"),
    "q_b10": _variant(normalize="none", bins=10, edges="quantile"),
    "q_s5": _variant(normalize="none", steps=(1, 2, 3, 4, 5), edges="quantile"),
    "q_s7": _variant(normalize="none", steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "q_over": _variant(normalize="none", stride=400, edges="quantile"),
    "q_w1600_over": _variant(normalize="none", length=1600, stride=800, edges="quantile"),
    "q_w1600": _variant(normalize="none", length=1600, stride=1600, edges="quantile"),
    "q_b4_w1600_over": _variant(normalize="none", bins=4, length=1600, stride=800, edges="quantile"),
    "q_b4_s5": _variant(normalize="none", bins=4, steps=(1, 2, 3, 4, 5), edges="quantile"),
    # ---- round 2: attack the hard gear 30-2 pair (Miss vs Surface) -------------
    # longer window + overlap: finer time-texture resolution on the transition map.
    "zq_s7_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                 steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_s9_over": _variant(normalize="zscore", stride=400,
                           steps=(1, 2, 3, 4, 5, 6, 7, 8, 9), edges="quantile"),
    "zq_s12_over": _variant(normalize="zscore", stride=400,
                            steps=tuple(range(1, 13)), edges="quantile"),
    "zq_row_s7_over": _variant(normalize="zscore", weight="row_probability", stride=400,
                               steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    # more symbols + many steps + overlap: richer state alphabet for fine faults.
    "zq_b8_s7_over": _variant(normalize="zscore", bins=8, stride=400,
                              steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    "zq_b4_s9_over": _variant(normalize="zscore", bins=4, stride=400,
                              steps=(1, 2, 3, 4, 5, 6, 7, 8, 9), edges="quantile"),
    # block-window overlap variant (200-sample hop) for max temporal density.
    "zq_s7_w800_stride200": _variant(normalize="zscore", stride=200,
                                     steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile"),
    # ---- round 3 (2026-10-02, 导师问题1/2)：延迟步数上限 & 是否需要间隔 --------
    # 统一固定在最终窗口设置 (L=1600 / stride=800) 下，只变 transition_steps（单变量）。
    # 连续组 -> 回答「延迟一共取到多少性能最好」；稀疏组 -> 回答「是否需要间隔」。
    "zq_s1_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                 steps=(1,), edges="quantile"),
    "zq_s3_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                 steps=(1, 2, 3), edges="quantile"),
    "zq_s5_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                 steps=(1, 2, 3, 4, 5), edges="quantile"),
    "zq_s9_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                 steps=tuple(range(1, 10)), edges="quantile"),
    "zq_s12_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                  steps=tuple(range(1, 13)), edges="quantile"),
    "zq_s15_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                  steps=tuple(range(1, 16)), edges="quantile"),
    # 稀疏（带间隔）：最大延迟与 s5 / s7 / s9 对齐，但步数更少 -> 隔离「间隔」的作用
    "zq_s135_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                   steps=(1, 3, 5), edges="quantile"),
    "zq_s1357_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                    steps=(1, 3, 5, 7), edges="quantile"),
    "zq_s13579_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                     steps=(1, 3, 5, 7, 9), edges="quantile"),
    "zq_s147_w1600_over": _variant(normalize="zscore", length=1600, stride=800,
                                   steps=(1, 4, 7), edges="quantile"),
}

# representation keys that force the window cache to be rebuilt
REP_KEYS = ("normalize", "bin_edges_mode", "transition_weight", "transition_steps",
            "symbol_bins", "include_self_transition", "window_stride", "epoch_samples")


def rep_dir_name(variant: str) -> str:
    v = VARIANTS[variant]
    steps = "".join(str(s) for s in v["transition_steps"])
    self_tag = "self" if v["include_self_transition"] else "noself"
    norm = "" if str(v["normalize"]).lower() == "none" else f"{v['normalize']}_"
    edge_tag = "" if str(v["bin_edges_mode"]).lower() == "quantile" else f"{v['bin_edges_mode']}_"
    return (f"{norm}{edge_tag}{variant}_{v['transition_weight']}_b{v['symbol_bins']}"
            f"_s{steps}_{self_tag}_w{v['window_stride']}_L{v['epoch_samples']}")


def apply_config(cfg, variant: str, lr: float, dropout: float, task: str, arch: str, out_root: Path):
    v = VARIANTS[variant]
    OmegaConf.set_struct(cfg, False)
    cfg.method.normalize = v["normalize"]
    cfg.method.bin_edges_mode = v["bin_edges_mode"]
    cfg.method.state_stats = v["state_stats"]
    cfg.method.transition_weight = v["transition_weight"]
    cfg.method.transition_steps = list(v["transition_steps"])
    cfg.method.symbol_bins = v["symbol_bins"]
    cfg.method.include_self_transition = v["include_self_transition"]
    cfg.method.attention = v["attention"]
    cfg.model.gcn_attention = bool(v["gcn_attention"])
    cfg.data.epoch_samples = v["epoch_samples"]
    cfg.data.window_stride = v["window_stride"]
    cfg.data.output_root = str(out_root / rep_dir_name(variant))
    cfg.train.output_dir = str(out_root / "ckpt" / variant / task / arch)
    cfg.train.checkpoint_name = f"best_{variant}_{arch}.pt"
    cfg.train.lr = float(lr)
    cfg.train.dropout = float(dropout)
    cfg.train.evaluate_test = False
    cfg.train.seed = 42
    cfg.train.seeds = [42]
    return cfg


def trial_key(task, condition, arch, variant, lr, dropout):
    return f"{task}|{condition}|{arch}|{variant}|lr{lr:g}_drop{dropout:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="SEU tuning driver, val only.")
    parser.add_argument("--task", default="bearing")
    parser.add_argument("--condition", default="20_0")
    parser.add_argument("--archs", default="cnn,gcn")
    parser.add_argument("--max-trials", type=int, default=99)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--variant", default=None,
                        help="run only this variant, bypassing SEARCH_SPACE order")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = args.task
    condition = args.condition
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    if args.variant:
        SEARCH_SPACE.setdefault(task, {}).setdefault(condition, []).insert(
            0, {"w": args.variant, "lr": args.lr, "dropout": args.dropout}
        )
    out_root = Path(ensure_dir(str(resolve_path(TUNE_ROOT))))
    log_path = out_root / "log.json"
    log = load_json(str(log_path)) if log_path.exists() else {"trials": []}
    done = {t["key"] for t in log["trials"]}

    trials = SEARCH_SPACE.get(task, {}).get(condition, [])
    ran = 0
    for spec in trials:
        variant, lr, dropout = spec["w"], spec["lr"], spec["dropout"]
        hit = []
        for arch in archs:
            key = trial_key(task, condition, arch, variant, lr, dropout)
            if key in done:
                prev = [t for t in log["trials"] if t["key"] == key][0]
                hit.append((arch, prev["val_acc"]))
                print(f"skip {key} val={prev['val_acc']:.4f}", flush=True)
                continue
            if ran >= args.max_trials:
                break
            cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_{arch}.yaml"))
            apply_config(cfg, variant, lr, dropout, task, arch, out_root)
            rep_dir = Path(cfg.data.output_root) / task / condition
            if not (rep_dir / "meta.json").exists():
                print(f"build {variant} {task} {condition}", flush=True)
                build_condition(task, condition, cfg, force=False)
            print(f"train {key}", flush=True)
            res = train_one(task, condition, 42, cfg, evaluate_test=False,
                            confirm_test=False, write_paper_figures=False)
            row = {
                "key": key, "task": task, "condition": condition, "arch": arch,
                "variant": variant, "lr": lr, "dropout": dropout,
                "val_acc": float(res["best_val_acc"]),
                "val_macro_f1": float(res.get("val_macro_f1", 0.0)),
                "best_epoch": int(res["best_epoch"]),
                "rep_dir": str(cfg.data.output_root),
            }
            log["trials"].append(row)
            done.add(key)
            save_json(log, str(log_path))
            ran += 1
            hit.append((arch, row["val_acc"]))
            print(f"done {key} val={row['val_acc']:.4f} f1={row['val_macro_f1']:.4f}", flush=True)
        if hit and all(v >= args.target for _, v in hit):
            print(f"TARGET reached at {variant} lr={lr:g} drop={dropout:g}", flush=True)
            break
        if all(a in [h[0] for h in hit] for a in archs) and hit:
            worst = min(v for _, v in hit)
            print(f"  {variant} lr={lr:g} drop={dropout:g}: worst={worst:.4f}", flush=True)

    ranked = sorted(log["trials"], key=lambda t: t["val_acc"], reverse=True)
    print("\n=== top ===", flush=True)
    for t in ranked[:8]:
        print(f"{t['val_acc']:.4f} f1={t['val_macro_f1']:.4f} {t['key']}", flush=True)


if __name__ == "__main__":
    main()
