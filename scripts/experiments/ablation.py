"""Ablation gap-filling + consolidation for SEU (state count / steps / stride).

Fills the missing ablation points on the hardest task (gear 30-2, CNN) so the
teacher gets a complete "one-variable-at-a-time" table:

  - symbol_bins (state count): 2 / 4 / 6 / 8 / 10   (transition_steps=7 fixed)
  - transition_steps (步长):    1 / 3 / 5 / 7 / 9     (symbol_bins=6 fixed)
  - window stride / length (跳步): w400/L800, w800/L800, w800/L1600 (s7,b6)

Existing points are read from datas/experiments_seu/tune/log.json; only the
missing ones are trained (val only, test sealed). Output -> teacher_extra/ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.train_runtime import resolve_path
from scripts.experiments.tune_seu import (
    VARIANTS,
    _variant,
    apply_config,
    build_condition,
    train_one,
    TUNE_ROOT,
)

OUT_DIR = ROOT / "datas/experiments_seu/teacher_extra"

# add missing variants to the shared VARIANTS registry
# (bins=2 is omitted: a 2x2 state matrix collapses under the CNN's two 2x2 pools)
VARIANTS["zq_s1"] = _variant(normalize="zscore", steps=(1,), edges="quantile")
VARIANTS["zq_s7_b8"] = _variant(normalize="zscore", bins=8, steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile")
VARIANTS["zq_s7_b10"] = _variant(normalize="zscore", bins=10, steps=(1, 2, 3, 4, 5, 6, 7), edges="quantile")


def load_existing():
    log_path = resolve_path(TUNE_ROOT) / "log.json"
    if not log_path.exists():
        return {}
    d = json.loads(log_path.read_text(encoding="utf-8"))
    out = {}
    for t in d["trials"]:
        if t["task"] == "gear" and t["condition"] == "30_2" and t["arch"] == "cnn":
            out[t["variant"]] = t["val_acc"]
    return out


def run_variant(task, condition, arch, variant, lr, dropout):
    cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_{arch}.yaml"))
    out_root = resolve_path(TUNE_ROOT)
    apply_config(cfg, variant, lr, dropout, task, arch, out_root)
    rep_dir = Path(cfg.data.output_root) / task / condition
    if not (rep_dir / "meta.json").exists():
        build_condition(task, condition, cfg, force=False)
    res = train_one(task, condition, 42, cfg, evaluate_test=False,
                    confirm_test=False, write_paper_figures=False)
    return float(res["best_val_acc"])


def main():
    existing = load_existing()
    # fill gaps
    gaps = ["zq_s1", "zq_s9", "zq_s7_b8", "zq_s7_b10"]
    for v in gaps:
        if v in existing:
            print(f"skip {v} (have {existing[v]:.4f})", flush=True)
            continue
        acc = run_variant("gear", "30_2", "cnn", v, 5e-4, 0.2)
        existing[v] = acc
        print(f"ran {v} val={acc:.4f}", flush=True)

    # build clean one-variable tables
    def grab(v):
        return round(existing[v], 4) if v in existing else None

    bins_table = [
        {"symbol_bins": 4, "variant": "zq_s7_b4", "val_acc": grab("zq_s7_b4")},
        {"symbol_bins": 6, "variant": "zq_s7", "val_acc": grab("zq_s7")},
        {"symbol_bins": 8, "variant": "zq_s7_b8", "val_acc": grab("zq_s7_b8")},
        {"symbol_bins": 10, "variant": "zq_s7_b10", "val_acc": grab("zq_s7_b10")},
    ]
    steps_table = [
        {"transition_steps": 1, "variant": "zq_s1", "val_acc": grab("zq_s1")},
        {"transition_steps": 3, "variant": "zq", "val_acc": grab("zq")},
        {"transition_steps": 5, "variant": "zq_s5", "val_acc": grab("zq_s5")},
        {"transition_steps": 7, "variant": "zq_s7", "val_acc": grab("zq_s7")},
        {"transition_steps": 9, "variant": "zq_s9", "val_acc": grab("zq_s9")},
    ]
    stride_table = [
        {"window": "w400/L800 (overlap)", "variant": "zq_s7_over", "val_acc": grab("zq_s7_over")},
        {"window": "w800/L800", "variant": "zq_s7", "val_acc": grab("zq_s7")},
        {"window": "w800/L1600 (long+overlap)", "variant": "zq_s7_w1600_over", "val_acc": grab("zq_s7_w1600_over")},
    ]
    result = {
        "note": "gear 30-2 (hardest), CNN, lr=5e-4 drop=0.2, val-only (test sealed)",
        "symbol_bins_ablation": bins_table,
        "transition_steps_ablation": steps_table,
        "window_stride_ablation": stride_table,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 状态数消融 (s7固定) =====")
    for r in bins_table:
        print(f"  bins={r['symbol_bins']:2d}  val_acc={r['val_acc']}")
    print("===== 步长消融 (b6固定) =====")
    for r in steps_table:
        print(f"  steps={r['transition_steps']:2d}  val_acc={r['val_acc']}")
    print("===== 跳步/窗口消融 (s7,b6) =====")
    for r in stride_table:
        print(f"  {r['window']:24s}  val_acc={r['val_acc']}")
    print(f"wrote {OUT_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
