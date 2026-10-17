"""线性判别复核：判别力到底来自"表示"还是"分类器"？

做法（与训练代码完全独立，只用缓存的 X/y）：
  把 [30,6,6] 转移矩阵展平成 1080 维 → 直接喂给 LDA（sklearn，收缩解）。
  LDA 是完全线性的，没有任何表示学习或非线性能力。

注意：噪声实验下必须【按同口径取样】——
  逐窗口径与严格口径的 X 缓存不同，不能混用（第一版曾用受污染的数据得出过高结论）。

用法：
  python src/analysis/lda_check.py --variants base,snrm6,snrm10,gsnrm6,gsnrm10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

CONDS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]


def cache_dir(task: str, cond: str, variant: str) -> Path | None:
    """从任意一次该变体运行的 manifest.json 里反查数据缓存目录。"""
    root = PROJECT_ROOT / "artifacts/variant_scan" / variant
    if not root.exists():
        return None
    for run in sorted(root.iterdir()):
        mf = run / "manifest.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text(encoding="utf-8"))
        ds = meta.get("dataset_meta")
        if not ds:
            continue
        p = Path(ds)
        if p.exists() and f"\\{task}\\" in str(p) and cond in str(p):
            return p.parent
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="base,snrm6,snrm10,gsnrm6,gsnrm10")
    args = p.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    safe_print(f"{'变体':>10} | {'工况':>12} | {'N_train':>7} | {'LDA acc':>8} | {'macroF1':>8}")
    safe_print("-" * 62)
    for variant in variants:
        for task, cond in CONDS:
            d = cache_dir(task, cond, variant)
            if d is None:
                safe_print(f"{variant:>10} | {task} {cond:>6} | 缓存缺失，跳过")
                continue
            Xtr = np.load(d / "X_train.npy").reshape(len(np.load(d / "y_train.npy")), -1)
            ytr = np.load(d / "y_train.npy")
            Xva = np.load(d / "X_val.npy").reshape(len(np.load(d / "y_val.npy")), -1)
            yva = np.load(d / "y_val.npy")
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)
            acc = accuracy_score(yva, pred)
            f1 = f1_score(yva, pred, average="macro")
            rows.append({"variant": variant, "task": task, "condition": cond,
                         "n_train": int(len(ytr)), "lda_acc": float(acc), "lda_macro_f1": float(f1)})
            safe_print(f"{variant:>10} | {task} {cond:>6} | {len(ytr):7d} | {100*acc:7.2f}% | {100*f1:7.2f}%")

    out = PROJECT_ROOT / "reports" / "lda_check.csv"
    import csv
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    safe_print(f"\n-> {out}")


if __name__ == "__main__":
    main()
