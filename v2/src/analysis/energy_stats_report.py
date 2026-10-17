# -*- coding: utf-8 -*-
"""幅值补充（representation.energy_stats）前后对照报告。

回答的问题：等频符号化抹掉的幅值信息，用每通道一个 6x6 区块补回来之后，
齿轮任务"缺齿 ↔ 齿面磨损"的重叠有没有被分开？

三项对照（都在同一划分/窗/训练预算下）：
  1. 平台精度（末 20 轮均值）      —— before = 主协议 v5，after = rms3 / state3
  2. 混淆矩阵：缺齿(2) ↔ 齿面磨损(4) 的错分次数（3 个种子汇总）
  3. 表示可分性：类心间距 / 类内散布，以及"最像的一对"
另出图：fig_tsne_energy.png（齿轮 20-0：before / rms3 / state3 三格）

用法： python src/analysis/energy_stats_report.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import dataset_dir, load_protocol  # noqa: E402
from data.seu_dds import class_names  # noqa: E402
from reporting.class_labels import cn_name  # noqa: E402
from reporting.make_noise_figure import setup_font  # noqa: E402
from utils.console import setup_utf8_console  # noqa: E402

setup_utf8_console()   # 控制台默认 GBK 会因 ↗/↔ 之类字符报 UnicodeEncodeError
from tuning.scan_variants import build_variant_protocol, data_id  # noqa: E402

SCAN = PROJECT_ROOT / "artifacts" / "variant_scan_energy"
MAIN = PROJECT_ROOT / "artifacts" / "main_70_15_15_v5"
CONDS = (("gear", "20_0"), ("gear", "30_2"))
MODELS = ("cnn",)   # GCN 的 relational 布局要求"通道数 = 通道 × 滞后"，33 通道不满足，故本消融只在 CNN 上做
SEEDS = (42, 43, 44)


def proto() -> dict:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")


def plateau(run: Path) -> float:
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    s = m.get("stability") or {}
    v = s.get("last20_val_acc_mean")
    return float(v) * 100 if v is not None else float("nan")


def confusion(run: Path) -> np.ndarray:
    cm = np.zeros((5, 5), dtype=int)
    p = run / "val_predictions.csv"
    if not p.exists():
        return cm
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cm[int(r["true_label"]), int(r["pred_label"])] += 1
    return cm


def main() -> None:
    setup_font()
    print("=" * 96)
    print("幅值补充前后对照（齿轮任务；平台精度 = 末 20 轮均值，3 个种子）")
    print("=" * 96)
    print(f"{'任务-工况':<12}{'模型':<5}{'before(主协议)':>16}{'rms3':>10}{'state3':>10}{'Δ(rms3)':>10}{'Δ(state3)':>11}")
    for task, cond in CONDS:
        for mdl in MODELS:
            b = st.mean([plateau(MAIN / f"{task}_{mdl}_{cond}_seed{s}") for s in SEEDS])
            a1 = st.mean([plateau(SCAN / "rms3" / f"{task}_{mdl}_{cond}_rms3_seed{s}") for s in SEEDS])
            a2 = st.mean([plateau(SCAN / "state3" / f"{task}_{mdl}_{cond}_state3_seed{s}") for s in SEEDS])
            print(f"{task+' '+cond:<12}{mdl:<5}{b:>16.3f}{a1:>10.3f}{a2:>10.3f}"
                  f"{a1-b:>+10.3f}{a2-b:>+11.3f}")

    print()
    print("=" * 96)
    print("齿轮 20-0：缺齿(2) ↔ 齿面磨损(4) 的错分次数（3 个种子汇总；越小越好）")
    print("=" * 96)
    for mdl in MODELS:
        cms = {"before": np.zeros((5, 5), int), "rms3": np.zeros((5, 5), int),
               "state3": np.zeros((5, 5), int)}
        for s in SEEDS:
            cms["before"] += confusion(MAIN / f"gear_{mdl}_20_0_seed{s}")
            cms["rms3"] += confusion(SCAN / "rms3" / f"gear_{mdl}_20_0_rms3_seed{s}")
            cms["state3"] += confusion(SCAN / "state3" / f"gear_{mdl}_20_0_state3_seed{s}")
        for k, cm in cms.items():
            err = int(cm.sum() - np.trace(cm))
            pair = int(cm[2, 4] + cm[4, 2])
            print(f"  {mdl.upper():<4}{k:<8} 总错分 {err:>3}   其中 缺齿↔齿面磨损 {pair:>3}"
                  f"   准确率 {100*np.trace(cm)/cm.sum():.2f}%")

    # ---- 表示可分性 ----
    print()
    print("=" * 96)
    print("表示可分性：类心间距 / 类内散布（越大越好；<2 基本重叠）")
    print("=" * 96)
    prot = proto()
    variants = {}
    for name in ("rms3", "state3"):
        pid = data_id(build_variant_protocol(name, {"representation.energy_stats":
                                                    "rms" if name == "rms3" else "state"}))
        variants[name] = PROJECT_ROOT / "artifacts" / pid / "dataset"
    for task, cond in CONDS:
        names = list(class_names(task))
        rows = [("before", dataset_dir(prot, task, cond))]
        rows += [(k, v / task / cond) for k, v in variants.items()]
        for tag, d in rows:
            X = np.load(d / "X_val.npy").astype(np.float32)
            y = np.load(d / "y_val.npy").astype(np.int64)
            Xf = X.reshape(len(X), -1)
            Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
            cen, spread = {}, []
            for c in range(5):
                Z = Xf[y == c]
                cen[c] = Z.mean(0)
                spread.append(np.linalg.norm(Z - cen[c], axis=1).mean())
            d24 = float(np.linalg.norm(cen[2] - cen[4]))
            dists = [float(np.linalg.norm(cen[a] - cen[b]))
                     for a in range(5) for b in range(a + 1, 5)]
            print(f"  {task+' '+cond:<12}{tag:<8} 类内散布 {np.mean(spread):>6.1f}   "
                  f"类间均值 {np.mean(dists):>6.1f}   比值 {np.mean(dists)/np.mean(spread):>5.2f}   "
                  f"缺齿↔齿面磨损 {d24:>6.1f}")

    # ---- t-SNE 前后对照（齿轮 20-0）----
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color_of = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
    names = list(class_names("gear"))
    panels = [("before（30 通道，无幅值）", dataset_dir(prot, "gear", "20_0"))]
    panels += [(f"{k}（33 通道，+幅值）", v / "gear" / "20_0") for k, v in variants.items()]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (title, d) in zip(axes, panels):
        X = np.load(d / "X_val.npy").astype(np.float32)
        y = np.load(d / "y_val.npy").astype(np.int64)
        Xf = X.reshape(len(X), -1)
        Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
        Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                 random_state=42, max_iter=1000).fit_transform(Xf)
        for ci in range(5):
            m = y == ci
            ax.scatter(Z[m, 0], Z[m, 1], s=12, color=color_of[ci], alpha=0.75,
                       edgecolors="none")
        cen = np.stack([Z[y == ci].mean(0) for ci in range(5)])
        d24 = float(np.linalg.norm(cen[2] - cen[4]))
        for ci in range(5):
            ax.text(cen[ci, 0], cen[ci, 1], cn_name("gear", names[ci]), fontsize=9,
                    color=color_of[ci], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_of[ci], alpha=0.8))
        ax.set_title(f"{title}\n缺齿↔齿面磨损 类心距 = {d24:.1f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("齿轮 20-0：补回幅值信息前后的表示 t-SNE（红色标签=缺齿，紫色=齿面磨损）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PROJECT_ROOT / "reports" / "fig_tsne_energy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
