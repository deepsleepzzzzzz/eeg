"""Round-3 advisor figures: full-SNR noise curve + symbol-bins ablation (Chinese).

Reads experiment JSONs and produces:
  noise_full_cn.png      -- train-noisy/test-noisy accuracy across SNR 40..-4 dB
  bins_ablation_cn.png   -- test accuracy vs number of symbol states (4..10)
  noise_snr_bars.png     -- (optional) injected-vs-actual SNR verification bars

Outputs to doc/figures/seu/teacher/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
TE = ROOT / "datas" / "experiments_seu" / "teacher_extra"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"bearing_20_0": "#c0392b", "bearing_30_2": "#e67e22",
          "gear_20_0": "#2c5f8a", "gear_30_2": "#2e7d4f"}
CN = {"bearing": "轴承", "gear": "齿轮箱"}


def fig_noise_full():
    d = json.loads((TE / "noise_robustness_full.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, task in zip(axes, ["bearing", "gear"]):
        for r in d:
            if r["task"] != task:
                continue
            key = f"{r['task']}_{r['condition']}"
            xs = [p["snr_db"] for p in r["curve"]]
            ys = [p["acc"] * 100 for p in r["curve"]]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                    color=COLORS[key], label=f"{CN[task]} {r['condition'].replace('_','-')}")
            for x, y in zip(xs, ys):
                if x in (40, 10, 0, -4):
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, -12 if y < 60 else 7), fontsize=7.5,
                                color=COLORS[key], ha="center")
        ax.set_xlabel("信噪比 SNR (dB)", fontsize=10)
        ax.set_ylabel("测试准确率 (%)", fontsize=10)
        ax.set_title(f"{'轴承' if task=='bearing' else '齿轮箱'}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("加噪训练 / 加噪测试：从近干净（40 dB）到强噪声（−4 dB）的完整曲线", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "noise_full_cn.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


def fig_bins():
    d = json.loads((TE / "ablation_bins_zscore.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, task in zip(axes, ["bearing", "gear"]):
        rows = [r for r in d["bins"] if r["task"] == task]
        if not rows:
            continue
        for r in rows:
            key = f"{r['task']}_{r['condition']}"
            xs = [p["bins"] for p in r["curve"]]
            ys = [p["acc"] * 100 for p in r["curve"]]
            ax.plot(xs, ys, marker="o", ms=4.5, lw=1.6, color=COLORS[key],
                    label=f"{r['condition'].replace('_','-')}")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, color=COLORS[key], ha="center")
        ax.set_xlabel("符号状态数（分组数）", fontsize=10)
        ax.set_ylabel("测试准确率 (%)", fontsize=10)
        ax.set_title(CN[task], fontsize=11)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("符号数消融：不同分组数下的测试准确率（6 为本文默认值）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "bins_ablation_cn.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("noise", "all"):
        fig_noise_full()
    if which in ("bins", "all"):
        fig_bins()
