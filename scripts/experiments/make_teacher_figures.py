"""Generate supplementary figures for the teacher's data request.

Reads teacher_extra/*.json and emits:
  - ablation.png   (3 subplots: symbol_bins / transition_steps / window stride)
  - efficiency.png (params + FLOPs, CNN vs GCN)
into doc/figures/seu/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TD = ROOT / "datas/experiments_seu/teacher_extra"
FIG = ROOT / "doc" / "figures" / "seu"


def fig_ablation():
    d = json.loads((TD / "ablation.json").read_text(encoding="utf-8"))
    bins = d["symbol_bins_ablation"]
    steps = d["transition_steps_ablation"]
    win = d["window_stride_ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    xs = [b["symbol_bins"] for b in bins]
    ys = [b["val_acc"] * 100 for b in bins]
    ax.plot(xs, ys, marker="o", color="tab:blue")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("symbol_bins (状态数)"); ax.set_ylabel("val acc (%)")
    ax.set_title("状态数消融 (steps=7 固定)"); ax.grid(alpha=0.3)
    ax.set_xticks(xs)

    ax = axes[1]
    xs = [s["transition_steps"] for s in steps]
    ys = [s["val_acc"] * 100 for s in steps]
    ax.plot(xs, ys, marker="o", color="tab:green")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("transition_steps (步长)"); ax.set_ylabel("val acc (%)")
    ax.set_title("步长消融 (bins=6 固定)"); ax.grid(alpha=0.3)
    ax.set_xticks(xs)

    ax = axes[2]
    labels = [w["window"] for w in win]
    ys = [w["val_acc"] * 100 for w in win]
    xs = range(len(labels))
    ax.bar(xs, ys, color="tab:orange")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 4), fontsize=8, ha="center")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=12, fontsize=8)
    ax.set_ylabel("val acc (%)"); ax.set_title("窗口/跳步消融 (steps=7,bins=6)"); ax.grid(alpha=0.3)

    fig.suptitle("齿轮 30-2 (最难工况) 单变量消融 — 验证集", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=130)
    plt.close(fig)
    print("saved ablation.png")


def fig_efficiency():
    d = json.loads((TD / "efficiency.json").read_text(encoding="utf-8"))
    cnn = [r for r in d if r["arch"] == "cnn"][0]
    gcn = [r for r in d if r["arch"] == "gcn"][0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.bar(["CNN", "GCN"], [cnn["params"], gcn["params"]], color=["tab:blue", "tab:green"])
    ax.set_ylabel("参数量")
    ax.set_title("参数量")
    for i, v in enumerate([cnn["params"], gcn["params"]]):
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax = axes[1]
    ax.bar(["CNN", "GCN"], [cnn["flops_per_sample"], gcn["flops_per_sample"]], color=["tab:blue", "tab:green"])
    ax.set_ylabel("FLOPs (每样本)")
    ax.set_title("计算量")
    for i, v in enumerate([cnn["flops_per_sample"], gcn["flops_per_sample"]]):
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.suptitle("模型轻量化硬指标 (6×6 转移矩阵输入)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "efficiency.png", dpi=130)
    plt.close(fig)
    print("saved efficiency.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_ablation()
    fig_efficiency()
