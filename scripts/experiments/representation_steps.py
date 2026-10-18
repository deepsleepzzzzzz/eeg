"""Step-by-step schematic of the state-transition representation layer.

Advisor: "方法流程每一步都要配示意图" — three-channel signal -> windowing ->
symbolization -> state transition, each step with a small picture.

4 panels (left -> right):
  1. raw three-channel signal, one 1600-pt window highlighted
  2. the window (ch0) with 6 quantile (equal-frequency) symbol edges
  3. the resulting symbol sequence A..F as a color strip
  4. the 6x6 one-step transition probability matrix

Pure numpy + matplotlib (no torch). Output: doc/figures/seu/teacher/representation_steps.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"

from data.seu_dds import class_file, cut_windows, load_seu_csv, task_dir, vibration_xyz, window_starts

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BINS = 6
SYM = ["A", "B", "C", "D", "E", "F"]
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]


def transition_matrix(states: np.ndarray, bins: int) -> np.ndarray:
    mat = np.zeros((bins, bins), dtype=np.float64)
    src, dst = states[:-1], states[1:]
    np.add.at(mat, (src, dst), 1.0)
    total = mat.sum()
    return mat / total if total > 0 else mat


def main():
    folder = task_dir(Path("datas/seu"), "bearing")
    raw = load_seu_csv(folder / class_file("bearing", "outer", "20_0"))
    xyz = vibration_xyz(raw)  # [3, T]
    n = int(xyz.shape[1])
    # pick a mid-file window (stable running section, clear impulses)
    all_starts = window_starts(0, n, 1600, 1600)
    start = all_starts[int(len(all_starts) * 0.6)]
    win = xyz[:, start : start + 1600].copy()
    x = win[0]

    # z-score per window/channel (the actual first step of the method), then
    # equal-frequency symbolization on the normalized signal
    x = (x - x.mean()) / (x.std() + 1e-12)

    # quantile edges on the window (equal-frequency symbolization)
    edges = np.quantile(x, np.linspace(0.0, 1.0, BINS + 1)[1:-1])
    states = np.searchsorted(edges, x, side="right").astype(int)
    M = transition_matrix(states, BINS)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7),
                             gridspec_kw={"width_ratios": [1.05, 1.15, 1.0, 0.95]})

    # 1) raw three-channel signal + window
    ax = axes[0]
    seg = xyz[:, start : start + 3000]
    t = np.arange(seg.shape[1]) / 5120.0
    for k, lab in enumerate(["x", "y", "z"]):
        ax.plot(t, seg[k], lw=0.6, alpha=0.8, label=lab)
    ax.axvspan(0, 1600 / 5120.0, color="red", alpha=0.12)
    ax.set_title("① 三通道原始信号（红区 = 取出的窗口）", fontsize=10)
    ax.set_xlabel("时间 (s)", fontsize=8); ax.set_ylabel("幅值", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, ncol=3, loc="upper right")

    # 2) window + symbol edges
    ax = axes[1]
    tt = np.arange(x.size)
    ax.plot(tt, x, lw=0.7, color="#333")
    for e in edges:
        ax.axhline(e, color="#888", lw=0.7, ls="--", alpha=0.7)
    ax.set_title("② 等频符号化：6 条分位边界", fontsize=10)
    ax.set_xlabel("采样点", fontsize=8); ax.set_ylabel("幅值", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(x.min() - 0.2, x.max() + 0.2)

    # 3) symbol sequence color strip (decimated to fit)
    ax = axes[3 - 1]
    step = 10
    seg_s = states[::step][:80]
    ax.imshow(seg_s[None, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(COLORS),
              vmin=0, vmax=BINS - 1, interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, len(seg_s), 20))
    ax.set_xticklabels([str(i * step) for i in range(0, len(seg_s), 20)], fontsize=7)
    ax.set_title("③ 符号序列 A–F（每 10 点取 1）", fontsize=10)
    ax.set_xlabel("采样点", fontsize=8)
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[i], label=SYM[i]) for i in range(BINS)]
    ax.legend(handles=handles, fontsize=7, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.38))

    # 4) transition matrix
    ax = axes[3]
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=float(M.max()))
    ax.set_xticks(range(BINS)); ax.set_yticks(range(BINS))
    ax.set_xticklabels(SYM, fontsize=8); ax.set_yticklabels(SYM, fontsize=8)
    ax.set_xlabel("to state", fontsize=8); ax.set_ylabel("from state", fontsize=8)
    ax.set_title("④ 一步转移概率矩阵 6×6", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle("状态转移表示层：从原始信号到转移矩阵的分步示意", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "representation_steps.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    main()
