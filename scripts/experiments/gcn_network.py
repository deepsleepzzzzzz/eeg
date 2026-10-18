"""GCN network-structure figure (advisor: draw the 6-state graph, who connects to whom).

Shows that the "state-transition representation" is exactly a directed, weighted
graph: 6 amplitude states are 6 nodes, and M[i][j] (one-step transition
probability) is the weighted edge i -> j. The GCN then performs message passing
on this graph (node i aggregates its neighbours weighted by the edge weights).

Left: real one-step transition matrix (bearing health, 20 Hz-0 V, ch-mean).
Right: the same matrix drawn as a 6-node directed graph (edge width ~ probability,
self-loops show the dominant "state persistence" on the diagonal).

Pure numpy + matplotlib (no torch). Output: doc/figures/seu/teacher/gcn_network.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"
BINS, STEPS, CH = 6, 7, 3

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_step1(task: str, cond: str, cls_idx: int = 0) -> np.ndarray:
    X = np.load(REP / task / cond / "X_train.npy").astype(np.float32)  # [B,21,6,6]
    y = np.load(REP / task / cond / "y_train.npy").astype(np.int64)
    B = X.shape[0]
    Xr = X.reshape(B, CH, STEPS, BINS, BINS)          # [B,3,7,6,6]
    step1 = Xr[:, :, 0, :, :]                          # [B,3,6,6]
    return step1[y == cls_idx].mean(axis=(0, 1))       # [6,6] ch-mean, class-mean


def draw_graph(ax, M: np.ndarray):
    n = M.shape[0]
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 1.0
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    node_r = 0.14
    # normalizer for edge width/alpha
    offdiag = M[~np.eye(n, dtype=bool)]
    wmax = float(offdiag.max()) if offdiag.size else 1.0
    wmin = float(offdiag.min()) if offdiag.size else 0.0

    # draw nodes
    for i, (x, y) in enumerate(pos):
        c = plt.Circle((x, y), node_r, fc="#eaf2fb", ec="#1f3b63", lw=1.8, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, f"S{i + 1}", ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=4)

    # draw edges (skip self-loops here; self-loops drawn separately)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(M[i, j])
            if w < 1e-4:
                continue
            # normalize
            lw = 0.4 + 4.0 * (w - wmin) / (wmax - wmin + 1e-9)
            alpha = 0.15 + 0.85 * (w - wmin) / (wmax - wmin + 1e-9)
            p0 = pos[i]; p1 = pos[j]
            d = p1 - p0
            L = np.hypot(*d)
            u = d / (L + 1e-9)
            s = p0 + u * node_r
            e = p1 - u * node_r
            # slight curvature to separate opposite directions
            rad = 0.12
            ax.add_patch(FancyArrowPatch(s, e, connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>", mutation_scale=10,
                                         lw=lw, color="#c0392b", alpha=alpha, zorder=2))

    # self-loops: ring around the node whose thickness ~ diagonal probability
    diag = np.diag(M)
    dmax = float(diag.max()) if diag.size else 1.0
    for i in range(n):
        w = float(diag[i])
        lw = 0.6 + 4.5 * (w / (dmax + 1e-9))
        c = plt.Circle(pos[i], node_r * 1.6, fill=False, ec="#1f6f3a",
                       lw=lw, alpha=0.9, zorder=1)
        ax.add_patch(c)

    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    # representative: bearing health (diagonal most obvious)
    M = load_step1("bearing", "20_0", cls_idx=0)
    names = json.loads((REP / "bearing" / "20_0" / "meta.json").read_text(encoding="utf-8"))["class_names"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    ax0, ax1 = axes

    im = ax0.imshow(M, cmap="viridis", vmin=0.0, vmax=float(M.max()))
    ax0.set_xticks(range(BINS)); ax0.set_yticks(range(BINS))
    ax0.set_xticklabels([f"S{i+1}" for i in range(BINS)])
    ax0.set_yticklabels([f"S{i+1}" for i in range(BINS)])
    ax0.set_xlabel("to state (j)", fontsize=10)
    ax0.set_ylabel("from state (i)", fontsize=10)
    ax0.set_title(f"一步状态转移概率矩阵 M[i][j]\n(轴承 {names[0]}，20 Hz-0 V，三轴平均)", fontsize=10.5)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, float(M.max())), cmap="viridis"),
                      ax=ax0, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("转移概率", fontsize=9)

    draw_graph(ax1, M)
    ax1.set_title("同一矩阵作为「复杂网络」\n节点 = 6 个幅值状态；有向边 i→j 的粗细 ∝ 转移概率\n"
                  "（外圈绿环 = 自环，粗细 ∝ 状态维持概率）", fontsize=10.5)

    fig.suptitle("GCN 的图结构：符号状态之间的有向加权转移网络", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "gcn_network.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    main()
