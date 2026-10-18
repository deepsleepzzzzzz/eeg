# -*- coding: utf-8 -*-
"""重绘「不同工况的状态转移矩阵」——响应导师"色差要明显、多档颜色"的意见。

导师原话（录音）：
  「你看你有黄色有蓝色，黄色代表什么……要把那个颜色条搞出来」
  「这么看不明显，把他们区别搞大一点，色彩搞大一点」
  「你搞七种颜色、十种颜色变化，那同样变化也明显——你就红黄成绿金蓝」

相对上一版（teacher_figs_round2.py::fig_condition_matrices）的改动：
  1. 转移概率矩阵由连续 viridis 改为 **离散 10 档高对比色图（Spectral_r）**，
     统一绝对色标 0 → 全局最大值，让细微差异也能跨档、看得见。
  2. 保留 x/y/z 分轴 + 差值行 + 两处颜色条。

输出（覆盖）：doc/figures/seu/teacher/cond_matrices_{gear,bearing}.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
for _cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    try:
        font_manager.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"
BINS, STEPS, CH = 6, 7, 3
COND_LABEL = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}
CH_LABEL = ["x 轴", "y 轴", "z 轴"]

N_LEVELS = 10
CMAP = ListedColormap(plt.get_cmap("Spectral_r")(np.linspace(0.03, 1.0, N_LEVELS)))


def _class_means(task: str, cond: str):
    X = np.load(REP / task / cond / "X_train.npy").astype(np.float32)     # [B,21,6,6]
    y = np.load(REP / task / cond / "y_train.npy").astype(np.int64)
    meta = json.loads((REP / task / cond / "meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    Xr = X.reshape(len(X), CH, STEPS, BINS, BINS)
    step1 = Xr[:, :, 0, :, :]                                            # [B,3,6,6]
    return names, {names[c]: step1[y == c].mean(axis=0) for c in range(len(names))}


def fig_condition_matrices(task: str) -> None:
    names, m20 = _class_means(task, "20_0")
    _, m30 = _class_means(task, "30_2")
    pretty = {"gear": "齿轮箱", "bearing": "轴承"}[task]

    vmax = max(float(m20[n].max()) for n in names + list(m30))
    vmax = float(max(vmax, 1e-8))
    dmax = max(float(np.abs(m30[n] - m20[n]).max()) for n in names) or 1e-8

    bounds = np.linspace(0.0, vmax, N_LEVELS + 1)
    norm = BoundaryNorm(bounds, CMAP.N)

    fig, axes = plt.subplots(CH * 2 + 1, len(names),
                             figsize=(3.05 * len(names), 2.05 * (CH * 2 + 1)),
                             constrained_layout=True)
    ticks = list(range(BINS))
    tlabels = [str(i + 1) for i in ticks]

    im_top = im_diff = None
    for j, name in enumerate(names):
        a3, b3 = m20[name], m30[name]
        d3 = b3 - a3
        for c in range(CH):
            for i, (mat, tag) in enumerate([(a3[c], COND_LABEL["20_0"]),
                                            (b3[c], COND_LABEL["30_2"])]):
                row = c * 2 + i
                ax = axes[row, j]
                im = ax.imshow(mat, cmap=CMAP, norm=norm)
                im_top = im
                ax.set_xticks(ticks); ax.set_yticks(ticks)
                ax.set_xticklabels(tlabels, fontsize=6.5)
                ax.set_yticklabels(tlabels, fontsize=6.5)
                ax.tick_params(length=0)
                if row == 0:
                    ax.set_title(name, fontsize=10.5)
                if j == 0:
                    ax.set_ylabel(f"{CH_LABEL[c]}·{tag}", fontsize=7.5)
                if row == CH * 2 - 1:
                    ax.set_xlabel("to", fontsize=7)
        # 差值行（发散色，保留连续以体现正负）
        ax = axes[CH * 2, j]
        d = d3.mean(axis=0)
        im = ax.imshow(d, cmap="coolwarm", vmin=-dmax, vmax=dmax)
        im_diff = im
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=6.5)
        ax.set_yticklabels(tlabels, fontsize=6.5)
        ax.tick_params(length=0)
        tvd = float(np.abs(d).sum() / 2.0)
        ax.set_title(f"差值 (30-2 − 20-0)\nTVD={tvd:.3f}", fontsize=9)
        if j == 0:
            ax.set_ylabel("差值\n(三轴均值)", fontsize=8)
        ax.set_xlabel("to", fontsize=7)

    if im_top is not None:
        cb1 = fig.colorbar(im_top, ax=list(axes[: CH * 2, :].ravel()),
                           fraction=0.013, pad=0.012)
        cb1.set_label(f"转移概率（{N_LEVELS} 档）", fontsize=8)
        cb1.set_ticks(np.linspace(0, vmax, 4))
        cb1.ax.set_yticklabels([f"{v:.3f}" for v in np.linspace(0, vmax, 4)],
                               fontsize=6.5)
        cb1.outline.set_visible(False)
    if im_diff is not None:
        cb2 = fig.colorbar(im_diff, ax=list(axes[CH * 2, :].ravel()),
                           fraction=0.013, pad=0.012)
        cb2.set_label("概率差\n(红=30-2 高\n蓝=20-0 高)", fontsize=7.5)
        cb2.set_ticks(np.linspace(-dmax, dmax, 5))
        cb2.ax.set_yticklabels([f"{v:.3f}" for v in np.linspace(-dmax, dmax, 5)],
                               fontsize=6.5)
        cb2.outline.set_visible(False)

    fig.suptitle(f"SEU DDS {pretty}：两工况下各类的一步状态转移矩阵"
                 f"（x/y/z 分轴 + {N_LEVELS} 档色标 + 差值）", fontsize=13)
    out = FIG / f"cond_matrices_{task}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    for t in ("gear", "bearing"):
        fig_condition_matrices(t)
