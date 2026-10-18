# -*- coding: utf-8 -*-
"""导师第 3 轮意见：重绘「方法总流程图」。

两点诉求（导师原话）：
  ① 提高美感、紧凑一点（原图卡片太大太浮夸）；
  ② 每一步都要配示意图举例——分窗画框、符号化用 A–F 等分、状态转移配矩阵、
     噪声增广配波形对比……「别人一看就知道这一步干了啥」。

产出：doc/figures/seu/teacher/flowchart_v2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEU = ROOT / "datas" / "seu" / "gearbox"
REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"

FS = 5120


# --------------------------------------------------------------------------- #
# 真实素材
# --------------------------------------------------------------------------- #
def _read_signal(path: Path, n: int = 4000, cols=(1, 2, 3)) -> np.ndarray:
    start, delim = None, "\t"
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.lstrip()
            if s and (s[0].isdigit() or s[0] in "+-."):
                start = i
                delim = "\t" if s.count("\t") >= 7 else ","
                break
    if start is None:
        return np.empty((0, 3))
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(rows) >= n:
                break
            parts = line.strip().split(delim)
            if len(parts) < 8:
                continue
            try:
                rows.append([float(parts[c]) for c in cols])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


SIG = _read_signal(SEU / "gearset" / "Health_20_0.csv", n=4200)   # [N,3]
SIG = SIG[200:]                                                   # 去掉起始毛刺
_names, _X = None, None
_m = json.loads((REP / "gear" / "20_0" / "meta.json").read_text(encoding="utf-8"))
CLASSES = _m["class_names"]
_XT = np.load(REP / "gear" / "20_0" / "X_train.npy").astype(np.float32)   # [B,21,6,6]
MAT = _XT[0].reshape(3, 7, 6, 6)[0, 0]                            # 一步转移矩阵示例


# --------------------------------------------------------------------------- #
# 绘图基元
# --------------------------------------------------------------------------- #
CARD_STYLE = [
    ("#eaf1fb", "#3b7ddd"),   # 1 原始信号
    ("#eaf1fb", "#3b7ddd"),   # 2 分窗
    ("#e9f5ec", "#2f9e63"),   # 3 符号化
    ("#e9f5ec", "#2f9e63"),   # 4 状态转移
    ("#fdf3e2", "#e8930c"),   # 5 张量
    ("#fdf3e2", "#e8930c"),   # 6 模型
    ("#f3ebfb", "#8e5bd6"),   # 7 噪声增广
    ("#fdeceb", "#d9534f"),   # 8 分类
]


def mini(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


# --- 示例 1：三轴信号 ------------------------------------------------------ #
def ex_signal(fig, rect):
    ax = mini(fig, rect)
    v = SIG[:1800]
    t = np.arange(len(v)) / FS
    for k, c in enumerate(["#3b7ddd", "#e8930c", "#2f9e63"]):
        ax.plot(t, v[:, k], lw=0.6, color=c, alpha=0.9)
    ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("时间 (s)", fontsize=6.5, labelpad=0.5)
    ax.text(0.02, 0.86, "x / y / z", transform=ax.transAxes, fontsize=6.5,
            color="#555", va="top")


# --- 示例 2：分窗 + z-score ------------------------------------------------ #
def ex_window(fig, rect):
    ax = mini(fig, rect)
    v = SIG[:2600, 0]
    ax.plot(v, lw=0.6, color="#3b7ddd")
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    for x0 in (120, 920, 1720):
        ax.add_patch(Rectangle((x0, lo), 1600, rng, facecolor="#e8930c",
                               alpha=0.16, edgecolor="#e8930c", lw=0.8))
    ax.set_xlim(0, len(v)); ax.set_ylim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.9, "窗长1600 · 步长800", transform=ax.transAxes,
            fontsize=6.5, color="#555", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


# --- 示例 3：等频符号化 A–F ------------------------------------------------ #
def ex_symbol(fig, rect):
    ax = mini(fig, rect)
    v = SIG[1000:1900, 0]
    n = len(v)
    ax.plot(v, lw=0.7, color="#2b2b2b")
    edges = np.quantile(v, np.linspace(0, 1, 7))
    ax.set_xlim(0, n); ax.set_xticks([]); ax.set_yticks([])
    labels = "ABCDEF"
    for j in range(6):
        y0, y1 = edges[j], edges[j + 1]
        ax.add_patch(Rectangle((0, y0), n, y1 - y0, facecolor=plt.cm.tab10(j),
                               alpha=0.13, lw=0))
        if j > 0:
            ax.axhline(y0, ls="--", lw=0.6, color="#999")
    for j in range(6):
        yc = (edges[j] + edges[j + 1]) / 2
        ax.text(n * 0.975, yc, labels[j], fontsize=6.8, va="center", ha="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.14", fc=plt.cm.tab10(j),
                          ec="none", alpha=0.9))
    ax.set_xlim(0, n)
    ax.text(0.02, 0.92, "等频分位 → A–F", transform=ax.transAxes,
            fontsize=6.5, color="#555", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


# --- 示例 4：一步转移矩阵 -------------------------------------------------- #
def ex_matrix(fig, rect):
    ax = mini(fig, rect)
    im = ax.imshow(MAT, cmap="viridis", vmin=0, vmax=float(MAT.max()))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels("ABCDEF", fontsize=6)
    ax.set_yticklabels("ABCDEF", fontsize=6)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)
    ax.set_title("一步转移概率 P(i→j)", fontsize=6.5, pad=2, color="#444")


# --- 示例 5：多通道堆叠张量 ------------------------------------------------ #
def ex_tensor(fig, rect):
    ax = mini(fig, rect)
    ax.set_xlim(0, 9.2); ax.set_ylim(0, 3.7)
    ch_color = ["#3b7ddd", "#e8930c", "#2f9e63"]
    for r in range(3):
        for c in range(7):
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88,
                                   facecolor=ch_color[r], alpha=0.18 + 0.03 * r,
                                   edgecolor=ch_color[r], lw=0.6))
    ax.text(7.35, 2.5, "3 通道", fontsize=6.5, color="#555", va="center")
    ax.text(3.5, -0.05, "× 7 时间步  →  张量 [21, 6, 6]", fontsize=6.5,
            color="#555", ha="center", va="top")


# --- 示例 6：CNN / GCN ----------------------------------------------------- #
def ex_model(fig, rect):
    ax = mini(fig, rect)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    # left: CNN 卷积块
    for i, (w, h, x) in enumerate([(1.0, 2.6, 0.4), (0.8, 2.0, 1.9), (0.6, 1.4, 3.1)]):
        ax.add_patch(Rectangle((x, 2 - h / 2), w, h, facecolor="#e8930c",
                               alpha=0.25, edgecolor="#e8930c", lw=0.8))
    ax.text(2.2, 0.35, "CNN 局部卷积", fontsize=6.5, color="#666", ha="center")
    # right: GCN 6-node ring
    cx, cy, R = 8.0, 2.0, 1.2
    pts = [(cx + R * np.cos(a), cy + R * np.sin(a))
           for a in np.linspace(0.5 * np.pi, 2.5 * np.pi, 6, endpoint=False)]
    for i in range(6):
        for j in range(6):
            if i != j and (i + j) % 3 == 0:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        lw=0.5, color="#8e5bd6", alpha=0.4)
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), 0.28, facecolor="#f3ebfb",
                            edgecolor="#8e5bd6", lw=0.9))
    ax.text(8.0, 0.35, "GCN 图传播", fontsize=6.5, color="#666", ha="center")


# --- 示例 7：训练阶段噪声增广 ---------------------------------------------- #
def ex_noise(fig, rect):
    ax = mini(fig, rect)
    v = SIG[:1600, 0]
    seg = v - v.mean()
    snr = 0.0
    noise = np.random.default_rng(0).normal(0, seg.std() * 10 ** (-snr / 20), seg.shape[0])
    vn = seg + noise
    t = np.arange(len(seg)) / FS * 1000
    ax.plot(t, vn, lw=0.5, color="#8e5bd6", alpha=0.55)
    ax.plot(t, seg, lw=0.8, color="#333")
    ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("时间 (ms)", fontsize=6.5, labelpad=0.5)
    ax.text(0.02, 0.9, "干净信号 + 白噪声(多SNR)", transform=ax.transAxes,
            fontsize=6.5, color="#555", va="top")


# --- 示例 8：故障分类 ------------------------------------------------------ #
CN = {"Health": "正常", "Chipped": "齿面剥落", "Miss": "缺齿",
      "Root": "齿根裂纹", "Surface": "齿面磨损"}


def ex_class(fig, rect):
    ax = mini(fig, rect)
    vals = [99.5, 98.7, 99.1, 97.8, 98.3]
    cols = ["#2f9e63", "#d9534f", "#d9534f", "#d9534f", "#d9534f"]
    y = np.arange(5)[::-1]
    ax.barh(y, vals, color=cols, alpha=0.85, height=0.68)
    ax.set_xlim(88, 100.8); ax.set_ylim(-0.6, 4.6)
    ax.set_xticks([]); ax.set_yticks([])
    for yy, vv, c in zip(y, vals, CLASSES):
        ax.text(88.5, yy, CN[c], fontsize=6.4, va="center", ha="left", color="white")
        ax.text(vv - 0.2, yy, f"{vv:.1f}", fontsize=6.2, va="center", ha="right",
                color="white")
    ax.set_title("单工况 5 类准确率 (%)", fontsize=6.8, pad=2, color="#444")


# --------------------------------------------------------------------------- #
# 主图
# --------------------------------------------------------------------------- #
def fig_flowchart_v2():
    steps = [
        ("原始振动信号", "三轴通道，5120 Hz；每类 104 万点（≈3.4 min）", ex_signal),
        ("分窗 + 逐窗 z-score", "窗长 1600、步长 800（50% 重叠）；逐窗标准化", ex_window),
        ("等频符号化（6 状态）", "按分位数把幅值划成 A–F 六档，边界仅由训练集拟合", ex_symbol),
        ("多时间步状态转移", "步长 1→7；统计状态 i→j 的转移概率，得 6×6 矩阵", ex_matrix),
        ("多通道堆叠张量", "3 个振动通道 × 7 个时间步", ex_tensor),
        ("CNN / GCN 特征学习", "CNN 学矩阵局部模式；GCN 按图结构传播聚合", ex_model),
        ("训练阶段噪声增广（抗噪）", "训练信号注入高斯白噪声后重新编码", ex_noise),
        ("故障分类（单工况 5 类）", "正常 + 4 类故障，两工况各自独立建模", ex_class),
    ]

    fig = plt.figure(figsize=(9.6, 10.8))
    fig.patch.set_facecolor("white")

    top, bottom = 0.945, 0.012
    n = len(steps)
    h = (top - bottom) / n
    card_x, card_w = 0.017, 0.585
    ex_x, ex_w = 0.625, 0.362
    cx = card_x + card_w / 2

    fig.text(0.5, 0.978, "多时间步状态转移网络 + CNN / GCN 方法总流程",
             ha="center", va="center", fontsize=14, fontweight="bold", color="#1f2d3d")
    fig.text(0.5, 0.958, "每一步右侧配真实数据示例", ha="center", va="center",
             fontsize=9, color="#8a94a6")

    for i, (title, sub, exfn) in enumerate(steps):
        y_top = top - i * h
        y_bot = top - (i + 1) * h
        ch = h * 0.82
        cy0 = y_bot + h * 0.09
        fc, ac = CARD_STYLE[i]

        fig.add_artist(FancyBboxPatch((card_x, cy0), card_w, ch,
                                      boxstyle="round,pad=0.002,rounding_size=0.010",
                                      linewidth=0.8, edgecolor="#d3dce8",
                                      facecolor=fc, transform=fig.transFigure,
                                      zorder=1))
        fig.add_artist(FancyBboxPatch((card_x + 0.005, cy0 + 0.007), 0.0065, ch - 0.014,
                                      boxstyle="round,pad=0,rounding_size=0.003",
                                      linewidth=0, facecolor=ac,
                                      transform=fig.transFigure, zorder=2))
        # step number badge
        fig.add_artist(Circle((card_x + 0.032, cy0 + ch * 0.5), 0.0125,
                              transform=fig.transFigure, facecolor=ac,
                              edgecolor="white", linewidth=1.2, zorder=3))
        fig.text(card_x + 0.032, cy0 + ch * 0.5, str(i + 1), ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white", zorder=4)
        fig.text(card_x + 0.052, cy0 + ch * 0.62, title, ha="left", va="center",
                 fontsize=10.5, fontweight="bold", color="#1f2d3d")
        fig.text(card_x + 0.052, cy0 + ch * 0.26, sub, ha="left", va="center",
                 fontsize=7.6, color="#5b6472")

        exfn(fig, [ex_x, cy0, ex_w, ch])

        if i < n - 1:
            fig.add_artist(FancyArrowPatch((cx, cy0), (cx, y_bot - h * 0.015),
                                           arrowstyle="-|>", mutation_scale=11,
                                           lw=1.3, color="#9aa7b8",
                                           transform=fig.transFigure, zorder=1))

    out = FIG / "flowchart_v2.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_flowchart_v2()
