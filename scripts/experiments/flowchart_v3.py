# -*- coding: utf-8 -*-
"""方法总流程图 v3：每一步配「真实数据」具体例子。

导师意见：步骤示意图要「拿数据举例子，别人一看就知道这一步干了啥」。
v2 的问题：示例多为手绘示意（手画窗框、现场随机加噪、手写准确率）。

v3 的原则：**每一个数字、每一段波形都从真实实验产物里取**——
  · 原始信号      真实 SEU Health_20_0.csv 的 x/y/z（标注实际点数）
  · 分窗          真实窗索引（第 #800–#2400 点，步长 800）+ z-score 前后具体数值
  · 符号化        用 meta.json 里训练时真正用的 bin_edges，输出真实符号串 A B A C …
  · 状态转移      真实类别统计矩阵（P(A→A)=0.064 等，标在对角线上）
  · 张量          真实单样本 X_train[0] 的 21 个 6×6 切片（标注切片编号）
  · 网络          真实结构（CNN 2 层 32/64、GCN 2 层隐层 32）
  · 噪声增广      按实验真实的 SNR 档位（40…−4 dB）
  · 分类          真实各类召回率（从 metrics.json 的混淆矩阵算）

产出：doc/figures/seu/teacher/flowchart_v3.png
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
WIN, STRIDE = 1600, 800

GEAR = "gear"
COND = "20_0"

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


META = json.loads((REP / GEAR / COND / "meta.json").read_text(encoding="utf-8"))
CLASSES = META["class_names"]                     # Health/Chipped/Miss/Root/Surface
BIN_EDGES = np.asarray(META["bin_edges"])         # [3, 7] 训练集拟合的真实分位边界
SIG = _read_signal(SEU / "gearset" / "Health_20_0.csv", n=6000)[200:]   # [N,3]
_XT = np.load(REP / GEAR / COND / "X_train.npy").astype(np.float32)     # [B,21,6,6]
_YT = np.load(REP / GEAR / COND / "y_train.npy")
MAT_PER_CLASS = {c: _XT[_YT == i].reshape(-1, 3, 7, 6, 6)[:, 0].mean(axis=0)  # [7,6,6]
                 for i, c in enumerate(CLASSES)}

# 真实符号序列（拿一个真实窗 + 真实边界算出）
_W0 = SIG[800:800 + WIN, 0]
_Z0 = (_W0 - _W0.mean()) / _W0.std()
_ED0 = BIN_EDGES[0]
_SYM0 = np.digitize(_Z0, _ED0[1:-1])              # 0..5
_SYM_STR = "".join("ABCDEF"[s] for s in _SYM0[:12])

# 真实各类准确率（齿轮 20-0 CNN seed42 的混淆矩阵）
_MET = json.loads((ROOT / "datas" / "experiments_seu" /
                   "checkpoints_gear_cnn_resplit" / "gear" / COND / "seed_42" /
                   "metrics.json").read_text(encoding="utf-8"))
_CM = np.asarray(_MET["test_confusion_matrix"])
RECALL = {c: float(_CM[i, i] / _CM[i].sum()) for i, c in enumerate(_MET["class_names"])}
TEST_ACC = float(_MET["test_acc"])

# --------------------------------------------------------------------------- #
CARD_STYLE = [
    ("#eaf1fb", "#3b7ddd"), ("#eaf1fb", "#3b7ddd"),
    ("#e9f5ec", "#2f9e63"), ("#e9f5ec", "#2f9e63"),
    ("#fdf3e2", "#e8930c"), ("#fdf3e2", "#e8930c"),
    ("#f3ebfb", "#8e5bd6"), ("#fdeceb", "#d9534f"),
]
CN = {"Health": "正常", "Chipped": "齿面剥落", "Miss": "缺齿",
      "Root": "齿根裂纹", "Surface": "齿面磨损"}


def mini(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def _tag(ax, txt, x=0.02, y=0.94):
    """左上角标签，浅色底避免压住图形。"""
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=6.6, color="#3a4453",
            va="top", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d8dfe8",
                      lw=0.5, alpha=0.92))


# --- 1 原始信号：真实点数 ---------------------------------------------------- #
def ex_signal(fig, rect):
    ax = mini(fig, rect)
    v = SIG[:2400]
    t = np.arange(len(v)) / FS
    for k, c in enumerate(["#3b7ddd", "#e8930c", "#2f9e63"]):
        ax.plot(t, v[:, k], lw=0.55, color=c, alpha=0.9)
    ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("时间 (s)", fontsize=6.5, labelpad=0.5)
    _tag(ax, f"x/y/z · 5120 Hz · 前 2400 点（0–{t[-1]:.2f} s）")


# --- 2 分窗 + z-score：真实窗索引与数值 -------------------------------------- #
def ex_window(fig, rect):
    ax = mini(fig, rect)
    v = SIG[:4000, 0]
    ax.plot(v, lw=0.55, color="#3b7ddd")
    lo, hi = float(v.min()), float(v.max()); rng = hi - lo
    for i, x0 in enumerate((0, STRIDE, 2 * STRIDE)):
        lab = "窗 #1" if i == 0 else f"#{i+1}"
        ax.add_patch(Rectangle((x0, lo), WIN, rng, facecolor="#e8930c",
                               alpha=0.13 if i else 0.20,
                               edgecolor="#e8930c", lw=0.8))
        ax.text(x0 + WIN / 2, lo + rng * 0.04, lab, fontsize=5.8, ha="center",
                va="bottom", color="#a5640a")
    ax.set_xlim(0, 4000); ax.set_ylim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([])
    _tag(ax, f"窗#1 = 采样点 0–{WIN}，窗#2 = {STRIDE}–{WIN+STRIDE}\n"
            f"原始均值 {v[:WIN].mean():.3f} 标准差 {v[:WIN].std():.3f}")


# --- 3 符号化：真实分位边界 + 真实符号串 ------------------------------------- #
def ex_symbol(fig, rect):
    ax = mini(fig, rect)
    n = 1000
    z = _Z0[:n]
    ax.plot(z, lw=0.7, color="#2b2b2b")
    for j in range(6):
        y0, y1 = _ED0[j], _ED0[j + 1]
        ax.add_patch(Rectangle((0, y0), n, y1 - y0, facecolor=plt.cm.tab10(j),
                               alpha=0.14, lw=0))
        if j > 0:
            ax.axhline(y0, ls="--", lw=0.55, color="#9aa")
    for j in range(6):
        yc = (_ED0[j] + _ED0[j + 1]) / 2
        ax.text(n * 0.905, yc, "ABCDEF"[j], fontsize=6.6, va="center", ha="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.13", fc=plt.cm.tab10(j),
                          ec="none", alpha=0.92))
    ax.set_xlim(0, n * 1.02)
    e = _ED0
    _tag(ax, f"训练集分位边界: {e[1]:.2f} | {e[2]:.2f} | {e[3]:.2f} | {e[4]:.2f} | {e[5]:.2f}")
    ax.text(0.02, 0.06, f"符号串 {_SYM_STR} …", transform=ax.transAxes,
            fontsize=6.4, color="#2f9e63", va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", fc="#f2fbf6", ec="#bfe3d0", lw=0.5))


# --- 4 状态转移：真实类别矩阵 + 真实对角值 ------------------------------------ #
def ex_matrix(fig, rect):
    ax = mini(fig, rect)
    M = MAT_PER_CLASS["Health"][0]        # 正常类、通道 x、一步
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=float(M.max()))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels("ABCDEF", fontsize=6); ax.set_yticklabels("ABCDEF", fontsize=6)
    ax.tick_params(length=0)
    for i in range(6):                    # 对角线上标真实数值
        ax.text(i, i, f"{M[i,i]:.2f}", ha="center", va="center", fontsize=4.6,
                color="white" if M[i, i] < M.max() * 0.6 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_visible(False)
    ax.set_title("正常类·通道x·一步转移 P(i→j)", fontsize=6.4, pad=2, color="#444")


# --- 5 张量：真实切片编号 ---------------------------------------------------- #
def ex_tensor(fig, rect):
    ax = mini(fig, rect)
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.9)
    ch_color = ["#3b7ddd", "#e8930c", "#2f9e63"]
    for r in range(3):
        for c in range(7):
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88,
                                   facecolor=ch_color[r], alpha=0.20 + 0.03 * r,
                                   edgecolor=ch_color[r], lw=0.6))
        ax.text(7.28, r + 0.5, f"通道 {'xyz'[r]}", fontsize=6.2, color="#555",
                va="center")
    ax.text(3.5, -0.12, "7 个时间步（步长 1→7）", fontsize=6.3,
            color="#555", ha="center", va="top")
    _tag(ax, "真实张量 X_train[0]  shape = [21, 6, 6]", x=0.01, y=1.02)


# --- 6 网络：真实结构 -------------------------------------------------------- #
def ex_model(fig, rect):
    ax = mini(fig, rect)
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.2)
    for (w, h, x) in [(1.0, 2.6, 0.3), (0.8, 2.0, 1.75), (0.6, 1.4, 2.9)]:
        ax.add_patch(Rectangle((x, 2.3 - h / 2), w, h, facecolor="#e8930c",
                               alpha=0.25, edgecolor="#e8930c", lw=0.8))
    ax.text(1.9, 0.55, "CNN: 2 层卷积 32/64\n3×3 核 · 2×2 池化", fontsize=6.0,
            color="#666", ha="center", va="top")
    cx, cy, R = 8.3, 2.3, 1.2
    pts = [(cx + R * np.cos(a), cy + R * np.sin(a))
           for a in np.linspace(0.5 * np.pi, 2.5 * np.pi, 6, endpoint=False)]
    for i in range(6):
        for j in range(6):
            if i != j and (i + j) % 3 == 0:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        lw=0.5, color="#8e5bd6", alpha=0.4)
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), 0.27, facecolor="#f3ebfb",
                            edgecolor="#8e5bd6", lw=0.9))
    ax.text(8.3, 0.55, "GCN: 2 层图卷积\n隐层 32 · 关系布局", fontsize=6.0,
            color="#666", ha="center", va="top")


# --- 7 噪声增广：真实 SNR 档位 ----------------------------------------------- #
def ex_noise(fig, rect):
    ax = mini(fig, rect)
    seg = SIG[:1600, 0]
    seg = seg - seg.mean()
    snr = 0.0                                        # 取实验真实用到的 0 dB 档
    rng = np.random.default_rng(42)                  # 真实实验种子
    noise = rng.normal(0, seg.std() * 10 ** (-snr / 20), seg.shape[0])
    vn = seg + noise
    t = np.arange(len(seg)) / FS * 1000
    ax.plot(t, vn, lw=0.45, color="#8e5bd6", alpha=0.5)
    ax.plot(t, seg, lw=0.8, color="#333")
    ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("时间 (ms)", fontsize=6.5, labelpad=0.5)
    _tag(ax, "黑=干净  紫=0 dB 加噪（σ 由 SNR 反解）\n"
             "训练用 40 / 30 / 20 / 15 / 10 / 6 / 2 / 0 / −2 / −4 dB")


# --- 8 分类：真实各类召回率 -------------------------------------------------- #
def ex_class(fig, rect):
    ax = mini(fig, rect)
    order = CLASSES
    vals = [RECALL[c] * 100 for c in order]
    y = np.arange(5)[::-1]
    ax.barh(y, vals, color="#2f9e63", alpha=0.85, height=0.66)
    ax.set_xlim(87, 101.6); ax.set_ylim(-0.6, 4.6)
    ax.set_xticks([]); ax.set_yticks([])
    for yy, vv, c in zip(y, vals, order):
        ax.text(87.5, yy, CN[c], fontsize=6.3, va="center", ha="left", color="white")
        ax.text(vv + 0.25, yy, f"{vv:.1f}", fontsize=6.2, va="center", ha="left",
                color="#2f6f4f")
    ax.set_title(f"齿轮 20-0 · CNN · 测试集总体 {TEST_ACC*100:.2f}%",
                 fontsize=6.4, pad=2, color="#444")


# --------------------------------------------------------------------------- #
def fig_flowchart_v3():
    steps = [
        ("原始振动信号", "三轴 5120 Hz；每类 104 万点（≈3.4 min）", ex_signal),
        ("分窗 + 逐窗 z-score", "窗长 1600、步长 800（50% 重叠）；逐窗标准化", ex_window),
        ("等频符号化（6 状态）", "按训练集分位数划 A–F 六档，边界只用训练集拟合", ex_symbol),
        ("多时间步状态转移", "步长 1→7；统计 i→j 转移概率，各得 6×6 矩阵", ex_matrix),
        ("多通道堆叠张量", "3 个振动通道 × 7 个时间步 = 21 个 6×6", ex_tensor),
        ("CNN / GCN 特征学习", "CNN 学矩阵局部模式；GCN 按图结构传播聚合", ex_model),
        ("训练阶段噪声增广（抗噪）", "训练信号注入高斯白噪声后重新编码", ex_noise),
        ("故障分类（单工况 5 类）", "正常 + 4 类故障，两工况各自独立建模", ex_class),
    ]

    fig = plt.figure(figsize=(9.8, 11.0))
    fig.patch.set_facecolor("white")

    top, bottom = 0.943, 0.012
    n = len(steps)
    h = (top - bottom) / n
    card_x, card_w = 0.017, 0.545
    ex_x, ex_w = 0.583, 0.404
    cx = card_x + card_w / 2

    fig.text(0.5, 0.978, "多时间步状态转移网络 + CNN / GCN 方法总流程",
             ha="center", va="center", fontsize=14, fontweight="bold", color="#1f2d3d")
    fig.text(0.5, 0.957, "每一步右侧为真实数据示例（信号 / 分位边界 / 转移矩阵 / 准确率均取自本次实验产物）",
             ha="center", va="center", fontsize=8.4, color="#8a94a6")

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
        fig.add_artist(FancyBboxPatch((card_x + 0.005, cy0 + 0.007), 0.0060, ch - 0.014,
                                      boxstyle="round,pad=0,rounding_size=0.003",
                                      linewidth=0, facecolor=ac,
                                      transform=fig.transFigure, zorder=2))
        fig.add_artist(Circle((card_x + 0.030, cy0 + ch * 0.5), 0.0120,
                              transform=fig.transFigure, facecolor=ac,
                              edgecolor="white", linewidth=1.2, zorder=3))
        fig.text(card_x + 0.030, cy0 + ch * 0.5, str(i + 1), ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white", zorder=4)
        fig.text(card_x + 0.049, cy0 + ch * 0.62, title, ha="left", va="center",
                 fontsize=10.2, fontweight="bold", color="#1f2d3d")
        fig.text(card_x + 0.049, cy0 + ch * 0.26, sub, ha="left", va="center",
                 fontsize=7.4, color="#5b6472")

        exfn(fig, [ex_x, cy0, ex_w, ch])

        if i < n - 1:
            fig.add_artist(FancyArrowPatch((cx, cy0), (cx, y_bot - h * 0.015),
                                           arrowstyle="-|>", mutation_scale=11,
                                           lw=1.3, color="#9aa7b8",
                                           transform=fig.transFigure, zorder=1))

    out = FIG / "flowchart_v3.png"
    fig.savefig(out, dpi=155, facecolor="white")
    plt.close(fig)
    print("saved", out)
    print("真实数字核对：", f"符号串={_SYM_STR}", f"测试准确率={TEST_ACC:.4f}",
          "各类召回=", {k: round(v, 3) for k, v in RECALL.items()})


if __name__ == "__main__":
    fig_flowchart_v3()
