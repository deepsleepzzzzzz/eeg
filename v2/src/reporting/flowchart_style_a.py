# -*- coding: utf-8 -*-
"""方法总流程图 —— 文献风格版（元素 + 粗体标签 + 细箭头，横向回流布局）。

背景
----
2026-10-09 用户贴来一篇 EEG 癫痫检测论文（Shankar et al., BSPC 2021）的方法总流程图，
要求「把我的流程图改成这种风格」。经确认：
  · 布局 = 两行回流 3+3 + 底部输出收口（第 1–3 步 左→右；第 4–6 步 右→左；
    末行原为「7 噪声增广 + 8 故障分类」，2026-10-10 按用户要求改成一块**输出收口**
    `ex_result()`：虚线外框内并排 5 个类别标签框，照参考文献末端的写法）；
  · 配色 = 保留彩色（热图 / 柱状图 / 三轴波形本身带颜色，去掉会丢信息）。

与 `flowchart_v3_clean.py` 的关系
--------------------------------
本脚本是**新版式**，不改动 `flowchart_v3_clean.py` 与其产物。
数据加载与 8 个绘图函数（ex_*）照搬，仅：
  · 画布由竖版 9.8×11.0 改为横版 15.0×9.0；
  · 去掉左侧「圆角卡片 + 彩色序号圆」，改为**每个元素上方一行粗体标题**；
    （原来标题下还有一行灰色小字说明，按用户 2026-10-10 要求已全部撤下，
      文案仍保留在 `steps` 元组的第 2 项里，可直接挪作图注。）
  · 8 个元素按回流路径排布，元素之间用**细箭头**连接（行内横向、行间竖向）；
  · ex_matrix / ex_tensor 因新格宽高比不同，几何重算（矩阵改为正方形居中；张量按宽度自适应缩放）；
  · 去掉 ex_matrix / ex_class 自带的 set_title（信息已移到标题下方的灰色小字）。

产出：doc/figures/seu/teacher/flowchart_style_a.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# 素材全部来自**本项目自己**的目录：
#   data/seu/gearbox/...                      原始 CSV（已核验与官方仓库 blob sha 一致）
#   artifacts/<主协议>/dataset/gear/20_0/     主协议 v5 的表示缓存（滞后 1-10）
#   artifacts/<主协议>/gear_cnn_20_0_seed42/  运行指标（验证集，测试集封存）
# 此前这份脚本读的是旧项目缓存，其滞后数是 7，与主协议 v5 的 1-10 不一致，
# 画出来会与协议打架（图注写 7 个时间步、协议写 10 个）。
OUT_ROOT = Path(__file__).resolve().parents[2]   # 项目根
SRC_ROOT = OUT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.protocol import dataset_dir, load_protocol, window_cfg  # noqa: E402
from reporting.class_labels import GEAR_CN as CN  # noqa: E402  # 类别中文名唯一真源

FIG = OUT_ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)


def _main_protocol() -> dict:
    """取 status: main 的主协议（不写死文件名）。"""
    for f in sorted((OUT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GEAR = "gear"
COND = "20_0"

PROTOCOL = _main_protocol()                 # 主协议 v5（唯一真源）
SEU = OUT_ROOT / "data" / "seu" / "gearbox"
DS = dataset_dir(PROTOCOL, GEAR, COND)      # 表示缓存（滞后数由协议决定）
RUN = OUT_ROOT / "artifacts" / str(PROTOCOL["name"]) / f"{GEAR}_cnn_{COND}_seed42"
FS = 5120
WIN, STRIDE = window_cfg(PROTOCOL)
LAGS = [int(x) for x in PROTOCOL["representation"]["lags"]]
S = len(LAGS)                               # 滞后个数（v5 = 10）
FW, FH = 13.2, 8.0                          # 横版画布（英寸）


# --------------------------------------------------------------------------- #
# 真实素材（与 flowchart_v3_clean.py 一致）
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


META = json.loads((DS / "meta.json").read_text(encoding="utf-8"))
CLASSES = META["class_names"]
BIN_EDGES = np.asarray(META["bin_edges"])
SIG = _read_signal(SEU / "gearset" / "Health_20_0.csv", n=6000)[200:]
_XT = np.load(DS / "X_train.npy").astype(np.float32)
_YT = np.load(DS / "y_train.npy")
MAT_PER_CLASS = {c: _XT[_YT == i].reshape(-1, 3, S, 6, 6)[:, 0].mean(axis=0)
                 for i, c in enumerate(CLASSES)}

_W0 = SIG[800:800 + WIN, 0]
_Z0 = (_W0 - _W0.mean()) / _W0.std()
_ED0 = BIN_EDGES[0]
_SYM0 = np.digitize(_Z0, _ED0[1:-1])
_SYM_STR = "".join("ABCDEF"[s] for s in _SYM0[:12])

_MET = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
VM = _MET["val_metrics"]                   # 验证集：测试集封存，只用验证集
_CM = np.asarray(VM["confusion_matrix"])
RECALL = {c: float(_CM[i, i] / _CM[i].sum()) for i, c in enumerate(CLASSES)}
VAL_ACC = float(VM["accuracy"])

# 类别中文名统一从 reporting.class_labels 取（见该模块说明）。


def mini(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


# --- 1 原始信号：三行堆叠子图（共用纵轴刻度，只底行给时间轴）----------------- #
def ex_signal(fig, rect):
    v = SIG[:2400]
    t = np.arange(len(v)) / FS
    x0, y0, w, h = rect
    lx = w * 0.014
    top = h * 0.055
    bot = h * 0.035
    band = h - top - bot
    gap = band * 0.045
    rh = (band - 2 * gap) / 3.0
    cols = ["#3b7ddd", "#e8930c", "#2f9e63"]
    ymax = float(np.abs(v).max()) * 1.10
    for k in range(3):
        sub = fig.add_axes([x0 + lx, y0 + bot + (2 - k) * (rh + gap), w - lx, rh])
        sub.plot(t, v[:, k], lw=0.55, color=cols[k])
        sub.set_xlim(0, t[-1]); sub.set_ylim(-ymax, ymax)
        sub.set_xticks([]); sub.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            sub.spines[s].set_visible(False)
    return x0 + lx, x0 + w          # 内容左右边界（figure 坐标）→ 供流程箭头对齐


# --- 2 / 3 步专用：示意波形 -------------------------------------------------- #
def _sch_master(n=2048):
    rng = np.random.RandomState(7)
    u = np.linspace(0.0, 1.0, n)
    y = 0.10 * (0.55 * np.sin(2 * np.pi * 9.0 * u + 0.5)
                + 0.30 * np.sin(2 * np.pi * 16.0 * u + 1.4)
                + 0.15 * np.sin(2 * np.pi * 27.0 * u + 2.3))
    y += 0.030 * rng.randn(n)
    for i, c in enumerate((0.06, 0.25, 0.44, 0.63, 0.82)):
        m = u >= c
        d = u[m] - c
        amp = 0.78 + 0.22 * np.sin(2 * np.pi * 1.3 * u[m] + i)
        y[m] += amp * np.exp(-d / 0.045) * np.sin(2 * np.pi * 28.0 * d)
    return u, y / np.abs(y).max()


def _sch_window():
    u, y = _sch_master()
    seg = y[u <= 0.5]
    seg = seg / np.abs(seg).max()
    return np.linspace(0.0, 1.0, len(seg)), seg


# --- 2 分窗 + z-score：示意（相邻两窗 · 实线 + 虚线，贴 Sensors 2020 Fig.6）--- #
def ex_window(fig, rect):
    ax = mini(fig, rect)
    u, y = _sch_master()
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)

    W, S = 0.50, 0.25
    # 2026-10-10 用户要求「各子图高度一致、对齐」：框体上下撑满面板，
    # 与第 1/3 步的波形、坐标轴内容等高（原框 0.150~0.775 偏矮）。
    ylo, yhi = 0.030, 0.770
    yc, ya = 0.395, 0.285
    C = "#c07a12"

    ax.plot(u, yc + ya * y, lw=0.95, color="#2b6cb0", zorder=3)
    ax.add_patch(Rectangle((0.0, ylo), W, yhi - ylo, facecolor="none",
                           edgecolor=C, lw=1.5, zorder=4))
    ax.add_patch(Rectangle((S, ylo), W, yhi - ylo, facecolor="none",
                           edgecolor=C, lw=1.5, linestyle=(0, (4, 2)), zorder=4))
    for i, x0 in enumerate((0.0, S)):
        ax.text(x0 + 0.028, 0.722, f"窗 #{i + 1}", fontsize=9.0, ha="left",
                va="center", color="#7a4a06", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=C,
                          lw=0.7, alpha=0.94))

    ay = 0.925
    for xx in (0.0, W):
        ax.plot([xx, xx], [yhi, ay], lw=0.6, ls=":", color=C, zorder=2)
    ax.annotate("", xy=(W, ay), xytext=(0.0, ay), annotation_clip=False,
                arrowprops=dict(arrowstyle="<->", lw=1.3, mutation_scale=ARROW_MS,
                                color="#b06f10"))
    ax.text(W / 2, ay + 0.016, "窗长 W = 2048 点", fontsize=7.5, ha="center",
            va="bottom", color="#96600d", fontweight="bold")

    by = 0.792
    ax.annotate("", xy=(S, by), xytext=(0.0, by), annotation_clip=False,
                arrowprops=dict(arrowstyle="<->", lw=1.3, mutation_scale=ARROW_MS,
                                color="#2f9e63"))
    ax.text(S / 2, by + 0.016, "步长 B = 1024 点", fontsize=7.5, ha="center",
            va="bottom", color="#2f9e63", fontweight="bold")

    ax.set_xticks([]); ax.set_yticks([])
    return rect[0], rect[0] + rect[2]      # 波形铺满整个面板宽度


# --- 3 等频符号化：单个窗逐点分档 → A~F（SAX 官方示例画法）------------------ #
def _sch_sax(n=20):
    t = np.linspace(0.0, 1.0, n)
    y = (0.55 * np.sin(2 * np.pi * 1.20 * t + 2.56)
         + 0.42 * np.sin(2 * np.pi * 2.89 * t + 2.08)
         + 0.20 * np.sin(2 * np.pi * 4.22 * t + 5.43))
    return y / np.abs(y).max() * 1.30


def ex_symbol(fig, rect):
    x0, y0, w, h = rect
    # 上下边距对称（各 6.2% 面板高）→ 顶边与第 1/2 步对齐（原上方只留 1.9%，顶边偏高）
    ax = fig.add_axes([x0, y0 + 0.0145, w - 0.008, h - 0.0290])
    ax.set_zorder(15)
    brk = np.array([-0.90, -0.38, 0.01, 0.40, 0.90])
    rep = np.array([-1.20, -0.64, -0.185, 0.205, 0.65, 1.20])
    y = _sch_sax()
    x = np.arange(len(y))
    b = np.digitize(y, brk)

    ax.set_xlim(-0.8, len(y) - 0.2); ax.set_ylim(-1.62, 1.62)
    for e in brk:
        ax.axhline(e, ls="--", lw=0.7, color="#2f9e63", zorder=1)
    # 2026-10-10 用户选 A：删去「所在档位」橙线及其图例 —— 档位代表值与原始波形
    # 几乎重合（量化误差小），信息完全由每点的 A~F 字母承担，橙色曲线属冗余。
    ax.plot(x, y, ls="--", lw=0.9, color="#3b7ddd", marker="o", ms=3.0,
            mfc="white", mec="#3b7ddd", mew=0.9, zorder=3)
    for i in range(len(y)):
        # 字母锚到**原始点**外侧（原锚在档位代表值上，橙线删后字母会悬空）
        ax.text(x[i], y[i] + (0.20 if y[i] >= 0 else -0.20),
                "ABCDEF"[b[i]], fontsize=7.5, ha="center", va="center",
                color=plt.cm.tab10(b[i]), fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.86))
    ax.set_yticks(brk)
    # ⚠️ 必须用 Unicode 减号 U+2212：ASCII hyphen "−" 在 DejaVu Sans 里只有 0.36 em，
    # 而 "+" 有 0.84 em → 正负标签宽度不等，五个刻度左右都错位（用户截图指认）。
    ax.set_yticklabels([f"{v:+.2f}".replace("-", "\u2212") for v in brk],
                       fontsize=7.5)
    ax.yaxis.tick_right()          # 刻度移到右侧：左侧留给行内箭头，否则「+0.01」被箭头穿过
    ax.set_xticks([0, 5, 10, 15])
    ax.tick_params(labelsize=7.5, length=1.8, width=0.5, color="#9aa7b8")
    for sp in ax.spines.values():
        sp.set_color("#c07a12"); sp.set_linewidth(0.9)
    return x0, x0 + w - 0.008              # 坐标轴右边界比面板略收 0.008


# --- 4 状态转移：真实类别矩阵（新格为宽扁，故热图取正方形居中）--------------- #
def ex_matrix(fig, rect):
    x0, y0, w, h = rect
    ww = min(h * FH / FW, w * 0.60)          # 英寸坐标下让热图恒为正方形
    ax = fig.add_axes([x0 + (w - ww * 1.10) / 2.0, y0, ww, h])   # 连色标一起居中
    M = MAT_PER_CLASS["Health"][0]
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=float(M.max()))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels("ABCDEF", fontsize=9); ax.set_yticklabels("ABCDEF", fontsize=9)
    ax.tick_params(length=0, pad=1.5)
    for i in range(6):
        ax.text(i, i, f"{M[i,i]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if M[i, i] < M.max() * 0.6 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7.5, length=2); cb.outline.set_visible(False)
    off = (w - ww * 1.10) / 2.0              # 热图+色标整体居中 → 内容边界对称
    return x0 + off, x0 + off + ww * 1.10


# --- 5 张量：三通道 × S 时间步，真实 6×6 错位堆叠（按新格宽高自适应缩放）------ #
_TENSOR = _XT[_YT == 0].reshape(-1, 3, S, 6, 6)[0]
_TVLO, _TVHI = (float(np.percentile(_TENSOR, 5)), float(np.percentile(_TENSOR, 95)))


def _heat_card(ax, x, y, s, M, ec, z=4):
    pad = s * 0.014
    ax.add_patch(Rectangle((x - pad, y - pad), s + 2 * pad, s + 2 * pad,
                           facecolor="white", edgecolor="none", zorder=z))
    cs = s / 6.0
    for r in range(6):
        for c in range(6):
            v = (M[r, c] - _TVLO) / max(_TVHI - _TVLO, 1e-9)
            ax.add_patch(Rectangle((x + c * cs, y + (5 - r) * cs), cs * 1.005,
                                   cs * 1.005,
                                   facecolor=plt.cm.viridis(np.clip(v, 0, 1)),
                                   lw=0, zorder=z + 1))
    ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=ec,
                           lw=1.0, zorder=z + 2))


def ex_tensor(fig, rect):
    ax = mini(fig, rect)
    W, Hh = rect[2] * FW, rect[3] * FH
    ax.set_xlim(0, W); ax.set_ylim(0, Hh)
    chc = ["#3b7ddd", "#e8930c", "#2f9e63"]
    n = S
    # 2026-10-10：减小横向层叠偏移、加大纵向偏移 → 卡片组不再是一条扁带，
    # 高度撑到面板 ~86%，与左右邻居（热图 / 网络图）等高。
    oxf, ov = 0.100, 0.170
    a = 1 + (n - 1) * oxf                   # 每组卡片横向跨度系数
    b = 1 + (n - 1) * ov                    # 纵向跨度系数
    s = min(W * 0.97 / (3.0 * a), Hh * 0.92 / b)   # 同时受宽、高约束
    ox, oy = s * oxf, s * ov
    gw = s + (n - 1) * ox
    gapx = (W - 3 * gw) / 4.0
    ybase = (Hh - s * b) * 0.52          # 张量垂直居中（下方留「通道 xyz」标签）
    for r in range(3):
        x0 = gapx + r * (gw + gapx)
        for k in range(n - 1, -1, -1):
            _heat_card(ax, x0 + k * ox, ybase + k * oy, s, _TENSOR[r, k],
                       ec=chc[r], z=4 + (n - k))
        ax.text(x0 + gw / 2, ybase - 0.022, f"通道 {'xyz'[r]}",
                fontsize=9.0, color=chc[r], ha="center", va="top",
                fontweight="bold")
    return rect[0] + gapx / FW, rect[0] + rect[2] - gapx / FW


# --- 6 网络：CNN = 网格 + 局部卷积；GCN = 圆环图 + 邻域聚合 -------------------- #
# 两者刻意用**不同的几何**区分：CNN 的输入是「图像/矩阵」，卷积核只看 3×3 邻域；
# GCN 的输入是「6 个状态构成的图」，每层沿图的邻域做聚合。
def _cnn_net(ax, cx, cy, accent, n=4, g=0.105, r=0.032, dx=0.52):
    """3 列 4×4 网格点；列间**只连 3×3 邻域** → 局部感受野。"""
    xs = cx + np.array([-dx, 0.0, dx])
    grid = []
    for x in xs:
        col = {}
        for i in range(n):
            for j in range(n):
                col[(i, j)] = (x + (j - (n - 1) / 2.0) * g,
                               cy + (i - (n - 1) / 2.0) * g)
        grid.append(col)
    for k in range(len(xs) - 1):                # 列间：3×3 邻域（局部卷积）
        for (i, j), (x1, y1) in grid[k].items():
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    x2y2 = grid[k + 1].get((i + di, j + dj))
                    if x2y2 is not None:
                        ax.plot([x1, x2y2[0]], [y1, x2y2[1]], lw=0.22,
                                color="#d9c3a5", alpha=0.85, zorder=1)
    for k in range(len(xs)):                    # 圆点（连线在下、圆点在上）
        cs = accent if k != 1 else "#9aa7b8"
        for (xx, yy) in grid[k].values():
            ax.add_patch(Circle((xx, yy), r, facecolor=cs, edgecolor="white",
                                lw=0.35, zorder=3))


def _gcn_net(ax, cx, cy, accent, n=6, R=0.270, r=0.048, dx=0.66):
    """3 列，每列 6 个节点排成圆环（= 6 个符号状态）；列内画图边、列间全连接。"""
    xs = cx + np.array([-dx, 0.0, dx])
    ang = 0.5 * np.pi + np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    cols = [[(x + R * np.cos(a), cy + R * np.sin(a)) for a in ang] for x in xs]
    for c in cols:                              # 列内图边：一环 + 隔一节点
        for i in range(n):
            for step in (1, 2):
                j = (i + step) % n
                ax.plot([c[i][0], c[j][0]], [c[i][1], c[j][1]], lw=0.22,
                        color="#b9aede", alpha=0.9, zorder=1)
    for k in range(len(xs) - 1):                # 列间：全连接（邻域聚合）
        for (x1, y1) in cols[k]:
            for (x2, y2) in cols[k + 1]:
                ax.plot([x1, x2], [y1, y2], lw=0.20, color="#c6b9e6",
                        alpha=0.7, zorder=1)
    for k, c in enumerate(cols):
        cs = accent if k != 1 else "#9aa7b8"
        for (xx, yy) in c:
            ax.add_patch(Circle((xx, yy), r, facecolor=cs, edgecolor="white",
                                lw=0.4, zorder=3))


def ex_model(fig, rect):
    ax = mini(fig, rect)
    W, Hh = rect[2] * FW, rect[3] * FH          # 英寸坐标：圆点恒为正圆
    ax.set_xlim(0, W); ax.set_ylim(0, Hh)
    # 2026-10-10 用户要求「各子图高度一致」：两张网络取**同一内容高度 H**，并排总宽不超面板。
    #   单列宽：CNN = 3g（4×4 网格）、GCN = 2R（6 节点圆环）。
    #   ⚠️ 列间距必须大于单列宽，且**环列之间要留空隙**（原取 2R 相切 → 三环粘连、用户说「干涉」）：
    #      CNN 取 4.2g（列间圆点留 1.2g 缝）、GCN 取 2.75R（环间留 0.75R 缝）。
    #   ⇒ CNN 总宽 = 11.4g = 3.80H，GCN 总宽 = 7.5R = 3.75H，合计 7.55H ≤ 可用宽度。
    XM, MID = 0.10, 0.30                        # 外侧边距 / CNN–GCN 中缝（英寸，中缝要明显大于内部空隙）
    H = min(Hh * 0.42, (W - 2 * XM - MID) / 7.55)
    g, R = H / 3.0, H / 2.0
    dxc, dxg = 4.2 * g, 2.75 * R
    wc, wg = 2 * dxc + 3 * g, 2 * dxg + 2 * R
    ccx, gcx = XM + wc / 2.0, XM + wc + MID + wg / 2.0
    lab_gap = 0.18                              # 标签与最底部节点的间距（原来 0.10，与节点干涉）
    cy = Hh * 0.5 + (lab_gap + 0.09) / 2.0      # 「网络图 + 下方标签」整体居中
    _cnn_net(ax, ccx, cy, "#d9791a", g=g, r=0.28 * g, dx=dxc)
    _gcn_net(ax, gcx, cy, "#7a4bcf", R=R, r=0.17 * R, dx=dxg)
    for cx, lab, col in [(ccx, "CNN", "#d9791a"), (gcx, "GCN", "#7a4bcf")]:
        ax.text(cx, cy - R - lab_gap, lab, fontsize=12.0, fontweight="bold",
                ha="center", va="top", color=col)
    return rect[0] + XM / FW, rect[0] + rect[2] - XM / FW


# 末行收口块的大标题。**不能写"齿轮任务"**：这套流程同时用于齿轮与轴承两个
# 任务（两类分别独立建模），写死成齿轮会与论文正文自相矛盾。
RESULT_TITLE = "故障种类"


# --- 收口：判别输出（5 类）----------------------------------------------------- #
# 2026-10-10 按用户要求，用本块**替代**原来的「7 噪声增广」+「8 故障分类柱状图」。
# 写法照参考文献末端的收口：虚线外框内并排放**类别标签框**（原来是三行说明文字）。
# 下面 ex_noise / ex_class 两个函数保留未删，改回来只需在 steps 里重新挂上即可。
def ex_result(fig, rect):
    ax = mini(fig, rect)
    W, Hh = rect[2] * FW, rect[3] * FH
    ax.set_xlim(0, W); ax.set_ylim(0, Hh)
    pad = 0.16
    ax.add_patch(FancyBboxPatch((pad, pad), W - 2 * pad, Hh - 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f6f9fc", edgecolor="#8f9bab",
                                lw=1.2, linestyle=(0, (5, 3)), zorder=2))
    # 类别标签框并排居中：学参考文献末端的「虚线外框 + 实线类别框」收口写法
    # 收口块只留一行大字「故障种类」，不列任何具体类别名、也不画类别框：
    # 齿轮任务与轴承任务各有一套 5 类，列任何一套都会让另一套看起来
    # "不属于本流程"；空框同样会被读成"忘了填字"。类别清单放在正文与图注里。
    ax.text(W / 2.0, Hh * 0.50, RESULT_TITLE, fontsize=19.0, fontweight="bold",
            ha="center", va="center", color="#1f2d3d", zorder=4)
    return rect[0] + pad / FW, rect[0] + rect[2] - pad / FW


# --- 7 噪声增广：真实 SNR 档位（当前未挂进 steps）----------------------------- #
def ex_noise(fig, rect):
    ax = mini(fig, rect)
    seg = SIG[:1600, 0]
    seg = seg - seg.mean()
    rng = np.random.default_rng(42)
    noise = rng.normal(0, seg.std() * 1.0, seg.shape[0])     # 0 dB
    vn = seg + noise
    t = np.arange(len(seg)) / FS * 1000
    ax.plot(t, vn, lw=0.45, color="#8e5bd6", alpha=0.5)
    ax.plot(t, seg, lw=0.8, color="#333")
    ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("时间 (ms)", fontsize=6.5, labelpad=0.5)


# --- 8 分类：真实各类召回率（当前未挂进 steps）------------------------------- #
def ex_class(fig, rect):
    ax = mini(fig, rect)
    vals = [RECALL[c] * 100 for c in CLASSES]
    y = np.arange(5)[::-1]
    ax.barh(y, vals, color="#2f9e63", alpha=0.85, height=0.66)
    ax.set_xlim(87, 102.2); ax.set_ylim(-0.6, 4.6)
    ax.set_xticks([]); ax.set_yticks([])
    for yy, vv, c in zip(y, vals, CLASSES):
        ax.text(87.5, yy, CN[c], fontsize=7.6, va="center", ha="left", color="white")
        ax.text(vv + 0.3, yy, f"{vv:.1f}", fontsize=7.4, va="center", ha="left",
                color="#2f6f4f")
    ax.text(87.5, -0.55, f"验证集总体 {VAL_ACC*100:.2f}%", fontsize=7.6,
            va="center", ha="left", color="#5a6472")


# --------------------------------------------------------------------------- #
# 版式：三行回流 3+3+2
# --------------------------------------------------------------------------- #
COL_X = [0.022, 0.3467, 0.6714]
PANEL_W = 0.2987
PANEL_H = 0.235
ROW_Y = [0.628, 0.325, 0.022]        # 三行面板的下边界
TITLE_Y = [0.891, 0.588, 0.285]
# 6 个面板落在 (列, 行) 上：行内先左→右、再右→左；第 3 行放文字收口块
PLACE = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]

ARROW = "#4a5568"
ARROW_IN = 0.176                       # 流程箭头统一物理长度（英寸）
ARROW_MS = 13                          # 箭头头部尺寸（pt）—— 全图统一


def _arrow(fig, p, q):
    fig.add_artist(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ARROW_MS,
                                   lw=1.3, color=ARROW, transform=fig.transFigure,
                                   zorder=1, shrinkA=0, shrinkB=0))


def _harrow(fig, xa, xb, y):
    """横向流程箭头：自 xa 指向 xb，物理长度统一为 ARROW_IN，居中于两列之间的缝隙。
    注意 figure 的 x 单位 ≠ y 单位（画幅 13.2×8.0），必须按英寸换算。
    ⚠️ 方向由 xa→xb 决定，不能对 xa/xb 排序（第 2 行是回流，须朝左）。"""
    L = min(ARROW_IN / FW, abs(xb - xa) - 0.006)
    cx = (xa + xb) / 2.0
    s = 1.0 if xb >= xa else -1.0                     # 指向方向
    _arrow(fig, (cx - s * L / 2.0, y), (cx + s * L / 2.0, y))


def _varrow(fig, x, y_low, y_high):
    """竖向流程箭头（朝下，y_high 在上）：物理长度统一为 ARROW_IN。"""
    L = min(ARROW_IN / FH, (y_high - y_low) - 0.004)
    cy = (y_low + y_high) / 2.0
    _arrow(fig, (x, cy + L / 2.0), (x, cy - L / 2.0))


def _col_cx(c):
    return COL_X[c] + PANEL_W / 2.0


def _col_right(c):
    return COL_X[c] + PANEL_W


def _row_cy(r):
    return ROW_Y[r] + PANEL_H / 2.0


def fig_flowchart_style_a():
    steps = [
        ("原始振动信号", "三轴 5120 Hz；每类 104 万点（≈3.4 min）", ex_signal),
        ("分窗 + 逐窗 z-score", f"窗长 {WIN}、步长 {STRIDE}（50% 重叠）；训练集固定 μ/σ 归一化", ex_window),
        ("等频符号化（6 状态）", "按训练集分位数划 A–F 六档，边界只用训练集拟合", ex_symbol),
        ("多时间步状态转移", f"步长 1→{S}；统计 i→j 转移概率，各得 6×6 矩阵", ex_matrix),
        ("多通道堆叠张量", f"3 个振动通道 × {S} 个时间步 = {3 * S} 个 6×6", ex_tensor),
        ("CNN / GCN 特征学习", "CNN：3×3 卷积 32→64；GCN：2 层图卷积 · 隐层 32", ex_model),
    ]
    # 文字收口块（替代原来的「噪声增广」+「故障分类」两格）；标题见模块级 RESULT_TITLE

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.938, "多时间步状态转移网络 + CNN / GCN 方法总流程",
             ha="center", va="center", fontsize=21, fontweight="bold", color="#1f2d3d")

    spans = []
    for i, (title, sub, exfn) in enumerate(steps):
        col, row = PLACE[i]
        # 步骤标题（「1. 原始振动信号」等）按用户 2026-10-10 要求**不再上屏**；
        # 标题文案仍留在 steps 元组第 1 项里，备作图注。
        # 每个 ex_* 返回 (内容左边界, 内容右边界)（figure 坐标）→ 供流程箭头对齐。
        spans.append(exfn(fig, [COL_X[col], ROW_Y[row], PANEL_W, PANEL_H]))

    # 第 7 块：输出收口，横跨整行（标题同样不再上屏）
    spans.append(ex_result(fig, [COL_X[0], ROW_Y[2],
                                 COL_X[2] + PANEL_W - COL_X[0], PANEL_H]))

    # 行内横向箭头（沿回流方向）
    # ⚠️ 锚点是**相邻两图内容的真实边界**，不是面板边界 —— 否则第 4 步热图（内容居中、
    #    左右各留 0.94 in 空白）会让箭头紧贴张量、却离热图很远（用户：「箭头与图之间的距离要一致」）。
    _harrow(fig, spans[0][1], spans[1][0], _row_cy(0))
    _harrow(fig, spans[1][1], spans[2][0], _row_cy(0))
    _harrow(fig, spans[3][0], spans[4][1], _row_cy(1))
    _harrow(fig, spans[4][0], spans[5][1], _row_cy(1))
    # 行间竖向箭头（折返）：行1 末(列2) ↓ 行2 首(列2)；行2 末(列0) ↓ 收口块
    # （标题删除后，箭头改为直接跨两行面板之间的缝隙）
    _varrow(fig, _col_cx(2), ROW_Y[1] + PANEL_H, ROW_Y[0])
    _varrow(fig, _col_cx(0), ROW_Y[2] + PANEL_H, ROW_Y[1])

    out = FIG / "fig_pipeline_style_a.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print("saved", out)
    print("核对：", f"符号串={_SYM_STR}", f"测试准确率={VAL_ACC:.4f}",
          "各类召回=", {k: round(v, 3) for k, v in RECALL.items()})


if __name__ == "__main__":
    fig_flowchart_style_a()
