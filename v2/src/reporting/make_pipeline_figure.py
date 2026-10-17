"""图 1：按期刊标准重画（对照 DDGF-Net / ADAMFFN 的插图规则）。

遵循的 10 条规则：
  1 扁平化：纯色块 + 细边框，无阴影无夸张渐变
  2 色系区分模块类型：表示构造=蓝系 / 分类器=紫系 / 输出=绿系
  3 虚线框分组 + 组名
  4 每块标超参（卷积核、通道数、激活、池化）
  5 箭头细、方向统一（分组才用虚线）
  6 真实数据嵌入（信号、切窗、符号化电平、真实转移矩阵）
  7 无衬线字体、字号统一
  8 留白充足
  9 流程单向：从左到右、自上而下
  10 图注一句话
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from reporting.make_noise_figure import setup_font  # noqa: E402
from reporting.class_labels import cn_names  # noqa: E402
from data.seu_dds import class_names  # noqa: E402
from utils.console import setup_utf8_console  # noqa: E402
setup_utf8_console(); setup_font()
plt.rcParams["font.size"] = 9

# ---- 配色（规则 2）----
C_REP_BG, C_REP_EC = "#eaf3fb", "#3d7ebf"     # 表示构造：蓝系
C_CLS_BG, C_CLS_EC = "#f4ecf7", "#7d3c98"     # 分类器：紫系
C_OUT_BG, C_OUT_EC = "#eef7ee", "#3d8b40"     # 输出：绿系
C_GROUP = "#8a8a8a"
C_ARROW = "#555555"

# ---- 示意信号 ----
rng = np.random.default_rng(3)
fs, T = 5120, 4096
t = np.arange(T) / fs
base = (np.sin(2*np.pi*620*t) + .45*np.sin(2*np.pi*1240*t + .7) + .22*np.sin(2*np.pi*155*t))
imp = np.zeros(T); imp[::300] = 1.0
imp = np.convolve(imp, np.exp(-np.arange(50)/7.0), mode="same")
core = base + 2.4*imp
chan = [core + .30*rng.standard_normal(T),
        (core + .30*rng.standard_normal(T))*1.9,
        (core + .30*rng.standard_normal(T))*0.6]
COL = ["#1f77b4", "#e8850c", "#2ca02c"]
LE = "ABCDEF"

fig = plt.figure(figsize=(16, 8.4))
fig.patch.set_facecolor("white")

def grp(x, y, w, h, label):
    """规则 3：虚线分组框 + 组名（组名放左上角外侧）。"""
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                  fc="none", ec=C_GROUP, lw=1.1, ls=(0, (5, 4)),
                                  transform=fig.transFigure))
    fig.text(x + 0.008, y + h + 0.012, label, fontsize=10.5, color="#444", fontweight="bold")

def arr(x1, y1, x2, y2):
    """规则 5：细线 + 实心三角。"""
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                   mutation_scale=13, lw=1.2, color=C_ARROW,
                                   transform=fig.transFigure,
                                   shrinkA=0, shrinkB=0))

# ================= 分组一：表示构造 =================
grp(0.022, 0.560, 0.956, 0.375, "表示构造")

# ① 三通道信号
ax1 = fig.add_axes([0.045, 0.640, 0.135, 0.230])
for i, (s, c) in enumerate(zip(chan, COL)):
    y = s[:700]; ax1.plot(np.arange(700), y - y.mean() + (2-i)*8, color=c, lw=0.65)
ax1.set_title("① 三通道振动信号\n行星齿轮箱 x / y / z", fontsize=9, pad=4)
ax1.set_xticks([]); ax1.set_yticks([])
for sp in ax1.spines.values(): sp.set_color("#c8c8c8"); sp.set_linewidth(0.8)

arr(0.187, 0.752, 0.216, 0.752)

# ② 切窗
ax2 = fig.add_axes([0.222, 0.640, 0.180, 0.230])
seg = chan[0][:1500]
ax2.plot(np.arange(1500), seg, color="#555", lw=0.7)
lo, hi = float(seg.min()), float(seg.max())
ax2.add_patch(Rectangle((20, lo), 620, hi-lo, fill=False, ec="#d62728", lw=1.6))
ax2.add_patch(Rectangle((330, lo), 620, hi-lo, fill=False, ec="#1f77b4", lw=1.6, ls="--"))
ax2.text(330, hi*1.30, "窗 #1", color="#d62728", fontsize=8.5, ha="center")
ax2.text(660, hi*1.30, "窗 #2", color="#1f77b4", fontsize=8.5, ha="center")
ax2.set_title("② 切窗\n窗长 2048 点 / 步长 1024 点", fontsize=9, pad=4)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_ylim(lo*1.15, hi*1.70)
for sp in ax2.spines.values(): sp.set_color("#c8c8c8"); sp.set_linewidth(0.8)

arr(0.408, 0.752, 0.437, 0.752)

# ③ 符号化
ax3 = fig.add_axes([0.443, 0.625, 0.185, 0.255])
n = 110
s3 = chan[0][:n]
edges = np.quantile(chan[0], np.linspace(0, 1, 7)[1:-1])
vmin, vmax = float(s3.min()), float(s3.max())
cent = [(vmin+edges[0])/2] + [(edges[i]+edges[i+1])/2 for i in range(4)] + [(edges[4]+vmax)/2]
for b in range(6):
    ax3.axhline(cent[b], color="#b5b5b5", lw=0.8, ls=":")
    ax3.text(n+3, cent[b], LE[b], fontsize=9.5, va="center", color="#444", fontweight="bold")
ax3.plot(np.arange(n), s3, color="#333", lw=1.0)
sy = np.searchsorted(edges, s3, side="right")
ax3.scatter(np.arange(0, n, 8), s3[::8], s=[24]*len(s3[::8]),
            c=[plt.cm.tab10(i) for i in sy[::8]], zorder=5,
            edgecolors="white", linewidths=0.4)
ax3.set_xlim(-4, n+16); ax3.set_ylim(vmin-0.4, vmax+0.4)
ax3.set_title("③ 符号化：6 档等频分位", fontsize=9, pad=4)
ax3.set_xticks([]); ax3.set_yticks([])
for sp in ax3.spines.values(): sp.set_color("#c8c8c8"); sp.set_linewidth(0.8)

arr(0.634, 0.752, 0.663, 0.752)

# ④ 转移矩阵
ax4 = fig.add_axes([0.670, 0.660, 0.115, 0.215])
M = np.zeros((6, 6))
full = np.searchsorted(edges, chan[0], side="right")
for a, b in zip(full[:-1], full[1:]): M[a, b] += 1
M /= M.sum()
ax4.imshow(M, cmap="viridis")
ax4.set_xticks(range(6)); ax4.set_xticklabels(list(LE), fontsize=7.5)
ax4.set_yticks(range(6)); ax4.set_yticklabels(list(LE), fontsize=7.5)
ax4.tick_params(length=0)
ax4.set_title("④ 联合转移概率\n矩阵 6 × 6", fontsize=9, pad=4)
for b in range(6):
    for a in range(6):
        if M[a, b] >= 0.10:
            ax4.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=6, color="white")

# ⑤ 堆叠（放在 ④ 右侧下方，形成"展开"感）
ax5 = fig.add_axes([0.800, 0.600, 0.165, 0.310])
ax5.set_xlim(0, 10); ax5.set_ylim(0, 10); ax5.axis("off")
ax5.set_title("⑤ 堆叠：3 通道 × 10 滞后", fontsize=9, pad=4)
for ci in range(3):
    x0 = 0.4 + ci*3.2
    for k in range(3):
        ax5.add_patch(Rectangle((x0 + k*0.45, 3.3 + k*0.45), 2.0, 2.0,
                                fc=plt.cm.viridis(0.12 + 0.26*k), ec="#4a4a4a", lw=0.7,
                                alpha=0.97, zorder=k))
    ax5.text(x0 + 1.1, 2.75, ("x", "y", "z")[ci], fontsize=9, ha="center",
             color=COL[ci], fontweight="bold")
ax5.text(5.0, 1.0, "→ 30 × 6 × 6 = 1080 元素", fontsize=8.5, ha="center", color="#c0392b")

arr(0.788, 0.752, 0.796, 0.752)

# ================= 分组二：分类与输出 =================
grp(0.022, 0.120, 0.956, 0.360, "分类与输出")

# ⑥ 分类器
ax6 = fig.add_axes([0.045, 0.300, 0.560, 0.170])
ax6.set_xlim(0, 13); ax6.set_ylim(-2.0, 10); ax6.axis("off")
for i, (name, struct) in enumerate([
        ("CNN", "Conv 3×3 / 32 → BN → ReLU → Pool 2×2 → Conv 3×3 / 64 → BN → ReLU → Pool 2×2 → GAP → FC / 5"),
        ("GCN", "关系图卷积 ×2（6 节点，隐层 32）→ 全局池化 → FC / 5")]):
    y = 6.2 - i*5.0
    ax6.add_patch(FancyBboxPatch((0.2, y), 9.6, 3.4, boxstyle="round,pad=0.12",
                                 fc=C_CLS_BG, ec=C_CLS_EC, lw=1.2))
    ax6.text(0.55, y + 1.7, f"⑥ {name}", fontsize=10, va="center", fontweight="bold", color=C_CLS_EC)
    ax6.text(2.6, y + 1.7, struct, fontsize=8, va="center", color="#444")
ax6.text(5.0, -1.0, "参数量 27,685（CNN）/ 54,941（GCN）", fontsize=8.5, ha="center", color="#666")

ax6.text(11.8, 3.4, "同数据集下\n参数量较最强基线\n小 1–2 个数量级",
         fontsize=8.5, ha="center", va="center", color="#666", linespacing=1.6)
arr(0.700, 0.395, 0.700, 0.250)

# ⑦ 输出
ax7 = fig.add_axes([0.045, 0.145, 0.900, 0.090])
ax7.set_xlim(0, 10); ax7.set_ylim(0, 10); ax7.axis("off")
names = cn_names("gear", class_names("gear"))  # 类别中文名唯一真源（健康/齿面剥落/缺齿/齿根裂纹/齿面磨损）
for i, c in enumerate(names):
    x = 0.30 + i*1.95
    ax7.add_patch(FancyBboxPatch((x, 2.2), 1.72, 5.4, boxstyle="round,pad=0.14",
                                 fc=C_OUT_BG, ec=C_OUT_EC, lw=1.2))
    ax7.text(x+0.86, 4.9, c, ha="center", va="center", fontsize=9.5)
# "故障种类"：本流程同时用于齿轮与轴承两个任务，不能写成"齿轮任务"
ax7.text(5.0, 9.4, "⑦ 故障种类", fontsize=10, ha="center", fontweight="bold")

fig.text(0.5, 0.035, "图 1  多时间步状态转移网络 + CNN / GCN 故障诊断方法总流程",
         ha="center", fontsize=11.5)

out = PROJECT_ROOT / "reports" / "fig_pipeline.png"
fig.savefig(out, bbox_inches="tight", dpi=160, facecolor="white")
print("[ok]", out)
