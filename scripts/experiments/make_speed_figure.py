# -*- coding: utf-8 -*-
"""速度/效率对比图：本文 vs 同源文献（参数量、FLOPs，对数坐标）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "doc/figures/seu/speed_compare.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 方法, 参数量(M), FLOPs(M), 是否本文
methods = [
    ("本文 CNN",      0.0251, 0.384,  True),
    ("本文 GCN",      0.0185, 0.558,  True),
    ("DSMC-ECA",      0.204,  10.037, False),
    ("WDCNN",         2.87,   386.52, False),
    ("STCSE",         2.10,   800.0,  False),
    ("MGE-ResNet",    None,   1990.0, False),
    ("ResNet-18",     9.18,   1728.64,False),
]
names  = [m[0] for m in methods]
params = [m[1] for m in methods]
flops  = [m[2] for m in methods]
isus   = [m[3] for m in methods]
colors = ["#c0392b" if u else "#5b8db8" for u in isus]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# 左：参数量
ax = axes[0]
idx = [i for i, p in enumerate(params) if p is not None]
ax.bar([names[i] for i in idx], [params[i] for i in idx],
       color=[colors[i] for i in idx], edgecolor="#333", linewidth=0.6)
ax.set_yscale("log")
ax.set_title("参数量（百万，对数坐标）", fontsize=13, fontweight="bold")
ax.set_ylabel("Params (M)", fontsize=11)
ax.tick_params(axis="x", rotation=25, labelsize=9)
for i in idx:
    ax.text(list(idx).index(i), params[i]*1.15, f"{params[i]:g}M",
            ha="center", fontsize=8.5)
ax.grid(axis="y", ls="--", alpha=0.35)

# 右：FLOPs
ax = axes[1]
ax.bar(names, flops, color=colors, edgecolor="#333", linewidth=0.6)
ax.set_yscale("log")
ax.set_title("单样本算量 FLOPs（百万，对数坐标）", fontsize=13, fontweight="bold")
ax.set_ylabel("FLOPs (M)", fontsize=11)
ax.tick_params(axis="x", rotation=25, labelsize=9)
for i, v in enumerate(flops):
    ax.text(i, v*1.15, f"{v:g}M", ha="center", fontsize=8.5)
ax.grid(axis="y", ls="--", alpha=0.35)

# 图例
from matplotlib.patches import Patch
handles = [Patch(facecolor="#c0392b", edgecolor="#333", label="本文"),
           Patch(facecolor="#5b8db8", edgecolor="#333", label="同源文献")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.99))

fig.suptitle("速度/效率对比：本文 vs 同源文献（对数坐标）", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
