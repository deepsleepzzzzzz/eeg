"""故障形态示意图（导师问题3 的图示部分）。

注意：本图是**机理示意图**，不是实物照片。实物照片请从数据集原论文
(Shao S, McAleer S, Yan R, Baldi P. Highly Accurate Machine Fault Diagnosis
Using Deep Transfer Learning. IEEE TII, 2019, 15(4):2446-2455) 的台架/故障件图中获取。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#e74c3c"
STEEL = "#cfd8dc"
EDGE = "#546e7a"


def draw_gear(ax, missing=(), worn=(), pitting=(), cracked=(), teeth=20,
              r=1.0, tooth_h=0.20):
    seg = 2 * np.pi / teeth
    m = teeth * 8
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    tid = np.floor(ang / seg).astype(int) % teeth
    pos = (ang % seg) / seg
    rad = np.where(pos < 0.45, r + tooth_h, r)
    for t in missing:
        rad[tid == t] = r - 0.02
    for t in worn:
        rad[(tid == t) & (pos < 0.45)] = r + tooth_h * 0.30
    ax.fill(rad * np.cos(ang), rad * np.sin(ang), color=STEEL, ec=EDGE, lw=1.0, zorder=2)
    ax.add_patch(Circle((0, 0), r * 0.30, fc="white", ec=EDGE, lw=1.2, zorder=3))

    for t in missing:
        a = (t + 0.2) * seg
        ax.annotate("", xy=(r * 1.05 * np.cos(a), r * 1.05 * np.sin(a)),
                    xytext=(r * 1.55 * np.cos(a), r * 1.55 * np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6), zorder=5)
    for t in pitting:
        for k in (0.15, 0.3, 0.42):
            a = (t + k) * seg
            ax.plot([(r + tooth_h * 0.55) * np.cos(a)], [(r + tooth_h * 0.55) * np.sin(a)],
                    "o", color=RED, ms=3.6, zorder=5)
    for t in cracked:
        a = t * seg
        ax.plot([r * 0.99 * np.cos(a), r * 0.80 * np.cos(a)],
                [r * 0.99 * np.sin(a), r * 0.80 * np.sin(a)],
                "-", color=RED, lw=2.2, zorder=5)
    for t in worn:
        a = t * seg
        ax.annotate("", xy=((r + tooth_h * 0.4) * np.cos(a), (r + tooth_h * 0.4) * np.sin(a)),
                    xytext=(r * 1.5 * np.cos(a), r * 1.5 * np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6), zorder=5)

    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal"); ax.axis("off")


def draw_bearing(ax, fault=None, n_roll=8, R=1.0):
    r_out, r_in = R, R * 0.55
    ax.add_patch(Circle((0, 0), r_out, fill=False, ec=EDGE, lw=7, zorder=2))
    ax.add_patch(Circle((0, 0), r_in, fill=False, ec=EDGE, lw=7, zorder=2))
    rc = (r_out + r_in) / 2
    rr = (r_out - r_in) / 2 * 0.72
    for i in range(n_roll):
        a = 2 * np.pi * i / n_roll
        bad = (fault == "ball" and i == 0) or (fault == "comb" and i in (0, 4))
        ax.add_patch(Circle((rc * np.cos(a), rc * np.sin(a)), rr,
                            fc=RED if bad else "#b0bec5", ec=EDGE, lw=1.0, zorder=3))
    if fault in ("inner", "comb"):
        ax.add_patch(Wedge((0, 0), r_in + 0.075, 25, 70, width=0.15, fc=RED, zorder=4))
    if fault in ("outer", "comb"):
        ax.add_patch(Wedge((0, 0), r_out, 195, 240, width=0.15, fc=RED, zorder=4))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    gear_rows = [
        ("Health 正常", {}),
        ("Chipped 齿面剥落", dict(pitting=(3,))),
        ("Miss 缺齿", dict(missing=(10,))),
        ("Root 齿根裂纹", dict(cracked=(15,))),
        ("Surface 齿面磨损", dict(worn=(7,))),
    ]
    bearing_rows = [
        ("health 正常", None),
        ("ball 滚动体故障", "ball"),
        ("inner 内圈故障", "inner"),
        ("outer 外圈故障", "outer"),
        ("comb 复合故障", "comb"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8))
    for j, (title, kw) in enumerate(gear_rows):
        ax = axes[0, j]
        draw_gear(ax, **kw)
        ax.set_title(title, fontsize=11.5)
    for j, (title, f) in enumerate(bearing_rows):
        ax = axes[1, j]
        draw_bearing(ax, fault=f)
        ax.set_title(title, fontsize=11.5)

    fig.suptitle("SEU DDS 数据集：齿轮箱与轴承各故障形态示意图", fontsize=14.5)
    fig.text(0.5, 0.015,
             "注：本图为机理示意图（红色标出故障位置），非实物照片。实物照片请取自数据集原论文 "
             "Shao et al., IEEE TII 2019, 15(4):2446-2455 的台架与故障件插图。",
             ha="center", fontsize=8.6, color="#8e2b2b")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    out = FIG / "fault_schematic.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    main()
