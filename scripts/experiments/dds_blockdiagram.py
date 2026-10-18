"""东南大学 DDS 传动系统动态模拟器 —— 结构框图与测点布置图（导师问题 3）。

产出：doc/figures/seu/teacher/dds_platform_diagram.png
说明：本图为按数据集官方说明（README / 数据集说明文档）重绘的结构示意图，
      与 dds_platform_photo.png（数据集官方台架实物图）配套使用。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BOX_FC = "#e8eef7"
BOX_EC = "#3c5a86"
ACC_FC = "#fdeceb"
ACC_EC = "#c0392b"
TXT = "#1f2933"

CHAIN = [
    ("电机控制器\nMotor controller", False),
    ("电机\nMotor", True),
    ("行星齿轮箱\nPlanetary gearbox", True),
    ("平行轴齿轮箱\nParallel gearbox", True),
    ("制动器（负载）\nBrake / Load", True),
    ("制动器控制器\nBrake controller", False),
]
# 每个部件下方的传感器标注（None = 无测点）
SENSOR = [
    None,
    "测点 1：电机 z 向振动\n＋电机扭矩",
    "测点 2–4：\nx / y / z 三向振动",
    "测点 6–8：\nx / y / z 三向振动",
    None,
    None,
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 5.4))
    ax.set_xlim(0, 134); ax.set_ylim(0, 54); ax.axis("off")

    n = len(CHAIN)
    bw, gap = 17.5, 4.0
    total = n * bw + (n - 1) * gap
    x0 = (134 - total) / 2
    ytop, bh = 34.0, 11.0

    for i, (label, is_fault) in enumerate(CHAIN):
        x = x0 + i * (bw + gap)
        fc, ec = (ACC_FC, ACC_EC) if is_fault else (BOX_FC, BOX_EC)
        ax.add_patch(FancyBboxPatch((x, ytop), bw, bh,
                                    boxstyle="round,pad=0.5,rounding_size=1.2",
                                    linewidth=1.6, facecolor=fc, edgecolor=ec))
        ax.text(x + bw / 2, ytop + bh / 2, label, ha="center", va="center",
                fontsize=11.5, color=TXT, linespacing=1.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.4, ytop + bh / 2),
                                         (x + bw + gap - 0.4, ytop + bh / 2),
                                         arrowstyle="-|>", mutation_scale=16,
                                         linewidth=1.8, color=BOX_EC))
        # 传感器标注
        s = SENSOR[i]
        if s:
            ax.text(x + bw / 2, ytop - 3.2, s, ha="center", va="top",
                    fontsize=9.5, color=ACC_EC, linespacing=1.5)
            ax.add_patch(FancyArrowPatch((x + bw / 2, ytop - 1.2),
                                         (x + bw / 2, ytop - 0.2),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.2, color=ACC_EC))

    # 传感器汇总
    ax.text(67, 15.0,
            "传感器：7 个 608A11 加速度计（行星齿轮箱 x/y/z、平行轴齿轮箱 x/y/z、电机 z 向）＋ 电机扭矩，采样频率 5120 Hz，"
            "每个状态文件含 8 路信号",
            ha="center", va="center", fontsize=10.5, color=TXT,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f4f7fb", edgecolor=BOX_EC, linewidth=1.1))

    # 工况
    ax.text(67, 9.0,
            "工况设置：20 Hz-0 V（1200 r/min、0 N·m，空载）　|　30 Hz-2 V（1800 r/min、7.32 N·m，带载）",
            ha="center", va="center", fontsize=10.5, color=TXT,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f4f7fb", edgecolor=BOX_EC, linewidth=1.1))

    # 故障类型
    ax.text(67, 3.2,
            "齿轮 5 类状态：健康 / 断齿(Chipped) / 缺齿(Miss) / 齿根裂纹(Root) / 齿面磨损(Surface)　"
            "　轴承 5 类状态：健康 / 滚动体 / 内圈 / 外圈 / 内外圈复合",
            ha="center", va="center", fontsize=10.5, color=ACC_EC,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#fdf4f3", edgecolor=ACC_EC, linewidth=1.1))

    ax.text(2, 51.5, "东南大学 DDS（Drivetrain Dynamics Simulator）结构框图与测点布置",
            ha="left", va="center", fontsize=14, color="#1f3b63", fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = FIG / "dds_platform_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
