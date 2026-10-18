# -*- coding: utf-8 -*-
"""统一重绘「图 4 齿轮箱 / 图 5 轴承」的原始振动信号图——按图 7 的配色与版式风格。

为什么重绘
----------
原图（teacher_figs_round2.py::fig_signals）把 x/y/z 三轴叠画在同一坐标系里，
互相遮挡严重；且轴承数据存在一个更致命的量级问题：
  · 轴承「正常 / 复合故障」含约 **0.35 V/s** 的直流慢漂移 → 幅值 ±0.4 V；
  · 轴承三类故障（滚珠/内圈/外圈）斜率仅约 0.001 V/s → 幅值 ±0.02 V。
  两者相差 20 倍，统一 y 轴后故障类波形被压成一条直线，根本看不出故障特征。
  （齿轮箱子集不存在该问题：所有类别斜率都在 ±0.0007 V/s 以内，量级也一致。）

本次统一改法（两图同一套版式语言，并与图 7 对齐）
--------------------------------------------------
  ① 配色沿用图 7 通道色：x=#3b7ddd、y=#e8930c、z=#2f9e63；网格/边框/字号同图 7；
  ② **x/y/z 分通道显示**（行 = 通道，列 = 故障类别），彻底消除遮挡；
  ③ **逐通道去直流 + 去线性趋势**（消除各文件的直流偏置与慢漂移，只留交流振动成分；
     与后续 z-score + 等频符号化的处理口径一致）——对齿轮箱同样适用，
     经核对不会改变其相对量级关系（齿轮箱本无漂移，去趋势≈仅减直流）；
  ④ 每格纵轴按该通道真实幅值范围自动缩放并留 16% 空白，"每个通道都看得清"；
  ⑤ 类别标题改中文，左侧竖排标工况，每行左上角标"通道 x/y/z"；
  ⑥ 每格取文件前 2000 点（≈0.39 s），十格同一区间，保证横向可比。

数据全部取自真实 SEU 记录（gearset / bearingset 的 *_20_0.csv、*_30_2.csv 第 1–3 通道），
类别顺序直读实验产物的 meta.json，无任何硬编码或合成数据。

输出：
  doc/figures/seu/teacher/signals_gear_v2.png
  doc/figures/seu/teacher/signals_bearing_v2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

for _cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    try:
        font_manager.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEU = ROOT / "datas" / "seu" / "gearbox"
REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"

FS = 5120
NSHOW = 2000                       # 每格采样点数（≈0.39 s）
CONDS = ["20_0", "30_2"]
COND_LABEL = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}

# 与图 7 完全一致的通道配色
CH_COLORS = ["#3b7ddd", "#e8930c", "#2f9e63"]
CH_NAMES = ["x", "y", "z"]

TASK_CFG = {
    "gear": {
        "subdir": "gearset",
        "pretty": "齿轮箱",
        "cn": {"Health": "正常", "Chipped": "齿面剥落", "Miss": "缺齿",
               "Root": "齿根裂纹", "Surface": "齿面磨损"},
    },
    "bearing": {
        "subdir": "bearingset",
        "pretty": "轴承",
        "cn": {"health": "正常", "ball": "滚珠故障", "inner": "内圈故障",
               "outer": "外圈故障", "comb": "复合故障"},
    },
}


# --------------------------------------------------------------------------- #
def _read_signal(path: Path, n: int = NSHOW, cols=(1, 2, 3)) -> np.ndarray:
    """读 DDS CSV 数值段，自动识别 tab / 逗号分隔。返回 [n, 3]。"""
    start, delim = None, "\t"
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.lstrip()
            if s and (s[0].isdigit() or s[0] in "+-."):
                start = i
                delim = "\t" if s.count("\t") >= 7 else ","
                break
    if start is None:
        return np.empty((0, 3), dtype=np.float64)
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


def _match_csv(subdir: str, cls: str, cond: str) -> Path | None:
    for f in (SEU / subdir).glob(f"*_{cond}.csv"):
        stem = f.stem[: -(len(cond) + 1)]
        if stem.lower() == cls.lower():
            return f
    return None


# --------------------------------------------------------------------------- #
def fig_signals(task: str) -> None:
    cfg = TASK_CFG[task]
    subdir, pretty, cn = cfg["subdir"], cfg["pretty"], cfg["cn"]

    # 类别顺序直读真实实验产物，不硬编码
    meta = json.loads((REP / task / "30_2" / "meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    nc = len(names)
    nrow = len(CONDS) * len(CH_NAMES)                 # 2 工况 × 3 通道 = 6 行

    fig, axes = plt.subplots(nrow, nc,
                             figsize=(2.62 * nc, 1.34 * nrow),
                             sharex=True, sharey=False)
    fig.patch.set_facecolor("white")

    for ci, cond in enumerate(CONDS):
        for j, name in enumerate(names):
            p = _match_csv(subdir, name, cond)
            sig = _read_signal(p) if p is not None else np.empty((0, 3))
            # 去直流 + 去线性趋势：消除各文件的直流偏置与慢漂移，只留交流振动成分。
            # 轴承子集必须做（正常/复合 ~0.35 V/s vs 故障类 ~0.001 V/s，相差 20 倍，
            # 不去趋势则故障波形被压平）；齿轮箱子集本就无漂移，去趋势等价于仅减直流。
            if sig.size:
                tt = np.arange(sig.shape[0]) / FS
                for k in range(3):
                    sig[:, k] = sig[:, k] - np.polyval(np.polyfit(tt, sig[:, k], 1), tt)
            t = np.arange(sig.shape[0]) / FS
            for k in range(3):
                ax = axes[ci * 3 + k, j]
                if sig.size:
                    ax.plot(t, sig[:, k], lw=0.5, color=CH_COLORS[k], alpha=0.95)
                    lo, hi = float(sig[:, k].min()), float(sig[:, k].max())
                    pad = (hi - lo) * 0.16 + 1e-9
                    ax.set_ylim(lo - pad, hi + pad)
                ax.grid(alpha=0.22, lw=0.5)
                ax.tick_params(labelsize=6.0, length=2.0, pad=0.8)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                    sp.set_color("#c8d0da")
                if j == 0:
                    ax.text(0.015, 0.90, f"通道 {CH_NAMES[k]}", transform=ax.transAxes,
                            fontsize=6.8, color=CH_COLORS[k], fontweight="bold",
                            va="top", ha="left", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec=CH_COLORS[k], lw=0.5, alpha=0.92))
            if ci == 0:
                axes[0, j].set_title(cn.get(name, name), fontsize=10.0, pad=4.0)
        y_mid = 1.0 - (ci * 3 + 1.5) / nrow
        fig.text(0.006, y_mid + 0.012, COND_LABEL[cond], rotation=90, fontsize=9.0,
                 va="center", ha="left", color="#33475e", fontweight="bold")
        for j in range(nc):
            axes[ci * 3 + 2, j].set_xlabel("时间 (s)", fontsize=6.5, labelpad=1.0)

    fig.suptitle(f"SEU DDS {pretty}数据集：两种工况下五类状态的原始振动信号（x / y / z 分通道）",
                 fontsize=12.5, y=0.9965)
    fig.text(0.5, 0.9695,
             f"每格取文件前 {NSHOW} 点（≈{NSHOW / FS:.2f} s）；蓝 = x、橙 = y、绿 = z；"
             "已逐通道去直流 + 去线性趋势（消除各文件的直流偏置与慢漂移），三通道分行显示以避免遮挡",
             ha="center", va="center", fontsize=7.8, color="#8a94a6")

    fig.tight_layout(rect=(0.012, 0, 1, 0.9605))
    out = FIG / f"signals_{task}_v2.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    for t in ("gear", "bearing"):
        fig_signals(t)
