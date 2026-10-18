"""导师第二轮要求的图（2026-10-02）。

产出（-> doc/figures/seu/teacher/）：
  signals_gear.png / signals_bearing.png   问题3：各工况信号
  flowchart.png                            问题4：方法总流程
  cond_matrices_gear.png / _bearing.png    问题7：不同工况的状态转移矩阵对比
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEU = ROOT / "datas" / "seu" / "gearbox"
REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"
BINS, STEPS, CH = 6, 7, 3
CONDS = ["20_0", "30_2"]
COND_LABEL = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}


# --------------------------------------------------------------------------- #
# 问题 3：各工况信号图
# --------------------------------------------------------------------------- #
def _read_signal(path: Path, n: int = 2000, cols=(1, 2, 3)) -> np.ndarray:
    """读 DDS CSV 数值段，自动识别 tab / 逗号（两种格式混用）。返回 [n, len(cols)]。"""
    start, delim = None, "\t"
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.lstrip()
            if s and (s[0].isdigit() or s[0] in "+-."):
                start = i
                delim = "\t" if s.count("\t") >= 7 else ","
                break
    if start is None:
        return np.empty((0, len(cols)), dtype=np.float64)
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


def fig_signals(task: str) -> None:
    meta = json.loads((REP / task / "30_2" / "meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    subdir = "gearset" if task == "gear" else "bearingset"
    pretty = {"gear": "齿轮箱", "bearing": "轴承"}[task]

    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.0),
                             sharex=True)
    for ci, cond in enumerate(CONDS):
        for j, name in enumerate(names):
            ax = axes[ci, j]
            p = _match_csv(subdir, name, cond)
            if p is None:
                ax.set_axis_off()
                continue
            sig = _read_signal(p, n=2000)
            t = np.arange(sig.shape[0]) / 5120.0
            for k, (c, lab) in enumerate(zip(range(3), ["x", "y", "z"])):
                ax.plot(t, sig[:, k], lw=0.7, alpha=0.65,
                        label=lab if (ci == 0 and j == 0) else None)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{COND_LABEL[cond]}\n幅值", fontsize=9)
            if ci == 1:
                ax.set_xlabel("时间 (s)", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.7)
    fig.suptitle(f"SEU DDS {pretty}数据集：两种工况下五类状态的原始振动信号（三轴）",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / f"signals_{task}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


# --------------------------------------------------------------------------- #
# 问题 4：方法总流程图
# --------------------------------------------------------------------------- #
def fig_flowchart() -> None:
    steps = [
        ("原始振动信号", "三轴振动通道，5120 Hz；每类 1,048,560 点（≈3.4 min 连续信号）", "#e8f1fb"),
        ("分窗 + 逐窗 z-score", "窗长 1600 点，步长 800（50% 重叠）；逐窗标准化消除类间幅度量级差", "#e8f1fb"),
        ("等频符号化（6 状态）", "按分位数划分幅度区间 → 状态 {1,…,6}；区间边界仅由训练集拟合", "#eaf6ec"),
        ("多时间步状态转移", "步长 1→7：有向、带权，统计状态 i→j 的转移概率，得 6×6 概率矩阵", "#eaf6ec"),
        ("多通道堆叠张量", "3 个振动通道 × 7 个时间步 → 张量 [21, 6, 6]", "#fff4e0"),
        ("CNN / GCN 自动特征学习", "CNN 视矩阵为图像学局部模式；GCN 直接按图结构传播聚合", "#fff4e0"),
        ("训练阶段噪声增广（抗噪）", "对训练信号注入高斯白噪声（多 SNR 混合）再编码，使模型在训练时见过噪声分布，获得抗噪能力", "#f3e8ff"),
        ("故障分类（单工况 5 类）", "正常 / 4 类故障；两工况各自独立建模", "#fdeaea"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.52 * len(steps) + 0.45)
    ax.axis("off")
    for i, (title, sub, color) in enumerate(steps):
        y = 1.52 * len(steps) - 0.76 - i * 1.52
        ax.add_patch(FancyBboxPatch((0.7, y - 0.55), 8.6, 1.10,
                                    boxstyle="round,pad=0.08",
                                    fc=color, ec="#4a4a4a", lw=1.1))
        ax.text(5.0, y + 0.24, f"{i + 1}. {title}", ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(5.0, y - 0.26, sub, ha="center", va="center",
                fontsize=8.0, color="#3a3a3a", wrap=True)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - 0.63), (5.0, y - 0.97),
                                         arrowstyle="-|>", mutation_scale=14,
                                         lw=1.5, color="#555555"))
    ax.set_title("多时间步状态转移网络 + CNN/GCN 方法总流程", fontsize=12, pad=5)
    fig.tight_layout()
    out = FIG / "flowchart.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


# --------------------------------------------------------------------------- #
# 问题 7：不同工况的状态转移矩阵对比
# --------------------------------------------------------------------------- #
def _class_step1_means(task: str, cond: str):
    X = np.load(REP / task / cond / "X_train.npy").astype(np.float32)   # [B,21,6,6]
    y = np.load(REP / task / cond / "y_train.npy").astype(np.int64)
    meta = json.loads((REP / task / cond / "meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    B = X.shape[0]
    Xr = X.reshape(B, CH, STEPS, BINS, BINS).mean(axis=1)              # [B,7,6,6]
    return names, {names[c]: Xr[y == c].mean(axis=0)[0] for c in range(len(names))}


def _class_means(task: str, cond: str):
    """Return class names and per-class mean matrices for all channels.

    Returns (names, means) where means[cls] is [CH, BINS, BINS] (channel-major,
    no averaging across channels)."""
    X = np.load(REP / task / cond / "X_train.npy").astype(np.float32)   # [B,21,6,6]
    y = np.load(REP / task / cond / "y_train.npy").astype(np.int64)
    meta = json.loads((REP / task / cond / "meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    B = X.shape[0]
    Xr = X.reshape(B, CH, STEPS, BINS, BINS)                            # [B,3,7,6,6]
    step1 = Xr[:, :, 0, :, :]                                           # [B,3,6,6]
    return names, {names[c]: step1[y == c].mean(axis=0) for c in range(len(names))}


def fig_condition_matrices(task: str) -> None:
    names, m20 = _class_means(task, "20_0")
    _, m30 = _class_means(task, "30_2")
    pretty = {"gear": "齿轮箱", "bearing": "轴承"}[task]
    CH_LABEL = ["x 轴", "y 轴", "z 轴"]

    # layout: rows = 2 conditions x 3 channels + 1 diff row; cols = classes
    fig, axes = plt.subplots(CH * 2 + 1, len(names),
                             figsize=(3.0 * len(names), 2.05 * (CH * 2 + 1)),
                             constrained_layout=True)
    ticks = list(range(BINS))
    tlabels = [str(i + 1) for i in ticks]

    # shared color limits
    vmax = max(float(m20[n].max()) for n in names + list(m30))
    vmax = float(max(vmax, 1e-8))
    dmax = max(float(np.abs(m30[n] - m20[n]).max()) for n in names)
    dmax = float(dmax) or 1e-8

    im_top = None
    im_diff = None
    tvd_rows = []
    for j, name in enumerate(names):
        a3, b3 = m20[name], m30[name]           # [3,6,6]
        d3 = b3 - a3
        for c in range(CH):
            for i, (mat, tag) in enumerate([(a3[c], f"{COND_LABEL['20_0']}"),
                                            (b3[c], f"{COND_LABEL['30_2']}")]):
                row = c * 2 + i
                ax = axes[row, j]
                im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=vmax)
                im_top = im
                ax.set_xticks(ticks); ax.set_yticks(ticks)
                ax.set_xticklabels(tlabels, fontsize=6.5); ax.set_yticklabels(tlabels, fontsize=6.5)
                if row == 0:
                    ax.set_title(name, fontsize=10.5)
                if j == 0:
                    ax.set_ylabel(f"{CH_LABEL[c]}·{tag}", fontsize=7.5)
                if row == CH * 2 - 1:
                    ax.set_xlabel("to", fontsize=7)
        # diff row: channel-mean difference for readability + per-channel TVD
        ax = axes[CH * 2, j]
        d = d3.mean(axis=0)
        im = ax.imshow(d, cmap="coolwarm", vmin=-dmax, vmax=dmax)
        im_diff = im
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=6.5); ax.set_yticklabels(tlabels, fontsize=6.5)
        tvds = [float(np.abs(d3[c]).sum() / 2.0) for c in range(CH)]
        tvd_mean = float(np.abs(d).sum() / 2.0)
        tvd_rows.append((name, tvd_mean, tvds))
        ax.set_title(f"差值 (30-2−20-0)\nTVD={tvd_mean:.3f}", fontsize=9)
        if j == 0:
            ax.set_ylabel("差值\n(三轴均值)", fontsize=8)
        ax.set_xlabel("to", fontsize=7)

    if im_top is not None:
        cb1 = fig.colorbar(im_top, ax=list(axes[: CH * 2, :].ravel()), fraction=0.012, pad=0.01)
        cb1.set_label("转移概率", fontsize=9)
        cb1.ax.tick_params(labelsize=8)
    if im_diff is not None:
        cb2 = fig.colorbar(im_diff, ax=list(axes[CH * 2, :].ravel()), fraction=0.012, pad=0.01)
        cb2.set_label("概率差（红=30-2 更高，蓝=20-0 更高）", fontsize=9)
        cb2.ax.tick_params(labelsize=8)

    fig.suptitle(f"SEU DDS {pretty}：两工况下各类的一步状态转移概率矩阵（x/y/z 分轴 + 差值）",
                 fontsize=13)
    out = FIG / f"cond_matrices_{task}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)
    # per-channel TVD summary for the docx text
    print(f"  per-channel TVD ({task}):")
    for name, tvd_mean, tvds in tvd_rows:
        print(f"    {name:8s} mean={tvd_mean:.4f}  x={tvds[0]:.4f} y={tvds[1]:.4f} z={tvds[2]:.4f}")


if __name__ == "__main__":
    for t in ("gear", "bearing"):
        fig_signals(t)
    fig_flowchart()
    for t in ("gear", "bearing"):
        fig_condition_matrices(t)
