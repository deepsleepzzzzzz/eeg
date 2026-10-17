# -*- coding: utf-8 -*-
"""导师补充要求（2026-10-02）对应的四组图 —— 全部基于当前主协议 v5，只读，不碰测试集。

  ① 各工况的信号（转速/负载差异 + 故障幅值差异）        -> fig_condition_signals.png
  ② 故障类型示意图（齿轮 5 类 / 轴承 5 类，自绘）        -> fig_fault_schematic.png
  ③ 表示特征的 t-SNE（同任务下两种工况画在一张图里）    -> fig_tsne.png
  ④ 不同工况的转移矩阵 + 差值 + 总变差距离（每任务一张） -> fig_cond_matrices_gear.png / _bearing.png

用法： python src/reporting/make_teacher_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import dataset_dir, load_protocol, raw_root, window_cfg  # noqa: E402
from data.seu_dds import class_file, class_names, normalize_condition, task_dir, load_seu_csv  # noqa: E402
from reporting.class_labels import cn_name  # noqa: E402
from reporting.make_noise_figure import setup_font  # noqa: E402

OUT = PROJECT_ROOT / "reports"
COMBOS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]
COND_CN = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}
TASK_CN = {"gear": "齿轮", "bearing": "轴承"}
FS = 5120


def main_protocol() -> dict:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")


PROTO = main_protocol()
CH = [int(c) for c in PROTO["dataset"]["channel_indices"]]
LAGS = [int(x) for x in PROTO["representation"]["lags"]]
BINS = int(PROTO["preprocessing"]["bins"])


# --------------------------------------------------------------------------- #
# ① 各工况的信号
# --------------------------------------------------------------------------- #
def fig_condition_signals() -> None:
    setup_font()
    n = 1536                      # 300 ms
    start = 20000
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 9.6), sharex=True)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    for r, (task, cond) in enumerate(COMBOS):
        names = list(class_names(task))
        d = raw_root(PROTO) / "gearbox" / ("gearset" if task == "gear" else "bearingset")
        base = load_seu_csv(d / class_file(task, names[0], cond))[:, CH]
        fault = load_seu_csv(d / class_file(task, names[1], cond))[:, CH]
        seg_b = base[start:start + n]
        seg_f = fault[start:start + n]
        t = np.arange(n) / FS * 1000.0                       # ms
        for c in range(3):
            ax = axes[r, c]
            ax.plot(t, seg_b[:, c], lw=0.7, color="#8c8c8c", alpha=0.95, label="健康")
            ax.plot(t, seg_f[:, c], lw=0.7, color=colors[c], alpha=0.85,
                    label=f"{cn_name(task, names[1])}")
            ax.tick_params(labelsize=7.5, length=2)
            ax.grid(alpha=0.18, lw=0.5)
            if r == 0:
                ax.set_title(f"{'xyz'[c]} 通道", fontsize=10)
            if c == 0:
                rpm = 1200 if cond == "20_0" else 1800
                ax.set_ylabel(f"{TASK_CN[task]} {COND_CN[cond]}\n（{rpm} r/min）", fontsize=9)
            if r == 3:
                ax.set_xlabel("时间 (ms)", fontsize=9)
            if r == 0 and c == 2:
                ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        # 标出一个旋转周期
        per = 1000.0 / (20.0 if cond == "20_0" else 30.0)
        axes[r, 2].annotate("", xy=(per, axes[r, 2].get_ylim()[0]),
                            xytext=(0, axes[r, 2].get_ylim()[0]),
                            arrowprops=dict(arrowstyle="<->", lw=1.0, color="#444"))
        axes[r, 2].text(per, axes[r, 2].get_ylim()[0], f"  1 转 = {per:.1f} ms",
                        fontsize=7.5, va="bottom", color="#444")
    fig.suptitle("四个任务-工况组合的振动信号（健康 vs 故障，x/y/z 三通道）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_condition_signals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[ok] reports/fig_condition_signals.png")


# --------------------------------------------------------------------------- #
# ② 故障类型示意图（自绘）
# --------------------------------------------------------------------------- #
def _gear(ax, kind: str) -> None:
    """画一个齿轮示意；kind ∈ health/chipped/miss/root/surface。"""
    R, teeth = 1.0, 18
    ax.add_patch(Circle((0, 0), R, fill=False, lw=1.6, color="#333"))
    ax.add_patch(Circle((0, 0), R * 0.30, fill=False, lw=1.2, color="#333"))
    for k in range(teeth):
        a = 2 * np.pi * k / teeth
        w, h = R * 0.11, R * 0.20
        cx, cy = np.cos(a), np.sin(a)
        pts = []
        for dx, dy in ((-w, 0), (-w * 0.75, h), (w * 0.75, h), (w, 0)):
            pts.append((cx * (R + dy) - np.sin(a) * dx, cy * (R + dy) + np.cos(a) * dx))
        skip = (kind == "miss" and k == 4)
        if not skip:
            ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor="#333", lw=1.0))
        if kind == "chipped" and k == 4:
            ax.add_patch(Polygon([pts[1], pts[2], (pts[1][0] * 0.92, pts[1][1] * 0.92)],
                                 closed=True, facecolor="#d62728", edgecolor="none", alpha=0.85))
        if kind == "root" and k == 12:
            a = 2 * np.pi * 12 / teeth
            x0, y0 = np.cos(a) * R * 0.97, np.sin(a) * R * 0.97
            ax.add_patch(FancyArrowPatch((x0 * 1.35, y0 * 1.35), (x0, y0),
                                         arrowstyle="-", lw=1.8, color="#d62728"))
            ax.text(x0 * 1.45, y0 * 1.45, "裂纹", fontsize=7.5, color="#d62728", ha="center")
        if kind == "surface" and k == 9:
            a = 2 * np.pi * 9 / teeth
            ax.add_patch(Wedge((0, 0), R * 1.0, np.degrees(a) - 7, np.degrees(a) + 7,
                               width=R * 0.22, facecolor="#d62728", edgecolor="none", alpha=0.28))
            ax.text(np.cos(a) * R * 1.5, np.sin(a) * R * 1.5, "磨损", fontsize=7.5,
                    color="#d62728", ha="center")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def _bearing(ax, kind: str) -> None:
    """画一个深沟球轴承示意；kind ∈ health/outer/inner/ball/comb。"""
    Rout, Rin = 1.0, 0.52
    ax.add_patch(Circle((0, 0), Rout, fill=False, lw=2.0, color="#333"))
    ax.add_patch(Circle((0, 0), Rout * 0.86, fill=False, lw=2.0, color="#333"))
    ax.add_patch(Circle((0, 0), Rin, fill=False, lw=2.0, color="#333"))
    ax.add_patch(Circle((0, 0), Rin * 0.72, fill=False, lw=1.2, color="#333"))
    for k in range(8):
        a = 2 * np.pi * k / 8
        ax.add_patch(Circle((np.cos(a) * 0.77, np.sin(a) * 0.77), 0.105,
                            facecolor="none", edgecolor="#555", lw=1.0))
    marks = {
        "outer": [(0.0, 0.93, "外圈滚道", 0.0, 1.55)],
        "inner": [(0.0, 0.47, "内圈滚道", 0.0, -1.55)],
        "ball": [(np.cos(np.pi / 4) * 0.77, np.sin(np.pi / 4) * 0.77, "滚动体", 1.15, 1.15)],
        "comb": [(np.cos(np.pi) * 0.90, 0.10, "保持架/组合", -1.5, 0.35)],
    }
    for (x, y, lab, tx, ty) in marks.get(kind, []):
        ax.add_patch(Circle((x, y), 0.14, facecolor="#d62728", edgecolor="none", alpha=0.75))
        ax.add_patch(FancyArrowPatch((tx * 0.92, ty * 0.92), (x, y), arrowstyle="-",
                                     lw=1.1, color="#d62728"))
        ax.text(tx * 1.02, ty * 1.02, lab, fontsize=7.5, color="#d62728",
                ha="center", va="center")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_fault_schematic() -> None:
    setup_font()
    gear_kinds = ["health", "chipped", "miss", "root", "surface"]
    # 顺序必须与 class_names("bearing") = (health, ball, inner, outer, comb) 一致，
    # 否则会出现"标题写滚动体、标记画在外圈"这种错位（首版就踩了）。
    bear_kinds = ["health", "ball", "inner", "outer", "comb"]
    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.6))
    for c, k in enumerate(gear_kinds):
        ax = axes[0, c]
        _gear(ax, k)
        ax.set_title(cn_name("gear", class_names("gear")[c]), fontsize=11)
    for c, k in enumerate(bear_kinds):
        ax = axes[1, c]
        _bearing(ax, k)
        ax.set_title(cn_name("bearing", class_names("bearing")[c]), fontsize=11)
    axes[0, 0].text(-1.62, 1.42, "齿轮任务（gearset/）", fontsize=10, color="#555")
    axes[1, 0].text(-1.62, 1.42, "轴承任务（bearingset/）", fontsize=10, color="#555")
    fig.suptitle("五类健康状态示意（自绘示意，非实物照片）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # 注意：自 2026-10-16 起论文的 图 3(b) 使用**用户提供的**机理示意图
    # （reports/fig_fault_modes.png）。本函数自绘的那张保留备用，改名为 _selfmade，
    # 避免重跑本脚本时把用户给的图覆盖掉。
    fig.savefig(OUT / "fig_fault_schematic_selfmade.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[ok] reports/fig_fault_schematic_selfmade.png（备用；论文用的是 fig_fault_modes.png）")


# --------------------------------------------------------------------------- #
# ③ t-SNE：同一任务下两种工况画在一张图里
# --------------------------------------------------------------------------- #
def fig_tsne() -> None:
    """表示特征的 t-SNE：两种工况叠加在一张图里，并画出每类的"工况漂移箭头"。

    为什么要画箭头：同一类别的 20-0 与 30-2 会各自聚成一团、两团相距很远，
    看上去"很散"。这其实是**工况漂移主导**的直接体现——把每一类的两个类心连起来，
    读者就能立刻分清"哪些距离是类别差异、哪些距离是工况差异"，而不是把它们看成噪声。
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    setup_font()
    color_of = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.6))
    for ax, task in zip(axes, ("gear", "bearing")):
        feats, labels, conds = [], [], []
        for cond in ("20_0", "30_2"):
            d = dataset_dir(PROTO, task, cond)
            X = np.load(d / "X_val.npy").astype(np.float32)
            y = np.load(d / "y_val.npy").astype(np.int64)
            feats.append(X.reshape(len(X), -1))
            labels.append(y)
            conds.append(np.full(len(y), 0 if cond == "20_0" else 1))
        Xf = np.concatenate(feats)
        yf = np.concatenate(labels)
        cf = np.concatenate(conds)
        # perplexity=50：本任务 1520 个点（两种工况各 760），sqrt(n)≈39，50 属常规取值；
        # 与 30 相比簇更紧凑、边界更清楚（参数只影响画面整洁度，重叠与否是数据性质）。
        Z = TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto",
                 random_state=42, max_iter=1500).fit_transform(
            StandardScaler().fit_transform(Xf))
        names = list(class_names(task))
        for ci in range(5):
            for cnd, mk, alpha, size in ((0, "o", 0.80, 16), (1, "^", 0.60, 16)):
                m = (yf == ci) & (cf == cnd)
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], s=size, marker=mk,
                               color=color_of[ci], alpha=alpha, edgecolors="none")
        # 每类的两个工况类心 + 漂移箭头
        shifts = []
        for ci in range(5):
            a = Z[(yf == ci) & (cf == 0)].mean(axis=0)
            b = Z[(yf == ci) & (cf == 1)].mean(axis=0)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color_of[ci],
                                        alpha=0.85, shrinkA=3, shrinkB=3))
            ax.scatter([a[0], b[0]], [a[1], b[1]], s=64, marker="D",
                       facecolor="white", edgecolor=color_of[ci], lw=1.4, zorder=5)
            mid = (a + b) / 2   # 类心中点（论文版不标注文字，避免与图例重复）
            shifts.append(float(np.linalg.norm(b - a)))
        # 类内散布（同类同工况内到自身类心的平均距离），用来和工况漂移比大小
        spread = []
        for ci in range(5):
            for cnd in (0, 1):
                m = (yf == ci) & (cf == cnd)
                if m.sum() > 1:
                    c0 = Z[m].mean(axis=0)
                    spread.append(float(np.linalg.norm(Z[m] - c0, axis=1).mean()))
        ax.set_title(f"{TASK_CN[task]}任务（圆=20-0，三角=30-2）", fontsize=11)
        ax.text(0.015, 0.015,
                f"工况漂移（同类两项工况类心距）均值 {np.mean(shifts):.1f}\n"
                f"类内散布（同类同工况内）均值 {np.mean(spread):.1f}\n"
                f"→ 漂移 ≈ 散布的 {np.mean(shifts) / np.mean(spread):.1f} 倍",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=0.9))
        ax.set_xticks([]); ax.set_yticks([])
        handles = [plt.Line2D([], [], marker="o", ls="", color=color_of[ci], markersize=8)
                   for ci in range(5)]
        ax.legend(handles, [cn_name(task, nm) for nm in names], fontsize=9,
                  loc="best", framealpha=0.92)
    fig.suptitle("表示特征（1080 维转移概率张量展平）的 t-SNE：每个类别内部都被两种工况拉成两团——"
                 "工况漂移是特征空间里最大的变异来源", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_tsne.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[ok] reports/fig_tsne.png（含工况漂移箭头）")


# --------------------------------------------------------------------------- #
# ④ 不同工况的转移矩阵 + 差值 + 总变差距离
# --------------------------------------------------------------------------- #
def fig_cond_matrices() -> None:
    setup_font()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    levels = np.linspace(0.0, 1.0, 11)
    cmap = ListedColormap(plt.get_cmap("Spectral_r")(np.linspace(0.05, 1.0, 10)))
    for task in ("gear", "bearing"):
        means, vmax = {}, 0.0
        for cond in ("20_0", "30_2"):
            d = dataset_dir(PROTO, task, cond)
            X = np.load(d / "X_train.npy").astype(np.float32)
            y = np.load(d / "y_train.npy").astype(np.int64)
            Xr = X.reshape(len(X), len(CH), len(LAGS), BINS, BINS)
            per_class = [Xr[y == c][:, 0, 0].mean(axis=0) for c in range(5)]
            means[cond] = per_class
            vmax = max(vmax, float(np.max(per_class)))
        norm = BoundaryNorm(levels * vmax, cmap.N)
        fig, axes = plt.subplots(3, 5, figsize=(14.6, 8.4))
        names = list(class_names(task))
        for c in range(5):
            for r, cond in enumerate(("20_0", "30_2")):
                ax = axes[r, c]
                im = ax.imshow(means[cond][c], cmap=cmap, norm=norm)
                if r == 0:
                    ax.set_title(cn_name(task, names[c]), fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{COND_CN[cond]}", fontsize=10)
                ax.set_xticks(range(BINS), list("ABCDEF"), fontsize=7)
                ax.set_yticks(range(BINS), list("ABCDEF"), fontsize=7)
            diff = np.abs(means["20_0"][c] - means["30_2"][c])
            tv = 0.5 * diff.sum()
            ax = axes[2, c]
            ax.imshow(diff, cmap="magma_r", vmin=0, vmax=float(vmax) * 0.4)
            ax.text(0.5, -0.28, f"TV = {tv:.3f}", transform=ax.transAxes,
                    ha="center", fontsize=8.5, color="#333")
            ax.set_xticks(range(BINS), list("ABCDEF"), fontsize=7)
            ax.set_yticks(range(BINS), list("ABCDEF"), fontsize=7)
            if c == 0:
                ax.set_ylabel("|20-0 − 30-2|", fontsize=9)
        fig.suptitle(f"{TASK_CN[task]}任务：两类工况的一步联合转移概率矩阵（x 通道）与差值"
                     f"（TV = 总变差距离，越大说明该类的工况差异越大）", fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / f"fig_cond_matrices_{task}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] reports/fig_cond_matrices_{task}.png")


if __name__ == "__main__":
    fig_condition_signals()
    fig_fault_schematic()
    fig_tsne()
    fig_cond_matrices()
    print("done")
