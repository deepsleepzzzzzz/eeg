"""导师问题1/2 的作图：从 tune/log.json 读延迟消融结果。

图1 lag_steps_curve.png     连续组 acc vs 最大延迟  -> 回答"延迟取到多少性能最好"
图2 lag_sparse_vs_dense.png 稀疏(vs)连续 柱状对比    -> 回答"是否需要间隔"
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "datas" / "experiments_seu" / "tune" / "log.json"
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# variant -> (标签, 最大延迟, 类型)
VMAP = {
    "zq_s1_w1600_over":   ("1",          1,  "dense"),
    "zq_s3_w1600_over":   ("1-3",        3,  "dense"),
    "zq_s5_w1600_over":   ("1-5",        5,  "dense"),
    "zq_s7_w1600_over":   ("1-7",        7,  "dense"),
    "zq_s9_w1600_over":   ("1-9",        9,  "dense"),
    "zq_s12_w1600_over":  ("1-12",      12,  "dense"),
    "zq_s15_w1600_over":  ("1-15",      15,  "dense"),
    "zq_s135_w1600_over": ("1,3,5",      5,  "sparse"),
    "zq_s1357_w1600_over": ("1,3,5,7",   7,  "sparse"),
    "zq_s13579_w1600_over": ("1,3,5,7,9", 9, "sparse"),
    "zq_s147_w1600_over": ("1,4,7",      7,  "sparse"),
}
TASK_CN = {"bearing": "轴承", "gear": "齿轮箱"}
COLOR = {"bearing": "#c0392b", "gear": "#2471a3"}


def load() -> dict:
    log = json.loads(LOG.read_text(encoding="utf-8"))
    out = {}
    for t in log["trials"]:
        if t.get("arch") != "cnn" or t.get("condition") != "30_2":
            continue
        if t["variant"] not in VMAP:
            continue
        out[(t["task"], t["variant"])] = float(t["val_acc"])
    return out


def fig_curve(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for task in ("bearing", "gear"):
        pts = sorted({(VMAP[v][1], d[(task, v)]) for v in VMAP
                      if VMAP[v][2] == "dense" and (task, v) in d})
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys, marker="o", lw=2, ms=6, color=COLOR[task],
                label=f"{TASK_CN[task]} 30 Hz-2 V")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha="center", color=COLOR[task])
    ax.set_xlabel("最大延迟步数（连续取 1…N）", fontsize=12)
    ax.set_ylabel("验证集准确率 (%)", fontsize=12)
    ax.set_title("多时间步延迟：连续取到多少步性能最好（30 Hz-2 V，CNN，仅验证集）", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted({v[1] for v in VMAP.values()}))
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    out = FIG / "lag_steps_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


def fig_sparse_vs_dense(d: dict) -> None:
    # 三种"最大延迟"下比较 连续 vs 稀疏
    groups = [
        ("最大延迟 5", ["zq_s5_w1600_over"], ["zq_s135_w1600_over"]),
        ("最大延迟 7", ["zq_s7_w1600_over"], ["zq_s1357_w1600_over", "zq_s147_w1600_over"]),
        ("最大延迟 9", ["zq_s9_w1600_over"], ["zq_s13579_w1600_over"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, task in zip(axes, ("bearing", "gear")):
        labels, dense_vals, sparse_list = [], [], []
        for gname, dv, sv in groups:
            labels.append(gname)
            dense_vals.append(d[(task, dv[0])] * 100 if (task, dv[0]) in d else None)
            sparse_list.append([(VMAP[v][0], d[(task, v)] * 100) for v in sv if (task, v) in d])
        x = range(len(labels))
        w = 0.26
        ax.bar([i - w / 2 for i in x], dense_vals, w, color="#95a5a6", label="连续取（1…N）")
        for i, sl in enumerate(sparse_list):
            for k, (tag, val) in enumerate(sl):
                off = w / 2 + k * w
                ax.bar([i + off], [val], w, color="#2471a3" if k == 0 else "#1e8449",
                       label=("稀疏取（1,3,5…）" if i == 0 and k == 0 else
                              ("稀疏取（1,4,7）" if tag == "1,4,7" else None)))
                ax.annotate(f"{val:.2f}", (i + off, val), textcoords="offset points",
                            xytext=(0, 4), fontsize=7.5, ha="center")
        for i, v in enumerate(dense_vals):
            if v:
                ax.annotate(f"{v:.2f}", (i - w / 2, v), textcoords="offset points",
                            xytext=(0, 4), fontsize=7.5, ha="center")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(f"{TASK_CN[task]} 30 Hz-2 V", fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(88, 101.5)
    axes[0].set_ylabel("验证集准确率 (%)", fontsize=12)
    h, lb = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            if b not in lb:
                h.append(a); lb.append(b)
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=10.5, framealpha=0.9)
    fig.suptitle("是否需要间隔：稀疏取步 vs 连续取步（同最大延迟下对比）", fontsize=13.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    out = FIG / "lag_sparse_vs_dense.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    data = load()
    print("找到", len(data), "条结果")
    for k in sorted(data):
        print("  ", k, f"{data[k]:.4f}")
    fig_curve(data)
    fig_sparse_vs_dense(data)
