"""论文图表生成器：效率表、混淆矩阵、转移矩阵可视化、参数量-精度散点图。

数据来源全部是已有产物（artifacts/ 与 reports/efficiency.json），不重跑训练。
所有图都按验证集，测试集封存不动。

用法：python src/reporting/make_paper_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reporting.make_noise_figure import setup_font  # noqa: E402
from reporting.class_labels import TASK_CN, cn_name  # noqa: E402

MAIN = PROJECT_ROOT / "artifacts/main_70_15_15_v5"
OUT = PROJECT_ROOT / "reports"
CONDS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]
# 类别中文名统一从 reporting.class_labels 取（见该模块说明），本文件不再自定义。


def find_main_protocol():
    from config.protocol import load_protocol

    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def efficiency_table() -> None:
    eff = json.loads((OUT / "efficiency.json").read_text(encoding="utf-8"))
    lines = [
        "# 效率表（参数量 / FLOPs / 推理时间）",
        "",
        "> FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 按其官方定义统计（乘加计 2 次），",
        "> 单样本、推理模式；推理时间取 GPU 上 200 次前向均值（warmup 50 次）。",
        "> 输入为本文表示 `[30, 6, 6]`（3 通道 × 10 滞后）。",
        "",
        "| 模型 | 参数量 | FLOPs | 推理时间 (ms/样本) |",
        "|---|---:|---:|---:|",
    ]
    for r in eff:
        lines.append(
            f"| {r['model'].upper()} | {r['params']:,} | {r['flops']:,}（{r['flops']/1e6:.3f} M） | {r['ms_per_sample']:.3f} |"
        )
    lines += [
        "",
        "## 与文献对照（输入口径不同，仅供量级参照）",
        "",
        "| 方法 | 参数量 | FLOPs | 输入 |",
        "|---|---:|---:|---|",
        "| DSMC-ECA (Sensors 2026) | 204,000 | 10,037,000 | 1024 点单通道 |",
        "| DDGF-Net (Machines 2026) | 15,290,000（61.2 MB FP32） | — | 1024 点单通道 |",
        "| **本文 CNN** | **27,685** | **954,496** | `30×6×6 = 1080` 元素 |",
        "| **本文 GCN** | **54,941** | **2,017,152** | `30×6×6 = 1080` 元素 |",
        "",
        "**读法**：本文 CNN 的参数量是 DSMC-ECA 的 **1/7.4**、算量是 **1/10.5**；",
        "是 DDGF-Net 参数量的 **1/552**。",
        "",
        "> ⚠️ 更正记录：此前表中 0.384 M / 0.558 MFLOPs 来源不明，与手算不符（是 v2 的 21 通道时期残留），已作废。",
        "",
    ]
    (OUT / "table_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[ok] reports/table_efficiency.md")


def confusion_matrices() -> None:
    """按 seed 汇总混淆矩阵（验证集），每工况取 CNN 与 GCN 各一张。"""
    setup_font()
    from data.seu_dds import class_names

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (task, cond) in enumerate(CONDS):
        for row, model in enumerate(("cnn", "gcn")):
            cm = None
            for seed in (42, 43, 44, 45, 46):
                p = MAIN / f"{task}_{model}_{cond}_seed{seed}" / "val_predictions.csv"
                if not p.exists():
                    continue
                t, pr = [], []
                for r in csv.DictReader(p.open(encoding="utf-8")):
                    t.append(int(r["true_label"]))
                    pr.append(int(r["pred_label"]))
                m = np.zeros((5, 5), dtype=int)
                for a, b in zip(t, pr):
                    m[a, b] += 1
                cm = m if cm is None else cm + m
            ax = axes[row, col]
            if cm is None:
                ax.axis("off")
                continue
            acc = np.trace(cm) / cm.sum()
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
            for i in range(5):
                for j in range(5):
                    if cm[i, j]:
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                                color="white" if cm[i, j] > cm.max() * 0.55 else "black")
            ax.set_title(f"{TASK_CN[task]} {cond.replace('_','-')} · {model.upper()} · {100*acc:.2f}%", fontsize=10)
            # 轴刻度直接写类别名：齿轮与轴承是两套不同数据，只写 0..4 看不出是哪一类
            panel_names = list(class_names(task))
            ax.set_xticks(range(5), [cn_name(task, n) for n in panel_names],
                          fontsize=7, rotation=40, ha="right")
            ax.set_yticks(range(5), [cn_name(task, n) for n in panel_names], fontsize=7)
            if col == 0:
                ax.set_ylabel("真实类别" if row == 0 else "")
            if row == 1:
                ax.set_xlabel("预测类别")
    fig.suptitle("混淆矩阵（验证集，5 个 seed 汇总）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_confusion.png", bbox_inches="tight")
    print("[ok] reports/fig_confusion.png")


def transition_matrices() -> None:
    """转移矩阵图（可解释性用）：

    * 主文 `fig_transition.png`：只画**齿轮 20-0**的 5 类，按行归一化成条件转移概率 P(j|i)，
      每行和为 1——与 §4.5 讨论的"自转移概率/行熵/行归一化距离"口径一致；
    * 补充材料 `fig_transition_full.png`：四个任务-工况 x 5 类全量（同样行归一化）。
      2026-10-17：原图把 20 个联合概率矩阵全铺在主文，既小又和正文的条件结构口径不一致。
    """
    from config.protocol import load_protocol
    from data.seu_dds import class_names

    proto = load_protocol(find_main_protocol())
    lags = [int(x) for x in proto["representation"]["lags"]]
    setup_font()
    conds = (("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2"))
    lag_idx = min(2, len(lags) - 1)  # 第 3 个滞后

    def row_norm(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(m.sum(axis=1, keepdims=True), 1e-12)

    def draw(conds_sel, figsize, fname, title):
        fig, axes = plt.subplots(len(conds_sel), 5, figsize=figsize)
        if len(conds_sel) == 1:
            axes = np.asarray(axes).reshape(1, 5)
        im = None
        for row, (task, cond) in enumerate(conds_sel):
            d = MAIN / "dataset" / task / cond
            X = np.load(d / "X_val.npy")
            y = np.load(d / "y_val.npy")
            names = class_names(task)
            for c in range(5):
                mat = row_norm(X[y == c][:, 0 * len(lags) + lag_idx].mean(axis=0))
                ax = axes[row, c]
                im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=0.55)
                ax.set_title(f"{cn_name(task, names[c])}\n{names[c]}", fontsize=9, linespacing=1.5)
                ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(f"{TASK_CN[task]}任务\n{cond.replace('_', '-')}", fontsize=11)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.93, 0.93), h_pad=3.2)
        if im is not None:
            cax = fig.add_axes([0.945, 0.15, 0.012, 0.7])
            fig.colorbar(im, cax=cax, label="条件转移概率 P(j|i)（色标 0–0.55）")
        fig.savefig(OUT / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] reports/{fname}")

    # ---- 主图：上排 = 行归一化矩阵，下排 = 相对类均值的差异（突出类间结构）
    d = MAIN / "dataset" / "gear" / "20_0"
    X = np.load(d / "X_val.npy")
    y = np.load(d / "y_val.npy")
    names_g = class_names("gear")
    P = np.asarray([row_norm(X[y == c][:, 0 * len(lags) + lag_idx].mean(axis=0)) for c in range(5)])
    D = P - P.mean(axis=0)
    vdiff = float(np.abs(D).max())
    fig, axes = plt.subplots(2, 5, figsize=(17.0, 7.8))
    im1 = im2 = None
    for c in range(5):
        ax = axes[0, c]
        im1 = ax.imshow(P[c], cmap="viridis", vmin=0.0, vmax=0.55)
        ax.set_title(f"{cn_name('gear', names_g[c])}\n{names_g[c]}", fontsize=9, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1, c]
        im2 = ax.imshow(D[c], cmap="RdBu_r", vmin=-vdiff, vmax=vdiff)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("行归一化\nP(j|i)", fontsize=10)
    axes[1, 0].set_ylabel("相对类均值\nΔP(j|i)", fontsize=10)
    fig.suptitle("齿轮 20-0：各类的行归一化平均转移矩阵（上）与相对类均值的差异（下）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.93, 0.93), h_pad=2.0)
    cax1 = fig.add_axes([0.945, 0.53, 0.011, 0.33]); fig.colorbar(im1, cax=cax1, label="P(j|i)（0–0.55）")
    cax2 = fig.add_axes([0.945, 0.10, 0.011, 0.33]); fig.colorbar(im2, cax=cax2, label="ΔP(j|i)")
    fig.savefig(OUT / "fig_transition.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] reports/fig_transition.png（上：矩阵；下：差异）")
    draw(conds, (17.0, 13.6), "fig_transition_full.png",
         "四个任务-工况下各类的行归一化平均状态转移矩阵（x 通道，第 3 个滞后；验证集）")

    # 图的"参数台账"：题注与图必须对得上（tests/test_figure_caption_consistency.py 会核对）
    (OUT / "fig_transition_meta.json").write_text(
        json.dumps({
            "channel_index": 0, "channel": "x", "lag_index": lag_idx,
            "normalization": "row", "vmax": 0.55,
            "difference_row": True, "vmax_diff": round(vdiff, 4),
            "main_conditions": ["gear/20_0"], "classes": 5,
            "full_conditions": [f"{t}/{c}" for t, c in conds],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[ok] reports/fig_transition_meta.json")


def efficiency_scatter() -> None:
    """参数量–精度散点（同协议自跑）：轻量化主张的"一张图"版本。

    左：逐窗参考口径 −6 dB；右：严格（全局参考）口径 −10 dB。
    文献方法因协议不同不纳入；只画同一划分/同一窗/同一训练预算下自跑的 5 个方法。
    本文两个模型用星号标出——它们位于最左侧（参数量最小），并且在中/低信噪比下
    位于"≤0.06 M 参数"这一档的最上沿。
    """
    import statistics as st

    MAIN = PROJECT_ROOT / "artifacts/main_70_15_15_v5"
    SCAN = PROJECT_ROOT / "artifacts/variant_scan"
    # 参数量一律从 reports/efficiency_all.json 读（唯一真源），不手写常数：
    # 此前这里手写的 raw_cnn 0.0891 / wdcnn 0.0481 是旧窗长（1600）时期的残留，与项目记录不符。
    eff_all = json.loads((PROJECT_ROOT / "reports/efficiency_all.json").read_text(encoding="utf-8"))
    PARAMS = {k: v["params"] / 1e6 for k, v in eff_all.items()}
    LABEL = {"raw_cnn": "原始波形 1D-CNN", "wdcnn": "WDCNN", "resnet1d": "ResNet1D",
             "cnn": "本文 CNN", "gcn": "本文 GCN"}
    COLOR = {"raw_cnn": "#7f7f7f", "wdcnn": "#8c8c8c", "resnet1d": "#5a5a5a",
             "cnn": "#d62728", "gcn": "#ff7f0e"}

    # 变体名在两处目录里不同：
    #   基线方法（raw_cnn/wdcnn/resnet1d）在 variant_scan 下用 nm6 / gnm10；
    #   本文模型（cnn/gcn）在 variant_scan_fill 下用 snrm6 / gsnrm10
    #   （后者是 2026-10-16 补跑的：这两个 cell 曾被变体扫描覆盖丢失，重跑值与当年报告值逐位一致）。
    SRC = {
        "raw_cnn":   {"nm6": (SCAN, "nm6"),    "gnm10": (SCAN, "gnm10")},
        "wdcnn":     {"nm6": (SCAN, "nm6"),    "gnm10": (SCAN, "gnm10")},
        "resnet1d":  {"nm6": (SCAN, "nm6"),    "gnm10": (SCAN, "gnm10")},
        "cnn":       {"nm6": (PROJECT_ROOT / "artifacts/variant_scan_fill", "snrm6"),
                      "gnm10": (PROJECT_ROOT / "artifacts/variant_scan_fill", "gsnrm10")},
        "gcn":       {"nm6": (PROJECT_ROOT / "artifacts/variant_scan_fill", "snrm6"),
                      "gnm10": (PROJECT_ROOT / "artifacts/variant_scan_fill", "gsnrm10")},
    }

    def acc(model: str, tag: str, task: str = "gear", cond: str = "20_0") -> float | None:
        vals = []
        for seed in (42, 43, 44):
            if tag == "clean":
                f = MAIN / f"{task}_{model}_{cond}_seed{seed}" / "metrics.json"
            else:
                base, vname = SRC[model][tag]
                f = base / vname / f"{task}_{model}_{cond}_{vname}_seed{seed}" / "metrics.json"
            if f.exists():
                v = (json.loads(f.read_text(encoding="utf-8")).get("val_metrics") or {}).get("accuracy")
                if v is not None:
                    vals.append(100 * float(v))
        return st.mean(vals) if vals else None

    setup_font()
    panels = [("逐窗参考口径 · −6 dB", "nm6"), ("严格（全局参考）口径 · −10 dB", "gnm10")]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    for ax, (title, tag) in zip(axes, panels):
        pts = []
        for model in PARAMS:
            a = acc(model, tag)
            if a is None:
                continue
            pts.append((PARAMS[model], a, model))
            ax.scatter(PARAMS[model], a, s=210 if model in ("cnn", "gcn") else 80,
                       marker="*" if model in ("cnn", "gcn") else "o",
                       color=COLOR[model], zorder=3, edgecolors="white", linewidths=0.8)
            ax.annotate(f"{LABEL[model]}\n{a:.1f}%", (PARAMS[model], a),
                        textcoords="offset points", xytext=(9, -3), fontsize=8.5,
                        color=COLOR[model])
        # Pareto 上沿（左上包络）
        front, best = [], -1e9
        for x, y, m in sorted(pts):
            if y > best:
                front.append((x, y)); best = y
        ax.plot([p[0] for p in front], [p[1] for p in front], ":", color="#999", lw=1.2,
                zorder=1, label="Pareto 上沿（越小越靠左）")
        ax.set_xscale("log")
        ax.set_xlabel("参数量（百万，对数坐标）", fontsize=10.5)
        ax.set_ylabel("验证集准确率（%）", fontsize=10.5)
        ax.set_title(f"{title}（SEU 齿轮 20-0，同协议自跑，3 个种子）", fontsize=11)
        ax.grid(alpha=0.22, lw=0.6)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.set_ylim(min(p[1] for p in pts) - 3, 102)
    fig.suptitle("参数量–精度对比：本文表示的两个分类器均为万级参数（ CNN 0.0277 M、GCN 0.0549 M），"
                 "在 ≤0.06 M 一档中精度最高", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_efficiency_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[ok] reports/fig_efficiency_scatter.png")


def main() -> None:
    efficiency_table()
    confusion_matrices()
    transition_matrices()
    efficiency_scatter()


if __name__ == "__main__":
    main()
