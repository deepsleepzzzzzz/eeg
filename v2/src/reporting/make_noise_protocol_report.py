"""噪声实验报告生成器：表 A（与文献同口径）、表 B（口径效应）、三线对照图。

为什么需要它（2026-10-15 的重大修正）：
  历史 add_noise 按【每个窗自己的功率】缩放噪声，使加噪窗能量 = 干净窗能量 × 常数，
  "窗能量"因此成为与表示无关的捷径。已改为可选的 global 口径（所有窗共用参考功率）。
  本脚本把两种口径的同一模型结果并排报出，让读者自行判断。

数据来源：
  artifacts/variant_scan               逐窗口径（snr*）+ 严格口径（gsnr*）
  artifacts/variant_scan_noise_v5      齿轮 30-2 的逐窗口径（分批跑过）
  reports/energy_only_*.csv            纯能量分类器对照（零学习）

用法：python src/reporting/make_noise_protocol_report.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reporting.make_noise_figure import setup_font  # noqa: E402

SCAN_DIRS = [PROJECT_ROOT / "artifacts/variant_scan", PROJECT_ROOT / "artifacts/variant_scan_noise_v5"]
RUN_RE = re.compile(r"^(?P<task>[a-z]+)_(?P<model>[a-z]+)_(?P<cond>\d+_\d+)_(?P<tag>[A-Za-z0-9]+)_seed\d+$")

# tag -> SNR(dB)；snr* = 逐窗口径，gsnr* = 严格（全局参考）口径
def tag_snr(tag: str):
    strict = tag.startswith("gsnr")
    core = tag[1:] if strict else tag
    if core == "base":
        return None, strict
    m = re.match(r"snr(m?)(\d+)$", core)
    if not m:
        return None, strict
    return (-1 if m.group(1) else 1) * float(m.group(2)), strict


CONDS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]
TASK_CN = {"gear": "齿轮", "bearing": "轴承"}

# 文献值（DDGF-Net, Machines 2026, 14(9):1013, Table 3/4；SEU 齿轮 20-0，逐样本参考功率口径）
LIT = [
    ("MLP + 8 手工特征", 23.9, 21.3, 25.7, "0.003 M", "否"),
    ("WDCNN", 45.4, 64.8, 90.4, "0.048 M", "否"),
    ("BiLSTM-Attn", 36.9, 53.9, 81.1, "0.159 M", "否"),
    ("ViT-1D", 36.1, 52.0, 75.1, "—", "否"),
    ("DConformer", 42.2, 61.6, 87.6, "—", "否"),
    ("STFT-2D", 46.5, 69.5, 94.7, "—", "否"),
    ("ResNet1D", 56.3, 82.7, 99.3, "3.85 M", "否"),
    ("DRSN", 56.5, 83.1, 99.3, "—", "软阈值（内置）"),
    ("DDGF-Net", 71.2, 93.5, 99.9, "15.29 M", "CVAE 降噪器 8.49 M"),
]
LIT_EXTRA = {  # 正文/图另给的点（不在 Table 4）
    "DDGF-Net": {-8: 84.3},
}


def collect():
    """返还 {(task, cond, strict, snr): [acc, ...]}"""
    buckets = {}
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            for run in sorted(sub.iterdir()):
                m = RUN_RE.match(run.name)
                if not m:
                    continue
                snr, strict = tag_snr(m.group("tag"))
                if snr is None and m.group("tag") != "base":
                    continue
                f = run / "metrics.json"
                if not f.exists():
                    continue
                metrics = json.loads(f.read_text(encoding="utf-8"))
                if metrics.get("test_metrics"):
                    raise SystemExit(f"[FAIL] {f} 含测试集指标，本报告只读验证集")
                acc = (metrics.get("val_metrics") or {}).get("accuracy")
                if acc is None:
                    continue
                key = (m.group("task"), m.group("cond"), strict, snr)
                buckets.setdefault(key, []).append(float(acc))
    return buckets


def get(buckets, task, cond, strict, snr):
    vals = buckets.get((task, cond, strict, snr))
    return 100.0 * statistics.mean(vals) if vals else None


def load_energy():
    out = {}
    for ref in ("per_window", "global"):
        p = PROJECT_ROOT / f"reports/energy_only_{ref}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            out[(r["task"], r["condition"], ref, float(r["snr_db"]))] = 100.0 * float(r["acc_pooled"])
    return out


SNRS_FIG = [20, 10, 4, 0, -2, -4, -6, -8, -10, -12, -16]


def main() -> None:
    buckets = collect()
    energy = load_energy()
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    # ---------- 表 A：与文献同口径的对比（齿轮 20-0，逐窗口径）----------
    lines = [
        "# 表 A · 与文献同口径对比（SEU 齿轮 20-0）",
        "",
        "> 全部在**逐样本参考功率**口径下（DDGF-Net 原文口径），训练与测试同 SNR。",
        "> 文献值取自 DDGF-Net (Machines 2026, 14(9):1013) Table 3/4，其 8 个基线为原文自跑。",
        "> 本文值为本文方法在**同一口径**下重跑（验证集，CNN+GCN 均值，seed 42/43/44）。",
        "",
        "| 方法 | −10 dB | −6 dB | 0 dB | 参数量 | 降噪模块 |",
        "|---|---:|---:|---:|---|---|",
    ]
    lit_rows = []
    for name, a10, a6, a0, prm, den in LIT:
        lines.append(f"| {name} | {a10} | {a6} | {a0} | {prm} | {den} |")
        lit_rows.append((name, a10, a6, a0))
    cnn10, cnn6, cnn0 = (get(buckets, "gear", "20_0", False, s) for s in (-10, -6, 0))
    lines.append(
        f"| **本文（逐窗口径）** | **{cnn10:.2f}** | **{cnn6:.2f}** | **{cnn0:.2f}** | "
        f"**0.0277 M (CNN) / 0.0549 M (GCN)** | **无** |"
    )
    g10, g6, g0 = (get(buckets, "gear", "20_0", True, s) for s in (-10, -6, 0))
    lines.append(
        f"| 本文（严格口径，供参照，**不可与本表其它行直接比较**） | {g10:.2f} | {g6:.2f} | {g0:.2f} | 同上 | 无 |"
    )
    lines += [
        "",
        "**读法**：同口径下本文 −10 dB 高于 DDGF-Net 13.2 点；但该口径使噪声功率随窗自适应，",
        "存在与表示无关的\"窗能量\"捷径（见表 B 与图）。严格口径下本文为 "
        f"{g10:.2f}%，与其报告值同水平。",
        "",
    ]
    (out_dir / "table_lit_same_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 表 B：口径效应对照 ----------
    lines = [
        "# 表 B · 噪声参考功率口径的效应（同一模型，唯一变量 = 噪声口径）",
        "",
        "> 逐窗 = `noise.reference: per_window`（噪声功率按每个窗自己的功率缩放）",
        "> 严格 = `noise.reference: global`（所有窗共用训练集算出的参考功率）",
        "> 训练与验证同 SNR；其余全部相同（划分 70/15/15、窗 2048/1024、表示、模型、seed）。",
        "",
    ]
    for task, cond in CONDS:
        lines += [f"## {TASK_CN[task]} {cond.replace('_', '-')}", "",
                  "| SNR (dB) | 逐窗口径 | 严格口径 | 差 | 纯能量(逐窗) | 纯能量(严格) |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for snr in SNRS_FIG:
            a = get(buckets, task, cond, False, float(snr))
            b = get(buckets, task, cond, True, float(snr))
            e1 = energy.get((task, cond, "per_window", float(snr)))
            e2 = energy.get((task, cond, "global", float(snr)))
            f = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            d = "—" if (a is None or b is None) else f"**{b - a:+.2f}**"
            lines.append(f"| {snr:+d} | {f(a)} | {f(b)} | {d} | {f(e1)} | {f(e2)} |")
        lines.append("")
    lines += [
        "## 结论",
        "",
        "1. 逐窗口径下，**纯能量分类器（零学习）与本文模型的准确率基本相同**"
        "（齿轮 20-0，−10 dB：84.0 vs 84.4），说明该口径下模型未提供超出窗能量的信息。",
        "2. 换成严格口径后，同一模型正常下降（−10 dB 由 84.4 降至 70.5；−16 dB 由 84.5 降至 43.6）。",
        "3. 因此**报告抗噪结果必须写明噪声参考功率的口径**，否则跨论文不可比。",
        "",
    ]
    (out_dir / "table_noise_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 图：三（四）线对照 ----------
    setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    for ax, (task, cond) in zip(axes.ravel(), CONDS):
        xs = [s for s in SNRS_FIG]
        def series(strict, src):
            ys = []
            for s in xs:
                v = get(buckets, task, cond, strict, float(s)) if src == "model" else energy.get((task, cond, "global" if strict else "per_window", float(s)))
                ys.append(v)
            return xs, ys
        x1, y1 = series(False, "model")
        x2, y2 = series(True, "model")
        ax.plot(x1, [v for v in y1], "o-", color="tab:orange", label="本文（逐窗口径）")
        ax.plot(x2, [v for v in y2], "s-", color="tab:blue", label="本文（严格口径）")
        _, ye1 = series(False, "energy")
        _, ye2 = series(True, "energy")
        def _clean(xs, ys):
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            return [p[0] for p in pts], [p[1] for p in pts]
        ex1, ey1 = _clean(x1, ye1)
        if ex1:
            ax.plot(ex1, ey1, "--", color="tab:orange", alpha=0.55, marker="x",
                    markersize=4, label="纯窗能量分类器（逐窗）")
        ex2, ey2 = _clean(x2, ye2)
        if ex2:
            ax.plot(ex2, ey2, ":", color="tab:blue", alpha=0.8, marker="+",
                    markersize=5, label="纯窗能量分类器（严格）")
        if task == "gear" and cond == "20_0":
            lit_x = [-10, -8, -6, 0]
            lit_y = [71.2, 84.3, 93.5, 99.9]
            ax.plot(lit_x, lit_y, "*-", color="black", markersize=11, label="DDGF-Net（原文口径）")
        ax.set_title(f"{TASK_CN[task]} {cond.replace('_', '-')}", fontsize=11)
        ax.set_xlabel("信噪比 SNR (dB)")
        ax.set_ylabel("验证集准确率 (%)")
        ax.set_xticks([20, 10, 4, 0, -4, -8, -12, -16])
        ax.set_xlim(-17.5, 21.5)
        ax.set_ylim(15, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("噪声参考功率口径对抗噪曲线的影响（SEU DDS，验证集）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "fig_noise_protocol.png", bbox_inches="tight")
    print("[ok] reports/table_lit_same_protocol.md")
    print("[ok] reports/table_noise_protocol.md")
    print("[ok] reports/fig_noise_protocol.png")


if __name__ == "__main__":
    main()
