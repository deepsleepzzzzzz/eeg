"""抗噪曲线图：从扫描目录读取匹配加噪结果，叠加文献报告值。

用法：
    python src/reporting/make_noise_figure.py --out reports/fig_noise_robustness.png

默认合并两个扫描目录（不同工况分批跑的）：
    artifacts/variant_scan           轴承 20-0 / 轴承 30-2 / 齿轮 20-0
    artifacts/variant_scan_noise_v5  齿轮 30-2

约束（见 AGENTS.md）：
- 只读验证集指标，若某个 run 含 test_metrics 直接报错退出；
- 文献数字是硬编码常量（来自原文），改动必须在表注里同步说明。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tag -> SNR(dB)；base 为无噪声
TAG_TO_SNR = {
    "base": None,
    "snr20": 20.0,
    "snr10": 10.0,
    "snr4": 4.0,
    "snr0": 0.0,
    "snrm2": -2.0,
    "snrm4": -4.0,
    "snrm6": -6.0,
    "snrm8": -8.0,
    "snrm10": -10.0,
}

DEFAULT_SCAN_DIRS = ["artifacts/variant_scan", "artifacts/variant_scan_noise_v5"]

RUN_RE = re.compile(
    r"^(?P<task>[a-z]+)_(?P<model>[a-z]+)_(?P<cond>\d+_\d+)_(?P<tag>[A-Za-z0-9]+)_seed\d+$"
)

TASK_LABEL = {"gear": "齿轮", "bearing": "轴承"}

# 文献报告值（原文数字，仅作量级参照；口径见论文表注）
LITERATURE = [
    {
        "label": "DDGF-Net（齿轮 20-0）",
        "points": [(-10.0, 71.2), (-8.0, 84.3), (-6.0, 93.5)],
        "style": {"color": "black", "marker": "*", "linestyle": "None", "markersize": 13},
    },
    {
        "label": "DRSN（齿轮 20-0）",
        "points": [(-10.0, 56.5), (-6.0, 83.1), (-4.0, 92.5)],
        "style": {"color": "black", "marker": "v", "linestyle": "None", "markersize": 8},
    },
    {
        "label": "ResNet-1D（齿轮 20-0）",
        "points": [(-10.0, 56.3), (-4.0, 92.4)],
        "style": {"color": "dimgray", "marker": "^", "linestyle": "None", "markersize": 8},
    },
    {
        "label": "ViT-1D（齿轮 20-0）",
        "points": [(-10.0, 36.1)],
        "style": {"color": "dimgray", "marker": "s", "linestyle": "None", "markersize": 8},
    },
    {
        "label": "Dual-Channel（轴承 30-2）",
        "points": [(4.0, 93.27), (0.0, 91.67), (-4.0, 86.73), (-6.0, 85.87)],
        "style": {
            "color": "crimson",
            "marker": "x",
            "linestyle": "--",
            "linewidth": 1.8,
            "markersize": 7,
        },
    },
]

CURVE_STYLE = {
    ("gear", "20_0"): {"color": "tab:blue", "marker": "o"},
    ("gear", "30_2"): {"color": "tab:orange", "marker": "s"},
    ("bearing", "20_0"): {"color": "tab:green", "marker": "^"},
    ("bearing", "30_2"): {"color": "tab:red", "marker": "d"},
}

SNR_TICKS = [-10, -8, -6, -4, -2, 0, 4]
CLEAN_X = 10.0  # 无噪声点单独摆在右侧，与 SNR 轴用竖线隔开


def setup_font() -> None:
    """Windows 下让中文正常显示；缺字体时回退并给出提示。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[warn] 未找到中文字体，图中中文会显示为方框；请安装 Microsoft YaHei 或 SimHei")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def collect(scan_dirs: list[Path]) -> dict:
    """{(task, cond, model, snr)：[acc, ...]}；snr 为 None 表示无噪声。"""
    buckets: dict = {}
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"[warn] 跳过不存在的扫描目录：{scan_dir}")
            continue
        for sub in sorted(scan_dir.iterdir()):
            if not sub.is_dir():
                continue
            for run in sorted(sub.iterdir()):
                m = RUN_RE.match(run.name)
                if not m or m.group("tag") not in TAG_TO_SNR:
                    continue
                mpath = run / "metrics.json"
                if not mpath.exists():
                    print(f"[warn] 缺少 metrics.json：{run}")
                    continue
                metrics = json.loads(mpath.read_text(encoding="utf-8"))
                if metrics.get("test_metrics"):
                    raise SystemExit(f"[FAIL] {mpath} 含测试集指标，抗噪图只能读验证集")
                acc = metrics.get("val_metrics", {}).get("accuracy")
                if acc is None:
                    continue
                key = (m.group("task"), m.group("cond"), m.group("model"), TAG_TO_SNR[m.group("tag")])
                buckets.setdefault(key, []).append(float(acc))
    return buckets


def model_mean(buckets: dict, task: str, cond: str, snr: float | None) -> float | None:
    """同一 (任务,工况,SNR) 下：先按模型对 seed 求均值，再对 CNN/GCN 求均值。"""
    per_model = [
        statistics.mean(buckets[(task, cond, model, snr)])
        for model in ("cnn", "gcn")
        if buckets.get((task, cond, model, snr))
    ]
    return statistics.mean(per_model) if per_model else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan-dir", nargs="+", default=DEFAULT_SCAN_DIRS)
    p.add_argument("--out", default="reports/fig_noise_robustness.png")
    args = p.parse_args()

    scan_dirs = [Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d for d in args.scan_dir]
    buckets = collect(scan_dirs)
    if not buckets:
        raise SystemExit("[FAIL] 没有找到可用的验证集指标")

    setup_font()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for (task, cond), style in CURVE_STYLE.items():
        xs, ys = [], []
        for snr in [float(s) for s in SNR_TICKS]:
            v = model_mean(buckets, task, cond, snr)
            if v is not None:
                xs.append(snr)
                ys.append(100.0 * v)
        clean = model_mean(buckets, task, cond, None)
        if clean is not None:
            xs.append(CLEAN_X)
            ys.append(100.0 * clean)
        if xs:
            label = f"本文 · {TASK_LABEL[task]} {cond.replace('_', '-')}（CNN/GCN 均值）"
            ax.plot(xs, ys, label=label, **style)

    for entry in LITERATURE:
        ax.plot(
            [pt[0] for pt in entry["points"]],
            [pt[1] for pt in entry["points"]],
            label=entry["label"],
            **entry["style"],
        )

    ax.axvline(7.0, color="lightgray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("信噪比 SNR (dB)")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_title("匹配加噪 AWGN 下的抗噪性能（SEU DDS，验证集）", fontsize=12)
    ax.set_xticks(SNR_TICKS + [CLEAN_X])
    ax.set_xticklabels([str(s) for s in SNR_TICKS] + ["无噪声"])
    ax.set_xlim(-11.5, 12.0)
    ax.set_ylim(30, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for d in scan_dirs:
        print(f"  来源: {d.relative_to(PROJECT_ROOT) if d.is_relative_to(PROJECT_ROOT) else d}")
    print("  说明: 曲线=验证集（测试集封存），文献点=原文报告值")


if __name__ == "__main__":
    main()
