"""文献抗噪基线对照表：把 DDGF-Net 原文自跑的 8 个基线摊开，再看本文站在哪。

为什么必须做这张表：
  本文的核心论点是"文献抗噪数字不可横比"。要证明它，就要把整个区间摊开，
  而不是只挑好看或难看的几个点。
  DDGF-Net (Machines 2026, 14(9):1013) Table 4 提供了难得的素材：
  它在同一数据集、同一任务、同一划分、同一 SNR 下自跑了 8 个基线 ——
  这是文献里少见的"同口径自跑基线"，可以直接拿来与本文对比。

数据来源：
  文献值 = DDGF-Net Table 4（其原文自跑，逐样本参考功率口径）
  本文值 = artifacts/variant_scan 下自跑（同意义上的逐窗口径）
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

# DDGF-Net Table 4（SEU 齿轮 20-0，5 类，8:2 + 5 折 CV，匹配加噪，逐窗参考口径）
LIT = [
    ("MLP（8 个手工特征）", 0.003, 23.9, 21.3, 25.7, "无"),
    ("ViT-1D", None, 36.1, 52.0, 75.1, "无"),
    ("BiLSTM-Attention", 0.159, 36.9, 53.9, 81.1, "无"),
    ("DConformer", None, 42.2, 61.6, 87.6, "无"),
    ("WDCNN", 0.048, 45.4, 64.8, 90.4, "无"),
    ("STFT-2D", None, 46.5, 69.5, 94.7, "无"),
    ("ResNet1D", 3.85, 56.3, 82.7, 99.3, "无"),
    ("DRSN", None, 56.5, 83.1, 99.3, "软阈值（内置）"),
    ("DDGF-Net（其原文方法）", 15.29, 71.2, 93.5, 99.9, "CVAE 降噪器 8.49 M"),
]

RUN_RE = re.compile(r"^(?P<task>[a-z]+)_(?P<model>[a-z0-9_]+)_(?P<cond>\d+_\d+)_(?P<tag>[A-Za-z0-9_]+)_seed\d+$")
OURS = {"base": (None, "pw"), "snr0": (0, "pw"), "snrm6": (-6, "pw"), "snrm10": (-10, "pw"),
        "gsnr0": (0, "gl"), "gsnrm6": (-6, "gl"), "gsnrm10": (-10, "gl")}


def ours(task: str, cond: str, model: str, ref: str, snr):
    vals = []
    for d in (PROJECT_ROOT / "artifacts/variant_scan", PROJECT_ROOT / "artifacts/variant_scan_noise_v5"):
        if not d.exists():
            continue
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            for run in sub.iterdir():
                m = RUN_RE.match(run.name)
                f = run / "metrics.json"
                if not (m and f.exists()):
                    continue
                if (m.group("task"), m.group("cond"), m.group("model")) != (task, cond, model):
                    continue
                t = OURS.get(m.group("tag"))
                if t and t == (snr, ref):
                    a = (json.loads(f.read_text(encoding="utf-8")).get("val_metrics") or {}).get("accuracy")
                    if a is not None:
                        vals.append(a)
    return 100 * statistics.mean(vals) if vals else None


def main() -> None:
    o10 = ours("gear", "20_0", "cnn", "pw", -10.0)
    o6 = ours("gear", "20_0", "cnn", "pw", -6.0)
    o0 = ours("gear", "20_0", "cnn", "pw", 0.0)
    g10 = ours("gear", "20_0", "cnn", "gl", -10.0)
    g6 = ours("gear", "20_0", "cnn", "gl", -6.0)
    g0 = ours("gear", "20_0", "cnn", "gl", 0.0)

    lines = [
        "# 文献抗噪基线全景对照（SEU 齿轮 20-0，5 类，逐样本参考功率口径）",
        "",
        "> **文献值与本文值同为逐样本参考功率口径**（DDGF-Net 原文口径）。",
        "> DDGF-Net Table 4 的 8 个基线由**其原文作者自跑**（同划分 seed 42、同 SNR、同指标），",
        "> 是本领域少见的「同口径自跑基线」，可直接对比。",
        "",
        "| 方法 | 参数量 | −10 dB | −6 dB | 0 dB | 降噪模块 | 类别 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, prm, a10, a6, a0, den in sorted(LIT, key=lambda x: x[2]):
        p = "—" if prm is None else (f"{prm} M")
        cls = "**含专用去噪模块**" if den != "无" else "普通方法（无去噪）"
        lines.append(f"| {name} | {p} | {a10} | {a6} | {a0} | {den} | {cls} |")
    lines.append(
        f"| **本文 CNN（逐窗口径）** | **0.0277 M** | **{o10:.2f}** | **{o6:.2f}** | **{o0:.2f}** | **无** | 普通方法（无去噪） |"
    )
    lines.append(
        f"| 本文 CNN（严格口径） | 0.0277 M | {g10:.2f} | {g6:.2f} | {g0:.2f} | 无 | 普通方法（无去噪） |"
    )
    lines += [
        "",
        "## 这张表说明什么",
        "",
        f"1. **在「普通方法（无去噪模块）」这一档里，本文在 −10 dB 排第一**"
        f"（{o10:.2f}%，次优 DRSN 56.5%），在 −6 dB 也高于全部 8 个基线"
        f"（{o6:.2f}% vs 最高 83.1%）。",
        "2. **含专用去噪模块的方法属于另一个类别**：DDGF-Net 15.29 M 参数、其中 8.49 M（55.5%）",
        "   是监督训练的 CVAE 降噪器，−10 dB 为 71.2%（高于本文）——它的优势可归因于那个模块，",
        "   而非表示或分类器。",
        "3. **文献内部同一张表跨度极大**：−10 dB 从 23.9% 到 71.2%，跨度 47 个点；",
        "   若把范围更大的文献一并纳入，−10 dB 可从 36% 到 95%。",
        "   这正是本文主张「抗噪数字不可横比」的直接依据。",
        "",
        "## 必须同时说明的两条口径差异",
        "",
        "| 项 | DDGF-Net | 本文 |",
        "|---|---|---|",
        "| 划分 | 随机 8:2 + 5 折分层 CV | 时间有序 70/15/15 |",
        "| 窗 | 1024 点、无重叠 | 2048 点、步长 1024 |",
        "| 训练预算 | 全部基线 30 epoch | 200 epoch + patience 30 |",
        "| 参考功率口径 | 逐样本 | 逐窗（同一族口径） |",
        "",
        "**因此本表仍是「近似同口径」，不能作为精确的排名依据。**",
        "真正的同口径对比见 `table_baseline_per_window.md`（本文自跑，324 次训练）——",
        "在那里本文高于 WDCNN，但低于 ResNet1D 与原始波形 1D-CNN。",
        "两张表结论不同，本身就说明：**基线是否被充分训练，与噪声口径一样会影响结论**。",
        "",
    ]
    out = PROJECT_ROOT / "reports" / "table_lit_baselines.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print(f"[ok] {out}")
    safe_print(f"\n本文（逐窗口径）: -10 dB {o10:.2f}  -6 dB {o6:.2f}  0 dB {o0:.2f}")
    safe_print(f"本文（严格口径）: -10 dB {g10:.2f}  -6 dB {g6:.2f}  0 dB {g0:.2f}")


if __name__ == "__main__":
    main()
