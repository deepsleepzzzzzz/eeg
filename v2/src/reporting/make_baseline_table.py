"""同协议基线表：本文方法 vs 自跑基线（同一划分、同一窗、同一加噪、同一训练预算）。

为什么必须自跑：文献报告的基线值口径各不相同（随机划分 / 不同窗长 / 不同噪声参考功率），
直接引用无法回答"我们到底比谁强"。本表只统计由本项目管线亲自跑出来的数字。

数据来源：
  artifacts/variant_scan/<变体>/<task>_<model>_<cond>_<变体>_seed<seed>/metrics.json
  —— 基线变体：clean / n0 / n4 / nm4 / nm6 / nm10（逐窗参考）
                gnm4 / gnm6 / gnm10（全局参考，严格口径）
  —— 本文方法：base / snr* / snrm*（逐窗参考）、gsnr*（全局参考）
     （齿轮 30-2 的逐窗参考结果在 artifacts/variant_scan_noise_v5 下，本脚本一并合并）

用法：python src/reporting/make_baseline_table.py
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

RUN_RE = re.compile(
    r"^(?P<task>[a-z]+)_(?P<model>raw_cnn|wdcnn|resnet1d|cnn|gcn)_"
    r"(?P<cond>\d+_\d+)_(?P<tag>[A-Za-z0-9]+)_seed\d+$"
)

# 本文方法在各口径下的变体名 -> SNR
OURS = {
    "base": (None, "per_window"), "snr20": (20, "per_window"), "snr10": (10, "per_window"),
    "snr4": (4, "per_window"), "snr0": (0, "per_window"), "snrm2": (-2, "per_window"),
    "snrm4": (-4, "per_window"), "snrm6": (-6, "per_window"), "snrm8": (-8, "per_window"),
    "snrm10": (-10, "per_window"), "snrm12": (-12, "per_window"), "snrm14": (-14, "per_window"),
    "snrm16": (-16, "per_window"),
    "gsnr20": (20, "global"), "gsnr10": (10, "global"), "gsnr4": (4, "global"),
    "gsnr0": (0, "global"), "gsnrm2": (-2, "global"), "gsnrm4": (-4, "global"),
    "gsnrm6": (-6, "global"), "gsnrm8": (-8, "global"), "gsnrm10": (-10, "global"),
    "gsnrm12": (-12, "global"), "gsnrm16": (-16, "global"),
}
# 基线变体名 -> (SNR, 口径)
BASE = {
    "clean": (None, "per_window"), "n4": (4, "per_window"), "n0": (0, "per_window"),
    "nm4": (-4, "per_window"), "nm6": (-6, "per_window"), "nm10": (-10, "per_window"),
    "gnm4": (-4, "global"), "gnm6": (-6, "global"), "gnm10": (-10, "global"),
}
BASE_CN = {"raw_cnn": "原始波形 1D-CNN", "wdcnn": "WDCNN", "resnet1d": "ResNet1D"}
CONDS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]
TASK_CN = {"gear": "齿轮", "bearing": "轴承"}
MAIN_MODELS = ("cnn", "gcn")

SCAN_DIRS = [PROJECT_ROOT / "artifacts/variant_scan", PROJECT_ROOT / "artifacts/variant_scan_noise_v5"]


def collect():
    """{(task, cond, model, reference, snr): [acc, ...]}"""
    out = defaultdict(list)
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            for run in sub.iterdir():
                m = RUN_RE.match(run.name)
                if not m:
                    continue
                f = run / "metrics.json"
                if not f.exists():
                    continue
                tag = m.group("tag")
                if tag in BASE:
                    ref, snr = BASE[tag][1], BASE[tag][0]
                elif tag in OURS:
                    ref, snr = OURS[tag][1], OURS[tag][0]
                else:
                    continue
                metrics = json.loads(f.read_text(encoding="utf-8"))
                if metrics.get("test_metrics"):
                    raise SystemExit(f"[FAIL] {f} 含测试集指标，本表只读验证集")
                acc = (metrics.get("val_metrics") or {}).get("accuracy")
                if acc is None:
                    continue
                out[(m.group("task"), m.group("cond"), m.group("model"), ref, snr)].append(float(acc))
    return out


def mean(b, task, cond, model, ref, snr):
    vals = b.get((task, cond, model, ref, snr))
    return 100.0 * statistics.mean(vals) if vals else None


def n_of(b, task, cond, model, ref, snr):
    return len(b.get((task, cond, model, ref, snr), []))


def snapshot(b):
    """本轮已落地多少（用于跑批中途查看）。"""
    done = sum(len(v) for v in b.values())
    return done


def main() -> None:
    b = collect()
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    snr_levels = [None, 0, -4, -6, -10]

    for ref, ref_cn in (("per_window", "逐窗参考（与文献同口径）"),
                        ("global", "全局参考（严格口径）")):
        lines = [
            f"# 同协议基线表 · {ref_cn}",
            "",
            "> **全部由本项目管线自跑**：同一划分（时间有序 70/15/15）、同一窗（2048/1024）、",
            "> 同一加噪口径、同一训练预算（200 epoch / patience 30 / ReduceLROnPlateau）。",
            "> 唯一变量是输入表示与网络结构。验证集，seed 42/43/44。",
            "",
        ]
        for task, cond in CONDS:
            lines += [f"## {TASK_CN[task]} {cond.replace('_', '-')}", ""]
            head = "| 方法 | 参数量 | 干净 | " + " | ".join(
                f"{s:+d} dB" for s in snr_levels[1:]) + " |"
            lines += [head, "|---|---:|" + "---:|" * (len(snr_levels))]
            base_models = [("raw_cnn", "原始波形 1D-CNN", "—"), ("wdcnn", "WDCNN", "—"),
                           ("resnet1d", "ResNet1D", "3.85 M")]
            for m_key, m_cn, prm in base_models:
                cells = []
                for snr in snr_levels:
                    v = mean(b, task, cond, m_key, ref, snr)
                    cells.append("—" if v is None else f"{v:.2f}")
                lines.append(f"| {m_cn} | {prm} | " + " | ".join(cells) + " |")
            for m_key, m_cn, prm in (("cnn", "**本文 CNN**", "**0.0277 M**"),
                                     ("gcn", "**本文 GCN**", "**0.0549 M**")):
                if ref == "global" and task == "gear" and cond == "30_2":
                    pass
                cells = []
                for snr in snr_levels:
                    v = mean(b, task, cond, m_key, ref, snr)
                    if v is None and ref == "global":
                        v = mean(b, task, cond, m_key, "per_window", snr) if snr is None else None
                    cells.append("—" if v is None else (f"**{v:.2f}**" if v is not None else "—"))
                lines.append(f"| {m_cn} | {prm} | " + " | ".join(cells) + " |")
            lines.append("")
        (out_dir / f"table_baseline_{ref}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        safe_print(f"[ok] reports/table_baseline_{ref}.md")

    # 汇总：每个 (工况, 口径) 有多少次基线训练已落地
    safe_print("\n已落地的基线训练次数（按工况）：")
    for task, cond in CONDS:
        cnt = sum(n_of(b, task, cond, m, ref, snr)
                  for m in ("raw_cnn", "wdcnn", "resnet1d")
                  for ref in ("per_window", "global") for snr in (None, 4, 0, -4, -6, -10))
        safe_print(f"  {task:8s} {cond}: {cnt}  (满额 3 模型 × 9 变体 × 3 seed = 81)")
    safe_print(f"\n总计 {snapshot(b)} 条记录")


if __name__ == "__main__":
    main()
