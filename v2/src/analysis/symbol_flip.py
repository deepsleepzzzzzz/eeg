"""符号翻转率分析：检验"符号化本身是否提供抗噪性"。

为什么需要它：
  抗噪曲线在 -6 ~ -10 dB 几乎是平的（86.1 -> 84.4），这与"噪声会淹没故障特征"的
  常识冲突。可能的解释是：6 档分位符号化丢弃了幅值细节，只保留状态转移顺序，
  因此加性噪声对符号序列的扰动有限。本脚本把这个说法变成可测的数字。

严格复刻 build_dataset.py 的处理顺序（顺序错一步结论就无效）：
  切窗（原始幅值）->（训练集固定尺度）归一化 -> 加噪 -> 符号化
对【同一个窗】做两次符号化（干净 / 带噪），两次共用同一套【训练集拟合】的边界，
统计符号序列发生变化的采样点比例 = 翻转率。

用法：
    python src/analysis/symbol_flip.py --task gear --condition 20_0
    python src/analysis/symbol_flip.py --all
输出：
    reports/symbol_flip_<task>_<condition>.csv
    reports/table_symbol_flip.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from config.protocol import (  # noqa: E402
    load_protocol,
    method_cfg_from_protocol,
    window_cfg,
    channel_indices,
    raw_root as protocol_raw_root,
    dataset_dir,
)
from data.build_dataset import split_intervals  # noqa: E402
from data.seu_dds import (  # noqa: E402
    add_noise,
    class_file,
    class_names,
    cut_windows,
    load_seu_csv,
    task_dir,
    vibration_xyz,
    window_starts,
)
from representation.recurrence_plot import preprocess_signal  # noqa: E402
from representation.state_transition import symbolize_with_edges  # noqa: E402

# 与 artifacts/variant_scan 的噪声档保持一致，另加更低档与一个"基本无噪"参考
SNR_LIST = [40.0, 4.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -16.0, -20.0]
# build_dataset 里 splits_to_build = ["train","val"]，val 是第 2 个 => 噪声种子 = noise.seed + 1
VAL_NOISE_SEED_OFFSET = 1


def find_main_protocol() -> Path:
    from config.protocol import load_protocol as _lp

    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(_lp(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def load_val_windows(protocol: dict, task: str, condition: str) -> list[np.ndarray]:
    """按协议切出验证窗（原始幅值，尚未归一化）。顺序与 build_dataset 完全一致。"""
    length, stride = window_cfg(protocol)
    intervals = split_intervals(protocol)
    lo, hi = intervals["val"]
    folder = task_dir(protocol_raw_root(protocol), task)
    windows: list[np.ndarray] = []
    for name in class_names(task):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw, channel_indices(protocol))
        n = int(xyz.shape[1])
        starts = window_starts(lo, min(int(hi), n), length, stride)
        windows.extend(cut_windows(xyz, starts, length))
    return windows


def symbols_of(windows: list[np.ndarray], method_cfg: dict, edges: np.ndarray) -> list[np.ndarray]:
    """归一化（不包含预处理里的滤波，v5 为 none）后用给定边界符号化。"""
    out = []
    for w in windows:
        arr = preprocess_signal(w, method_cfg)  # train_global_zscore
        syms = np.stack(
            [symbolize_with_edges(arr[c], edges[c]) for c in range(arr.shape[0])],
            axis=0,
        )
        out.append(syms)
    return out


def run_one(protocol: dict, task: str, condition: str) -> list[dict]:
    meta_path = dataset_dir(protocol, task, condition) / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"[FAIL] 缺少干净基线缓存 meta：{meta_path}；先跑 "
            f"python src/data/build_dataset.py --protocol {find_main_protocol()} "
            f"--task {task} --conditions {condition}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("noise"):
        raise SystemExit(f"[FAIL] {meta_path} 是加噪缓存，无法作为干净基线")

    method_cfg = method_cfg_from_protocol(protocol)
    method_cfg["global_mean"] = meta["global_mean"]
    method_cfg["global_std"] = meta["global_std"]
    edges = np.asarray(meta["bin_edges"], dtype=np.float64)
    noise_seed = 0

    windows = load_val_windows(protocol, task, condition)
    clean = symbols_of(windows, method_cfg, edges)
    safe_print(f"[{task} {condition}] 验证窗 {len(windows)} 个，边界档数 {edges.shape[1] - 1}")

    rows = []
    for snr in SNR_LIST:
        noisy_windows = add_noise(windows, snr, noise_seed + VAL_NOISE_SEED_OFFSET)
        noisy = symbols_of(noisy_windows, method_cfg, edges)
        flips, totals, win_flip = 0, 0, []
        for cs, ns in zip(clean, noisy):
            diff = cs != ns
            flips += int(diff.sum())
            totals += int(diff.size)
            win_flip.append(float(diff.mean()))
        rows.append({
            "task": task,
            "condition": condition,
            "snr_db": snr,
            "n_windows": len(windows),
            "flip_rate": flips / totals if totals else float("nan"),
            "window_flip_mean": float(np.mean(win_flip)) if win_flip else float("nan"),
            "window_flip_p95": float(np.percentile(win_flip, 95)) if win_flip else float("nan"),
            "unchanged_window_ratio": float(np.mean([f == 0.0 for f in win_flip])),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="gear", choices=["gear", "bearing"])
    p.add_argument("--condition", default="20_0", choices=["20_0", "30_2"])
    p.add_argument("--all", action="store_true", help="四个任务工况全跑")
    args = p.parse_args()

    protocol_path = find_main_protocol()
    protocol = load_protocol(protocol_path)
    safe_print(f"[symbol_flip] 基准协议: {protocol_path.name}（自动解析 status=main）")

    combos = (
        [(t, c) for t in ("gear", "bearing") for c in ("20_0", "30_2")]
        if args.all
        else [(args.task, args.condition)]
    )

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for task, condition in combos:
        rows = run_one(protocol, task, condition)
        all_rows.extend(rows)
        path = out_dir / f"symbol_flip_{task}_{condition}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        safe_print(f"  -> {path}")

    lines = [
        "# 符号翻转率（干净窗 vs 同窗加噪，共用训练集拟合的边界）",
        "",
        "| task | condition | SNR (dB) | 翻转率 | 窗内平均翻转率 | P95 | 完全未翻转的窗占比 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['task']} | {r['condition']} | {r['snr_db']:+.0f} | "
            f"{100*r['flip_rate']:.2f}% | {100*r['window_flip_mean']:.2f}% | "
            f"{100*r['window_flip_p95']:.2f}% | {100*r['unchanged_window_ratio']:.1f}% |"
        )
    (out_dir / "table_symbol_flip.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print(f"[symbol_flip] 完成 {len(all_rows)} 行 -> reports/table_symbol_flip.md")


if __name__ == "__main__":
    main()
