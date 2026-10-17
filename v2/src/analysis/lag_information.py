"""表示参数的论证（对应 Shankar et al., BSPC 2021 §4.1 的 AMI / FNN 做法）。

他的做法：
  - 时间延迟 τ：用平均互信息 AMI 随 τ 的曲线，取第一个局部极小
  - 嵌入维数 m：用假近邻 FNN 随 m 的曲线，取降到 0 的点
  - 阈值 ε：经验扫描 10 个值

我们的对应参数有两个：
  - 滞后阶数 k（1..10）   ≈ 他的 τ（时间延迟）
  - 量化档数 B（6）       ≈ 他的 ε（阈值），用信息量而非阈值来论证

本脚本给出两条曲线：
  1) 符号序列的互信息 I(k) 随滞后 k 的衰减 —— 论证为什么取 1..10
     I(k) = Σ_ij P_k(i,j) log[ P_k(i,j) / (p_i p_j) ]
     其中 P_k 就是本文的联合转移概率矩阵（第 3 节的表示直接给出了它，零额外计算）
  2) 归一化互信息 I(1)/H(s) 随量化档数 B 的变化 —— 论证为什么取 6 档
     档数太少 -> 信息损失；档数太多 -> 转移矩阵稀疏、估计噪声上升

用法：python src/analysis/lag_information.py --task gear --condition 20_0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from config.protocol import (  # noqa: E402
    load_protocol, channel_indices, window_cfg, raw_root as protocol_raw_root, dataset_dir,
)
from data.build_dataset import split_intervals  # noqa: E402
from data.seu_dds import class_file, class_names, cut_windows, load_seu_csv, task_dir, vibration_xyz, window_starts  # noqa: E402
from representation.recurrence_plot import preprocess_signal  # noqa: E402
from representation.state_transition import symbolize_with_edges  # noqa: E402


def find_main_protocol() -> Path:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def load_windows(protocol, task, condition, split):
    length, stride = window_cfg(protocol)
    lo, hi = split_intervals(protocol)[split]
    ch = channel_indices(protocol)
    folder = task_dir(protocol_raw_root(protocol), task)
    out = []
    for name in class_names(task):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw, ch)
        ends = min(int(hi), int(xyz.shape[1]))
        starts = window_starts(lo, ends, length, stride)
        out.extend(np.asarray(w, dtype=np.float64) for w in cut_windows(xyz, starts, length))
    return out


def symbol_matrices(windows, edges, method_cfg=None):
    """每条窗 -> 每通道的符号序列。method_cfg 必须带上协议里的归一化设置。"""
    arrs = []
    for w in windows:
        a = preprocess_signal(w, method_cfg or {"normalize": "zscore"})
        syms = np.stack([symbolize_with_edges(a[c], edges[c]) for c in range(a.shape[0])], axis=0)
        arrs.append(syms)
    return arrs


def mutual_info_from_joint(P, bins):
    """P: [B,B] 联合概率。返回 I(X;Y)（nat）。"""
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (P > 0) & (denom > 0)
    return float(np.sum(P[mask] * np.log(P[mask] / denom[mask])))


def occupancy_entropy(P):
    p = P.sum(axis=1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def fit_quantile_edges(arrs, bins):
    pooled = np.concatenate([a.reshape(a.shape[0], -1) for a in arrs], axis=1)  # [C, N]
    return np.stack([np.quantile(pooled[c], np.linspace(0, 1, bins + 1)[1:-1]) for c in range(pooled.shape[0])])


def joint_from_symbols(syms_list, bins, k):
    """汇总所有窗、所有通道的 (s_t, s_{t+k}) 联合直方图。"""
    P = np.zeros((bins, bins), dtype=np.float64)
    for syms in syms_list:
        for c in range(syms.shape[0]):
            s = syms[c]
            if s.size <= k:
                continue
            a = s[:-k]
            b = s[k:]
            idx = a.astype(np.int64) * bins + b.astype(np.int64)
            P += np.bincount(idx, minlength=bins * bins).reshape(bins, bins)
    total = P.sum()
    return P / total if total > 0 else P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gear")
    ap.add_argument("--condition", default="20_0")
    ap.add_argument("--bins", default="4,6,8,10,12")
    ap.add_argument("--max-lag", type=int, default=24)
    args = ap.parse_args()
    bins_list = [int(b) for b in args.bins.split(",")]

    proto = load_protocol(find_main_protocol())
    safe_print(f"[lag_info] 基准协议 {find_main_protocol().name}  task={args.task} {args.condition}")
    windows = load_windows(proto, args.task, args.condition, "train")
    safe_print(f"[lag_info] 训练窗 {len(windows)} 条（只用训练集，边界与模型完全一致）")

    # 归一化设置必须与主协议一致（训练集固定尺度），否则符号化尺度错了
    meta = json.loads((dataset_dir(proto, args.task, args.condition) / "meta.json").read_text(encoding="utf-8"))
    method_cfg = {"normalize": str(proto["preprocessing"]["normalize"]),
                  "global_mean": meta["global_mean"], "global_std": meta["global_std"]}

    rows = []
    for B in bins_list:
        arrs = [preprocess_signal(w, method_cfg) for w in windows]
        edges = fit_quantile_edges(arrs, B)
        syms = symbol_matrices(windows, edges, method_cfg)
        for k in range(1, args.max_lag + 1):
            P = joint_from_symbols(syms, B, k)
            I_k = mutual_info_from_joint(P, B)
            Hs = occupancy_entropy(P)
            rows.append({"bins": B, "lag": k, "I_nat": I_k,
                         "H_state_nat": Hs, "I_over_H": (I_k / Hs if Hs > 0 else 0.0),
                         "log2B": float(np.log2(B))})
        safe_print(f"  B={B:2d}  I(1)={rows[-args.max_lag]['I_nat']:.4f} nat  "
                   f"H(s)={rows[-args.max_lag]['H_state_nat']:.4f}  I(1)/H={rows[-args.max_lag]['I_over_H']:.3f}")

    out_csv = PROJECT_ROOT / "reports" / f"lag_information_{args.task}_{args.condition}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    safe_print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
