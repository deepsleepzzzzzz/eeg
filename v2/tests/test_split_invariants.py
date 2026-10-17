"""划分不变量：与**当前主协议**一致（协议驱动，不写死窗口/窗数）。

历史教训：本文件曾把 1600/800 与 916/195/195 写死，协议升到 2048/1024 后它仍"全绿"，
但测的已经不是当前协议了（AGENTS.md 记录过同类事故：审计写死 v2）。
"""
from __future__ import annotations

import re

from data.build_dataset import split_intervals
from data.seu_dds import class_names, window_starts
from _protocol import main_protocol, main_window


def test_split_intervals_match_protocol_declaration():
    """由比例算出的区间，必须等于协议里显式声明的区间。"""
    cfg = main_protocol()
    n = int(cfg["dataset"]["file_length"])
    got = split_intervals(cfg)
    for key in ("train", "val", "test"):
        declared = cfg["split"][f"{key}_interval"]
        a, b = (int(x) for x in re.findall(r"\d+", declared)[:2])
        assert got[key] == [a, b], f"{key}: 计算 {got[key]} vs 协议声明 {declared}"
    assert got["test"][1] == n


def test_window_counts_match_protocol_declaration():
    """每类窗数与总数必须等于协议里的声明（协议与实现不许各说各话）。"""
    cfg = main_protocol()
    length, stride = main_window()
    counts = {k: len(window_starts(lo, hi, length, stride))
              for k, (lo, hi) in split_intervals(cfg).items()}
    declared = cfg["split"]["windows_per_class"]
    for k, v in counts.items():
        assert v == int(declared[k]), f"{k}: 实际 {v} 窗 vs 协议声明 {declared[k]}"
    n_cls = len(class_names("gear"))
    for k, v in counts.items():
        assert v * n_cls == int(cfg["split"]["totals"][k]), f"{k} 总数不含"


# 历史协议（70/20/10）用的窗口参数。**故意保留**：记录"当年那版怎么算的"，
# 且用命名常量而不是字面量，避免与"当前协议参数"混淆（防陈旧守卫会扫字面量）。
HIST_LEN, HIST_STRIDE = 1600, 800


def test_historical_70_20_10_arithmetic():
    """历史协议（70/20/10）的算术自洽性 —— **只验证算术**，不代表当前协议。"""
    from data.seu_dds import time_split_bounds
    n = 1_048_560
    train_end, val_end = time_split_bounds(n, 0.70, 0.20)
    assert train_end == 733_992 and val_end == 943_704
    assert len(window_starts(0, train_end, HIST_LEN, HIST_STRIDE)) == 916
    assert len(window_starts(train_end, val_end, HIST_LEN, HIST_STRIDE)) == 261
    assert len(window_starts(val_end, n, HIST_LEN, HIST_STRIDE)) == 130
