"""无泄漏不变量：窗口不跨段、窗起点不重复（窗口/步长取自当前主协议）。"""
from __future__ import annotations

from data.build_dataset import split_intervals
from data.seu_dds import window_starts
from _protocol import main_protocol, main_window


def _bounds(lo: int, hi: int, length: int, stride: int):
    starts = window_starts(lo, hi, length, stride)
    return starts, starts + length


def test_no_window_crosses_split_boundary():
    cfg = main_protocol()
    length, stride = main_window()
    iv = split_intervals(cfg)
    tr_s, tr_e = _bounds(*iv["train"], length, stride)
    va_s, va_e = _bounds(*iv["val"], length, stride)
    te_s, te_e = _bounds(*iv["test"], length, stride)

    assert tr_e[-1] <= va_s[0], "训练窗越入验证段"
    assert va_e[-1] <= te_s[0], "验证窗越入测试段"
    assert tr_s[0] >= iv["train"][0] and tr_e[-1] <= iv["train"][1]
    assert va_s[0] >= iv["val"][0] and va_e[-1] <= iv["val"][1]
    assert te_s[0] >= iv["test"][0] and te_e[-1] <= iv["test"][1]


def test_no_duplicate_window_starts_across_splits():
    cfg = main_protocol()
    length, stride = main_window()
    iv = split_intervals(cfg)
    all_starts = []
    for lo, hi in iv.values():
        all_starts.extend(window_starts(lo, hi, length, stride).tolist())
    assert len(all_starts) == len(set(all_starts)), "不同段之间出现了重复的窗起点"
