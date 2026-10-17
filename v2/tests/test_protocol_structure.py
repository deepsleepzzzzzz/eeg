"""主协议结构自洽性：表示、输入形状、GCN 布局必须互相一致。

背景（真实 bug）：v3 用 8 通道 x 7 滞后 = 56 个矩阵，但 gcn_steps 仍是 3，
relational 布局要求 num_graphs % steps == 0，56 % 3 != 0 -> 直接崩溃。
更隐蔽的是 v2：21 个矩阵配 gcn_steps=3，能跑通但把 (通道, 时间步) 拆错了。
"""
from pathlib import Path

import numpy as np

from config.protocol import load_protocol, model_cfg_from_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "configs" / "protocols"


def _main_protocols():
    out = []
    for p in sorted(PROTOCOLS.glob("main_*.yaml")):
        cfg = load_protocol(p)
        if str(cfg.get("status")) == "main":
            out.append((p.name, cfg))
    return out


def test_there_is_exactly_one_main_protocol():
    mains = _main_protocols()
    assert len(mains) == 1, f"应当只有一个 status=main 的协议，实际: {[n for n, _ in mains]}"


def test_gcn_steps_equals_number_of_lags():
    for name, cfg in _main_protocols():
        lags = [int(x) for x in cfg["representation"]["lags"]]
        steps = int(model_cfg_from_protocol(cfg, "gcn")["gcn_steps"])
        assert steps == len(lags), (
            f"{name}: gcn_steps={steps} 但 representation.lags 有 {len(lags)} 个；"
            f"relational 布局要求二者相等，否则 (通道, 时间步) 会被拆错"
        )


def test_declared_input_channels_match_channels_times_lags():
    for name, cfg in _main_protocols():
        n_ch = len(cfg["representation"]["channels"])
        n_lags = len(cfg["representation"]["lags"])
        bins = int(cfg["preprocessing"]["bins"])
        idx = cfg["dataset"].get("channel_indices")
        # 必须显式声明（此前写成 `if idx is not None` 跳过 —— 那是假保障：
        # 测试看起来在查通道，实际什么都没查）
        assert idx is not None, (
            f"{name}: 协议未显式声明 dataset.channel_indices；"
            f"通道索引决定用哪几路物理信号，必须写死"
        )
        assert len(idx) == n_ch, f"{name}: channel_indices 有 {len(idx)} 个但声明了 {n_ch} 个通道"
        free = cfg["representation"].get("input_shape")
        if free is not None:
            assert int(free[0]) == n_ch * n_lags, (
                f"{name}: input_shape[0]={free[0]} != 通道 {n_ch} x 滞后 {n_lags}"
            )
            assert int(free[1]) == bins and int(free[2]) == bins


def test_channel_indices_within_csv_width():
    for name, cfg in _main_protocols():
        idx = cfg["dataset"].get("channel_indices")
        assert idx is not None, f"{name}: 未声明 channel_indices"
        assert min(idx) >= 0, f"{name}: 列号不能为负"
        assert max(idx) <= 7, f"{name}: SEU 每个 CSV 只有 8 列"


def test_main_protocol_declares_all_result_affecting_params():
    """主协议必须显式声明所有会影响结果的参数，不许靠代码默认值兜底。

    2026-10-15 事故：channel_indices 一直靠 DEFAULT_CHANNEL_INDICES 兜底，
    协议里没写 —— 一旦默认值改动，所有结果静默改变。
    """
    required = [
        ("dataset", "channel_indices"),
        ("representation", "attention"),
        ("representation", "state_stats"),
        ("training", "label_smoothing"),
    ]
    for name, cfg in _main_protocols():
        missing = [f"{sec}.{key}" for sec, key in required if cfg.get(sec, {}).get(key) is None]
        assert not missing, f"{name}: 这些参数没显式声明，会走代码默认值 -> {missing}"


def test_channel_indices_required_not_defaulted():
    """代码层面也要挡住：协议缺该字段必须报错，而不是静默用默认值。"""
    import pytest

    from config.protocol import channel_indices_with_source

    with pytest.raises(ValueError, match="channel_indices"):
        channel_indices_with_source({"name": "unit_test", "dataset": {}})


def test_declared_window_counts_match_reality():
    """协议里声明的窗数必须与按窗口参数实际算出来的一致（防止抄错）。"""
    import numpy as np

    from data.build_dataset import split_intervals
    from data.seu_dds import window_starts

    for name, cfg in _main_protocols():
        L, S = int(cfg["window"]["length"]), int(cfg["window"]["stride"])
        iv = split_intervals(cfg)
        real = {k: len(window_starts(lo, hi, L, S)) for k, (lo, hi) in iv.items()}
        n_cls = int(cfg["dataset"].get("num_classes", 5))
        decl = cfg["split"].get("windows_per_class") or {}
        for split in ("train", "val", "test"):
            assert int(decl[split]) == real[split], (
                f"{name}: 协议写的 {split} 每类窗数 {decl[split]} != 实际 {real[split]}"
            )
        totals = cfg["split"].get("totals") or {}
        for split in ("train", "val", "test"):
            assert int(totals[split]) == real[split] * n_cls, (
                f"{name}: 协议写的 {split} 总数 {totals[split]} != 实际 {real[split] * n_cls}"
            )


def test_main_protocol_declares_prefilter_explicitly():
    """prefilter 必须显式声明，不允许靠代码默认值（否则改代码会静默改变行为）。"""
    for name, cfg in _main_protocols():
        prep = cfg["preprocessing"]
        assert "prefilter" in prep, f"{name}: preprocessing 未显式声明 prefilter"
        kind = str(prep["prefilter"]).lower()
        if kind not in ("none", "", "false"):
            need = {"moving_average": ["prefilter_k"],
                    "fir_bandpass": ["prefilter_fs", "prefilter_lo", "prefilter_hi", "prefilter_taps"],
                    "wavelet": ["prefilter_wavelet", "prefilter_level", "prefilter_mode"]}.get(kind, [])
            for k in need:
                assert k in prep, f"{name}: 用了 prefilter={kind} 但未声明 {k}"
