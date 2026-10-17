"""幅值补充区块（representation.energy_stats）的形状与语义检查。

动机：等频符号化把幅值抹平，导致齿轮"缺齿 ↔ 齿面磨损"重叠（类心距 5.5 vs 类内散布 14.8）。
新增的 energy_stats 用每通道一个 6x6 区块把能量信息补回来。本测试保证：

  1. 不写 energy_stats 时行为与原来**完全一致**（形状不变）——现有结果不受影响；
  2. 写 "rms" 时通道数 +C，且区块值随信号幅度单调增（确实是 log-RMS）；
  3. 写 "state" 时通道数 +C，对角线按通道中心化（均值≈0），并保留"哪个状态更响"的剖面；
  4. 未知取值必须报错（不静默降级）。
"""
from __future__ import annotations

import numpy as np
import pytest

from representation.state_transition import build_state_transition, fit_train_edges, symbolize_with_edges


def _cfg(**over):
    cfg = {
        "symbol_bins": 6,
        "symbol_strategy": "quantile",
        "bin_scope": "fit_on_train_only",
        "bin_edges_mode": "uniform",
        "transition_steps": [1, 2, 3],
        "transition_weight": "probability",
        "transition_multichannel": "stack",
        "include_self_transition": True,
        "state_stats": "none",
        "normalize": "none",
        # 边界由 build_dataset 在训练集上拟合后写入 meta.json；测试里直接给定
        "bin_edges": np.array([np.linspace(-3, 3, 7)] * 3),
    }
    cfg.update(over)
    return cfg


def _signals(n=3, t=600, scale=1.0):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n, t))
    return (base * scale).astype(np.float64)


def test_absent_energy_stats_keeps_shape():
    x = _signals()
    cfg = _cfg()
    out = build_state_transition(x, cfg)
    assert out.shape == (3 * 3, 6, 6), out.shape      # C*K, 与原来一致


def test_rms_block_tracks_amplitude():
    x = _signals(scale=1.0)
    edges = {
        "bin_edges": np.array([np.linspace(-3, 3, 7)] * 3),
        "symbol_bins": 6,
        "bin_scope": "fit_on_train_only",
    }
    cfg = _cfg(energy_stats="rms", **edges)
    out = build_state_transition(x, cfg)
    assert out.shape == (3 * 3 + 3, 6, 6), out.shape
    v1 = float(out[9][0, 0])
    x2 = _signals(scale=10.0)
    out2 = build_state_transition(x2, cfg)
    v2 = float(out2[9][0, 0])
    assert v2 > v1 + 0.9, (v1, v2)                     # log10(10) = 1
    assert np.allclose(out[9], out[9][0, 0])           # rms 区块内是常数


def test_state_block_is_centered_and_uses_edges():
    x = _signals()
    edges = np.array([np.linspace(-3, 3, 7)] * 3)
    cfg = _cfg(energy_stats="state", bin_edges=edges)
    out = build_state_transition(x, cfg)
    assert out.shape == (3 * 3 + 3, 6, 6), out.shape
    for c in range(3):
        blk = out[9 + c]
        diag = np.diag(blk)
        assert abs(float(diag.mean())) < 1e-6          # 通道内中心化
        assert np.allclose(blk - np.diag(diag), 0)     # 只填对角线


def test_unknown_mode_raises():
    x = _signals()
    cfg = _cfg(energy_stats="nonsense", bin_edges=np.array([np.linspace(-3, 3, 7)] * 3))
    with pytest.raises(ValueError):
        build_state_transition(x, cfg)
