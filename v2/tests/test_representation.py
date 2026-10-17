import numpy as np

from representation.state_transition import build_state_transition, fit_train_quantile_edges


# 本文件是**表示机制的单元测试**（自带 cfg，不依赖主协议）；
# 长度用命名常量，避免被误读成"协议参数"。
_T = 2048          # 与当前主协议窗长一致，这里只用于造数据
_C = 3             # 通道数


def _method_cfg():
    return {
        "symbol_bins": 6,
        "bin_scope": "per_epoch",
        "bin_edges_mode": "quantile",
        "symbol_strategy": "quantile",
        "transition_steps": [1, 2, 3, 4, 5, 6, 7],
        "transition_weight": "probability",
        "transition_multichannel": "stack",
        "include_self_transition": True,
        "state_stats": "none",
        "normalize": "zscore",
    }


def test_transition_shape_and_normalization():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(_C, _T))
    X = build_state_transition(signal, _method_cfg())

    steps = len(_method_cfg()["transition_steps"])
    assert X.shape == (_C * steps, 6, 6)   # 形状随自己的 cfg 推导，不写死
    assert np.isfinite(X).all()
    for k in range(X.shape[0]):
        assert np.isclose(float(X[k].sum()), 1.0, atol=1e-6)


def test_bin_edges_fit_train_only():
    rng = np.random.default_rng(7)
    train_a = rng.normal(loc=0.0, scale=1.0, size=(_C, _T))
    train_b = rng.normal(loc=0.0, scale=1.0, size=(_C, _T))
    other = rng.normal(loc=100.0, scale=5.0, size=(_C, _T))

    cfg = {
        "symbol_bins": 6,
        "bin_scope": "train_global",
        "bin_edges_mode": "quantile",
        "normalize": "zscore",
    }
    edges_a = fit_train_quantile_edges([train_a, train_b], [0, 1], cfg)
    edges_b = fit_train_quantile_edges([train_a, train_b, other], [0, 1], cfg)
    assert np.allclose(edges_a, edges_b)

    edges_c = fit_train_quantile_edges([train_a, train_b, other], [0, 1, 2], cfg)
    assert edges_c.shape == (3, 7)
    assert not np.allclose(edges_a, edges_c)
