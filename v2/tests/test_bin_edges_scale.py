"""回归测试：分位边界与符号化必须在同一尺度上。

背景（真实 bug）：边界由 fit_train_quantile_edges 在 preprocess_signal(...) 之后
拟合（z-score 尺度），而 build_state_transition 曾直接对**原始幅值**做符号化。
后果是 6 档里只用得到中间 2 档，6x6 转移矩阵退化成 2x2。
这两个测试就是为这个 bug 设的。
"""
import numpy as np

from representation.state_transition import build_state_transition, fit_train_edges

# 单元测试自带 cfg；长度用命名常量，避免被误读成"协议参数"
_T = 2048
_C = 3


def _method_cfg(bins: int = 6) -> dict:
    return {
        "representation": "transition",
        "attention": "none",
        "normalize": "per_window_per_channel_zscore",
        "symbol_bins": bins,
        "bin_scope": "fit_on_train_only",
        "bin_edges_mode": "equal_frequency_quantile",
        "symbol_strategy": "quantile",
        "transition_steps": [1, 2, 3, 4, 5, 6, 7],
        "transition_weight": "probability",
        "transition_multichannel": "stack",
        "include_self_transition": True,
        "state_stats": "none",
    }


def _fit(windows, method):
    edges = fit_train_edges(windows, range(len(windows)), method)
    method = dict(method)
    method["bin_edges"] = edges.tolist()
    return method


def test_global_edges_use_full_state_space():
    """每类样本量很少时也不能只占用一两个状态。"""
    rng = np.random.default_rng(0)
    # 幅值刻意取小（与 SEU 原始信号同量级），但分布是宽带的
    windows = [rng.standard_normal((_C, _T)) * 0.02 for _ in range(40)]
    method = _fit(windows, _method_cfg(6))

    mat = build_state_transition(windows[0], method)
    steps = len(method["transition_steps"])
    assert mat.shape == (_C * steps, 6, 6)   # 形状随自己的 cfg 推导，不写死

    occupancy = mat.sum(axis=1).sum(axis=0)  # 每个目的状态的占用
    used = int((occupancy > 0).sum())
    assert used >= 5, (
        f"6 档只用到 {used} 个状态，说明边界与数据不在同一尺度上"
        f"（occupancy={np.round(occupancy, 4).tolist()}）"
    )


def test_pipeline_is_scale_invariant():
    """整体放大信号不应改变表示（因为逐窗 z-score 会抵消尺度）。"""
    rng = np.random.default_rng(1)
    windows = [rng.standard_normal((_C, _T)) * 0.02 for _ in range(20)]
    method = _fit(windows, _method_cfg(6))

    a = build_state_transition(windows[0], method)
    b = build_state_transition(windows[0] * 1000.0, method)
    assert np.allclose(a, b, atol=1e-6), "表示随幅值尺度变化 —— 归一化没有生效"


def test_more_bins_add_information():
    """提高分位档数必须带来新的信息，而不是原地重标号。"""
    rng = np.random.default_rng(2)
    windows = [rng.standard_normal((_C, _T)) * 0.02 for _ in range(30)]

    def occupancy(bins):
        method = _fit(windows, _method_cfg(bins))
        mat = build_state_transition(windows[0], method)
        occ = mat.sum(axis=1).sum(axis=0)
        return int((occ > 0).sum()), mat.shape

    used6, shape6 = occupancy(6)
    used12, shape12 = occupancy(12)
    assert shape6[-1] == 6 and shape12[-1] == 12
    assert used12 > used6, f"bins=12 只用到 {used12} 个状态，bins=6 用到 {used6} 个 —— 细化没有增加信息"
