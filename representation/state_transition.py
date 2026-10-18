"""State-transition matrix representation.

This module discretizes a time series into amplitude states, then counts
directed transitions across one or more time steps.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_steps(raw) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (int, np.integer)):
        steps = [int(raw)]
    else:
        steps = [int(s) for s in list(raw)]
    steps = sorted(set(s for s in steps if s > 0))
    if not steps:
        raise ValueError("transition_steps must contain at least one positive integer")
    return steps


def symbolize_uniform(x: np.ndarray, bins: int) -> np.ndarray:
    """Map one signal to integer states 0..bins-1 using per-epoch amplitude bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    bins = int(bins)
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    scaled = (x - lo) / (hi - lo)
    states = np.floor(scaled * bins).astype(np.int64)
    return np.clip(states, 0, bins - 1)


def symbolize_quantile(x: np.ndarray, bins: int) -> np.ndarray:
    """Map one signal to states using per-epoch quantile edges."""
    x = np.asarray(x, dtype=np.float64).ravel()
    bins = int(bins)
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    if edges.size and float(np.max(edges) - np.min(edges)) < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def symbolize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map one signal with fixed inclusive edges. Values outside map to the end bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    edges = np.asarray(edges, dtype=np.float64).ravel()
    if edges.size < 3:
        raise ValueError("bin edges must contain at least two boundaries")
    states = np.searchsorted(edges, x, side="right") - 1
    return np.clip(states, 0, edges.size - 2).astype(np.int64)


def symbolize(x: np.ndarray, bins: int, strategy: str = "uniform") -> np.ndarray:
    strategy = str(strategy).lower()
    if strategy in ("uniform", "equal", "amplitude"):
        return symbolize_uniform(x, bins)
    if strategy in ("quantile", "percentile"):
        return symbolize_quantile(x, bins)
    raise ValueError(f"unknown symbol_strategy: {strategy}")


def _scope(method_cfg) -> str:
    return str(method_cfg.get("bin_scope", "per_epoch")).lower()


def uses_train_edges(method_cfg) -> bool:
    return _scope(method_cfg) in ("train_global", "train")


def fit_train_amplitude_edges(epochs: Sequence[np.ndarray], train_indices, method_cfg) -> np.ndarray:
    """Per-channel equal-width edges from the training epochs only. Shape [C, bins+1]."""
    from representation.recurrence_plot import preprocess_signal

    bins = int(method_cfg.get("symbol_bins", 6))
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    indices = [int(i) for i in train_indices]
    if not indices:
        raise ValueError("train_global requires at least one training epoch")
    first = preprocess_signal(epochs[indices[0]], method_cfg)
    if first.ndim != 2:
        raise ValueError(f"train_global expects [C,T] epochs, got {first.shape}")
    n_channels = int(first.shape[0])
    lo = np.full(n_channels, np.inf, dtype=np.float64)
    hi = np.full(n_channels, -np.inf, dtype=np.float64)
    for index in indices:
        arr = preprocess_signal(epochs[index], method_cfg)
        if arr.shape[0] != n_channels:
            raise ValueError("channel count changed while fitting bin edges")
        lo = np.minimum(lo, np.min(arr, axis=1))
        hi = np.maximum(hi, np.max(arr, axis=1))
    edges = np.empty((n_channels, bins + 1), dtype=np.float64)
    for channel in range(n_channels):
        span = float(hi[channel] - lo[channel])
        if span < 1e-12:
            edges[channel] = np.linspace(float(lo[channel]), float(lo[channel]) + 1.0, bins + 1)
        else:
            edges[channel] = np.linspace(float(lo[channel]), float(hi[channel]), bins + 1)
    return edges


def fit_train_quantile_edges(epochs: Sequence[np.ndarray], train_indices, method_cfg) -> np.ndarray:
    """Per-channel quantile edges from training epochs only. Shape [C, bins+1].

    Equal-frequency edges: every symbol state receives roughly the same share of
    training samples, so the state space is fully used instead of collapsing into
    the one bin that happens to cover most of the amplitude range. Boundaries are
    still fit once on the training split and shared by every sample, which keeps
    the node encoding consistent across the dataset.
    """
    from representation.recurrence_plot import preprocess_signal

    bins = int(method_cfg.get("symbol_bins", 6))
    if bins < 2:
        raise ValueError("symbol_bins must be >= 2")
    indices = [int(i) for i in train_indices]
    if not indices:
        raise ValueError("train_global requires at least one training epoch")
    first = preprocess_signal(epochs[indices[0]], method_cfg)
    if first.ndim != 2:
        raise ValueError(f"train_global expects [C,T] epochs, got {first.shape}")
    n_channels = int(first.shape[0])
    pooled: list[list[np.ndarray]] = [[] for _ in range(n_channels)]
    for index in indices:
        arr = preprocess_signal(epochs[index], method_cfg)
        if arr.shape[0] != n_channels:
            raise ValueError("channel count changed while fitting bin edges")
        for channel in range(n_channels):
            pooled[channel].append(arr[channel])
    edges = np.empty((n_channels, bins + 1), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, bins + 1)
    for channel in range(n_channels):
        values = np.concatenate(pooled[channel])
        raw = np.quantile(values, qs)
        # enforce strictly increasing edges so searchsorted stays well defined
        for i in range(1, raw.size):
            if raw[i] <= raw[i - 1]:
                raw[i] = raw[i - 1] + 1e-9
        edges[channel] = raw
    return edges


def fit_train_edges(epochs: Sequence[np.ndarray], train_indices, method_cfg) -> np.ndarray:
    """Dispatch boundary fitting by method.bin_edges_mode: quantile (default) or uniform."""
    mode = str(method_cfg.get("bin_edges_mode", "quantile")).lower()
    if mode in ("quantile", "percentile", "equal_frequency", "eqfreq"):
        return fit_train_quantile_edges(epochs, train_indices, method_cfg)
    if mode in ("uniform", "equal", "amplitude", "equal_width"):
        return fit_train_amplitude_edges(epochs, train_indices, method_cfg)
    raise ValueError(f"unknown bin_edges_mode: {mode}")


def transition_matrix(
    states: np.ndarray,
    bins: int,
    step: int,
    weight: str = "probability",
    include_self: bool = True,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64).ravel()
    bins = int(bins)
    step = int(step)
    mat = np.zeros((bins, bins), dtype=np.float32)
    if step <= 0:
        raise ValueError("transition step must be positive")
    if states.size <= step:
        return mat
    src = states[:-step]
    dst = states[step:]
    valid = (src >= 0) & (src < bins) & (dst >= 0) & (dst < bins)
    if not include_self:
        valid = valid & (src != dst)
    if np.any(valid):
        np.add.at(mat, (src[valid], dst[valid]), 1.0)
    weight = str(weight).lower()
    if weight in ("count", "counts", "raw"):
        return mat
    if weight in ("probability", "prob", "global"):
        total = float(np.sum(mat))
        return mat / total if total > 0 else mat
    if weight in ("row_probability", "row_prob", "row"):
        rows = np.sum(mat, axis=1, keepdims=True)
        return np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)
    raise ValueError(f"unknown transition_weight: {weight}")


def _channel_edges(method_cfg, channel: int | None) -> np.ndarray | None:
    if not uses_train_edges(method_cfg):
        return None
    raw = method_cfg.get("bin_edges", None)
    if raw is None:
        raise RuntimeError("bin_scope=train_global requires bin_edges fit on the training split")
    edges = np.asarray(raw, dtype=np.float64)
    if edges.ndim == 1:
        return edges
    if channel is None:
        raise ValueError("per-channel bin_edges need a channel index")
    return edges[int(channel)]


def _occupancy(states: np.ndarray, bins: int) -> np.ndarray:
    """Normalized state-occupancy vector [bins]: how often each state is visited."""
    if states.size == 0:
        return np.zeros(bins, dtype=np.float32)
    counts = np.bincount(states, minlength=bins).astype(np.float32)
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def _append_state_stats(mat: np.ndarray, states: np.ndarray, bins: int,
                        mode: str) -> np.ndarray:
    """Append per-step state statistics as extra rows: [S,B,B] -> [S*(1+extra),B,B].

    mode:
      none      -> unchanged
      occupancy -> one extra row block holding the normalized state-occupancy vector
                   repeated down the columns, so a row-wise CNN kernel can read it.
    """
    mode = str(mode).lower()
    if mode in ("none", "", "false"):
        return mat
    if mode not in ("occupancy", "occ"):
        raise ValueError(f"unknown state_stats mode: {mode}")
    occ = _occupancy(states, bins)  # [B]
    blocks = []
    for step_index in range(mat.shape[0]):
        # put occupancy on the diagonal: same [B,B] shape as a transition matrix, so
        # the CNN reads it with the same kernels and the signal strength survives.
        row = np.diag(occ).astype(np.float32)
        blocks.append(mat[step_index])
        blocks.append(row)
    return np.stack(blocks, axis=0).astype(np.float32)


def _build_channel(signal: np.ndarray, method_cfg, channel: int | None = None) -> np.ndarray:
    bins = int(method_cfg.get("symbol_bins", 6))
    strategy = str(method_cfg.get("symbol_strategy", "uniform"))
    steps = _as_steps(method_cfg.get("transition_steps", [1, 2, 3]))
    weight = str(method_cfg.get("transition_weight", "probability"))
    include_self = bool(method_cfg.get("include_self_transition", True))
    stats_mode = str(method_cfg.get("state_stats", "none"))
    edges = _channel_edges(method_cfg, channel)
    if edges is None:
        states = symbolize(signal, bins=bins, strategy=strategy)
    else:
        states = symbolize_with_edges(signal, edges)
    maps = [
        transition_matrix(states, bins=bins, step=s, weight=weight, include_self=include_self)
        for s in steps
    ]
    mat = np.stack(maps, axis=0).astype(np.float32)
    return _append_state_stats(mat, states, bins, stats_mode)


def build_state_transition(signal: np.ndarray, method_cfg) -> np.ndarray:
    """Return [steps, bins, bins] or [channels*steps, bins, bins]."""
    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    how = str(method_cfg.get("transition_multichannel", "mean")).lower()
    if arr.ndim == 1:
        return _build_channel(arr, method_cfg, channel=0 if uses_train_edges(method_cfg) else None)
    if arr.ndim != 2:
        raise ValueError(f"transition input must be 1D or [C,T], got shape={arr.shape}")
    maps = [_build_channel(arr[c], method_cfg, channel=c) for c in range(arr.shape[0])]
    stacked = np.stack(maps, axis=0).astype(np.float32)  # [C,S,B,B]
    if how in ("mean", "avg", "average"):
        return np.mean(stacked, axis=0).astype(np.float32)
    if how in ("stack", "per_channel", "channel"):
        c, s, b, _ = stacked.shape
        return stacked.reshape(c * s, b, b).astype(np.float32)
    raise ValueError(f"unknown transition_multichannel: {how}")
