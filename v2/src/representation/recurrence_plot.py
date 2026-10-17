"""经典递归图（论文 1/2 共用表示；阶段 R）。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .phase_space import build_phase_space
from .rp_core import pairwise_distances, resolve_epsilon, resize_square, zscore_channels


def build_recurrence_plot(
    signal: np.ndarray,
    m: int = 3,
    tau: int = 1,
    epsilon: Optional[float] = None,
    recurrence_percentile: float = 0.1,
    image_size: Optional[int] = 64,
    epsilon_mode: str = "percentile",
    std_k: float = 0.25,
    diameter_frac: float = 0.1,
) -> np.ndarray:
    traj = build_phase_space(signal, m=m, tau=tau)
    dist = pairwise_distances(traj)
    eps = resolve_epsilon(
        dist,
        epsilon=epsilon,
        mode=epsilon_mode,
        percentile=recurrence_percentile,
        std_k=std_k,
        diameter_frac=diameter_frac,
    )
    rp = (dist <= eps).astype(np.float32)
    return resize_square(rp, image_size)


def preprocess_signal(arr: np.ndarray, method_cfg) -> np.ndarray:
    return _preprocess(np.asarray(arr, dtype=np.float64), method_cfg)


def _preprocess(arr: np.ndarray, method_cfg) -> np.ndarray:
    # 轻量前置滤波（抗噪用）：必须在 z-score 之前，且拟合边界与符号化共用本函数，
    # 因此两条路径的尺度天然一致。
    kind = str((method_cfg or {}).get("prefilter", "none"))
    if kind.lower() not in ("none", "", "false"):
        from .prefilter import apply_prefilter
        arr = apply_prefilter(arr, kind, method_cfg)
    norm = str(method_cfg.get("normalize", "zscore")).lower()
    if norm in ("train_global_zscore", "train_global", "global_zscore"):
        if method_cfg.get("global_mean") is None or method_cfg.get("global_std") is None:
            raise KeyError(
                "normalize=train_global_zscore 需要 global_mean/global_std；"
                "它们由 build_dataset 在用训练集建缓存时算出并写入 meta.json"
            )
        mu = np.asarray(method_cfg["global_mean"], dtype=np.float64)
        sd = np.asarray(method_cfg["global_std"], dtype=np.float64)
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            return (a - mu[0]) / sd[0]
        return (a - mu[: a.shape[0], None]) / sd[: a.shape[0], None]
    if norm in ("zscore", "z", "std", "per_window_per_channel_zscore"):
        arr = zscore_channels(arr)
    return arr


def _build_1d(signal: np.ndarray, method_cfg) -> np.ndarray:
    kind = str(method_cfg.representation).lower()
    if kind == "transition":
        from .state_transition import build_state_transition

        return build_state_transition(signal, method_cfg)
    if kind == "rp":
        eps = method_cfg.get("epsilon", None)
        return build_recurrence_plot(
            signal,
            m=int(method_cfg.embedding_dim),
            tau=int(method_cfg.time_delay),
            epsilon=None if eps in (None, "null") else float(eps),
            recurrence_percentile=float(method_cfg.get("recurrence_percentile", 0.1)),
            image_size=int(method_cfg.rp_image_size),
            epsilon_mode=str(method_cfg.get("epsilon_mode", "percentile")),
            std_k=float(method_cfg.get("epsilon_std_k", 0.25)),
            diameter_frac=float(method_cfg.get("diameter_frac", 0.1)),
        )
    raise ValueError(f"未知 representation: {kind}")


def build_representation(signal: np.ndarray, method_cfg) -> Tuple[np.ndarray, dict]:
    """1D → [H,W]；多通道 RP → [C,H,W]。"""
    arr = np.asarray(signal, dtype=np.float64)
    arr = np.squeeze(arr)
    arr = _preprocess(arr, method_cfg)
    meta = {}
    kind = str(method_cfg.representation).lower()
    if kind == "transition":
        from .state_transition import build_state_transition

        return build_state_transition(arr, method_cfg), meta
    if arr.ndim == 1:
        rp = _build_1d(arr, method_cfg)
    elif arr.ndim == 2:
        maps = [_build_1d(arr[c], method_cfg) for c in range(arr.shape[0])]
        rp = np.stack(maps, axis=0).astype(np.float32)
    else:
        raise ValueError(f"表示输入须为 1D 或 [C,T]，收到 {arr.shape}")
    return rp, meta
