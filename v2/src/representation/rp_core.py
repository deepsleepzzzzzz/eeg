"""RP/MRP 共用：距离、阈值、归一化、缩放。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def pairwise_distances(traj: np.ndarray) -> np.ndarray:
    sq = np.sum(traj * traj, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (traj @ traj.T), 0.0)
    return np.sqrt(d2)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    std = float(np.std(x))
    if std < 1e-12:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / std


def zscore_channels(arr: np.ndarray) -> np.ndarray:
    """[C, T] 或 1D。各导独立 z-score（论文1 归一化）。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return zscore_1d(a)
    out = np.empty_like(a)
    for c in range(a.shape[0]):
        out[c] = zscore_1d(a[c])
    return out


def resolve_epsilon(
    dist: np.ndarray,
    *,
    epsilon: Optional[float] = None,
    mode: str = "percentile",
    percentile: float = 0.1,
    std_k: float = 0.25,
    diameter_frac: float = 0.1,
) -> float:
    """绝对 epsilon 优先；否则按论文/文献常用规则。

    - percentile：上三角距离分位（固定递归率，当前工程默认）
    - std：论文1 的 ε = k σ，σ 为全部距离的标准差
    - diameter：Marwan：最大相空间直径的比例（常取 0.1）
    """
    if epsilon is not None:
        return max(float(epsilon), 1e-8)
    mode = str(mode).lower()
    iu = np.triu_indices_from(dist, k=1)
    off = dist[iu]
    if off.size == 0:
        return 1e-8
    if mode in ("std", "sigma", "paper1"):
        sig = float(np.std(off))
        return max(float(std_k) * sig, 1e-8)
    if mode in ("diameter", "max", "marwan"):
        dmax = float(np.max(off))
        return max(float(diameter_frac) * dmax, 1e-8)
    return max(float(np.quantile(off, float(percentile))), 1e-8)


def resize_square(img: np.ndarray, image_size: Optional[int]) -> np.ndarray:
    if image_size is None:
        return img.astype(np.float32)
    h, w = img.shape[-2], img.shape[-1]
    if h == image_size and w == image_size:
        return img.astype(np.float32)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255.0).astype(np.uint8), mode="L")
    pil = pil.resize((int(image_size), int(image_size)), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0
