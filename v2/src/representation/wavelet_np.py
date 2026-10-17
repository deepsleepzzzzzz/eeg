"""不依赖 PyWavelets 的离散小波阈值去噪（db 系列，正交）。

用循环卷积矩阵实现分析/综合，正交小波的完美重构条件为
    x = H^T (H x) + G^T (G x)
分析矩阵只与信号长度有关，建一次缓存复用，因此单窗耗时在微秒~毫秒级。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# db1..db4 尺度滤波器（sum = sqrt(2)）
DB = {
    1: [0.7071067811865476, 0.7071067811865476],
    2: [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
    3: [0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
        -0.1350110200102546, -0.0854412738820267, 0.0352262918857095],
    4: [0.2303778133088964, 0.7148465705529155, 0.6308807679298587, -0.0279837694168599,
        -0.1870348117188811, 0.0308413818355607, 0.0328830116668852, -0.0105974017849973],
}


def _analysis_matrix(n: int, h: np.ndarray) -> sp.csr_matrix:
    """循环卷积分析矩阵 H：(n/2) x n，H[row, (2*row + l) % n] = h[l]"""
    rows, cols, vals = [], [], []
    L = len(h)
    for r in range(n // 2):
        for l in range(L):
            rows.append(r); cols.append((2 * r + l) % n); vals.append(h[l])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n // 2, n))


_CACHE: dict[tuple[int, int], tuple] = {}


def _matrices(n: int, order: int):
    key = (n, order)
    if key not in _CACHE:
        h = np.asarray(DB[order], dtype=np.float64)
        g = np.array([(-1) ** i * h[len(h) - 1 - i] for i in range(len(h))])
        _CACHE[key] = (_analysis_matrix(n, h), _analysis_matrix(n, g))
    return _CACHE[key]


def _dwt(x: np.ndarray, levels: int, order: int):
    a = np.asarray(x, dtype=np.float64)
    details = []
    for _ in range(levels):
        n = len(a)
        if n % 2 or n < len(DB[order]) * 2:
            break
        H, G = _matrices(n, order)
        details.append(G @ a)
        a = H @ a
    return a, details


def _idwt(a: np.ndarray, details: list, order: int) -> np.ndarray:
    for d in reversed(details):
        n = len(d) * 2
        H, G = _matrices(n, order)
        a = H.T @ a + G.T @ d
    return a


def wavelet_denoise(x: np.ndarray, order: int = 4, levels: int = 3, mode: str = "soft") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    def one(c: np.ndarray) -> np.ndarray:
        a, details = _dwt(c, levels, order)
        if not details:
            return c
        sigma = np.median(np.abs(details[-1])) / 0.6745
        thr = sigma * np.sqrt(2.0 * np.log(len(c)))
        if thr <= 0:
            return c
        def shrink(d):
            if mode == "soft":
                return np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
            return np.where(np.abs(d) > thr, d, 0.0)
        return _idwt(a, [shrink(d) for d in details], order)

    if x.ndim == 1:
        return one(x)
    return np.stack([one(c) for c in x])
