"""相空间重构（Takens）。"""
from __future__ import annotations

import numpy as np


def build_phase_space(signal: np.ndarray, m: int, tau: int) -> np.ndarray:
    """1D → [L, m]；[C, T] 多导 → [L, C*m]（论文3 多通道相空间）。"""
    x = np.asarray(signal, dtype=np.float64)
    x = np.squeeze(x)
    m = int(m)
    tau = max(int(tau), 1)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2:
        if x.shape[0] < x.shape[1]:
            x = x.T
    else:
        raise ValueError(f"相空间输入须为 1D 或 2D，收到 {x.shape}")
    n, c = x.shape
    length = n - (m - 1) * tau
    if length <= 1:
        raise ValueError(f"信号过短无法嵌入: n={n}, m={m}, tau={tau}")
    traj = np.empty((length, c * m), dtype=np.float64)
    for i in range(m):
        traj[:, i * c : (i + 1) * c] = x[i * tau : i * tau + length]
    return traj
