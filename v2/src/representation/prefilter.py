"""符号化前的轻量滤波（抗噪用）。

放在 preprocess_signal 里，保证【拟合分位边界】和【符号化】两条路径
走一模一样的步骤 —— 否则又会犯"尺度不一致"那类 bug。
"""
from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, k: int = 20) -> np.ndarray:
    """窗内滑动平均，最简单的低通。O(T)，微秒级。"""
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(k))
    if k == 1:
        return x
    ker = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(x, ker, mode="same")
    return np.stack([np.convolve(c, ker, mode="same") for c in x])


def fir_bandpass(x: np.ndarray, fs: float = 5120.0, lo: float = 300.0, hi: float = 2200.0, taps: int = 101) -> np.ndarray:
    """线性相位 FIR 带通（窗函数法），用卷积实现，单窗几十微秒。"""
    x = np.asarray(x, dtype=np.float64)
    n = np.arange(taps) - (taps - 1) / 2.0
    lo = max(1e-6, float(lo)); hi = min(fs / 2 - 1e-6, float(hi))
    h = 2 * hi / fs * np.sinc(2 * hi / fs * n) - 2 * lo / fs * np.sinc(2 * lo / fs * n)
    h *= np.hamming(taps)
    h /= np.abs(h).sum() + 1e-12
    if x.ndim == 1:
        return np.convolve(x, h, mode="same")
    return np.stack([np.convolve(c, h, mode="same") for c in x])


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int = 3, mode: str = "soft") -> np.ndarray:
    """小波阈值去噪（自实现 db1-db4，无需 PyWavelets）。"""
    from .wavelet_np import wavelet_denoise as _wd
    order = {"db1": 1, "db2": 2, "db3": 3, "db4": 4}.get(str(wavelet).lower(), 4)
    return _wd(np.asarray(x, dtype=np.float64), order=order, levels=int(level), mode=str(mode))


def apply_prefilter(arr: np.ndarray, kind: str, cfg: dict) -> np.ndarray:
    kind = str(kind).lower()
    a = np.asarray(arr, dtype=np.float64)
    if kind in ("none", "", "false"):
        return a
    if kind in ("moving_average", "ma", "smooth"):
        return moving_average(a, int(cfg.get("prefilter_k", 20)))
    if kind in ("fir_bandpass", "fir", "bandpass"):
        return fir_bandpass(a, float(cfg.get("prefilter_fs", 5120.0)),
                            float(cfg.get("prefilter_lo", 300.0)),
                            float(cfg.get("prefilter_hi", 2200.0)),
                            int(cfg.get("prefilter_taps", 101)))
    if kind in ("wavelet", "wavelet_denoise", "wt"):
        return wavelet_denoise(a, str(cfg.get("prefilter_wavelet", "db4")),
                               int(cfg.get("prefilter_level", 3)),
                               str(cfg.get("prefilter_mode", "soft")))
    raise ValueError(f"unknown prefilter: {kind}")
