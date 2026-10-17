"""Southeast University DDS gearbox / bearing recordings."""
from __future__ import annotations

from pathlib import Path

import numpy as np

def normalize_condition(value) -> str:
    text = str(value).replace("-", "_")
    if text in {"20_0", "200"}:
        return "20_0"
    if text in {"30_2", "302"}:
        return "30_2"
    raise ValueError(f"unknown SEU condition: {value}")


CONDITIONS = ("20_0", "30_2")
VIBRATION_CHANNELS = (1, 2, 3)  # 默认: csv 列 2,3,4 = 行星齿轮箱 x,y,z (0-based)
# SEU 官方为 8 通道: 0=扭矩, 1-3=行星 xyz, 4-6=平行 xyz, 7=电机振动

BEARING_CLASSES = ("health", "ball", "inner", "outer", "comb")
GEAR_CLASSES = ("Health", "Chipped", "Miss", "Root", "Surface")

BEARING_FILES = {
    "health": "health_{cond}.csv",
    "ball": "ball_{cond}.csv",
    "inner": "inner_{cond}.csv",
    "outer": "outer_{cond}.csv",
    "comb": "comb_{cond}.csv",
}
GEAR_FILES = {
    "Health": "Health_{cond}.csv",
    "Chipped": "Chipped_{cond}.csv",
    "Miss": "Miss_{cond}.csv",
    "Root": "Root_{cond}.csv",
    "Surface": "Surface_{cond}.csv",
}


def class_names(task: str) -> tuple[str, ...]:
    task = str(task).lower()
    if task == "bearing":
        return BEARING_CLASSES
    if task == "gear":
        return GEAR_CLASSES
    raise ValueError(f"unknown SEU task: {task}")


def class_file(task: str, name: str, condition: str) -> str:
    cond = normalize_condition(condition)
    table = BEARING_FILES if str(task).lower() == "bearing" else GEAR_FILES
    return table[name].format(cond=cond)


def task_dir(raw_root: Path, task: str) -> Path:
    task = str(task).lower()
    if task == "bearing":
        return Path(raw_root) / "gearbox" / "bearingset"
    if task == "gear":
        return Path(raw_root) / "gearbox" / "gearset"
    raise ValueError(f"unknown SEU task: {task}")


def load_seu_csv(path: Path) -> np.ndarray:
    """Return float64 [T, 8] after the DDS header. Files mix tab and comma."""
    start = 0
    first_data = ""
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            stripped = line.lstrip()
            if stripped and (stripped[0].isdigit() or stripped[0] in "+-"):
                start = i
                first_data = stripped
                break
        else:
            raise ValueError(f"no numeric rows in {path}")
    delimiter = "\t" if first_data.count("\t") >= 7 else ","
    arr = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=start,
        usecols=range(8),
        dtype=np.float64,
        invalid_raise=False,
        filling_values=np.nan,
    )
    if arr.ndim == 1:
        arr = arr.reshape(-1, 8) if arr.size % 8 == 0 else arr.reshape(-1, 1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.shape[1] < 4:
        raise ValueError(f"{path} has {arr.shape[1]} columns, need at least 4")
    return arr


def vibration_xyz(arr: np.ndarray, channels=None) -> np.ndarray:
    """Selected raw CSV channels as [C, T].

    `channels` are 0-based column indices; defaults to planetary x/y/z.
    """
    if channels is None:
        channels = VIBRATION_CHANNELS
    channels = [int(c) for c in channels]
    if not channels:
        raise ValueError("channels must not be empty")
    if min(channels) < 0 or max(channels) >= arr.shape[1]:
        raise ValueError(f"channel index out of range for {arr.shape[1]} columns: {channels}")
    cols = arr[:, channels]
    return np.ascontiguousarray(cols.T)


def time_split_bounds(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n * float(train_ratio))
    val_end = train_end + int(n * float(val_ratio))
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError(f"cannot split n={n} with train={train_ratio} val={val_ratio}")
    return train_end, val_end


def window_starts(start: int, end: int, length: int, stride: int) -> np.ndarray:
    if end - start < length:
        return np.zeros((0,), dtype=np.int64)
    last = end - length
    return np.arange(start, last + 1, int(stride), dtype=np.int64)


def cut_windows(xyz: np.ndarray, starts: np.ndarray, length: int) -> list[np.ndarray]:
    return [xyz[:, s : s + length].copy() for s in starts]


def add_noise(
    windows: list[np.ndarray],
    snr_db: float,
    seed: int = 0,
    p_ref: float | None = None,
) -> list[np.ndarray]:
    """Add white Gaussian noise at a given SNR (dB) to each [C, T] window.

    p_ref=None   -> 每窗参考：p_noise = 该窗自己的功率 / ratio。
                    【历史默认】注意：此时加噪窗能量恒等于干净窗能量乘一个常数，
                    于是"窗能量"本身在任何 SNR 下都保持类别可分，会高估抗噪性能。
    p_ref=数值    -> 全局参考：所有窗用同一个噪声功率 p_ref / ratio。
                    噪声地板不随窗走，能量捷径失效，是物理上更严格的口径。
    """
    ratio = 10 ** (snr_db / 10.0)
    rng = np.random.default_rng(seed)
    out = []
    for w in windows:
        w = np.asarray(w, dtype=np.float64)
        p_sig = float(np.mean(w ** 2))
        base = p_sig if p_ref is None else float(p_ref)
        p_noise = base / ratio if base > 0 else 0.0
        noise = np.sqrt(p_noise) * rng.standard_normal(w.shape)
        out.append((w + noise).astype(np.float32))
    return out
