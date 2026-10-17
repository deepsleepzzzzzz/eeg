from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.protocol import (  # noqa: E402
    load_protocol,
    method_cfg_from_protocol,
    window_cfg,
    channel_indices,
    channel_indices_with_source,
    raw_root as protocol_raw_root,
    dataset_dir,
    sealed_test_dir,
)
from data.seu_dds import (  # noqa: E402
    add_noise,
    class_file,
    class_names,
    cut_windows,
    load_seu_csv,
    normalize_condition,
    task_dir,
    time_split_bounds,
    vibration_xyz,
    window_starts,
)
from representation.recurrence_plot import preprocess_signal  # noqa: E402
from representation.state_transition import build_state_transition, fit_train_edges  # noqa: E402
from utils.io_utils import ensure_dir, save_json  # noqa: E402

def split_intervals(protocol: dict) -> dict:
    n = int(protocol["dataset"].get("file_length", 1_048_560))
    split = protocol["split"]
    train_end, val_end = time_split_bounds(n, float(split["train_ratio"]), float(split["val_ratio"]))
    return {
        "train": [0, int(train_end)],
        "val": [int(train_end), int(val_end)],
        "test": [int(val_end), n],
    }


def split_hash(protocol: dict, task: str, condition: str, class_names_list: list[str]) -> str:
    length, stride = window_cfg(protocol)
    intervals = split_intervals(protocol)
    payload = {
        "protocol": protocol["name"],
        "task": task,
        "condition": condition,
        "channel_indices": channel_indices(protocol),
        "channel_indices_source": channel_indices_with_source(protocol)[1],
        "window_length": length,
        "window_stride": stride,
        "intervals": intervals,
        "classes": {},
    }
    for name in class_names_list:
        starts = {}
        for split, (lo, hi) in intervals.items():
            starts[split] = window_starts(lo, hi, length, stride).astype(int).tolist()
        payload["classes"][name] = starts
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dataset_hash(raw_root: Path) -> str:
    files_json = raw_root / "_files.json"
    if files_json.exists():
        return hashlib.sha256(files_json.read_bytes()).hexdigest()
    return "unknown"


def encode_windows(windows: list[np.ndarray], method_cfg: dict) -> np.ndarray:
    if not windows:
        raise RuntimeError("no windows to encode")
    kind = str(method_cfg.get("representation", "transition")).lower()
    if kind == "raw":
        # 基线用：直接送归一化后的原始波形 [C, T]
        maps = [np.asarray(preprocess_signal(win, method_cfg), dtype=np.float32) for win in windows]
    else:
        maps = [build_state_transition(win, method_cfg).astype(np.float32) for win in windows]
    X = np.stack(maps, axis=0)
    if not np.isfinite(X).all():
        raise RuntimeError("transition cache contains NaN or inf")
    return X


def class_count_dict(labels: np.ndarray, names: list[str]) -> dict:
    values, counts = np.unique(labels, return_counts=True)
    return {names[int(v)]: int(c) for v, c in zip(values, counts) if 0 <= int(v) < len(names)}


def noise_reference_mode(noise_cfg: dict, protocol_name: str = "") -> str:
    """返回噪声参考功率口径；缺失即报错（不允许静默默认）。

    为什么强制显式声明：per_window（逐窗参考）会让加噪窗能量恒等于干净窗能量的
    常数倍，使"窗能量"成为与表示无关的捷径，系统性抬高抗噪性能
    （实测：齿轮 20-0 -10 dB 虚高 13.9 点，轴承 20-0 虚高 64.7 点）。
    """
    ref = noise_cfg.get("reference")
    if ref is None:
        raise ValueError(
            "协议 %r 启用了噪声但未显式声明 noise.reference。\n"
            "  请在对应 yaml 里显式写 noise.reference: per_window 或 global，"
            "不要依赖代码默认值。\n"
            "  （per_window 有能量捷径，会抬高抗噪性能；新实验建议用 global）"
            % (protocol_name,)
        )
    return str(ref).lower()


def build_condition(protocol: dict, task: str, condition: str, force: bool = False, meta_only: bool = False, include_test: bool = False) -> dict:
    names = list(class_names(task))
    condition = normalize_condition(condition)
    raw_dir = protocol_raw_root(protocol)
    out_dir = dataset_dir(protocol, task, condition)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        print(f"{task} {condition}: meta exists, skip")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    length, stride = window_cfg(protocol)
    intervals = split_intervals(protocol)
    method_cfg = method_cfg_from_protocol(protocol)
    meta = {
        "protocol": protocol["name"],
        "task": task,
        "condition": condition,
        "class_names": names,
        "channel_indices": channel_indices(protocol),
        "channel_indices_source": channel_indices_with_source(protocol)[1],
        "window_length": length,
        "window_stride": stride,
        "split_intervals": intervals,
        "split_hash": split_hash(protocol, task, condition, names),
        "dataset_hash": dataset_hash(raw_dir),
        "meta_only": bool(meta_only),
        "test_sealed": bool(not include_test),
    }
    if meta_only:
        ensure_dir(str(out_dir))
        save_json(meta, str(meta_path))
        print(f"{task} {condition}: wrote metadata only")
        return meta

    folder = task_dir(raw_dir, task)
    splits_to_build = ["train", "val"] + (["test"] if include_test else [])
    by_split = {name: [] for name in splits_to_build}
    y_split = {name: [] for name in splits_to_build}
    for label, name in enumerate(names):
        path = folder / class_file(task, name, condition)
        print(f"load {path.name}", flush=True)
        raw = load_seu_csv(path)
        xyz = vibration_xyz(raw, channel_indices(protocol))
        n = int(xyz.shape[1])
        for split in splits_to_build:
            lo, hi = intervals[split]
            starts = window_starts(lo, min(int(hi), n), length, stride)
            by_split[split].extend(cut_windows(xyz, starts, length))
            y_split[split].extend([label] * len(starts))
        print(f"  {name}: train={len(by_split['train'])}", flush=True)

    # ---- 训练集固定尺度归一化：用【全部训练窗】统计一组 per-channel 均值/标准差
    #      与逐窗 z-score 的区别：这里不抹掉"振动强弱"这一幅值信息。
    if str(method_cfg.get("normalize")) == "train_global_zscore":
        pooled = np.concatenate([np.asarray(w, dtype=np.float64) for w in by_split["train"]], axis=1)
        method_cfg["global_mean"] = pooled.mean(axis=1).tolist()
        method_cfg["global_std"] = (pooled.std(axis=1) + 1e-12).tolist()
        print(f"  [norm] train_global 均值={np.round(method_cfg['global_mean'], 6).tolist()} "
              f"标准差={np.round(method_cfg['global_std'], 6).tolist()}", flush=True)
        meta["global_mean"] = method_cfg["global_mean"]
        meta["global_std"] = method_cfg["global_std"]

    # ---- 噪声注入（在切窗之后、表示之前；只对协议指定的 split）
    noise_cfg = protocol.get("noise") or {}
    snr_db = noise_cfg.get("snr_db")
    if bool(noise_cfg.get("enabled", False)) and snr_db is not None:
        splits_noisy = [s for s in str(noise_cfg.get("splits", "train,val")).split(",") if s in by_split]
        # 参考功率口径（决定噪声地板是否随窗走）：
        #   per_window（历史默认）：p_noise 按每个窗自己的功率缩放
        #       -> 加噪窗能量 = 干净窗能量 × 常数，"窗能量"成为任何 SNR 下都有效的捷径，
        #          会高估抗噪性能（已实测：齿轮 20-0 在 -10 dB 以下与纯能量分类器同分）。
        #   global：所有窗共用同一个参考功率（由训练集全部窗算得），噪声地板不随窗走。
        ref_mode = noise_reference_mode(noise_cfg, str(protocol.get("name", "")))
        p_ref = None
        if ref_mode == "global":
            pooled = np.concatenate(
                [np.asarray(w, dtype=np.float64) for w in by_split["train"]], axis=1)
            p_ref = float(np.mean(pooled ** 2))
            print(f"  [noise] 全局参考功率 p_ref={p_ref:.6e}（由训练集全部窗算得）", flush=True)
        for si, split in enumerate(splits_noisy):
            before = float(np.mean([np.mean(w ** 2) for w in by_split[split]]))
            by_split[split] = add_noise(by_split[split], float(snr_db),
                                        int(noise_cfg.get("seed", 0)) + si, p_ref=p_ref)
            after = float(np.mean([np.mean(w ** 2) for w in by_split[split]]))
            print(f"  [noise] {split}: SNR={snr_db} dB ({ref_mode})  功率 {before:.3e} -> {after:.3e}", flush=True)
        meta["noise"] = {"type": noise_cfg.get("type", "awgn"), "snr_db": float(snr_db),
                         "splits": splits_noisy, "seed": int(noise_cfg.get("seed", 0)),
                         "reference": ref_mode, "p_ref": p_ref}

    edges = fit_train_edges(by_split["train"], range(len(by_split["train"])), method_cfg)
    method_cfg["bin_edges"] = edges.tolist()

    ensure_dir(str(out_dir))
    arrays = {}
    for split in splits_to_build:
        X = encode_windows(by_split[split], method_cfg)
        y = np.asarray(y_split[split], dtype=np.int64)
        save_dir = out_dir if split != "test" else sealed_test_dir(protocol, task, condition)
        ensure_dir(str(save_dir))
        np.save(save_dir / f"X_{split}.npy", X)
        np.save(save_dir / f"y_{split}.npy", y)
        arrays[split] = {"n": int(len(y)), "X_shape": list(X.shape)}
        print(f"{task} {condition} {split}: {X.shape}", flush=True)

    meta["bin_edges"] = edges.tolist()
    meta["arrays"] = arrays
    meta["train_class_counts"] = class_count_dict(np.asarray(y_split["train"]), names)
    meta["val_class_counts"] = class_count_dict(np.asarray(y_split["val"]), names)
    test_count = len(window_starts(intervals["test"][0], intervals["test"][1], length, stride))
    meta["test_class_counts"] = {name: int(test_count) for name in names}
    save_json(meta, str(meta_path))
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--task", required=True, choices=["bearing", "gear"])
    p.add_argument("--conditions", default="20_0,30_2")
    p.add_argument("--force", action="store_true")
    p.add_argument("--meta-only", action="store_true")
    p.add_argument("--include-test", action="store_true", help="encode and save the sealed test split (only for final confirmed evaluation)")
    return p.parse_args()


def main():
    args = parse_args()
    protocol = load_protocol(Path(args.protocol))
    for c in [x.strip() for x in args.conditions.split(",") if x.strip()]:
        build_condition(protocol, args.task, c, force=args.force, meta_only=args.meta_only, include_test=args.include_test)


if __name__ == "__main__":
    main()

