"""Build SEU DDS windows as stacked transition matrices. Test stays sealed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import (
    CONDITIONS,
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
from data.seu_cache import CACHE_VERSION, cache_identity, cache_spec, digest, file_digest, validate_cache
from experiments.train_runtime import resolve_path
from representation.recurrence_plot import build_representation
from representation.state_transition import fit_train_edges, uses_train_edges
from utils.io_utils import ensure_dir, load_json, save_json


def class_count_dict(labels: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {names[int(v)]: int(c) for v, c in zip(values, counts) if 0 <= int(v) < len(names)}


def encode_windows(windows: list[np.ndarray], method_cfg) -> np.ndarray:
    if not windows:
        raise RuntimeError("no windows to encode")
    maps = []
    for win in windows:
        arr, _ = build_representation(win, method_cfg)
        maps.append(arr.astype(np.float32))
    X = np.stack(maps, axis=0)
    if not np.isfinite(X).all():
        raise RuntimeError("transition cache contains NaN or inf")
    return X


def split_starts(n: int, cfg) -> dict[str, np.ndarray]:
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    train_end, val_end = time_split_bounds(n, cfg.data.train_ratio, cfg.data.val_ratio)
    max_per = cfg.data.get("max_windows_per_class")
    out = {}
    ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }
    for name, (lo, hi) in ranges.items():
        starts = window_starts(lo, hi, length, stride)
        if max_per is not None:
            starts = starts[: int(max_per)]
        out[name] = starts
    return out


def split_windows(xyz: np.ndarray, cfg) -> dict[str, list[np.ndarray]]:
    return {name: cut_windows(xyz, starts, int(cfg.data.epoch_samples))
            for name, starts in split_starts(int(xyz.shape[1]), cfg).items()}


def build_condition(task: str, condition: str, cfg, force: bool) -> dict:
    names = class_names(task)
    raw_root = resolve_path(cfg.data.raw_root)
    folder = task_dir(raw_root, task)
    condition = normalize_condition(condition)
    out_dir = resolve_path(cfg.data.output_root) / task / condition
    meta_path = out_dir / "meta.json"
    spec = cache_spec(task, condition, cfg)
    sources = {name: {"file": class_file(task, name, condition),
                      "sha256": file_digest(folder / class_file(task, name, condition))}
               for name in names}
    if meta_path.exists() and not force:
        validate_cache(load_json(str(meta_path)), cfg, task, condition, out_dir,
                       splits=("train", "val", "test"), source_files=sources)
        print(f"{task} {condition}: verified cache, skip")
        return {"task": task, "condition": condition, "skipped": True}

    method = cfg.method
    # normalize=zscore is allowed: it is a per-window deterministic transform, so the
    # train_global bin edges (fit after preprocessing) still map every sample to the
    # same fixed state space. It only removes per-sample amplitude scale differences.

    by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    y_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    file_rows = {}
    for label, name in enumerate(names):
        path = folder / class_file(task, name, condition)
        print(f"load {path.name}", flush=True)
        raw = load_seu_csv(path)
        xyz = vibration_xyz(raw)
        file_rows[name] = int(xyz.shape[1])
        parts = split_windows(xyz, cfg)
        for split_name, windows in parts.items():
            by_split[split_name].extend(windows)
            y_split[split_name].extend([label] * len(windows))
        print(
            f"  {name}: n={xyz.shape[1]} "
            f"train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}",
            flush=True,
        )

    train_windows = by_split["train"]
    edges = fit_train_edges(train_windows, range(len(train_windows)), method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})

    ensure_dir(str(out_dir))
    # A failed rebuild must not leave an old completion marker over mixed arrays.
    meta_path.unlink(missing_ok=True)
    arrays = {}
    array_hashes = {}
    for split_name in ("train", "val", "test"):
        X = encode_windows(by_split[split_name], enc_cfg.method)
        y = np.asarray(y_split[split_name], dtype=np.int64)
        np.save(out_dir / f"X_{split_name}.npy", X)
        np.save(out_dir / f"y_{split_name}.npy", y)
        for kind in ("X", "y"):
            name = f"{kind}_{split_name}.npy"
            array_hashes[name] = file_digest(out_dir / name)
        arrays[split_name] = {"n": int(len(y)), "X_shape": list(X.shape), "y_shape": list(y.shape)}
        print(f"{task} {condition} {split_name}: {X.shape}", flush=True)

    sample_identity = {
        "task": str(task).lower(), "condition": condition, "channels": [2, 3, 4],
        "class_names": list(names), "sources": sources,
        "length": int(cfg.data.epoch_samples),
        "starts": {name: {split: starts.tolist() for split, starts in split_starts(file_rows[name], cfg).items()}
                   for name in names},
    }
    meta = {
        "cache_version": CACHE_VERSION,
        "cache_spec": spec,
        "source_files": sources,
        "array_hashes": array_hashes,
        "task": task,
        "condition": condition,
        "protocol": "seu_time_split",
        "class_names": list(names),
        "channels": [2, 3, 4],
        "epoch_samples": int(cfg.data.epoch_samples),
        "window_stride": int(cfg.data.get("window_stride", cfg.data.epoch_samples)),
        "overlap": int(cfg.data.get("window_stride", cfg.data.epoch_samples)) < int(cfg.data.epoch_samples),
        "train_ratio": float(cfg.data.train_ratio),
        "val_ratio": float(cfg.data.val_ratio),
        "test_ratio": float(cfg.data.test_ratio),
        "split_seed": int(cfg.data.split_seed),
        "split_kind": "time_contiguous",
        "file_length": file_rows,
        "split_sample_id_fingerprint": digest(sample_identity),
        "bin_scope": str(method.get("bin_scope", "train_global")),
        "bin_edges_mode": str(method.get("bin_edges_mode", "quantile")),
        "normalize": str(method.get("normalize", "none")),
        "bin_edges": edges.tolist(),
        "arrays": arrays,
        "train_class_counts": class_count_dict(np.asarray(y_split["train"]), names),
        "val_class_counts": class_count_dict(np.asarray(y_split["val"]), names),
        "test_class_counts": class_count_dict(np.asarray(y_split["test"]), names),
    }
    meta["cache_fingerprint"] = cache_identity(meta)
    save_json(meta, str(meta_path))
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Build SEU transition windows with a time split.")
    parser.add_argument("--config", default="config/seu_bearing_cnn.yaml")
    parser.add_argument("--task", default=None, help="bearing or gear")
    parser.add_argument("--conditions", default=None, help="comma separated, e.g. 20_0,30_2")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    task = str(args.task or cfg.data.task)
    if args.conditions:
        conditions = [normalize_condition(c.strip()) for c in args.conditions.split(",") if c.strip()]
    else:
        conditions = [normalize_condition(c) for c in cfg.data.get("conditions", list(CONDITIONS))]
    for condition in conditions:
        build_condition(task, condition, cfg, force=bool(args.force))


if __name__ == "__main__":
    main()
