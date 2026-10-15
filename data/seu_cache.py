"""Identity and integrity checks for shared SEU representation caches."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from data.seu_dds import class_names, normalize_condition

CACHE_VERSION = 2


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def cache_spec(task: str, condition: str, cfg) -> dict:
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    limit = cfg.data.get("max_windows_per_class")
    ratios = [float(cfg.data[key]) for key in ("train_ratio", "val_ratio", "test_ratio")]
    if length < 1 or stride < 1 or (limit is not None and int(limit) < 1):
        raise ValueError("window length, stride and optional window limit must be positive")
    if any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to one")
    method = OmegaConf.to_container(cfg.method, resolve=True)
    # Attention belongs to the classifier; bin_edges are fitted by the builder.
    method = {k: v for k, v in method.items() if k not in ("attention", "bin_edges")}
    return {
        "task": str(task).lower(), "condition": normalize_condition(condition),
        "class_names": list(class_names(task)), "channels": [2, 3, 4],
        "raw_root": str(Path(str(cfg.data.raw_root))),
        "epoch_samples": length, "window_stride": stride,
        "max_windows_per_class": None if limit is None else int(limit),
        "ratios": ratios, "method": method,
    }


def cache_identity(meta: dict) -> str:
    return digest({key: meta[key] for key in ("cache_version", "cache_spec", "source_files", "array_hashes", "split_sample_id_fingerprint")})


def validate_cache(meta: dict, cfg, task: str, condition: str, directory: Path,
                   splits=("train", "val"), source_files=None) -> None:
    """Fail closed; training checks only train/val unless test is explicitly opened."""
    reason = None
    if meta.get("cache_version") != CACHE_VERSION:
        reason = "legacy or incomplete cache metadata"
    elif meta.get("cache_spec") != cache_spec(task, condition, cfg):
        reason = "cache configuration differs from requested configuration"
    elif source_files is not None and meta.get("source_files") != source_files:
        reason = "raw source files changed"
    else:
        try:
            if meta.get("cache_fingerprint") != cache_identity(meta):
                reason = "cache identity is inconsistent"
            else:
                for split in splits:
                    for kind in ("X", "y"):
                        name = f"{kind}_{split}.npy"
                        path = directory / name
                        if not path.is_file() or file_digest(path) != meta["array_hashes"].get(name):
                            reason = f"missing or changed cache array: {name}"
                            break
                    if reason:
                        break
        except (KeyError, TypeError):
            reason = "incomplete cache metadata"
    if reason:
        raise RuntimeError(f"{reason} ({directory}); rebuild with build_seu_dataset.py --config <config> --force before training")
