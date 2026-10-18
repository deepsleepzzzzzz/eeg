"""Independent held-out final acceptance for SEU DDS.

Protocol (frozen-hyperparameter, single-shot):
  - Four contiguous time segments, in strict temporal order:
        train  [0, 0.65)   val [0.65, 0.80)   test [0.80, 0.90)   holdout [0.90, 1.00)
  - Symbolization bin edges are fit on the TRAIN split only (no leakage).
  - Model is trained on train, early-stopped on val.
  - test  = the segment used during tuning (reported for continuity).
  - holdout = the final segment, measured exactly ONCE after hyperparameters are frozen.

The holdout segment sits at the very tail of the time axis. It never enters
training, bin-edge fitting, or early stopping. It overlaps the segment that was
previously used as the tuning "test" (because the full time axis was already
partitioned), so it is *training-side* independent rather than "born unseen".
The strongest "unseen" evidence remains the 5-fold rolling-origin CV; this
script adds a single-shot frozen final measurement on the hardest tail segment.

Usage:
  python scripts/experiments/heldout_seu.py --config config/seu_gear_cnn.yaml \
      --condition 30_2 --seed 42 --out datas/experiments_seu/holdout/gear_cnn_30_2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import (
    class_file,
    class_names,
    cut_windows,
    load_seu_csv,
    normalize_condition,
    task_dir,
    vibration_xyz,
    window_starts,
)
from experiments.train_runtime import evaluate, get_device, make_loader, resolve_path, set_seed
from models.cnn import build_model
from representation.recurrence_plot import build_representation
from representation.state_transition import fit_train_edges

# (train_end, val_end, test_end, holdout_end) as fractions of the time axis.
SEGMENTS = (0.65, 0.80, 0.90, 1.00)


def encode_windows(windows: list[np.ndarray], method_cfg) -> np.ndarray:
    maps = []
    for win in windows:
        arr, _ = build_representation(win, method_cfg)
        maps.append(arr.astype(np.float32))
    X = np.stack(maps, axis=0)
    if not np.isfinite(X).all():
        raise RuntimeError("transition cache contains NaN or inf")
    return X


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.astype(int).tolist()


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    rec = {}
    for c in range(n_classes):
        mask = y_true == c
        n = int(mask.sum())
        rec[str(c)] = float((y_pred[mask] == c).sum() / n) if n else 0.0
    return rec


def build_segments(task: str, condition: str, cfg):
    names = class_names(task)
    folder = task_dir(resolve_path(cfg.data.raw_root), task)
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    tr_end, va_end, te_end, ho_end = SEGMENTS

    by_split = {sn: [] for sn in ("train", "val", "test", "holdout")}
    y_split = {sn: [] for sn in ("train", "val", "test", "holdout")}
    for label, name in enumerate(names):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw)
        n = int(xyz.shape[1])
        ranges = {
            "train": (0, int(n * tr_end)),
            "val": (int(n * tr_end), int(n * va_end)),
            "test": (int(n * va_end), int(n * te_end)),
            "holdout": (int(n * te_end), int(n * ho_end)),
        }
        for sn, (lo, hi) in ranges.items():
            starts = window_starts(lo, hi, length, stride)
            wins = cut_windows(xyz, starts, length)
            by_split[sn].extend(wins)
            y_split[sn].extend([label] * len(wins))

    edges = fit_train_edges(by_split["train"], range(len(by_split["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X = {sn: encode_windows(by_split[sn], enc_cfg.method) for sn in ("train", "val", "test", "holdout")}
    y = {sn: np.asarray(y_split[sn], dtype=np.int64) for sn in ("train", "val", "test", "holdout")}
    return X, y, names


def train_and_evaluate(cfg, X, y, num_classes: int, names, seed: int, device):
    set_seed(seed, deterministic=bool(cfg.train.get("deterministic", True)))
    batch_size = int(cfg.train.batch_size)
    train_loader = make_loader(X["train"], y["train"], batch_size, 0, True)
    val_loader = make_loader(X["val"], y["val"], batch_size, 0, False)
    test_loader = make_loader(X["test"], y["test"], batch_size, 0, False)
    holdout_loader = make_loader(X["holdout"], y["holdout"], batch_size, 0, False)

    model = build_model(
        num_classes, cfg,
        in_channels=int(X["train"].shape[1]),
        image_size=int(X["train"].shape[-1]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )

    best_val = -1.0
    best_state = None
    bad = 0
    patience = int(cfg.train.get("early_stop_patience", 30))
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        _, val_acc, _, _ = evaluate(model, val_loader, device, criterion, predictions=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    _, test_acc, test_true, test_pred = evaluate(model, test_loader, device, criterion, predictions=True)
    _, holdout_acc, ho_true, ho_pred = evaluate(
        model, holdout_loader, device, criterion, predictions=True
    )
    return {
        "best_val_acc": round(float(best_val), 4),
        "test_acc": round(float(test_acc), 4),
        "holdout_acc": round(float(holdout_acc), 4),
        "holdout_confusion_matrix": confusion_matrix(ho_true, ho_pred, num_classes),
        "holdout_per_class_recall": per_class_recall(ho_true, ho_pred, num_classes),
        "test_confusion_matrix": confusion_matrix(test_true, test_pred, num_classes),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Independent held-out final acceptance for SEU.")
    p.add_argument("--config", default="config/seu_bearing_cnn.yaml")
    p.add_argument("--task", default=None)
    p.add_argument("--condition", default="30_2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    task = str(args.task or cfg.data.task)
    condition = normalize_condition(args.condition)
    if args.device:
        cfg.train.device = args.device
    device = get_device(cfg)

    X, y, names = build_segments(task, condition, cfg)
    print(
        f"[{task} {condition} {cfg.model.arch}] "
        f"train={len(y['train'])} val={len(y['val'])} test={len(y['test'])} "
        f"holdout={len(y['holdout'])} device={device}",
        flush=True,
    )
    result = train_and_evaluate(cfg, X, y, len(names), names, args.seed, device)
    summary = {
        "task": task,
        "condition": condition,
        "arch": str(cfg.model.arch),
        "seed": args.seed,
        "segments": {
            "train": [0.0, SEGMENTS[0]],
            "val": [SEGMENTS[0], SEGMENTS[1]],
            "test": [SEGMENTS[1], SEGMENTS[2]],
            "holdout": [SEGMENTS[2], SEGMENTS[3]],
        },
        "class_names": list(names),
        "split_counts": {sn: int(len(y[sn])) for sn in ("train", "val", "test", "holdout")},
        "protocol": "frozen_hyperparameters_single_shot_holdout",
        **result,
    }
    print(
        f"=== {task} {condition} ({cfg.model.arch}) held-out acceptance ===\n"
        f"val={summary['best_val_acc']:.4f} test={summary['test_acc']:.4f} "
        f"holdout={summary['holdout_acc']:.4f}",
        flush=True,
    )
    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
