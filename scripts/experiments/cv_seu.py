"""Blocked rolling-origin time-series cross-validation for SEU DDS.

Each fold keeps train strictly *before* val/test on the time axis (no leakage),
and the test window slides forward across folds. This gives a mean ± std of the
test accuracy that no single holdout (or any test segment previously "peeked" at)
can bias.

Usage:
  python scripts/experiments/cv_seu.py --config config/seu_gear_gcn.yaml \
      --condition 30_2 --folds 5 --seed 42 --out datas/experiments_seu/cv/gear_gcn_30_2.json
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

# (train_end_frac, val_end_frac, test_end_frac) — rolling-origin, test slides forward.
FOLDS = [
    (0.60, 0.70, 0.80),
    (0.65, 0.75, 0.85),
    (0.70, 0.80, 0.90),
    (0.75, 0.85, 0.95),
    (0.80, 0.90, 1.00),
]


def encode_windows(windows: list[np.ndarray], method_cfg) -> np.ndarray:
    maps = []
    for win in windows:
        arr, _ = build_representation(win, method_cfg)
        maps.append(arr.astype(np.float32))
    X = np.stack(maps, axis=0)
    if not np.isfinite(X).all():
        raise RuntimeError("transition cache contains NaN or inf")
    return X


def build_fold(task: str, condition: str, cfg, train_end: float, val_end: float, test_end: float):
    names = class_names(task)
    folder = task_dir(resolve_path(cfg.data.raw_root), task)
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    by_split = {"train": [], "val": [], "test": []}
    y_split = {"train": [], "val": [], "test": []}
    for label, name in enumerate(names):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw)
        n = int(xyz.shape[1])
        ranges = {
            "train": (0, int(n * train_end)),
            "val": (int(n * train_end), int(n * val_end)),
            "test": (int(n * val_end), int(n * test_end)),
        }
        for sn, (lo, hi) in ranges.items():
            starts = window_starts(lo, hi, length, stride)
            wins = cut_windows(xyz, starts, length)
            by_split[sn].extend(wins)
            y_split[sn].extend([label] * len(wins))

    edges = fit_train_edges(by_split["train"], range(len(by_split["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X = {sn: encode_windows(by_split[sn], enc_cfg.method) for sn in ("train", "val", "test")}
    y = {sn: np.asarray(y_split[sn], dtype=np.int64) for sn in ("train", "val", "test")}
    return X, y, len(names)


def train_fold(task: str, cfg, X, y, num_classes: int, seed: int, device) -> tuple[float, float]:
    set_seed(seed, deterministic=bool(cfg.train.get("deterministic", True)))
    batch_size = int(cfg.train.batch_size)
    train_loader = make_loader(X["train"], y["train"], batch_size, 0, True)
    val_loader = make_loader(X["val"], y["val"], batch_size, 0, False)
    test_loader = make_loader(X["test"], y["test"], batch_size, 0, False)

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
    _, test_acc, _, _ = evaluate(model, test_loader, device, criterion, predictions=True)
    return float(best_val), float(test_acc)


def parse_args():
    p = argparse.ArgumentParser(description="Rolling-origin time-series CV for SEU.")
    p.add_argument("--config", default="config/seu_bearing_cnn.yaml")
    p.add_argument("--task", default=None)
    p.add_argument("--condition", default="30_2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="output JSON path")
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

    folds = FOLDS[: args.folds]
    results = []
    for fi, (te, ve, se) in enumerate(folds, 1):
        X, y, num_classes = build_fold(task, condition, cfg, te, ve, se)
        best_val, test_acc = train_fold(task, cfg, X, y, num_classes, args.seed, device)
        results.append({"fold": fi, "val_acc": round(best_val, 4), "test_acc": round(test_acc, 4)})
        print(
            f"[{task} {condition}] fold {fi}/{len(folds)} "
            f"train={len(y['train'])} val={len(y['val'])} test={len(y['test'])} "
            f"val_acc={best_val:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )

    test_accs = [r["test_acc"] for r in results]
    summary = {
        "task": task,
        "condition": condition,
        "arch": str(cfg.model.arch),
        "folds": len(results),
        "seed": args.seed,
        "folds_detail": results,
        "test_mean": round(float(np.mean(test_accs)), 4),
        "test_std": round(float(np.std(test_accs)), 4),
        "test_min": round(float(np.min(test_accs)), 4),
        "test_max": round(float(np.max(test_accs)), 4),
    }
    print(
        f"\n=== {task} {condition} ({cfg.model.arch}) CV summary ===\n"
        f"test mean={summary['test_mean']:.4f} ± {summary['test_std']:.4f} "
        f"min={summary['test_min']:.4f} max={summary['test_max']:.4f}",
        flush=True,
    )
    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
