"""Train SEU bearing / gear classifiers. Softmax via CrossEntropy; test sealed."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import CONDITIONS, class_names, normalize_condition
from data.seu_cache import validate_cache
from experiments.train_runtime import (
    evaluate,
    get_device,
    load_array_pair,
    make_loader,
    method_name,
    parse_seeds,
    resolve_path,
    save_predictions,
    set_seed,
)
from models.cnn import build_model
from scripts.experiments.seu_figures import paper_stem, paper_title, save_val_figures
from utils.io_utils import ensure_dir, load_json, save_json


def class_count_dict(labels: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {names[int(v)]: int(c) for v, c in zip(values, counts) if 0 <= int(v) < len(names)}


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str, names: tuple[str, ...]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    return {
        f"{prefix}_acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        f"{prefix}_balanced_acc": float(
            balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
        ),
        f"{prefix}_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
        ),
        f"{prefix}_confusion_matrix": cm.astype(int).tolist(),
        f"{prefix}_class_counts": class_count_dict(y_true, names),
    }


def train_one(
    task: str,
    condition: str,
    seed: int,
    cfg,
    evaluate_test: bool,
    confirm_test: bool,
    write_paper_figures: bool = True,
) -> dict:
    if evaluate_test and not confirm_test:
        raise RuntimeError("test evaluation requires --confirm-test after parameters are frozen")

    names = class_names(task)
    condition = normalize_condition(condition)
    data_dir = resolve_path(cfg.data.output_root) / task / condition
    meta = load_json(str(data_dir / "meta.json"))
    validate_cache(meta, cfg, task, condition, data_dir,
                   splits=("train", "val", "test") if evaluate_test else ("train", "val"))
    X_train, y_train = load_array_pair(data_dir, "train")
    X_val, y_val = load_array_pair(data_dir, "val")
    X_test = y_test = None
    if evaluate_test:
        X_test, y_test = load_array_pair(data_dir, "test")

    set_seed(seed, deterministic=bool(cfg.train.get("deterministic", True)))
    device = get_device(cfg)
    workers = int(cfg.train.get("num_workers", 0))
    batch_size = int(cfg.train.batch_size)
    train_loader = make_loader(X_train, y_train, batch_size, workers, True)
    val_loader = make_loader(X_val, y_val, batch_size, workers, False)

    criterion = nn.CrossEntropyLoss()
    model = build_model(
        len(names),
        cfg,
        in_channels=int(X_train.shape[1]),
        image_size=int(X_train.shape[-1]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    run_dir = Path(
        ensure_dir(str(resolve_path(cfg.train.output_dir) / task / condition / f"seed_{seed}"))
    )
    ckpt_path = run_dir / str(cfg.train.get("checkpoint_name", "best.pt"))
    stop_on = str(cfg.train.get("stop_on", "val_acc"))
    patience = int(cfg.train.get("early_stop_patience", 30))
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    bad_epochs = 0
    train_losses = []

    print(
        f"{task} {condition} seed={seed}: train={len(y_train)} val={len(y_val)} "
        f"test={'sealed' if not evaluate_test else len(y_test)} "
        f"shape={X_train.shape[1:]} classes={len(names)} device={device}",
        flush=True,
    )
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, val_acc, val_true, val_pred = evaluate(
            model, val_loader, device, criterion, predictions=True
        )
        train_losses.append(train_loss)
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        improved = val_loss < best_val_loss if stop_on == "val_loss" else val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_metrics = metrics_from_predictions(val_true, val_pred, "val", names)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "cache_fingerprint": meta["cache_fingerprint"],
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, _, val_true, val_pred = evaluate(model, val_loader, device, criterion, predictions=True)
    save_predictions(run_dir / "val_predictions.csv", val_true, val_pred)
    val_metrics = metrics_from_predictions(val_true, val_pred, "val", names)
    if write_paper_figures:
        title = paper_title(task, condition, method_name(cfg))
        paper_dir = Path(ensure_dir(str(ROOT / "doc" / "figures" / "seu")))
        figs = save_val_figures(
            model,
            val_loader,
            device,
            names,
            val_metrics["val_confusion_matrix"],
            run_dir,
            title,
            seed=seed,
        )
        stem = paper_stem(task, condition, method_name(cfg), seed=seed)
        shutil.copy2(figs["val_confusion"], paper_dir / f"{stem}_confusion.png")
        shutil.copy2(figs["val_tsne"], paper_dir / f"{stem}_tsne.png")

    result = {
        "task": task,
        "condition": condition,
        "seed": int(seed),
        "protocol": "seu_time_split",
        "method": method_name(cfg),
        "epoch_samples": int(cfg.data.epoch_samples),
        "window_stride": int(cfg.data.get("window_stride", cfg.data.epoch_samples)),
        "split_seed": int(meta["split_seed"]),
        "split_sample_id_fingerprint": str(meta["split_sample_id_fingerprint"]),
        "cache_fingerprint": meta["cache_fingerprint"],
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(train_losses)),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_evaluated": bool(evaluate_test),
        "input_shape": list(X_train.shape[1:]) if X_train.ndim else [],
        "class_names": list(names),
        "train_class_counts": class_count_dict(y_train, names),
        "val_class_counts": class_count_dict(y_val, names),
        **best_metrics,
    }
    if evaluate_test:
        test_loss, test_acc, test_true, test_pred = evaluate(
            model,
            make_loader(X_test, y_test, batch_size, workers, False),
            device,
            criterion,
            predictions=True,
        )
        save_predictions(run_dir / "test_predictions.csv", test_true, test_pred)
        result.update(
            {
                "test_loss": float(test_loss),
                "test_acc_once": float(test_acc),
                "test_evaluated_after_freeze": True,
                **metrics_from_predictions(test_true, test_pred, "test", names),
            }
        )
    save_json(result, str(run_dir / "metrics.json"))
    print(
        f"{task} {condition} seed={seed}: best_val_acc={best_val_acc:.4f} "
        f"best_epoch={best_epoch} test_evaluated={evaluate_test}",
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEU 5-class CNN/GCN with test sealed by default."
    )
    parser.add_argument("--config", default="config/seu_bearing_cnn.yaml")
    parser.add_argument("--task", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))
    task = str(args.task or cfg.data.task)
    if args.conditions:
        conditions = [normalize_condition(c.strip()) for c in args.conditions.split(",") if c.strip()]
    else:
        conditions = [normalize_condition(c) for c in cfg.data.get("conditions", list(CONDITIONS))]
    seeds = parse_seeds(cfg, args)
    for condition in conditions:
        for seed in seeds:
            train_one(
                task,
                condition,
                seed,
                cfg,
                evaluate_test=bool(args.evaluate_test),
                confirm_test=bool(args.confirm_test),
            )


if __name__ == "__main__":
    main()
