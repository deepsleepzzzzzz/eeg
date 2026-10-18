from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.seu_dds import class_names, normalize_condition  # noqa: E402
from data.build_dataset import validate_cache  # noqa: E402
from config.protocol import (  # noqa: E402
    artifact_root,
    build_train_omega,
    dataset_dir,
    sealed_test_dir,
)
from evaluation.metrics import compute_metrics  # noqa: E402
from models.cnn import build_model  # noqa: E402
from training.runtime import evaluate, get_device, make_loader, set_seed, save_predictions  # noqa: E402
from utils.io_utils import ensure_dir, save_json  # noqa: E402


def load_yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def load_arrays(data_dir: Path, split: str):
    X = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
    y = np.load(data_dir / f"y_{split}.npy").astype(np.int64)
    return X, y


def build_cfg(protocol: dict, model_name: str, device: str | None = None):
    """device=None 时自动选择；显式传入可强制执行 CPU（并行跑时必需）。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_train_omega(protocol, model_name, device=device)


def build_scheduler(cfg, optimizer):
    """Return (scheduler, record).  Unknown/absent name -> no scheduler."""
    sched = cfg.train.get("lr_scheduler") or {}
    name = str(sched.get("name", "none")).lower()
    if name in ("", "none", "null", "false"):
        return None, {"name": "none"}
    if name in ("reduce_on_plateau", "plateau", "reducelronplateau"):
        monitor = str(sched.get("monitor", "val_loss")).lower()
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError(f"unsupported lr_scheduler.monitor: {monitor}")
        mode = "min" if monitor == "val_loss" else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(sched.get("factor", 0.5)),
            patience=int(sched.get("patience", 8)),
            min_lr=float(sched.get("min_lr", 1e-6)),
        )
        return scheduler, {
            "name": "reduce_on_plateau",
            "monitor": monitor,
            "factor": float(sched.get("factor", 0.5)),
            "patience": int(sched.get("patience", 8)),
            "min_lr": float(sched.get("min_lr", 1e-6)),
        }
    raise ValueError(f"unknown lr_scheduler.name: {name}")


def train_one(experiment_cfg: dict, protocol_cfg: dict, evaluate_test: bool, confirm_test: bool,
              epochs_override: int | None = None, limit_train: int | None = None, output_root=None,
              device: str | None = None, save_model: bool = False):
    if evaluate_test and not confirm_test:
        raise RuntimeError("test evaluation requires --confirm-test")

    task = str(experiment_cfg["task"])
    condition = normalize_condition(experiment_cfg["condition"])
    model_name = str(experiment_cfg["model"])
    seed = int(experiment_cfg["seed"])
    experiment_id = str(experiment_cfg["experiment_id"])
    protocol_name = str(protocol_cfg["name"])

    data_dir = dataset_dir(protocol_cfg, task, condition)
    meta_path = data_dir / "meta.json"
    meta = validate_cache(protocol_cfg, task, condition, include_test=evaluate_test)
    names = list(class_names(task))

    X_train, y_train = load_arrays(data_dir, "train")
    X_val, y_val = load_arrays(data_dir, "val")
    if limit_train:
        X_train, y_train = X_train[:limit_train], y_train[:limit_train]

    cfg = build_cfg(protocol_cfg, model_name, device=device)

    # ---- 一致性校验：GCN relational 布局要求 in_channels == 通道数 x 时间步数
    rep = protocol_cfg.get("representation", {})
    n_lags = len(rep.get("lags", []) or [])
    n_ch = len(rep.get("channels", []) or [])
    if str(cfg.model.get("arch", "cnn")).lower() == "gcn":
        if str(cfg.model.get("gcn_layout", "relational")).lower() == "relational":
            steps = int(cfg.model.get("gcn_steps", 0))
            if n_lags and steps != n_lags:
                raise ValueError(
                    f"gcn_steps={steps} 与 representation.lags 的数量 {n_lags} 不一致；"
                    f"relational 布局下 gcn_steps 必须等于滞后阶数"
                )
            if n_ch and X_train.shape[1] != n_ch * n_lags:
                raise ValueError(
                    f"输入通道 {X_train.shape[1]} != 通道数 {n_ch} x 滞后数 {n_lags}"
                )
    if epochs_override:
        cfg.train.epochs = int(epochs_override)
    set_seed(seed, deterministic=True)
    device = get_device(cfg)

    train_loader = make_loader(X_train, y_train, cfg.train.batch_size, 0, True)
    val_loader = make_loader(X_val, y_val, cfg.train.batch_size, 0, False)
    model = build_model(len(names), cfg, in_channels=X_train.shape[1], image_size=X_train.shape[-1]).to(device)

    opt_name = str(cfg.train.get("optimizer", "adam")).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.train.get("label_smoothing", 0.0)))
    scheduler, scheduler_record = build_scheduler(cfg, optimizer)

    run_dir = (Path(output_root) / experiment_id) if output_root is not None else (artifact_root(protocol_cfg) / protocol_name / experiment_id)
    ensure_dir(str(run_dir))

    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    bad_epochs = 0
    patience = int(cfg.train.early_stop_patience)
    history: list[dict] = []

    for epoch in range(1, int(cfg.train.epochs) + 1):
        # ---- train (loss/acc accumulated during the same forward pass: no extra cost)
        model.train()
        # 同上：训练指标也在 GPU 上累加，去掉每 batch 的 .item() 同步
        tr_loss_t = torch.zeros((), dtype=torch.float64, device=device)
        tr_correct_t = torch.zeros((), dtype=torch.long, device=device)
        tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            n = int(yb.numel())
            tr_total += n
            tr_loss_t += loss.detach().double() * n
            tr_correct_t += (logits.argmax(dim=1) == yb).sum()
        train_loss = float(tr_loss_t.item()) / max(tr_total, 1)
        train_acc = float(tr_correct_t.item()) / max(tr_total, 1)

        # ---- validation
        # 每轮只要 loss/acc；predictions=True 会在每个 batch 做一次 GPU->CPU 同步
        # 并把 975 个整数逐个转成 Python 对象，白白让 GPU 空转等 CPU。
        # 预测值只在选出最佳权重之后需要一次（见下方）。
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, predictions=False)
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": lr_now,
        })
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.3e}",
            flush=True,
        )

        # ---- model selection / early stopping (unchanged from v1 on purpose)
        if val_acc > best["val_acc"]:
            best = {"val_acc": float(val_acc), "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # ---- lr decay (v2)
        if scheduler is not None:
            scheduler.step(val_loss if scheduler_record["monitor"] == "val_loss" else val_acc)

        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch} (patience={patience})", flush=True)
            break

    model.load_state_dict(best["state"])
    if save_model:
        # 可选：把最佳权重落盘，供 t-SNE 等分析脚本提取 embed()（默认关闭，不影响已有结果）
        torch.save({"state": best["state"], "experiment_id": experiment_id,
                    "protocol": protocol_name, "task": task, "condition": condition,
                    "model": model_name, "seed": seed,
                    "cache_fingerprint": meta["cache_fingerprint"]}, run_dir / "best_state.pt")
        print(f"saved state {run_dir / 'best_state.pt'}")
    best_val_loss, best_val_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion, predictions=True)
    best["val_acc"] = float(best_val_acc)
    val_metrics = compute_metrics(y_true, y_pred, n_classes=len(names))
    save_predictions(run_dir / "val_predictions.csv", y_true, y_pred)

    h = np.asarray([[r["train_loss"], r["train_acc"], r["val_loss"], r["val_acc"], r["lr"]] for r in history], dtype=np.float64)
    tail = h[-min(20, len(h)):]
    stability = {
        "epochs_run": int(len(history)),
        "last20_train_loss_mean": float(tail[:, 0].mean()),
        "last20_train_acc_mean": float(tail[:, 1].mean()),
        "last20_val_loss_mean": float(tail[:, 2].mean()),
        "last20_val_acc_mean": float(tail[:, 3].mean()),
        "last20_val_acc_std": float(tail[:, 3].std()),
        "gap_train_minus_val_acc": float(tail[:, 1].mean() - tail[:, 3].mean()),
        "final_lr": float(h[-1, 4]),
    }

    result = {
        "experiment_id": experiment_id,
        "protocol": protocol_name,
        "task": task,
        "condition": condition,
        "model": model_name,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_loss": float(best_val_loss),
        "val_metrics": val_metrics,
        "stability": stability,
        "lr_scheduler": scheduler_record,
        "dataset_split_hash": meta.get("split_hash"),
        "dataset_hash": meta.get("dataset_hash"),
        "cache_fingerprint": meta["cache_fingerprint"],
    }
    if evaluate_test:
        sdir = sealed_test_dir(protocol_cfg, task, condition)
        X_test, y_test = load_arrays(sdir, "test")
        test_loader = make_loader(X_test, y_test, cfg.train.batch_size, 0, False)
        _, test_acc, y_test_true, y_test_pred = evaluate(model, test_loader, device, criterion, predictions=True)
        test_metrics = compute_metrics(y_test_true, y_test_pred, n_classes=len(names))
        save_predictions(run_dir / "test_predictions.csv", y_test_true, y_test_pred)
        result["test_metrics"] = test_metrics
        result["test_evaluated_once"] = True

    save_json(result, str(run_dir / "metrics.json"))
    save_json(history, str(run_dir / "history.json"))
    manifest = {
        "experiment_id": experiment_id,
        "protocol": protocol_name,
        "task": task,
        "condition": condition,
        "model": model_name,
        "seed": seed,
        "dataset_meta": str(meta_path),
        "split_hash": meta.get("split_hash"),
        "dataset_hash": meta.get("dataset_hash"),
        "cache_fingerprint": meta["cache_fingerprint"],
        "metrics_path": str(run_dir / "metrics.json"),
        "history_path": str(run_dir / "history.json"),
        "lr_scheduler": scheduler_record,
        "test_access": "sealed_once" if evaluate_test else "sealed",
    }
    save_json(manifest, str(run_dir / "manifest.json"))
    print(f"saved {run_dir}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--output-root", default=None, help="把结果写到指定目录（默认 artifacts/<协议名>/）")
    p.add_argument("--save-model", action="store_true", help="保存最佳权重 best_state.pt（供 t-SNE 分析）")
    return p.parse_args()


def main():
    args = parse_args()
    experiment_cfg = load_yaml(Path(args.experiment))
    protocol_cfg = load_yaml(Path(args.protocol))
    train_one(experiment_cfg, protocol_cfg, args.evaluate_test, args.confirm_test, args.epochs,
              args.limit_train, output_root=args.output_root, save_model=args.save_model)


if __name__ == "__main__":
    main()
