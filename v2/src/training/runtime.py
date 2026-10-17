"""Shared training helpers for SEU classifiers."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def method_name(cfg) -> str:
    model_cfg = cfg.get("model") if cfg.get("model") is not None else {}
    arch = str(model_cfg.get("arch", "cnn")).lower()
    representation = str(cfg.method.get("representation", "transition")).lower()
    if arch == "gcn":
        if bool(model_cfg.get("gcn_attention", False)):
            return f"{representation}_gcn_attn"
        return f"{representation}_gcn"
    attention = str(cfg.method.get("attention", "none")).lower()
    return f"{representation}_{attention}_cnn"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cfg) -> torch.device:
    requested = str(cfg.train.get("device", "cuda"))
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if bool(cfg.train.get("require_gpu", False)):
            raise RuntimeError("CUDA is required but unavailable in this Python environment")
    return torch.device("cpu")


def load_array_pair(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"missing {split} arrays: {x_path}, {y_path}")
    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    return x, y


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
    )


def evaluate(model, data_loader, device, criterion=None, predictions: bool = False):
    """评估。指标在 GPU 上累加，整个 split 只做一次 GPU->CPU 同步。

    原实现每个 batch 调 2~3 次 .item()，每次都会强制同步，GPU 反复空转等 CPU。
    这里改为张量累加、循环结束统一 .item()，数值与原来一致（同样顺序的 float64 求和）。
    """
    model.eval()
    total = 0
    correct_t = torch.zeros((), dtype=torch.long, device=device)
    loss_t = torch.zeros((), dtype=torch.float64, device=device)
    true_all: list[torch.Tensor] = []
    pred_all: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct_t += (pred == yb).sum()
            total += int(yb.numel())
            if criterion is not None:
                loss_t += criterion(logits, yb).detach().double() * int(yb.numel())
            if predictions:
                true_all.append(yb)
                pred_all.append(pred)
    correct = int(correct_t.item())
    loss = float(loss_t.item()) / total if total else 0.0
    acc = float(correct) / total if total else 0.0
    if predictions:
        y_true = torch.cat(true_all).cpu().numpy().astype(np.int64)
        y_pred = torch.cat(pred_all).cpu().numpy().astype(np.int64)
        return loss, acc, y_true, y_pred
    return loss, acc


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack([np.arange(len(y_true)), y_true, y_pred]),
        delimiter=",",
        header="sample_index,true_label,pred_label",
        comments="",
        fmt="%d",
    )


def parse_seeds(cfg, args) -> list[int]:
    if getattr(args, "seeds", None):
        return [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    seeds = cfg.train.get("seeds", None)
    if seeds is not None:
        return [int(s) for s in seeds]
    return [int(cfg.train.get("seed", 42))]
