"""Ablation: symbol_bins sweep + z-score on/off (SEU DDS).

Answers two advisor questions:
  - Why 6 symbol states? What happens with 4/5/7/8/9/10?
  - Is z-score redundant given quantile (equal-frequency) symbolization?

Clean-training protocol (no noise): per (task, condition), cut windows 70/20/10,
fit train-global quantile edges, encode to stacked transition matrices, train CNN,
evaluate on the sealed test split. Reproduces the frozen baseline (bins=6, zscore).

Outputs: teacher_extra/ablation_bins_zscore.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.seu_dds import class_file, class_names, cut_windows, load_seu_csv, normalize_condition, task_dir, vibration_xyz, window_starts
from experiments.train_runtime import evaluate, get_device, make_loader, resolve_path, set_seed
from models.cnn import build_model
from representation.recurrence_plot import build_representation
from representation.state_transition import fit_train_edges

OUT_DIR = ROOT / "datas/experiments_seu/teacher_extra"

BINS_LIST = [4, 5, 6, 7, 8, 9, 10]
NORM_LIST = ["zscore", "none"]
STRATEGY_LIST = ["quantile", "uniform"]  # 等频 vs 等宽


def cut_raw(cfg, task, condition):
    names = class_names(task)
    folder = task_dir(resolve_path(cfg.data.raw_root), task)
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    tr_ratio = float(cfg.data.train_ratio)
    va_ratio = float(cfg.data.val_ratio)
    by = {"train": [], "val": [], "test": []}
    ys = {"train": [], "val": [], "test": []}
    for label, name in enumerate(names):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw)
        n = int(xyz.shape[1])
        ranges = {
            "train": (0, int(n * tr_ratio)),
            "val": (int(n * tr_ratio), int(n * (tr_ratio + va_ratio))),
            "test": (int(n * (tr_ratio + va_ratio)), n),
        }
        for sn, (lo, hi) in ranges.items():
            starts = window_starts(lo, hi, length, stride)
            wins = cut_windows(xyz, starts, length)
            by[sn].extend(wins)
            ys[sn].extend([label] * len(wins))
    return by, {sn: np.asarray(ys[sn], dtype=np.int64) for sn in ys}, names


def encode(windows, method_cfg):
    maps = []
    for win in windows:
        arr, _ = build_representation(win, method_cfg)
        maps.append(arr.astype(np.float32))
    return np.stack(maps, axis=0)


def train_clean(cfg, task, condition, device):
    """Clean training; return (test_acc, n_test, input_shape)."""
    by, y, names = cut_raw(cfg, task, condition)
    edges = fit_train_edges(by["train"], range(len(by["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X = {sn: encode(by[sn], enc_cfg.method) for sn in by}
    num_classes = len(names)
    set_seed(42, deterministic=True)
    bs = int(cfg.train.batch_size)
    tl = make_loader(X["train"], y["train"], bs, 0, True)
    vl = make_loader(X["val"], y["val"], bs, 0, False)
    model = build_model(num_classes, cfg, in_channels=int(X["train"].shape[1]),
                        image_size=int(X["train"].shape[-1])).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr),
                           weight_decay=float(cfg.train.weight_decay))
    best = -1.0
    best_state = None
    bad = 0
    patience = int(cfg.train.get("early_stop_patience", 30))
    for _ in range(1, int(cfg.train.epochs) + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        _, va, _, _ = evaluate(model, vl, device, crit, predictions=True)
        if va > best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    te = make_loader(X["test"], y["test"], bs, 0, False)
    _, acc, _, _ = evaluate(model, te, device, crit, predictions=True)
    return float(acc), int(len(y["test"])), list(X["train"].shape[1:])


def apply_override(cfg, key, value):
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(c, False)
    if key == "symbol_bins":
        c.method.symbol_bins = int(value)
    elif key == "normalize":
        c.method.normalize = str(value)
    elif key == "bin_edges_mode":
        c.method.bin_edges_mode = str(value)
    else:
        raise ValueError(key)
    return c


def run_bins(cfg, task, condition, device):
    out = []
    for b in BINS_LIST:
        t0 = time.perf_counter()
        c = apply_override(cfg, "symbol_bins", b)
        acc, n_test, shape = train_clean(c, task, condition, device)
        out.append({"bins": b, "acc": round(acc, 4), "n_test": n_test,
                    "input_shape": shape, "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"[bins] {task} {condition} bins={b} acc={acc:.4f}", flush=True)
    return out


def run_norm(cfg, task, condition, device):
    out = []
    for norm in NORM_LIST:
        t0 = time.perf_counter()
        c = apply_override(cfg, "normalize", norm)
        acc, n_test, shape = train_clean(c, task, condition, device)
        out.append({"normalize": norm, "acc": round(acc, 4), "n_test": n_test,
                    "input_shape": shape, "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"[norm] {task} {condition} normalize={norm} acc={acc:.4f}", flush=True)
    return out


def run_strategy(cfg, task, condition, device):
    """等频(quantile) vs 等宽(uniform) 符号化边界，在 bins=6 下对比。"""
    out = []
    for mode in STRATEGY_LIST:
        t0 = time.perf_counter()
        c = apply_override(cfg, "bin_edges_mode", mode)
        c.method.symbol_bins = 6
        acc, n_test, shape = train_clean(c, task, condition, device)
        out.append({"bin_edges_mode": mode, "acc": round(acc, 4), "n_test": n_test,
                    "input_shape": shape, "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"[strategy] {task} {condition} mode={mode} acc={acc:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--what", default="bins", choices=["bins", "norm", "strategy", "both"])
    args = ap.parse_args()

    if args.task and args.condition:
        jobs = [(args.task, args.condition)]
    else:
        jobs = [("bearing", "20_0"), ("bearing", "30_2"),
                ("gear", "20_0"), ("gear", "30_2")]

    bins_results, norm_results, strategy_results = [], [], []
    for task, condition in jobs:
        cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_cnn.yaml"))
        device = get_device(cfg)
        print(f"device={device}", flush=True)
        if args.what in ("bins", "both"):
            bins_results.append({"task": task, "condition": condition,
                                 "curve": run_bins(cfg, task, condition, device)})
        if args.what in ("norm", "both"):
            norm_results.append({"task": task, "condition": condition,
                                 "curve": run_norm(cfg, task, condition, device)})
        if args.what in ("strategy", "both"):
            strategy_results.append({"task": task, "condition": condition,
                                     "curve": run_strategy(cfg, task, condition, device)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation_bins_zscore.json"
    payload = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if bins_results:
        payload["bins"] = bins_results
    if norm_results:
        payload["normalize"] = norm_results
    if strategy_results:
        payload["strategy"] = strategy_results
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote ablation_bins_zscore.json")


if __name__ == "__main__":
    main()
