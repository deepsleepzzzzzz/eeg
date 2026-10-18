"""Noise robustness curve: SNR -4..+10 dB, TRAIN-NOISY / TEST-NOISY (comparable to 文献1).

Protocol matches 文献1 (Guan et al.): for each SNR level, additive white Gaussian
noise is added to the raw signal, quantile bin edges are re-fit on the NOISY train
split, and the CNN is trained & tested entirely on noisy data. This is the standard
"learn to handle noise" protocol (as opposed to the harder train-clean/test-noisy).

Per (task, condition): train [0,0.70) / val [0.70,0.90) / test [0.90,1.00).

Outputs: teacher_extra/noise_robustness.json + doc/figures/seu/noise_robustness_curve.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.seu_dds import class_file, class_names, cut_windows, load_seu_csv, normalize_condition, task_dir, vibration_xyz, window_starts
from experiments.train_runtime import evaluate, get_device, make_loader, resolve_path, set_seed
from models.cnn import build_model
from representation.recurrence_plot import build_representation
from representation.state_transition import fit_train_edges

OUT_DIR = ROOT / "datas/experiments_seu/teacher_extra"
FIG_DIR = ROOT / "doc" / "figures" / "seu"
SNRS = [-4, -2, 0, 2, 4, 6, 8, 10]


def add_noise(windows, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for w in windows:
        w = np.asarray(w, dtype=np.float64)
        p_sig = float(np.mean(w ** 2))
        p_noise = p_sig / (10 ** (snr_db / 10.0)) if p_sig > 0 else 0.0
        noise = np.sqrt(p_noise) * rng.standard_normal(w.shape)
        out.append((w + noise).astype(np.float32))
    return out


def encode(windows, method_cfg):
    maps = []
    for win in windows:
        arr, _ = build_representation(win, method_cfg)
        maps.append(arr.astype(np.float32))
    return np.stack(maps, axis=0)


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


def train_noisy(cfg, task, condition, snr_db, device):
    by, y, names = cut_raw(cfg, task, condition)
    # add noise to all splits
    noisy = {sn: add_noise(by[sn], snr_db, seed=0) for sn in by}
    edges = fit_train_edges(noisy["train"], range(len(noisy["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X = {sn: encode(noisy[sn], enc_cfg.method) for sn in noisy}
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
    return float(acc)


def run_one(cfg, task, condition, device):
    curve = []
    t0 = time.perf_counter()
    for snr in SNRS:
        acc = train_noisy(cfg, task, condition, snr, device)
        curve.append({"snr_db": snr, "acc": round(acc, 4)})
        print(f"  [{task} {condition}] SNR={snr:+d}dB acc={acc:.4f}", flush=True)
    return {
        "task": task, "condition": condition, "arch": "cnn",
        "protocol": "train-noisy / test-noisy (comparable to 文献1)",
        "curve": curve,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--condition", default=None)
    args = ap.parse_args()
    if args.task and args.condition:
        jobs = [(args.task, args.condition)]
    else:
        jobs = [("bearing", "20_0"), ("bearing", "30_2"), ("gear", "20_0"), ("gear", "30_2")]

    results = []
    for task, condition in jobs:
        cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_cnn.yaml"))
        device = get_device(cfg)
        results.append(run_one(cfg, task, condition, device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "noise_robustness.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"bearing_20_0": "tab:red", "bearing_30_2": "tab:orange",
              "gear_20_0": "tab:blue", "gear_30_2": "tab:green"}
    for r in results:
        key = f"{r['task']}_{r['condition']}"
        xs = [p["snr_db"] for p in r["curve"]]
        ys = [p["acc"] * 100 for p in r["curve"]]
        ax.plot(xs, ys, marker="o", label=f"{r['task']} {r['condition']}", color=colors.get(key))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Noise robustness (train-noisy / test-noisy)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_robustness_curve.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'noise_robustness.json'} + curve PNG")


if __name__ == "__main__":
    main()
