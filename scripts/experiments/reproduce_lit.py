"""Reproduce our method under the protocols of two SEU-comparable papers.

文献1 — Guan et al., "Improved Transformer", ACM AAIA 2023:
  gear 5-class, both working conditions (20-0 + 30-2) MERGED into one pool,
  80% train / 20% test (stratified), additive Gaussian noise at SNR=10 dB,
  reports 99.4%.

文献5 — Zhang et al., "MSRC+CBAM", IIETA / Traitement du Signal 2026:
  bearing & gearbox, per-condition (G1=20-0, G2=30-2) 4 tasks,
  train 60% / test 40% (time-contiguous), clean signal,
  reports G1 Bearing 99.4 / G1 Gearbox 99.8 / G2 Bearing 99.6 / G2 Gearbox 99.4.

We run OUR method (transition-matrix + CNN) under each protocol.
Output -> datas/experiments_seu/teacher_extra/reproduce_lit.json
"""
from __future__ import annotations

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


def add_noise_windows(windows, snr_db, seed=0):
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


def train_cnn(cfg, X_tr, y_tr, X_va, y_va, num_classes, device):
    set_seed(42, deterministic=True)
    bs = int(cfg.train.batch_size)
    tl = make_loader(X_tr, y_tr, bs, 0, True)
    vl = make_loader(X_va, y_va, bs, 0, False)
    model = build_model(num_classes, cfg, in_channels=int(X_tr.shape[1]),
                        image_size=int(X_tr.shape[-1])).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr),
                           weight_decay=float(cfg.train.weight_decay))
    best = -1.0
    best_state = None
    bad = 0
    patience = int(cfg.train.get("early_stop_patience", 30))
    t0 = time.perf_counter()
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
    return model, float(best), time.perf_counter() - t0


def load_class_windows(task, class_name, condition, cfg):
    folder = task_dir(resolve_path(cfg.data.raw_root), task)
    raw = load_seu_csv(folder / class_file(task, class_name, condition))
    xyz = vibration_xyz(raw)
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    starts = window_starts(0, int(xyz.shape[1]), length, stride)
    return cut_windows(xyz, starts, length)


def reproduce_lit1(cfg):
    """Gear 5-class, both conditions merged, 80/20, SNR=10 dB."""
    task = "gear"
    names = class_names(task)
    device = get_device(cfg)
    all_w = []
    all_y = []
    for label, name in enumerate(names):
        wins = load_class_windows(task, name, "20_0", cfg) + load_class_windows(task, name, "30_2", cfg)
        all_w.extend(wins)
        all_y.extend([label] * len(wins))
    # add SNR=10 dB noise to every window (matches 文献1 "noise added to increase difficulty")
    all_w = add_noise_windows(all_w, snr_db=10, seed=0)
    all_w = np.array(all_w, dtype=object)
    all_y = np.asarray(all_y, dtype=np.int64)
    # stratified 80/20 split with fixed seed
    rng = np.random.default_rng(42)
    tr_idx, te_idx = [], []
    for c in range(len(names)):
        ci = np.where(all_y == c)[0]
        rng.shuffle(ci)
        split = int(len(ci) * 0.8)
        tr_idx.extend(ci[:split])
        te_idx.extend(ci[split:])
    tr_idx = np.asarray(tr_idx); te_idx = np.asarray(te_idx)
    tr_w = [all_w[i] for i in tr_idx]
    te_w = [all_w[i] for i in te_idx]
    # fit edges on train, encode both
    edges = fit_train_edges(tr_w, range(len(tr_w)), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X_tr = encode(tr_w, enc_cfg.method)
    X_te = encode(te_w, enc_cfg.method)
    y_tr = all_y[tr_idx]
    y_te = all_y[te_idx]
    # small val carve-out from train for early stopping (10% of train)
    rng2 = np.random.default_rng(0)
    perm = rng2.permutation(len(y_tr))
    v_cut = int(len(y_tr) * 0.1)
    va_idx, tr_use = perm[:v_cut], perm[v_cut:]
    model, best_val, ttime = train_cnn(cfg, X_tr[tr_use], y_tr[tr_use], X_tr[va_idx], y_tr[va_idx], len(names), device)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(X_te, y_te, int(cfg.train.batch_size), 0, False)
    _, acc, _, _ = evaluate(model, loader, device, crit, predictions=True)
    return {
        "lit": 1, "protocol": "gear 5-class merged conditions, 80/20, SNR=10 dB",
        "n_train": int(len(tr_use)), "n_test": int(len(y_te)),
        "best_val_acc": round(best_val, 4), "test_acc": round(float(acc), 4),
        "train_time_s": round(ttime, 2),
    }


def reproduce_lit5(cfg, task, condition):
    """Per-condition task, 60% train / 40% test, clean signal."""
    names = class_names(task)
    device = get_device(cfg)
    folder = task_dir(resolve_path(cfg.data.raw_root), task)
    length = int(cfg.data.epoch_samples)
    stride = int(cfg.data.get("window_stride", length))
    by = {"train": [], "test": []}
    ys = {"train": [], "test": []}
    for label, name in enumerate(names):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw)
        n = int(xyz.shape[1])
        ranges = {"train": (0, int(n * 0.6)), "test": (int(n * 0.6), n)}
        for sn, (lo, hi) in ranges.items():
            starts = window_starts(lo, hi, length, stride)
            wins = cut_windows(xyz, starts, length)
            by[sn].extend(wins)
            ys[sn].extend([label] * len(wins))
    edges = fit_train_edges(by["train"], range(len(by["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})
    X_tr = encode(by["train"], enc_cfg.method)
    X_te = encode(by["test"], enc_cfg.method)
    y_tr = np.asarray(ys["train"], dtype=np.int64)
    y_te = np.asarray(ys["test"], dtype=np.int64)
    # carve a stratified val (10% of train) for early stopping
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_tr))
    v_cut = int(len(y_tr) * 0.1)
    va_idx = perm[:v_cut]
    tr_idx = perm[v_cut:]
    model, best_val, ttime = train_cnn(cfg, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx], y_tr[va_idx], len(names), device)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(X_te, y_te, int(cfg.train.batch_size), 0, False)
    _, acc, _, _ = evaluate(model, loader, device, crit, predictions=True)
    return {
        "lit": 5, "protocol": "per-condition 60% train / 40% test, clean",
        "task": task, "condition": condition,
        "n_train": int(len(y_tr) - v_cut), "n_test": int(len(y_te)),
        "best_val_acc": round(best_val, 4), "test_acc": round(float(acc), 4),
        "train_time_s": round(ttime, 2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    cfg_gear = OmegaConf.load(resolve_path("config/seu_gear_cnn.yaml"))
    r1 = reproduce_lit1(cfg_gear)
    results.append(r1)
    print(f"[lit1] test_acc={r1['test_acc']} (target 99.4% @ SNR10)", flush=True)

    for task, condition in [("bearing", "20_0"), ("bearing", "30_2"), ("gear", "20_0"), ("gear", "30_2")]:
        cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_cnn.yaml"))
        r = reproduce_lit5(cfg, task, condition)
        results.append(r)
        print(f"[lit5 {task} {condition}] test_acc={r['test_acc']}", flush=True)

    (OUT_DIR / "reproduce_lit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'reproduce_lit.json'}")


if __name__ == "__main__":
    main()
