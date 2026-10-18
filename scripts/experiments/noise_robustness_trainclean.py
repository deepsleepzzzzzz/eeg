"""Noise robustness, harder protocol: TRAIN-CLEAN / TEST-NOISY (zero-shot).

与 noise_robustness.py（train-noisy / test-noisy）成对：
  - 本脚本：训练阶段只用干净信号（不注入噪声），bin 边界在干净训练集上拟合，
    仅在测试阶段对测试集按各 SNR 加噪。用于回答「训练时没见过噪声，方法还能不能扛」。
  - noise_robustness.py：训练与测试都加噪（标准「学处理噪声」协议，可比文献1）。

协议说明：这是「零样本抗噪」的硬考验，掉点是预期内现象，用于说明
「噪声增广训练」是方法获得抗噪能力的必要环节（两脚本对比即证据）。

Per (task, condition): train [0,0.70) / val [0.70,0.90) / test [0.90,1.00).

Outputs: teacher_extra/noise_robustness_trainclean.json
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


def train_clean_then_eval_noisy(cfg, task, condition, snr_db, device):
    """训练用干净数据；bin 边界拟合于干净训练集；仅在测试集加噪评估。"""
    by, y, names = cut_raw(cfg, task, condition)
    edges = fit_train_edges(by["train"], range(len(by["train"])), cfg.method)
    enc_cfg = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})

    # 干净：train/val 不加噪，直接编码
    Xtr = encode(by["train"], enc_cfg.method)
    Xva = encode(by["val"], enc_cfg.method)

    num_classes = len(names)
    set_seed(42, deterministic=True)
    bs = int(cfg.train.batch_size)
    tl = make_loader(Xtr, y["train"], bs, 0, True)
    vl = make_loader(Xva, y["val"], bs, 0, False)
    model = build_model(num_classes, cfg, in_channels=int(Xtr.shape[1]),
                        image_size=int(Xtr.shape[-1])).to(device)
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

    # 测试集：干净 acc + 当前 SNR 加噪 acc
    Xte_clean = encode(by["test"], enc_cfg.method)
    te_clean = make_loader(Xte_clean, y["test"], bs, 0, False)
    _, clean_acc, _, _ = evaluate(model, te_clean, device, crit, predictions=True)

    noisy_test = add_noise(by["test"], snr_db, seed=0)
    Xte_noisy = encode(noisy_test, enc_cfg.method)
    te_noisy = make_loader(Xte_noisy, y["test"], bs, 0, False)
    _, noisy_acc, _, _ = evaluate(model, te_noisy, device, crit, predictions=True)
    return float(clean_acc), float(noisy_acc)


def run_one(cfg, task, condition, device):
    curve = []
    clean_test_acc = None
    best_val_acc = None
    t0 = time.perf_counter()
    for snr in SNRS:
        ca, na = train_clean_then_eval_noisy(cfg, task, condition, snr, device)
        if clean_test_acc is None:
            clean_test_acc = round(ca, 4)
        curve.append({"snr_db": snr, "acc": round(na, 4)})
        print(f"  [{task} {condition}] SNR={snr:+d}dB clean_test={ca:.4f} noisy_test={na:.4f}", flush=True)
    return {
        "task": task, "condition": condition, "arch": "cnn",
        "protocol": "train-clean / test-noisy (zero-shot, no noise augmentation)",
        "clean_test_acc": clean_test_acc,
        "n_test": 650,
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
    (OUT_DIR / "noise_robustness_trainclean.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'noise_robustness_trainclean.json'}")


if __name__ == "__main__":
    main()
