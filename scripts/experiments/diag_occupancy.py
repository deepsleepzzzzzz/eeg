"""Compare class separability with and without the occupancy channel (fast, no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import class_file, class_names, task_dir, load_seu_csv, vibration_xyz
from representation.recurrence_plot import build_representation
from representation.state_transition import fit_train_edges

TASK = sys.argv[1] if len(sys.argv) > 1 else "gear"
COND = sys.argv[2] if len(sys.argv) > 2 else "30_2"
WIN = 800
STRIDE = 800

base = OmegaConf.load(ROOT / "config" / f"seu_{TASK}_cnn.yaml")
OmegaConf.set_struct(base, False)
base.method.normalize = "zscore"
base.method.bin_edges_mode = "quantile"
base.method.symbol_bins = 6

root = Path("datas/seu")
folder = task_dir(root, TASK)
names = class_names(TASK)

# build raw windows for all classes (train + val splits together, split by time)
raw_train, y_train, raw_val, y_val = [], [], [], []
for label, name in enumerate(names):
    xyz = vibration_xyz(load_seu_csv(folder / class_file(TASK, name, COND)))
    n = xyz.shape[1]
    tr_end = int(n * 0.70)
    va_end = tr_end + int(n * 0.15)
    for lo, hi, xs, ys in ((0, tr_end, raw_train, y_train), (tr_end, va_end, raw_val, y_val)):
        nw = (hi - lo) // STRIDE
        for i in range(nw):
            xs.append(xyz[:, lo + i * STRIDE: lo + i * STRIDE + WIN])
            ys.append(label)

print(f"{TASK} {COND}: train={len(raw_train)} val={len(raw_val)}")

for stats_mode in ("none", "occupancy"):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.method.state_stats = stats_mode
    edges = fit_train_edges(raw_train, range(len(raw_train)), cfg.method)
    enc = OmegaConf.merge(cfg, {"method": {"bin_edges": edges.tolist()}})

    Xtr = np.stack([build_representation(w, enc.method)[0] for w in raw_train])
    Xva = np.stack([build_representation(w, enc.method)[0] for w in raw_val])
    ytr, yva = np.asarray(y_train), np.asarray(y_val)

    cent = np.stack([Xtr[ytr == c].reshape((ytr == c).sum(), -1).mean(axis=0)
                     for c in range(len(names))])
    d_val = ((Xva.reshape(len(Xva), -1)[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    pred = d_val.argmin(axis=1)
    acc = float((pred == yva).mean())

    # mean relative class distance
    rels = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = cent[i], cent[j]
            rels.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)))
    print(f"  state_stats={stats_mode:10s} 形状={Xtr.shape[1:]}  类间相对距离 mean={np.mean(rels):.4f} "
          f"min={np.min(rels):.4f}   最近中心 val_acc={acc:.4f}")

print("\n判读: occupancy 让类间相对距离变大、最近中心 val_acc 变高，说明幅值信息被找回。")
