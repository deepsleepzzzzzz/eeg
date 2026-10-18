"""Verify that z-score per window restores symbol-state diversity (SEU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import class_file, class_names, task_dir, load_seu_csv, vibration_xyz
from representation.recurrence_plot import preprocess_signal
from representation.state_transition import fit_train_edges, symbolize_with_edges

TASK = sys.argv[1] if len(sys.argv) > 1 else "bearing"
COND = sys.argv[2] if len(sys.argv) > 2 else "20_0"
WIN = 800
BINS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

cfg = OmegaConf.load(ROOT / "config" / f"seu_{TASK}_cnn.yaml")
OmegaConf.set_struct(cfg, False)
cfg.method.normalize = "zscore"
cfg.method.symbol_bins = BINS
cfg.method.bin_edges_mode = sys.argv[4] if len(sys.argv) > 4 else "quantile"

root = Path("datas/seu")
folder = task_dir(root, TASK)
names = class_names(TASK)

print(f"=== {TASK} {COND}, bins={BINS}, normalize=zscore ===")
for name in names:
    raw = load_seu_csv(folder / class_file(TASK, name, COND))
    xyz = vibration_xyz(raw)
    n = xyz.shape[1]
    nwin = n // WIN
    wins = xyz[:, : nwin * WIN].reshape(3, nwin, WIN)
    idx = np.linspace(0, nwin - 1, min(400, nwin)).astype(int)
    epochs = [wins[:, i] for i in idx]
    edges = fit_train_edges(epochs, range(len(epochs)), cfg.method)
    hist = np.zeros(BINS)
    for ep in epochs:
        arr = preprocess_signal(ep, cfg.method)
        for c in range(3):
            st = symbolize_with_edges(arr[c], edges[c])
            hist += np.bincount(st, minlength=BINS)
    hist = hist / hist.sum()
    top = np.sort(hist)[::-1][:3]
    nz = int((hist > 0.02).sum())
    print(f"  {name:8s} 有效状态数(>2%)={nz}  top3占比={[round(float(v),2) for v in top]}")

print("\n判读: 有效状态数应接近 bins（不再是 1），说明状态空间被真正用起来了。")
