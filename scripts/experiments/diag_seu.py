"""Diagnose SEU amplitude/state distributions per class to explain low val acc."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.seu_dds import class_file, class_names, task_dir, load_seu_csv, vibration_xyz
from representation.state_transition import symbolize_with_edges

TASK = sys.argv[1] if len(sys.argv) > 1 else "bearing"
COND = sys.argv[2] if len(sys.argv) > 2 else "20_0"
WIN = 800

cfg = OmegaConf.load(ROOT / "config" / f"seu_{TASK}_cnn.yaml")
root = Path("datas/seu")
folder = task_dir(root, TASK)
names = class_names(TASK)

# emulate fit_train_amplitude_edges on the train split of every class
edges = None
for name in names:
    raw = load_seu_csv(folder / class_file(TASK, name, COND))
    xyz = vibration_xyz(raw)
    n = xyz.shape[1]
    train_end = int(n * 0.70)
    # collect per-window stats from the train split
    nwin = train_end // WIN
    wins = xyz[:, : nwin * WIN].reshape(3, nwin, WIN)
    lo = wins.min(axis=(1, 2))
    hi = wins.max(axis=(1, 2))
    if edges is None:
        edges = np.stack([lo, hi], axis=1)
    else:
        edges = np.minimum(edges, np.stack([lo, hi], axis=1))
    print(f"{name:8s} 幅值 min={lo.round(3)} max={hi.round(3)}")

lo_hard = np.stack([np.minimum(edges[c, 0], edges[c, 0]) for c in range(3)])
print("\n全训练集统一幅值范围 (每通道):")
for c in range(3):
    print(f"  通道{c}: [{edges[c,0]:.3f}, {edges[c,1]:.3f}]  跨度={edges[c,1]-edges[c,0]:.3f}")

print("\n各状态(0..5)占比 —— 用统一边界编码后:")
bins = 6
lin = np.linspace(edges[:, 0], edges[:, 1], bins + 1, axis=1)  # [3, bins+1]
for name in names:
    raw = load_seu_csv(folder / class_file(TASK, name, COND))
    xyz = vibration_xyz(raw)
    nwin = (xyz.shape[1] // WIN)
    wins = xyz[:, : nwin * WIN].reshape(3, nwin, WIN)
    # sample 300 windows for speed
    idx = np.linspace(0, nwin - 1, min(300, nwin)).astype(int)
    hist = np.zeros(bins)
    for i in idx:
        for c in range(3):
            st = symbolize_with_edges(wins[c, i], lin[c])
            hist += np.bincount(st, minlength=bins)
    hist = hist / hist.sum()
    top = np.argsort(hist)[::-1][:3]
    print(f"  {name:8s} 占比 top3 状态: " + ", ".join(f"#{t}={hist[t]:.2f}" for t in sorted(top)))

print("\n判读: 若某类几乎全落在单一状态(占比>0.8)，说明统一等宽边界无法区分该类，自转移矩阵退化。")
