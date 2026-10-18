"""Interpretability: per-class mean transition matrices + inter-class difference heatmaps.

Loads the *exact* representation tensors the model sees (X_train.npy, y_train.npy),
then computes, per class:
  - mean step-1 transition matrix (channel-averaged) [6,6]
  - mean per-step matrix [7,6,6] (channel-averaged)
and inter-class difference heatmaps (fault class - health class) at step 1.

The channel/step ordering of the [21,6,6] tensor is: 3 channels x 7 steps stacked
as [ch0_s1..ch0_s7, ch1_s1..ch1_s7, ch2_s1..ch2_s7] (see representation/state_transition.py).

Outputs PNG heatmaps -> doc/figures/seu/interpret/ and JSON -> teacher_extra/interpretability.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data.seu_dds import class_names

OUT_DIR = ROOT / "datas/experiments_seu/teacher_extra"
FIG_DIR = ROOT / "doc" / "figures" / "seu" / "interpret"
REP_ROOT = ROOT / "datas/experiments_seu/representations"
BINS = 6
STEPS = 7
CHANNELS = 3


def load_class_means(task: str, condition: str):
    data_dir = REP_ROOT / task / condition
    X = np.load(data_dir / "X_train.npy").astype(np.float32)  # [B,21,6,6]
    y = np.load(data_dir / "y_train.npy").astype(np.int64)
    names = class_names(task)
    n_classes = len(names)
    # reshape [B, 3, 7, 6, 6]
    B = X.shape[0]
    Xr = X.reshape(B, CHANNELS, STEPS, BINS, BINS)
    # channel-averaged [B, 7, 6, 6]
    Xc = Xr.mean(axis=1)
    class_means = {}
    for c, name in enumerate(names):
        mask = y == c
        class_means[name] = Xc[mask].mean(axis=0)  # [7,6,6]
    return names, class_means


def plot_class_matrices(task, condition, names, class_means):
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        m = class_means[name][0]  # step-1
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=max(1e-6, m.max()))
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(BINS)); ax.set_yticks(range(BINS))
        ax.set_xticklabels(range(1, BINS + 1)); ax.set_yticklabels(range(1, BINS + 1))
        ax.set_xlabel("to state"); ax.set_ylabel("from state")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle(f"{task} {condition} — mean step-1 transition matrix per class", fontsize=12)
    fig.tight_layout()
    return fig


def plot_diff_heatmaps(task, condition, names, class_means):
    health = names[0]
    faults = names[1:]
    fig, axes = plt.subplots(1, len(faults), figsize=(3.2 * len(faults), 3.0))
    if len(faults) == 1:
        axes = [axes]
    vmax = 0.0
    for ax, name in zip(axes, faults):
        d = class_means[name][0] - class_means[health][0]
        vmax = max(vmax, float(np.abs(d).max()))
    vmax = max(vmax, 1e-6)
    for ax, name in zip(axes, faults):
        d = class_means[name][0] - class_means[health][0]
        im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name} − {health}", fontsize=10)
        ax.set_xticks(range(BINS)); ax.set_yticks(range(BINS))
        ax.set_xticklabels(range(1, BINS + 1)); ax.set_yticklabels(range(1, BINS + 1))
        ax.set_xlabel("to state"); ax.set_ylabel("from state")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle(f"{task} {condition} — step-1 transition matrix difference (fault − health)", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    for task in ("bearing", "gear"):
        for condition in ("20_0", "30_2"):
            names, class_means = load_class_means(task, condition)
            # figures
            fig1 = plot_class_matrices(task, condition, names, class_means)
            fig1.savefig(FIG_DIR / f"{task}_{condition}_class_matrices.png", dpi=130)
            plt.close(fig1)
            fig2 = plot_diff_heatmaps(task, condition, names, class_means)
            fig2.savefig(FIG_DIR / f"{task}_{condition}_diff_heatmaps.png", dpi=130)
            plt.close(fig2)
            # JSON (step-1 matrices + all-step, rounded)
            payload[f"{task}_{condition}"] = {
                "class_names": names,
                "step1_matrices": {n: np.round(class_means[n][0], 5).tolist() for n in names},
                "all_steps": {n: np.round(class_means[n], 5).tolist() for n in names},
            }
            print(f"done {task} {condition}", flush=True)
    (OUT_DIR / "interpretability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'interpretability.json'} + PNGs in {FIG_DIR}")


if __name__ == "__main__":
    main()
