"""训练曲线与混淆矩阵。"""
from __future__ import annotations

import os
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_train_curves(
    train_loss: List[float],
    val_acc: List[float],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(train_loss, color="C0", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(val_acc, color="C1", label="val_acc")
    ax2.set_ylabel("acc", color="C1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str,
    title: str = "Confusion matrix",
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mat = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(list(class_names), rotation=30, ha="right")
    ax.set_yticklabels(list(class_names))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = float(mat.max()) / 2.0 if mat.size else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                str(int(mat[i, j])),
                ha="center",
                va="center",
                color="white" if mat[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_tsne(
    xy: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    out_path: str,
    title: str = "t-SNE",
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labs = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(class_names):
        m = labs == i
        if not np.any(m):
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=18, alpha=0.75, label=name)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
