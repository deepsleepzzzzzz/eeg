"""Validation figures for SEU runs: confusion matrix and t-SNE. Does not open test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

from utils.visualizer import plot_confusion_matrix, plot_tsne


def paper_arch(method: str) -> str:
    return "gcn" if "_gcn" in method else "cnn"


def paper_title(task: str, condition: str, method: str) -> str:
    pretty = {"bearing": "Bearing", "gear": "Gear"}.get(task, task)
    label = {"transition_none_cnn": "CNN", "transition_gcn": "GCN"}.get(method, method.replace("_", " ").upper())
    return f"{pretty} {str(condition).replace('_', '-')} {label}"


def paper_stem(task: str, condition: str, method: str, seed: int = 42) -> str:
    tag = {"transition_none_cnn": "cnn", "transition_gcn": "gcn"}.get(method, method)
    suffix = "" if int(seed) == 42 else f"_seed{int(seed)}"
    return f"{task}_{tag}_{condition}{suffix}"


def collect_embeddings(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    vecs = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if not hasattr(model, "embed"):
                raise RuntimeError("model has no embed()")
            vecs.append(model.embed(xb).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(vecs, axis=0), np.concatenate(labels, axis=0)


def fit_tsne(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    n = int(embeddings.shape[0])
    perplexity = min(30.0, max(5.0, (n - 1) / 4.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    )
    return reducer.fit_transform(embeddings)


def save_val_figures(
    model,
    val_loader,
    device,
    class_names: list[str] | tuple[str, ...],
    confusion,
    out_dir: Path,
    title_prefix: str,
    seed: int = 42,
) -> dict:
    out_dir = Path(out_dir)
    cm_path = out_dir / "val_confusion.png"
    tsne_path = out_dir / "val_tsne.png"
    plot_confusion_matrix(
        confusion,
        class_names,
        str(cm_path),
        title=f"{title_prefix} val confusion",
    )
    emb, y = collect_embeddings(model, val_loader, device)
    xy = fit_tsne(emb, seed=seed)
    plot_tsne(xy, y, class_names, str(tsne_path), title=f"{title_prefix} val t-SNE")
    return {"val_confusion": str(cm_path), "val_tsne": str(tsne_path)}
