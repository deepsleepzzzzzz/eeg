from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1
    cm = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred, n_classes: int | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    recall = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=np.float64),
        where=cm.sum(axis=1) > 0,
    )
    return float(np.mean(recall)) if recall.size else 0.0


def macro_f1(y_true, y_pred, n_classes: int | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return float(np.mean(f1)) if f1.size else 0.0


def compute_metrics(y_true, y_pred, n_classes: int | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    return {
        "n": int(y_true.size),
        "correct": int(np.sum(y_true == y_pred)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes=cm.shape[0]),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes=cm.shape[0]),
        "confusion_matrix": cm.astype(int).tolist(),
    }
