import numpy as np

from evaluation.metrics import compute_metrics


def test_metrics_manual():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, n_classes=2)

    assert m["n"] == 4
    assert m["correct"] == 3
    assert np.isclose(m["accuracy"], 0.75)
    assert np.isclose(m["balanced_accuracy"], 0.75)
    assert np.isclose(m["macro_f1"], 0.7333333333333333)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


def test_metrics_perfect():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    m = compute_metrics(y, y, n_classes=5)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
