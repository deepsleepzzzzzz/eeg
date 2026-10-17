import json
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_metrics
from _bootstrap import PROJECT_ROOT


def test_golden_saved_predictions_match_saved_metrics():
    base = PROJECT_ROOT / "artifacts" / "golden" / "main_70_20_10"
    runs = sorted(base.glob("*/*/*/seed_42"))
    assert runs, f"no golden runs found under {base}"

    checked = 0
    for run in runs:
        metrics_path = run / "metrics.json"
        pred_path = run / "test_predictions.csv"
        if not metrics_path.exists() or not pred_path.exists():
            continue

        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        arr = np.loadtxt(pred_path, delimiter=",", skiprows=1, dtype=np.int64)
        y_true = arr[:, 1]
        y_pred = arr[:, 2]

        m = compute_metrics(y_true, y_pred, n_classes=5)
        assert m["n"] == int(len(y_true))
        assert np.isclose(m["accuracy"], saved["test_acc"], atol=1e-12)
        assert np.isclose(m["balanced_accuracy"], saved["test_balanced_acc"], atol=1e-12)
        assert np.isclose(m["macro_f1"], saved["test_macro_f1"], atol=1e-12)
        assert m["confusion_matrix"] == saved["test_confusion_matrix"]
        checked += 1

    assert checked >= 8, f"expected at least 8 golden runs, checked {checked}"
