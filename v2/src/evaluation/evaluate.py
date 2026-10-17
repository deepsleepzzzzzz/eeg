from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import compute_metrics


def load_predictions(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    y_true = np.asarray([int(r["true_label"]) for r in rows], dtype=np.int64)
    y_pred = np.asarray([int(r["pred_label"]) for r in rows], dtype=np.int64)
    return y_true, y_pred


def evaluate_predictions(path: Path, n_classes: int = 5) -> dict:
    y_true, y_pred = load_predictions(path)
    return compute_metrics(y_true, y_pred, n_classes=n_classes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--n-classes", type=int, default=5)
    args = p.parse_args()
    result = evaluate_predictions(Path(args.predictions), n_classes=args.n_classes)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
