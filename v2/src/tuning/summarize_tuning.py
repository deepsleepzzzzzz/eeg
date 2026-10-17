"""Summarize a tuning registry (validation metrics only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True)
    p.add_argument("--metric", default="macro_f1")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    rows = list(csv.DictReader(Path(args.registry).open(encoding="utf-8")))
    if not rows:
        print("empty registry")
        return
    key = args.metric
    rows.sort(key=lambda r: float(r[key]), reverse=True)
    print(f"{len(rows)} trials | sorted by {key} desc")
    print("-" * 100)
    for r in rows[: args.top]:
        print(
            f"{float(r[key]):.4f}  acc={float(r['val_acc']):.4f}  ep={r['best_epoch']:>4}  "
            f"{r['model']:<3}  {r['params']}"
        )
    print("-" * 100)
    for model in sorted({r["model"] for r in rows}):
        sub = [float(r[key]) for r in rows if r["model"] == model]
        print(f"{model}: n={len(sub)} best={max(sub):.4f} mean={sum(sub)/len(sub):.4f}")


if __name__ == "__main__":
    main()
