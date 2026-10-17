from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.protocol import artifact_root  # noqa: E402
from training.train import load_yaml, train_one  # noqa: E402

REGISTRY = PROJECT_ROOT / "registry.csv"
HEADER = [
    "experiment_id", "protocol", "comparison_group", "task", "condition", "model", "seed",
    "best_epoch", "best_val_acc", "plateau_acc", "plateau_std", "train_val_gap",
    "epochs_run", "final_lr", "metrics_path", "history_path", "prediction_path", "status",
]


def append_registry(row: dict) -> None:
    exists = REGISTRY.exists() and REGISTRY.stat().st_size > 0
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_list(text: str):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--tasks", default="bearing,gear")
    p.add_argument("--conditions", default="20_0,30_2")
    p.add_argument("--models", default="cnn,gcn")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    args = p.parse_args()

    protocol_cfg = load_yaml(Path(args.protocol))
    protocol_name = str(protocol_cfg["name"])
    for task in parse_list(args.tasks):
        for condition in parse_list(args.conditions):
            for model in parse_list(args.models):
                for seed in [int(s) for s in parse_list(args.seeds)]:
                    experiment_id = f"{task}_{model}_{condition}_seed{seed}"
                    experiment_cfg = {
                        "experiment_id": experiment_id,
                        "protocol": protocol_name,
                        "comparison_group": "main_clean_compare",
                        "task": task,
                        "condition": condition,
                        "model": model,
                        "seed": seed,
                    }
                    print(f"RUN {experiment_id}", flush=True)
                    result = train_one(
                        experiment_cfg,
                        protocol_cfg,
                        evaluate_test=False,
                        confirm_test=False,
                        epochs_override=args.epochs,
                        limit_train=args.limit_train,
                    )
                    st = result.get("stability", {}) or {}
                    run_dir = artifact_root(protocol_cfg) / protocol_name / experiment_id
                    append_registry({
                        "experiment_id": experiment_id,
                        "protocol": protocol_name,
                        "comparison_group": "main_clean_compare",
                        "task": task,
                        "condition": condition,
                        "model": model,
                        "seed": seed,
                        "best_epoch": result["best_epoch"],
                        "best_val_acc": round(float(result["best_val_acc"]), 6),
                        "plateau_acc": round(float(st.get("last20_val_acc_mean", 0.0)), 6),
                        "plateau_std": round(float(st.get("last20_val_acc_std", 0.0)), 6),
                        "train_val_gap": round(float(st.get("gap_train_minus_val_acc", 0.0)), 6),
                        "epochs_run": st.get("epochs_run"),
                        "final_lr": st.get("final_lr"),
                        "metrics_path": str(run_dir / "metrics.json"),
                        "history_path": str(run_dir / "history.json"),
                        "prediction_path": str(run_dir / "val_predictions.csv"),
                        "status": "done",
                    })
                    print(
                        f"DONE {experiment_id} best_acc={result['best_val_acc']:.4f} "
                        f"plateau_acc={st.get('last20_val_acc_mean', 0.0):.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
