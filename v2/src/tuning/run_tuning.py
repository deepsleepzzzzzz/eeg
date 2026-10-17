"""Validation-only hyper-parameter tuning for SEU DDS experiments.

Hard rules enforced here:
  * the sealed test set is never read (evaluate_test is always False)
  * every trial is written to a tuning registry (no selective reporting)
  * the protocol YAML stays the single source of truth; overrides are applied
    to an in-memory deep copy and recorded verbatim in the registry
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.protocol import artifact_root  # noqa: E402
from training.train import load_yaml, train_one  # noqa: E402


def coerce(value, ref):
    """Cast a grid value to the type of the current protocol value."""
    if ref is None:
        return value
    if isinstance(ref, bool):
        return str(value).lower() in ("1", "true", "yes")
    if isinstance(ref, int):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    if isinstance(ref, list):
        if isinstance(value, str):
            return [int(x) for x in value.split(",") if x.strip()]
        return list(value)
    return value


def apply_override(protocol: dict, key: str, value) -> None:
    if "." not in str(key):
        raise ValueError(f"override key must be '<section>.<name>', got {key!r}")
    section, name = str(key).split(".", 1)
    if section == "train":
        ref = protocol["training"].get(name)
        protocol["training"][name] = coerce(value, ref)
    elif section in ("cnn", "gcn"):
        ref = protocol["models"][section].get(name)
        protocol["models"][section][name] = coerce(value, ref)
    else:
        raise ValueError(f"unknown override section: {section!r}")


def write_registry(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True, help="YAML with tag/task/condition/models/seeds/grid")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    spec = load_yaml(Path(args.grid))
    protocol_base = load_yaml(Path(args.protocol))
    protocol_name = str(protocol_base["name"])

    tag = str(spec["tag"])
    task = str(spec["task"])
    condition = str(spec["condition"])
    models = [str(m) for m in spec.get("models", ["cnn"])]
    seeds = [int(s) for s in spec.get("seeds", [42])]
    grid = spec.get("grid", None) or {}

    out_root = Path(args.output_root) if args.output_root else (artifact_root(protocol_base) / "tuning" / tag)
    out_root.mkdir(parents=True, exist_ok=True)
    registry = out_root / "registry_tuning.csv"

    keys = sorted(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]
    total = len(combos) * len(models) * len(seeds)
    print(f"[config] tag={tag} task={task} condition={condition} models={models} seeds={seeds} trials={total}", flush=True)
    print(f"[config] grid={keys}", flush=True)
    print(f"[out] {out_root}", flush=True)

    rows: list[dict] = []
    idx = 0
    for model in models:
        for seed in seeds:
            for combo in combos:
                idx += 1
                protocol = deepcopy(protocol_base)
                over = {}
                for k, v in zip(keys, combo):
                    apply_override(protocol, k, v)
                    over[k] = v
                tune_id = f"{task}_{model}_{condition}_{tag}_{idx:03d}"
                experiment_cfg = {
                    "experiment_id": tune_id,
                    "protocol": protocol_name,
                    "comparison_group": f"tuning_{tag}",
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                }
                print(f"[{idx}/{total}] {tune_id} seed={seed} {over}", flush=True)
                result = train_one(
                    experiment_cfg,
                    protocol,
                    evaluate_test=False,
                    confirm_test=False,
                    output_root=out_root,
                )
                m = result["val_metrics"]
                st = result.get("stability", {}) or {}
                row = {
                    "tune_id": tune_id,
                    "tag": tag,
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "params": json.dumps(over, sort_keys=True),
                    "best_epoch": result["best_epoch"],
                    "val_acc": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "balanced_acc": m["balanced_accuracy"],
                    # 平台值：比"最高分尖峰"稳得多，作为选超参的主判据
                    "plateau_acc": st.get("last20_val_acc_mean"),
                    "plateau_std": st.get("last20_val_acc_std"),
                    "train_val_gap": st.get("gap_train_minus_val_acc"),
                    "epochs_run": st.get("epochs_run"),
                    "final_lr": st.get("final_lr"),
                    "run_dir": str(out_root / tune_id),
                }
                rows.append(row)
                write_registry(registry, rows)
                print(
                    f"    -> best_acc={m['accuracy']:.4f} plateau_acc={row['plateau_acc']:.4f} "
                    f"plateau_std={row['plateau_std']:.4f}({row['plateau_std']*975:.0f}win) "
                    f"gap={row['train_val_gap']:+.4f} epochs={row['epochs_run']}",
                    flush=True,
                )

    print(f"[done] {len(rows)} trials -> {registry}", flush=True)


if __name__ == "__main__":
    main()

