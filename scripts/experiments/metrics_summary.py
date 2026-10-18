"""Consolidate per-class metrics + multi-seed statistics for SEU DDS.

Reads every metrics.json under datas/experiments_seu/checkpoints_* (multi-seed
42/43/44) plus the 5-fold CV and held-out acceptance JSONs, then computes:

  - per-class sensitivity (recall), specificity, precision from confusion matrices
  - multi-seed mean +- std of val/test accuracy and per-class metrics
  - the same stats for the frozen-parameter held-out acceptance

Pure numpy; no GPU. Output -> datas/experiments_seu/teacher_extra/metrics_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CKPT_ROOTS = [
    "datas/experiments_seu/checkpoints_bearing_cnn",
    "datas/experiments_seu/checkpoints_bearing_gcn",
    "datas/experiments_seu/checkpoints_gear_cnn",
    "datas/experiments_seu/checkpoints_gear_gcn",
]
OUT_DIR = Path("datas/experiments_seu/teacher_extra")


def per_class(cm: np.ndarray, names) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.shape[0]
    total = cm.sum()
    out = {}
    for i, name in enumerate(names):
        tp = cm[i, i]
        row = cm[i, :].sum()          # actual class size
        col = cm[:, i].sum()          # predicted class size
        fn = row - tp
        fp = col - tp
        tn = total - tp - fn - fp
        sensitivity = tp / row if row > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / col if col > 0 else 0.0
        out[name] = {
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(precision), 4),
            "support": int(row),
        }
    return out


def agg_metric(list_of_dicts, metric):
    vals = [d[metric] for d in list_of_dicts if metric in d and d[metric] is not None]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}


def load_ckpt_results():
    rows = []
    for rel in CKPT_ROOTS:
        root = ROOT / rel
        if not root.exists():
            continue
        arch = "cnn" if "cnn" in rel else "gcn"
        task = "bearing" if "bearing" in rel else "gear"
        for task_dir in sorted((root / task).glob("*")):
            condition = task_dir.name
            for seed_dir in sorted(task_dir.glob("seed_*")):
                mj = seed_dir / "metrics.json"
                if not mj.exists():
                    continue
                d = json.loads(mj.read_text(encoding="utf-8"))
                seed = int(seed_dir.name.split("_")[1])
                names = d.get("class_names", [])
                row = {
                    "task": task,
                    "arch": arch,
                    "condition": condition,
                    "seed": seed,
                    "val_acc": round(float(d.get("val_acc", 0)), 4),
                    "test_acc": round(float(d.get("test_acc", 0)), 4) if d.get("test_acc") is not None else None,
                    "val_macro_f1": round(float(d.get("val_macro_f1", 0)), 4),
                    "test_macro_f1": round(float(d.get("test_macro_f1", 0)), 4) if d.get("test_macro_f1") is not None else None,
                    "val_per_class": per_class(np.asarray(d["val_confusion_matrix"]), names) if "val_confusion_matrix" in d else None,
                    "test_per_class": per_class(np.asarray(d["test_confusion_matrix"]), names) if "test_confusion_matrix" in d else None,
                    "class_names": names,
                }
                rows.append(row)
    return rows


def summarize_ckpts(rows):
    summary = {}
    for r in rows:
        key = (r["task"], r["arch"], r["condition"])
        summary.setdefault(key, []).append(r)
    out = []
    for (task, arch, condition), group in sorted(summary.items()):
        seeds = sorted(g["seed"] for g in group)
        names = group[0]["class_names"]
        # multi-seed mean+-std of overall acc / f1
        val_acc = agg_metric(group, "val_acc")
        test_acc = agg_metric(group, "test_acc")
        val_f1 = agg_metric(group, "val_macro_f1")
        test_f1 = agg_metric(group, "test_macro_f1")
        # per-class mean+-std aggregated across seeds (on test if available, else val)
        use_test = all(g.get("test_per_class") for g in group)
        per_class_agg = {}
        for i, name in enumerate(names):
            for m in ("sensitivity", "specificity", "precision"):
                vals = []
                for g in group:
                    src = g["test_per_class"] if use_test else g["val_per_class"]
                    if src and name in src:
                        vals.append(src[name][m])
                per_class_agg.setdefault(name, {})[m] = (
                    {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
                    if vals else None
                )
        out.append({
            "task": task, "arch": arch, "condition": condition,
            "seeds": seeds,
            "source": "test" if use_test else "val",
            "val_acc_mean_std": val_acc,
            "test_acc_mean_std": test_acc,
            "val_macro_f1_mean_std": val_f1,
            "test_macro_f1_mean_std": test_f1,
            "per_class_mean_std": per_class_agg,
        })
    return out


def load_holdout():
    out = []
    for jf in sorted((ROOT / "datas/experiments_seu/holdout").glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        names = d.get("class_names", [])
        out.append({
            "task": d["task"], "arch": d["arch"], "condition": d["condition"],
            "best_val_acc": d.get("best_val_acc"),
            "test_acc": d.get("test_acc"),
            "holdout_acc": d.get("holdout_acc"),
            "holdout_per_class": per_class(np.asarray(d["holdout_confusion_matrix"]), names),
            "test_per_class": per_class(np.asarray(d["test_confusion_matrix"]), names),
            "class_names": names,
        })
    return out


def load_cv():
    out = []
    for jf in sorted((ROOT / "datas/experiments_seu/cv").glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        out.append({
            "task": d["task"], "arch": d["arch"], "condition": d["condition"],
            "test_mean": d.get("test_mean"), "test_std": d.get("test_std"),
            "test_min": d.get("test_min"), "test_max": d.get("test_max"),
        })
    return out


def main():
    ckpt_rows = load_ckpt_results()
    ckpt_summary = summarize_ckpts(ckpt_rows)
    holdout = load_holdout()
    cv = load_cv()
    result = {
        "n_checkpoint_runs": len(ckpt_rows),
        "multi_seed_per_task": ckpt_summary,
        "holdout_acceptance": holdout,
        "rolling_origin_cv": cv,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / OUT_DIR / "metrics_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'metrics_summary.json'}  ({len(ckpt_rows)} runs)")

    # --- human-readable tables ---
    print("\n===== 多种子 test 准确率 mean±std (3 seeds: 42/43/44) =====")
    for r in ckpt_summary:
        ta = r["test_acc_mean_std"] or {}
        va = r["val_acc_mean_std"] or {}
        print(f"{r['task']:8s} {r['arch']:4s} {r['condition']:5s}  "
              f"val {va.get('mean')}±{va.get('std')}   "
              f"test {ta.get('mean')}±{ta.get('std')}")

    print("\n===== 独立留出(冻结参数) =====")
    for h in holdout:
        print(f"{h['task']:8s} {h['arch']:4s} {h['condition']:5s}  "
              f"val={h['best_val_acc']} test={h['test_acc']} holdout={h['holdout_acc']}")

    print("\n===== 5折 rolling-origin CV =====")
    for c in cv:
        print(f"{c['task']:8s} {c['arch']:4s} {c['condition']:5s}  "
              f"{c['test_mean']}±{c['test_std']}  [{c['test_min']},{c['test_max']}]")


if __name__ == "__main__":
    main()
