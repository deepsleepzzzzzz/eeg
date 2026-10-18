"""Aggregate multi-seed ablation: per-run metrics from confusion matrices, then mean +- std (ddof=1) across seeds.

val  <- outputs/ablation/summary_runs.json       (val_confusion_matrix)
test <- outputs/ablation/test_results_runs.json  (test_confusion_matrix)
Outputs (per split): <prefix>_per_run.csv, <prefix>.csv, <prefix>.md
  val prefix  = summary_val_mean_std ; test prefix = test_summary
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.ablation.ablation_metrics import cm_metrics  # noqa: E402

OUT = ROOT / "outputs" / "ablation"
METRICS = ("acc", "sens", "spec", "f1", "prec")
SPLITS = {"val": ("summary_runs.json", "val_confusion_matrix", "summary_val_mean_std"),
          "test": ("test_results_runs.json", "test_confusion_matrix", "test_summary")}


def order_key(tag):
    b, l = tag.split("_L")
    return (int(b[1:]), len(l))


def aggregate(split):
    fname, cmkey, prefix = SPLITS[split]
    runs = json.loads((OUT / fname).read_text(encoding="utf-8"))["runs"]
    per = {}  # (tag, task, arch, seed) -> {cond: metrics}
    meta = {}
    for r in runs.values():
        if not r.get("ok") or cmkey not in r:
            continue
        per.setdefault((r["tag"], r["task"], r["arch"], int(r["seed"])), {})[r["condition"]] = cm_metrics(r[cmkey])
        meta[r["tag"]] = r
    per_rows = []
    groups = defaultdict(list)  # (tag, task, arch, cond) -> list of metric dicts
    for (tag, task, arch, seed), conds in per.items():
        for cond, m in conds.items():
            groups[(tag, task, arch, cond)].append(m)
            per_rows.append({"split": split, "setting": tag, "task": task, "model": arch, "condition": cond, "seed": seed,
                             **{k: round(v, 4) for k, v in m.items()}})
        if {"20_0", "30_2"} <= set(conds):
            mm = {k: float(np.mean([conds[c][k] for c in ("20_0", "30_2")])) for k in METRICS}
            groups[(tag, task, arch, "mean")].append(mm)
            per_rows.append({"split": split, "setting": tag, "task": task, "model": arch, "condition": "mean", "seed": seed,
                             **{k: round(v, 4) for k, v in mm.items()}})
    per_rows.sort(key=lambda r: (r["task"], r["model"], order_key(r["setting"]), r["condition"], r["seed"]))
    rows = []
    for (tag, task, arch, cond), ms in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], order_key(kv[0][0]), kv[0][3])):
        m = meta[tag]
        row = {"split": split, "ablation": "both(baseline)" if tag == "b6_L123" else m["ablation"], "setting": tag,
               "bins": m["bins"], "lags": "-".join(map(str, m["lags"])), "lag_order": max(m["lags"]),
               "task": task, "model": arch, "condition": cond, "n_seeds": len(ms)}
        for k in METRICS:
            v = np.array([x[k] for x in ms])
            row[f"{k}_mean"] = round(float(v.mean()), 4)
            row[f"{k}_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        row["note"] = "CNN pool_size=1 at 3 bins" if (m["bins"] < 4 and arch == "cnn") else ""
        rows.append(row)
    for name, data in ((f"{prefix}_per_run.csv", per_rows), (f"{prefix}.csv", rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    label = "VALIDATION" if split == "val" else "TEST (evaluated once on the val-selected checkpoint; not used for selection)"
    seeds = sorted({r["seed"] for r in per_rows})
    lines = [f"# SEU ablation, {label}", "", f"Seeds: {seeds}. mean +- sample std (ddof=1) over seeds. "
             "Per seed, 'mean' condition = average of 20_0 and 30_2. sens = macro recall, spec = macro OvR specificity, f1/prec = macro.", ""]
    for cond in ("mean", "20_0", "30_2"):
        lines += [f"## condition: {cond}", "", "| task | model | setting | n | acc | sens | spec | f1 | prec | note |",
                  "|---|---|---|---:|---|---|---|---|---|---|"]
        for r in rows:
            if r["condition"] == cond:
                cells = " | ".join(f"{r[k+'_mean']:.4f}±{r[k+'_std']:.4f}" for k in METRICS)
                lines.append(f"| {r['task']} | {r['model']} | {r['setting']} | {r['n_seeds']} | {cells} | {r['note']} |")
        lines.append("")
    (OUT / f"{prefix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{split}: {len(per_rows)} per-run rows, {len(rows)} aggregate rows -> {prefix}.csv/.md")


if __name__ == "__main__":
    for sp in ("val", "test"):
        aggregate(sp)
