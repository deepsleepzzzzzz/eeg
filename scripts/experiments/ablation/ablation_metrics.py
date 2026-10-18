"""Metrics from saved val confusion matrices -> outputs/ablation/summary_seed42.{csv,md}."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "ablation"
METRICS = ("acc", "sens", "spec", "f1", "prec")


def cm_metrics(cm) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    rec = div(tp, tp + fn)
    prec = div(tp, tp + fp)
    spec = div(tn, tn + fp)
    f1 = div(2 * prec * rec, prec + rec)
    return {"acc": float(tp.sum() / n), "sens": float(rec.mean()), "spec": float(spec.mean()),
            "f1": float(f1.mean()), "prec": float(prec.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    runs = json.loads((OUT / "summary_runs.json").read_text(encoding="utf-8"))["runs"]
    groups: dict[tuple, dict] = {}
    for key, r in runs.items():
        if not r.get("ok") or int(r["seed"]) != a.seed:
            continue
        g = groups.setdefault((r["tag"], r["task"], r["arch"]), {"meta": r, "conds": {}})
        g["conds"][r["condition"]] = r
    rows = []
    for (tag, task, arch), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[1]["meta"]["bins"], len(kv[1]["meta"]["lags"]))):
        m = g["meta"]
        abl = "both(baseline)" if tag == "b6_L123" else m["ablation"]
        per = {}
        for cond, r in sorted(g["conds"].items()):
            per[cond] = cm_metrics(r["val_confusion_matrix"])
            rows.append({"ablation": abl, "setting": tag, "bins": m["bins"], "lags": "-".join(map(str, m["lags"])),
                         "lag_order": max(m["lags"]), "task": task, "model": arch, "condition": cond,
                         **{k: round(v, 4) for k, v in per[cond].items()},
                         "epochs_ran": r["epochs_ran"], "best_epoch": r["best_epoch"],
                         "train_s": round(r["train_s"], 1), "note": r.get("arch_note", "")})
        if len(per) == 2:
            mean = {k: float(np.mean([per[c][k] for c in per])) for k in METRICS}
            rows.append({"ablation": abl, "setting": tag, "bins": m["bins"], "lags": "-".join(map(str, m["lags"])),
                         "lag_order": max(m["lags"]), "task": task, "model": arch, "condition": "mean",
                         **{k: round(v, 4) for k, v in mean.items()},
                         "epochs_ran": "", "best_epoch": "", "train_s": "",
                         "note": m.get("arch_note", "")})
    csv_path = OUT / f"summary_seed{a.seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# SEU ablation, seed {a.seed}, VALIDATION set (test sealed)", "",
             "Metrics from val confusion matrices. sens = macro recall, spec = macro one-vs-rest specificity, "
             "f1/prec = macro. mean = arithmetic mean of 20_0 and 30_2.", ""]
    for cond in ("mean", "20_0", "30_2"):
        lines += [f"## condition: {cond}", "",
                  "| task | model | setting | bins | lags | acc | sens | spec | f1 | prec | note |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if r["condition"] == cond:
                lines.append(f"| {r['task']} | {r['model']} | {r['setting']} | {r['bins']} | {r['lags']} | "
                             f"{r['acc']:.4f} | {r['sens']:.4f} | {r['spec']:.4f} | {r['f1']:.4f} | {r['prec']:.4f} | {r['note']} |")
        lines.append("")
    (OUT / f"summary_seed{a.seed}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
