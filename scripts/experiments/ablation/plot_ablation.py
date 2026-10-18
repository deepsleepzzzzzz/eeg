"""Plot bins / lag ablation curves from outputs/ablation/summary_seed42.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "ablation"
FIG = OUT / "figures"
METRICS = (("acc", "Accuracy"), ("sens", "Sensitivity (macro recall)"), ("spec", "Specificity (macro OvR)"), ("f1", "Macro F1"))
CONDS = (("20_0", "20-0", "tab:blue", "o", "--"), ("30_2", "30-2", "tab:orange", "s", "--"), ("mean", "mean", "black", "D", "-"))
MODELS = (("cnn", "CNN"), ("gcn", "GCN"))
TASKS = (("bearing", "Bearing"), ("gear", "Gear"))


def load(seed):
    with (OUT / f"summary_seed{seed}.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def select(rows, abl, task, model, cond):
    xs = []
    for r in rows:
        if r["task"] != task or r["model"] != model or r["condition"] != cond:
            continue
        if abl == "bins" and r["lags"] == "1-2-3":
            xs.append((int(r["bins"]), r))
        elif abl == "lags" and r["bins"] == "6":
            xs.append((int(r["lag_order"]), r))
    return sorted(xs, key=lambda t: t[0])


def xlabel(abl):
    return "Number of amplitude bins (lags {1,2,3})" if abl == "bins" else "Max lag order L (lags {1..L}, 6 bins)"


def detail(rows, abl, task, tname, seed):
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(10, 12), sharex=True)
    for j, (m, mname) in enumerate(MODELS):
        for i, (k, kname) in enumerate(METRICS):
            ax = axes[i, j]
            for cond, clabel, col, mk, ls in CONDS:
                pts = select(rows, abl, task, m, cond)
                if pts:
                    ax.plot([x for x, _ in pts], [float(r[k]) for _, r in pts], ls, marker=mk, color=col, label=clabel, lw=2 if cond == "mean" else 1.2)
            if abl == "bins":
                ax.axvline(3, color="grey", lw=0.6, ls=":") if m == "cnn" else None
            ax.set_ylabel(kname)
            ax.set_xticks(sorted({x for x, _ in select(rows, abl, task, m, "mean")}))
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{tname} - {mname}")
            if i == len(METRICS) - 1:
                ax.set_xlabel(xlabel(abl))
    axes[0, 0].legend(title="condition", fontsize=8)
    note = "Validation set, seed %d." % seed + (" CNN at 3 bins uses pool_size=1." if abl == "bins" else "")
    fig.suptitle(f"SEU {tname}: {'bin-count' if abl == 'bins' else 'lag'} ablation\n{note}", fontsize=11)
    fig.tight_layout()
    path = FIG / f"{abl}_ablation_{task}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def overview(rows, abl, seed):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    styles = {"acc": ("tab:blue", "o"), "sens": ("tab:green", "s"), "spec": ("tab:red", "^"), "f1": ("tab:purple", "D")}
    for i, (task, tname) in enumerate(TASKS):
        for j, (m, mname) in enumerate(MODELS):
            ax = axes[i, j]
            pts = select(rows, abl, task, m, "mean")
            for k, kname in METRICS:
                col, mk = styles[k]
                ax.plot([x for x, _ in pts], [float(r[k]) for _, r in pts], "-", marker=mk, color=col, label=kname.split(" (")[0])
            ax.set_title(f"{tname} - {mname} (mean of 20-0 / 30-2)")
            ax.set_xticks([x for x, _ in pts])
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel(xlabel(abl))
            if j == 0:
                ax.set_ylabel("metric (val)")
    axes[0, 0].legend(fontsize=8)
    note = f"Validation set, seed {seed}." + (" CNN at 3 bins uses pool_size=1." if abl == "bins" else "")
    fig.suptitle(f"SEU {'bin-count' if abl == 'bins' else 'lag'} ablation overview. {note}", fontsize=11)
    fig.tight_layout()
    path = FIG / f"overview_{abl}_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load(a.seed)
    for abl in ("bins", "lags"):
        for task, tname in TASKS:
            print(detail(rows, abl, task, tname, a.seed))
        print(overview(rows, abl, a.seed))


if __name__ == "__main__":
    main()
