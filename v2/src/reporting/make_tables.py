"""从 registry.csv 生成论文表（逐次明细 + 按任务/工况/模型汇总的 均值±标准差）。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> float:
    return 100.0 * float(x)


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{pct(values[0]):.2f}"
    return f"{pct(statistics.mean(values)):.2f} ± {pct(statistics.pstdev(values)):.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    registry = PROJECT_ROOT / "registry.csv"
    rows = list(csv.DictReader(registry.open("r", encoding="utf-8", newline="")))
    selected = [r for r in rows if r.get("comparison_group") == args.group]

    detail, by_key = [], {}
    has_test = False
    for r in selected:
        mpath = Path(r["metrics_path"])
        if not mpath.is_absolute():
            mpath = PROJECT_ROOT / mpath
        metrics = load_metrics(mpath)
        vm = metrics.get("val_metrics", {}) or {}
        st = metrics.get("stability", {}) or {}
        tm = metrics.get("test_metrics")
        if tm:
            has_test = True
        detail.append({
            "experiment_id": r["experiment_id"],
            "task": r.get("task", ""),
            "condition": r.get("condition", ""),
            "model": r.get("model", ""),
            "seed": r.get("seed", ""),
            "best_epoch": metrics.get("best_epoch", r.get("best_epoch", "")),
            "epochs_run": st.get("epochs_run", r.get("epochs_run", "")),
            "best_acc": metrics.get("best_val_acc", float("nan")),
            "plateau_acc": st.get("last20_val_acc_mean", float("nan")),
            "macro_f1": vm.get("macro_f1", float("nan")),
            "balanced_acc": vm.get("balanced_accuracy", float("nan")),
            "gap": st.get("gap_train_minus_val_acc", float("nan")),
            "test_acc": (tm or {}).get("accuracy"),
        })
        key = (r.get("task", ""), r.get("condition", ""), r.get("model", ""))
        d = by_key.setdefault(key, {"plateau": [], "best": [], "macro_f1": [], "bal": [], "test": []})
        d["plateau"].append(st.get("last20_val_acc_mean", float("nan")))
        d["best"].append(metrics.get("best_val_acc", float("nan")))
        d["macro_f1"].append(vm.get("macro_f1", float("nan")))
        d["bal"].append(vm.get("balanced_accuracy", float("nan")))
        if tm:
            d["test"].append(tm.get("accuracy"))

    lines = [f"# {args.group}", ""]
    lines += ["## 汇总（验证集，跨 seed 均值 ± 标准差，单位 %）", ""]
    head = "| task | condition | model | n | plateau acc | best acc | macro F1 | balanced acc |"
    lines += [head, "|---|---|---|---:|---:|---:|---:|---:|"]
    for (task, cond, model), d in sorted(by_key.items()):
        lines.append(
            f"| {task} | {cond} | {model} | {len(d['best'])} | "
            f"{mean_std(d['plateau'])} | {mean_std(d['best'])} | "
            f"{mean_std(d['macro_f1'])} | {mean_std(d['bal'])} |"
        )
    lines += [""]

    if has_test:
        lines += ["## 封存测试集（每行只评估一次）", ""]
        lines += ["| task | condition | model | n | test acc |", "|---|---|---|---:|---:|"]
        for (task, cond, model), d in sorted(by_key.items()):
            if d["test"]:
                lines.append(f"| {task} | {cond} | {model} | {len(d['test'])} | {mean_std(d['test'])} |")
        lines += [""]

    lines += ["## 逐次明细（验证集）", ""]
    lines += [
        "| experiment_id | best_ep | epochs | best acc | plateau acc | macro F1 | balanced acc | train-val gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(detail, key=lambda x: (x["task"], x["condition"], x["model"], int(x["seed"]))):
        lines.append(
            f"| {d['experiment_id']} | {d['best_epoch']} | {d['epochs_run']} | "
            f"{pct(d['best_acc']):.2f} | {pct(d['plateau_acc']):.2f} | "
            f"{pct(d['macro_f1']):.2f} | {pct(d['balanced_acc']):.2f} | {pct(d['gap']):+.2f} |"
        )

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(selected)} runs)")


if __name__ == "__main__":
    main()
