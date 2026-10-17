"""生成 registry_scans.csv：主结果以外全部 run 的总账（自动扫描 artifacts/）。

为什么要它
----------
`registry.csv` 只登记主实验（40 个 run）。论文里的消融 / 噪声 / 基线 / 跨工况 /
表示扫描等上千个 run 此前只散落在 `artifacts/variant_scan*/` 里 ——
"哪个目录、什么协议、跑完没有"只能靠人记，而本项目反复踩的就是这个坑
（用目录数判断跑完、把未完成的结果当最终结果报错三次）。

本脚本把这件事变成一张可 diff 的表；`tests/test_registry_scans.py` 盯着它不许过期。

用法
----
    python src/reporting/make_registry_scans.py            # 重新生成 registry_scans.csv
    python src/reporting/make_registry_scans.py --check    # 只校验与磁盘是否一致，不写

规则
----
* 扫 `artifacts/*/`，跳过主结果 `main_70_15_15_v5`（在 registry.csv 里）、
  `golden/`（冻结回归基准）、`sealed_test/`（封存测试集，不许碰）、
  `tuning/`（调参中间记录）、以及单次运行缓存 `var_<hash>/`（按内容哈希命名，可清可重建）。
* 目录里有 `metrics.json` -> `status=done`；只有 `manifest.json`
  （train.py 先建目录、跑完才写 metrics）-> `status=incomplete`。
  残缺目录必须能被看见，而不是混在"跑完了"里。
* `metrics_path` 用**项目内相对路径**（registry.csv 里的历史绝对路径保持原样，不回改）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
OUT = PROJECT_ROOT / "registry_scans.csv"

#: 不登记进扫描总账的目录（各有各的台账）
SKIP_GROUPS = {"main_70_15_15_v5", "golden", "sealed_test", "tuning"}

FIELDS = [
    "experiment_id", "scan_group", "variant", "protocol", "task", "condition", "model", "seed",
    "best_epoch", "best_val_acc", "plateau_acc", "plateau_std", "train_val_gap",
    "epochs_run", "final_lr", "metrics_path", "status",
]


def _skip(name: str) -> bool:
    return name in SKIP_GROUPS or name.startswith("var_")


def _num(x, fmt: str = "") -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    if fmt:
        return format(float(x), fmt)
    return repr(float(x))


def build_rows(artifacts: Path | None = None) -> list[dict]:
    """扫描 artifacts/ 下所有非主实验目录，返回行列表（全部为字符串值）。"""
    artifacts = Path(artifacts or ARTIFACTS)
    rows: list[dict] = []
    for group in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        if _skip(group.name):
            continue
        for manifest in sorted(group.rglob("manifest.json")):
            run_dir = manifest.parent
            variant = run_dir.parent.relative_to(group).as_posix()
            if variant == ".":
                variant = ""
            row = {f: "" for f in FIELDS}
            row["experiment_id"] = run_dir.name
            row["scan_group"] = group.name
            row["variant"] = variant
            metrics_p = run_dir / "metrics.json"
            if not metrics_p.exists():
                row["status"] = "incomplete"
                rows.append(row)
                continue
            try:
                m = json.loads(metrics_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                row["status"] = "corrupt"
                rows.append(row)
                continue
            st = m.get("stability") or {}
            row.update({
                "protocol": str(m.get("protocol", "")),
                "task": str(m.get("task", "")),
                "condition": str(m.get("condition", "")),
                "model": str(m.get("model", "")),
                "seed": _num(m.get("seed")),
                "best_epoch": _num(m.get("best_epoch")),
                "best_val_acc": _num(m.get("best_val_acc"), ".6f"),
                "plateau_acc": _num(st.get("last20_val_acc_mean"), ".6f"),
                "plateau_std": _num(st.get("last20_val_acc_std"), ".6f"),
                "train_val_gap": _num(st.get("gap_train_minus_val_acc"), ".6f"),
                "epochs_run": _num(st.get("epochs_run")),
                "final_lr": _num(st.get("final_lr"), ".3g"),
                "metrics_path": metrics_p.relative_to(PROJECT_ROOT).as_posix(),
                "status": "done",
            })
            rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    import io

    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().replace("\r\n", "\n")


def main() -> int:
    setup_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验 registry_scans.csv 与磁盘一致，不写文件")
    a = ap.parse_args()

    rows = build_rows()
    text = render(rows)
    n_done = sum(1 for r in rows if r["status"] == "done")
    n_bad = len(rows) - n_done
    groups = sorted({r["scan_group"] for r in rows})
    protocols = sorted({r["protocol"] for r in rows if r["protocol"]})
    summary = (f"共 {len(rows)} 个 run（完成 {n_done}，异常 {n_bad}）；"
               f"{len(groups)} 个目录：{', '.join(groups)}；协议：{', '.join(protocols)}")

    if a.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur.replace("\r\n", "\n") != text:
            print(f"[FAIL] registry_scans.csv 与磁盘不一致，请重新生成：python src/reporting/make_registry_scans.py")
            print("       " + summary)
            return 1
        print("[OK  ] registry_scans.csv 与磁盘一致")
        print("       " + summary)
        return 0

    OUT.write_bytes(text.encode("utf-8"))
    print(f"[write] {OUT.relative_to(PROJECT_ROOT).as_posix()}")
    print("        " + summary)
    bad = [r for r in rows if r["status"] != "done"]
    for r in bad:
        print(f"        [{r['status']}] {r['scan_group']}/{r['variant']}/{r['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())