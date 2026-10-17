"""扫描实验总账 registry_scans.csv 必须与 artifacts/ 磁盘一致。

背景：论文里的消融 / 噪声 / 基线 / 跨工况数字散在 15 个扫描目录、1300+ 个 run 里。
"哪个目录、什么协议、跑完没有"此前只靠人记，已经因此三次把未跑完的 run 当最终结果报。
本测试让"跑完批忘了更新总账"当场失败（生成脚本：`python src/reporting/make_registry_scans.py`）。
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "registry_scans.csv"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_registry_scans", PROJECT_ROOT / "src" / "reporting" / "make_registry_scans.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_scans_matches_disk():
    assert OUT.exists(), "缺少 registry_scans.csv（先跑 python src/reporting/make_registry_scans.py）"
    rows = _load_generator().build_rows()
    assert len(rows) > 500, f"扫描总账只有 {len(rows)} 行，明显不对（项目里有上千个 run）"
    with OUT.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert [r["experiment_id"] for r in saved] == [r["experiment_id"] for r in rows], (
        "扫描总账与磁盘的 run 集合不一致，请重新生成 registry_scans.csv")
    assert saved == rows, "扫描总账内容与磁盘现状不一致，请重新生成 registry_scans.csv"


def test_scan_registry_has_no_incomplete_runs():
    bad = [r for r in _load_generator().build_rows() if r["status"] != "done"]
    detail = "\n".join(f"  [{r['status']}] {r['scan_group']}/{r['variant']}/{r['experiment_id']}"
                       for r in bad)
    assert not bad, ("有 run 只有 manifest.json 而没有 metrics.json（跑批没跑完或中途失败）：\n" + detail)


def test_scan_registry_protocols_are_declared():
    """总账里的协议名必须是 configs/protocols/ 里真实存在的协议（防止写成别名或笔误）。"""
    declared = {p.stem for p in (PROJECT_ROOT / "configs" / "protocols").glob("*.yaml")}
    used = {r["protocol"] for r in _load_generator().build_rows() if r["protocol"]}
    unknown = sorted(used - declared)
    assert not unknown, f"总账里出现 configs/protocols/ 未声明的协议：{unknown}"