"""artifacts/ 台账守卫：每个实验目录都必须在 artifacts/README.md 里登记。

背景（2026-10-16 核查发现）：`variant_scan_normtest2/`（33 run）与 `variant_scan_prefilter/`（24 run）
是**在 v4 协议下**跑的选型对照实验，但它们在 artifacts/README.md 的"当前可引用"表里
没有任何"对照/选型"标注 —— 只看台账会以为它们与主结果同协议（同一类"静默用错配置"）。

本测试强制两件事：
  1. artifacts/ 下每个目录（`var_<hash>` 缓存除外）都必须在台账里出现过；
  2. 跑的不是当前主协议的目录，台账里提到它的那几行必须带标注词（对照/旧/冻结/…）。
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
LEDGER = ARTIFACTS / "README.md"
#: 标注词：出现任意一个，就说明这行提醒了读者"此目录不是当前主协议的主结果"
FLAG_WORDS = ("对照", "历史", "旧", "冻结", "探针", "旁支", "可清", "归档", "机制", "选型", "不采用", "禁止引用", "为空", "未开")


def _groups() -> list[Path]:
    return sorted(p for p in ARTIFACTS.iterdir()
                  if p.is_dir() and not p.name.startswith("var_") and p.name != ".pytest_cache")


def _ledger_lines() -> list[str]:
    return LEDGER.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def test_every_artifacts_dir_is_registered_in_ledger():
    text = "\n".join(_ledger_lines())
    missing = [p.name for p in _groups() if p.name not in text]
    assert not missing, ("这些 artifacts 子目录没有在 artifacts/README.md 里登记：\n  "
                         + "\n  ".join(missing))


def test_non_main_protocol_dirs_are_flagged_in_ledger():
    lines = _ledger_lines()
    bad = []
    for g in _groups():
        protos = set()
        for mj in g.rglob("metrics.json"):
            try:
                protos.add(str(json.loads(mj.read_text(encoding="utf-8")).get("protocol", "")))
            except json.JSONDecodeError:
                continue
        if not protos or all(p.startswith("main_70_15_15_v5") for p in protos):
            continue
        hit = [l for l in lines if g.name in l and any(w in l for w in FLAG_WORDS)]
        if not hit:
            bad.append(f"{g.name}: 协议 {sorted(protos)}，台账里没有带标注词的行")
    assert not bad, ("以下目录用的不是当前主协议，却没有任何标注：\n  " + "\n  ".join(bad))