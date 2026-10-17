"""隔离测试：新项目必须自包含，不许引用旧项目。

用户 2026-10-16 明确：**不要理旧项目，怕干扰**（旧项目的绝对路径见 AGENTS.md）。
这条如果只写在文档里迟早会被违反（一次误写的绝对路径就够了），所以用测试强制：

  1. 代码/配置/脚本/JSON/CSV 里不得出现旧项目路径或名字（.md 允许——AGENTS.md 里那条规则本身要写它）；
  2. 协议里的 `paths.*` 必须是项目内相对路径（不许出现盘符），否则换机器/换目录就会静默读到别处；
  3. 项目的原始数据目录必须存在且非空（自包含的前提）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 关键字用拼接生成：本文件必须能通过自己这条检查（字面量不出现在源码里）
OLD_PROJECT_KEYS = tuple({"EEG" + "_SEU" + suffix for suffix in ("成果_20261005", "成果", "")})
SCAN_SUFFIX = {".py", ".yaml", ".yml", ".ps1", ".json", ".csv", ".bat", ".toml", ".ini"}


def _scan_files():
    for p in PROJECT_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in {".venv", ".git", "legacy_docs"} for part in p.parts):
            continue
        if p.suffix.lower() in SCAN_SUFFIX:
            yield p


def test_no_reference_to_old_project():
    bad = []
    for p in _scan_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for key in OLD_PROJECT_KEYS:
            if key in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if key in line:
                        bad.append(f"{p.relative_to(PROJECT_ROOT)}:{i}  {line.strip()[:90]}")
                break
    assert not bad, ("新项目里出现了旧项目路径（用户要求：不要理旧项目）——"
                     "请改成项目内相对路径或本项目内的副本：\n" + "\n".join(bad))


def test_protocol_paths_are_project_relative():
    bad = []
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for k, v in (cfg.get("paths") or {}).items():
            if isinstance(v, str) and re.match(r"^[A-Za-z]:", v):
                bad.append(f"{f.name}: paths.{k} = {v}")
    assert not bad, "协议里的路径必须是项目内相对路径：\n" + "\n".join(bad)


def test_raw_data_is_present_in_project():
    d = PROJECT_ROOT / "data" / "seu" / "gearbox"
    assert d.is_dir(), f"原始数据目录不存在：{d}"
    csvs = list(d.rglob("*.csv"))
    assert len(csvs) >= 20, f"原始 CSV 数量异常：{len(csvs)}"
    assert sum(p.stat().st_size for p in csvs) > 1e9, "原始 CSV 体量异常（数据可能不全）"
