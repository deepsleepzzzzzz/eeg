# -*- coding: utf-8 -*-
"""论文数字溯源审计：论文里的每个数字，都必须能追到**本项目**的 run，且协议是当前主协议 v5。

为什么单独做这件事：论文的数字分布在 9 张表与正文里，来源跨越多个扫描目录；
"数字对不上"往往不是算错，而是①取错目录（旧协议/被覆盖的旧 run）、②取错口径、
③引用了已归档的历史结果。本审计把这三件事变成可复核的检查。

检查项：
  A. 论文数字用到的每个 run 目录，必须位于本项目内（不得出现在 legacy/ 或项目外）；
  B. 这些 run 的 protocol 字段必须全部是当前主协议（v5）—— 出现 v1–v4 即为混用旧结果；
  C. 主协议主结果表（表 4）的 40 个 run 必须齐全（4 工况 × 2 模型 × 5 seed）；
  D. 表 8 依赖的加噪 run 必须齐全（4 工况 × 8 档 × 2 模型 × 3 seed）；
  E. 原始数据必须在本项目内，且与官方文件清单 sha 一致（沿用 audit_provenance 的做法）。

用法：python src/audit_paper_numbers.py      # 输出 reports/paper_numbers_audit.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import load_protocol  # noqa: E402
from utils.console import setup_utf8_console  # noqa: E402

setup_utf8_console()

# 论文数字的来源目录（与 tests/test_paper_*.py 一致）
SOURCES = [
    "artifacts/main_70_15_15_v5",          # 表 4、图 8、图 9、表 2/3 的缓存
    "artifacts/variant_scan",              # 表 5/表 9 的加噪行、表 6 基线、图 12
    "artifacts/variant_scan_rep_v5",       # 表 2/表 3 的参数扫描
    "artifacts/variant_scan_noise_v5",     # 表 8 齿轮 30-2
    "artifacts/variant_scan_noise_restore",# 表 8 其余三个工况（本次补齐）
    "artifacts/variant_scan_fill",         # 图 13 的本文值
    "artifacts/variant_scan_energy",       # §5.5(3) 幅值补充对照
    "artifacts/variant_scan_lagspace",     # 图 5 滞后取样对照
    "artifacts/cross_condition",           # §5.5(2) 跨工况
]
NOISE_COMBOS = [("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")]
NOISE_TAGS = ["base", "snr10", "snr4", "snr0", "snrm4", "snrm6", "snrm8", "snrm10"]


def main_protocol() -> dict:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")


def main() -> int:
    proto = main_protocol()
    name, version = str(proto["name"]), str(proto.get("version", "?"))
    lines = [f"# 论文数字溯源审计（主协议 {name}）", ""]
    pass_n = fail_n = 0

    def chk(ok: bool, title: str, detail: str = "") -> None:
        nonlocal pass_n, fail_n
        (lines.append(f"- ✅ {title}" + (f" —— {detail}" if detail else "")) if ok
         else lines.append(f"- ❌ {title}" + (f" —— {detail}" if detail else "")))
        if ok:
            pass_n += 1
        else:
            fail_n += 1

    lines += ["## A/B 每个来源目录的 run 归属与协议", ""]
    total_runs = 0
    protos = Counter()
    for rel in SOURCES:
        d = PROJECT_ROOT / rel
        if not d.exists():
            chk(False, f"{rel} 存在", "目录不存在")
            continue
        runs = sorted(p.parent for p in d.rglob("metrics.json"))
        outside = [str(r) for r in runs if PROJECT_ROOT.resolve() not in r.resolve().parents]
        legacy = [str(r) for r in runs if "legacy" in r.parts]
        chk(not outside, f"{rel} 的 {len(runs)} 个 run 都在本项目内",
            f"越界 {len(outside)} 个" if outside else "")
        chk(not legacy, f"{rel} 不来自 legacy/ 归档", f"{len(legacy)} 个" if legacy else "")
        local = Counter()
        for r in runs:
            try:
                p = json.loads((r / "metrics.json").read_text(encoding="utf-8")).get("protocol")
            except Exception:
                p = None
            local[p] += 1
            protos[p] += 1
        total_runs += len(runs)
        bad = {k: v for k, v in local.items() if k != name}
        chk(not bad, f"{rel} 全部为当前主协议 {name}",
            f"非 v5 协议：{bad}" if bad else f"{len(runs)} 个 run")

    lines += ["", "## C 主结果表（表 4）的 run 必须齐全", ""]
    missing = []
    for task in ("gear", "bearing"):
        for cond in ("20_0", "30_2"):
            for mdl in ("cnn", "gcn"):
                for seed in (42, 43, 44, 45, 46):
                    p = (PROJECT_ROOT / "artifacts/main_70_15_15_v5" /
                         f"{task}_{mdl}_{cond}_seed{seed}" / "metrics.json")
                    if not p.exists():
                        missing.append(str(p.relative_to(PROJECT_ROOT)))
    chk(not missing, "表 4 的 40 个 run 齐全（4 工况 × 2 模型 × 5 seed）",
        f"缺 {len(missing)} 个" if missing else "40/40")

    lines += ["", "## D 表 8 依赖的加噪 run 必须齐全", ""]
    for label, task, cond in (("齿轮 20-0", "gear", "20_0"), ("齿轮 30-2", "gear", "30_2"),
                              ("轴承 20-0", "bearing", "20_0"), ("轴承 30-2", "bearing", "30_2")):
        miss = []
        for tag in NOISE_TAGS:
            for mdl in ("cnn", "gcn"):
                for seed in (42, 43, 44):
                    ok = False
                    for base in ("variant_scan_noise_restore", "variant_scan_noise_v5",
                                 "variant_scan_fill"):
                        d = PROJECT_ROOT / "artifacts" / base / tag
                        if (d / f"{task}_{mdl}_{cond}_{tag}_seed{seed}" / "metrics.json").exists():
                            ok = True
                            break
                    if not ok:
                        miss.append(f"{tag}/{mdl}/s{seed}")
        chk(not miss, f"{label} 的 8 档 × 2 模型 × 3 seed 加噪 run 齐全",
            f"缺 {len(miss)} 个：{miss[:4]}" if miss else "48/48")

    lines += ["", "## E 原始数据在本项目内且与官方清单一致", ""]
    raw = PROJECT_ROOT / "data" / "seu"
    files_json = raw / "_files.json"
    chk(files_json.exists(), "原始数据清单存在", str(files_json.relative_to(PROJECT_ROOT)))
    if files_json.exists():
        official = {x["path"]: x["sha"] for x in json.loads(files_json.read_text(encoding="utf-8"))}
        mismatch, checked = [], 0
        for rel, sha in official.items():
            p = raw / rel
            if not p.exists():
                continue
            h = hashlib.sha1()
            h.update(b"blob %d\0" % p.stat().st_size)
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            checked += 1
            if h.hexdigest() != sha:
                mismatch.append(rel)
        chk(not mismatch, f"{checked} 个原始 CSV 与官方 git blob sha 一致",
            f"不一致：{mismatch}" if mismatch else "")

    lines += ["", "## 汇总", "",
              f"- 检查项：{pass_n} 通过 / {fail_n} 失败",
              f"- 论文数字涉及的 run 总数：{total_runs}",
              f"- 协议分布：{dict(protos)}",
              f"- 期望协议：{name}（version {version}）", ""]
    out = PROJECT_ROOT / "reports" / "paper_numbers_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ok] {out.relative_to(PROJECT_ROOT)}")
    return 1 if fail_n else 0


if __name__ == "__main__":
    raise SystemExit(main())
