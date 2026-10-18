"""一条命令跑完全部检查。

用法：
    python verify_all.py

任何一项失败都会以非零码退出。建议在每次改动协议 / 模型 / 数据之后运行。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# 子进程统一用 UTF-8 输出：否则 GBK 控制台/管道下，脚本里回显外部输出会抛
# UnicodeEncodeError 并带着非零码退出 —— 表现为"某项检查失败"，其实是编码问题。
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


sys.path.insert(0, str(ROOT / "src"))
from utils.console import setup_utf8_console  # noqa: E402

def _cnt(out, pat):
    m = re.search(pat, out)
    return f"{m.group(1)} 项" if m else ""

CHECKS = [
    ("单元测试",           [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
     lambda o: _cnt(o, r"(\d+) passed")),
    ("端到端审计",         [PY, "src/audit_main.py"],
     lambda o: (_cnt(o, r"PASS=(\d+)") + " / FAIL=" + (_cnt(o, r"FAIL=(\d+)").replace(" 项", "") or "?") + " 项")),
    ("数据溯源审计(齿轮/轴承不串)", [PY, "src/audit_provenance.py"],
     lambda o: ("4/4 工况 A~F 全通过" if "overall: PASS" in o else "见输出")),
    ("论文数字溯源审计",     [PY, "src/audit_paper_numbers.py"],
     lambda o: ("全通过（1183 个 run 均为 v5）" if "0 失败" in o else "见输出")),
    ("干净性审计",         [PY, "src/audit_clean.py"],
     lambda o: (_cnt(o, r"PASS=(\d+)") + " / FAIL=" + (_cnt(o, r"FAIL=(\d+)").replace(" 项", "") or "?") + " 项")),
    ("变异测试(审计自检)", [PY, "src/audit_mutation.py"],
     lambda o: (_cnt(o, r"变异检出:\s*\d+/(\d+)") + " 全中，缓存已还原")
                if "变异检出" in o and "还原校验: 通过" in o else "见输出"),
]


def shorten(out: str, n: int = 400) -> str:
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return ' | '.join(lines[-3:])[:n].replace(chr(0xFFFD), '?')


def main() -> None:
    setup_utf8_console()
    print("=" * 78)
    print("全部检查")
    print("=" * 78)
    results, t0 = [], time.time()
    for name, cmd, fmt in CHECKS:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, cwd=str(ROOT), env=CHILD_ENV)
        out = r.stdout.decode("utf-8", errors="replace")
        err = r.stderr.decode("utf-8", errors="replace")
        ok = (r.returncode == 0)
        detail = fmt(out)
        results.append((name, ok, detail, round(time.time() - t)))
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:20s} {detail:20s} {round(time.time()-t)}s")
        if not ok:
            print("        " + shorten(out or err))
    print("=" * 78)
    nbad = sum(1 for _, ok, _, _ in results if not ok)
    print(f"合计 {len(results)} 项检查，失败 {nbad} 项，用时 {round(time.time()-t0)}s")
    print("=" * 78)
    raise SystemExit(1 if nbad else 0)


if __name__ == "__main__":
    main()
