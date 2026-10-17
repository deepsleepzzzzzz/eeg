"""防陈旧守卫：审查/测试代码本身不许写死旧协议、旧参数、假检查。

用户要求"保证所有审查代码没问题、是与时俱进的、不是旧的"。项目历史上真的踩过：
  * 审计写死 v2 → "24 项全过"实际审的是旧协议；
  * test_split_invariants 写死 1600/800 与 916/195/195 → 协议升到 2048/1024 后仍"全绿"；
  * audit_clean 里 `len(sealed) == 0 or True` → 永远通过的假检查。
这些都不会让代码报错，只会让检查**悄悄失效**，所以必须用测试盯住。

规则（都在下面 RULES / 各 test 里）：
  1. src/**/*.py 不许出现协议文件名（main_70_15_15…）必须动态解析 status: main
     （例外：audit_main.py 里有一张"归档目录名"映射表，见注释）；
  2. src 与 tests 不许有永远为真的假检查/死代码：`or True` / `if False` / `assert True`；
  3. tests/ 不许写死旧窗长 `1600, 800`（当前主协议是 2048/1024，窗口参数一律从 _protocol 取）；
  4. 不许出现绝对路径（盘符）；
  5. 每个 src/audit_*.py 必须自己动态解析主协议。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
TESTS = PROJECT_ROOT / "tests"

#: 例外（每条都要有理由）
ALLOW_PROTOCOL_LITERAL = {
    "audit_main.py",   # LEGACY_ALIAS：把归档时改过名的旧协议目录映射回来，必须写名字
}


def _code_only(text: str) -> str:
    """剥掉 docstring 与整行注释后再扫。

    为什么：本文件与别处的"规则说明"会写在 docstring 里（里面当然会出现 or True、旧协议名等示例），
    那不是风险；真正危险的是**代码里**用它们。用 ast 精确定位 docstring 的行区间删掉，
    保留其它字符串字面量 —— 因为 `load_protocol("...main_70_15_15_v2.yaml")` 这类正是要抓的。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.splitlines()
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                for ln in range(first.lineno, (getattr(first, "end_lineno", first.lineno) or first.lineno) + 1):
                    drop.add(ln)
    return "\n".join("" if (i in drop or l.strip().startswith("#")) else l
                      for i, l in enumerate(lines, 1))


#: 本文件必须写出这些"坏模式"才能检查它们，故扫描时跳过自身
SELF = Path(__file__).resolve()


def _files(base: Path):
    for p in base.rglob("*.py"):
        if "__pycache__" in p.parts or p.resolve() == SELF:
            continue
        yield p


@pytest.mark.parametrize("rel", ["src", "tests"])
def test_no_hardcoded_protocol_filename(rel):
    bad = []
    for p in _files(PROJECT_ROOT / rel):
        if p.name in ALLOW_PROTOCOL_LITERAL:
            continue
        for i, l in enumerate(_code_only(p.read_text(encoding="utf-8", errors="replace")).splitlines(), 1):
            if l.strip().startswith("#"):
                continue
            if re.search(r"main_70_15_15\w*\.ya?ml", l):
                bad.append(f"{p.relative_to(PROJECT_ROOT)}:{i}  {l.strip()[:90]}")
    assert not bad, ("写死了协议文件名（应改为自动解析 status: main）：\n" + "\n".join(bad))


@pytest.mark.parametrize("rel", ["src", "tests"])
def test_no_vacuous_checks_or_dead_code(rel):
    bad = []
    for p in _files(PROJECT_ROOT / rel):
        for i, l in enumerate(_code_only(p.read_text(encoding="utf-8", errors="replace")).splitlines(), 1):
            if l.strip().startswith("#"):
                continue
            if re.search(r"\bor True\b|if False|assert True\b", l):
                bad.append(f"{p.relative_to(PROJECT_ROOT)}:{i}  {l.strip()[:90]}")
    assert not bad, ("出现永远为真的假检查或死代码：\n" + "\n".join(bad))


def test_tests_do_not_hardcode_legacy_window():
    """tests/ 里不许写死旧窗长 1600/800（当前主协议是 2048/1024）。"""
    bad = []
    for p in _files(TESTS):
        for i, l in enumerate(_code_only(p.read_text(encoding="utf-8", errors="replace")).splitlines(), 1):
            if l.strip().startswith("#"):
                continue
            # 只抓"调用点/形状"里的旧参数；用命名常量定义历史值是允许的
            if re.search(r"window_starts\([^)]*\b1600\s*,\s*800|"
                         r"\(\s*3\s*,\s*1600\s*\)", l):
                bad.append(f"{p.relative_to(PROJECT_ROOT)}:{i}  {l.strip()[:90]}")
    assert not bad, ("tests 里写死了旧窗长/旧形状（应从 _protocol 或命名常量取）：\n" + "\n".join(bad))


def test_no_absolute_paths_in_code():
    bad = []
    for base in (SRC, TESTS):
        for p in _files(base):
            for i, l in enumerate(_code_only(p.read_text(encoding="utf-8", errors="replace")).splitlines(), 1):
                if l.strip().startswith("#"):
                    continue
                if re.search(r"[\"'][A-Za-z]:\\\\", l):
                    bad.append(f"{p.relative_to(PROJECT_ROOT)}:{i}  {l.strip()[:90]}")
    assert not bad, ("代码里出现绝对路径（换机器/换目录会静默读别处）：\n" + "\n".join(bad))


def test_every_audit_resolves_protocol_dynamically():
    """每个 src/audit_*.py 都必须自己解析 status: main（否则升级协议后审的还是旧协议）。"""
    missing = []
    for p in sorted(SRC.glob("audit_*.py")):
        t = p.read_text(encoding="utf-8", errors="replace")
        if '"status"' not in t and "status" not in t:
            missing.append(p.name)
    assert not missing, f"这些审计脚本没有动态解析主协议：{missing}"
