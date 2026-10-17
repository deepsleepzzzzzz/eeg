"""所有源码模块必须能被 Python 解析（语法/缩进错误零容忍）。

背景：2026-10-16 我给 scan_variants.py 打补丁时把缩进改坏了（IndentationError），
但 46 项测试全绿 —— 因为没有任何测试会执行那个脚本。直到真正跑扫描才暴露。

实现说明：这里**不写 .pyc、也不用 pytest 的 tmp_path**。
早前一次沙箱内运行把 %TEMP%\pytest-of-liang 的权限弄坏，凡是用 tmp_path 的测试都会
PermissionError；而 compile() 只做解析、不落盘，既更稳也更快。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"


def test_all_source_modules_compile():
    files = sorted(SRC.rglob("*.py"))
    assert files, "src/ 下没有找到 Python 文件"
    bad = []
    for f in files:
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except SyntaxError as e:
            bad.append(f"{f.relative_to(PROJECT_ROOT)}: 第 {e.lineno} 行 {e.msg}")
    assert not bad, "以下模块无法编译：\n" + "\n".join(bad)


def test_top_level_scripts_compile():
    """仓库根目录下的脚本（如 verify_all.py）也要能解析 —— 它们不被 src/ 覆盖。"""
    bad = []
    for f in sorted(PROJECT_ROOT.glob("*.py")):
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except SyntaxError as e:
            bad.append(f"{f.name}: 第 {e.lineno} 行 {e.msg}")
    assert not bad, "以下脚本无法编译：\n" + "\n".join(bad)
