"""守卫：检查结果的真假不能由控制台编码决定。

真实踩过的坑（2026-10-15）：audit_mutation.py 会把 audit_main.py 的输出
（errors="replace" 解码，可能含 U+FFFD）再 print 一次。当子进程 stdout 是
GBK 管道时，print 抛 UnicodeEncodeError → 脚本以非零码退出 →
verify_all.py 报"变异测试失败"，其实是编码问题，不是审计失效。
"""
import os
import subprocess
import sys

from _bootstrap import PROJECT_ROOT

PY = sys.executable

ENTRY_POINTS = [
    "src/audit_main.py",
    "src/audit_clean.py",
    "src/audit_mutation.py",
    "verify_all.py",
]

SNIPPET = (
    "import sys;"
    "sys.path.insert(0, r'{src}');"
    "from utils.console import safe_print, setup_utf8_console;"
    "setup_utf8_console();"
    "safe_print('\ufffd 中文 ± − 混合回显');"
    "print('done')"
)


def test_safe_print_survives_gbk_pipe():
    """GBK 管道下输出替换字符也不能崩。"""
    env = {**os.environ, "PYTHONIOENCODING": "gbk"}
    code = SNIPPET.format(src=str(PROJECT_ROOT / "src"))
    res = subprocess.run([PY, "-c", code], capture_output=True, env=env)
    assert res.returncode == 0, res.stderr.decode("utf-8", errors="replace")
    assert b"done" in res.stdout


def test_entry_points_set_up_utf8_console():
    """审计/检查入口必须显式设置 UTF-8 输出（不能靠环境变量碰运气）。"""
    missing = [
        rel
        for rel in ENTRY_POINTS
        if "setup_utf8_console" not in (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"这些入口没有调用 setup_utf8_console()：{missing}"
