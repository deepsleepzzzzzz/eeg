"""控制台输出：强制 UTF-8，避免在 GBK 控制台/管道下崩溃。

真实踩过的坑（2026-10-15）：`audit_mutation.py` 会把 `audit_main.py` 的输出
（用 errors="replace" 解码，可能含 U+FFFD）再 print 一次。当子进程 stdout 是
管道、系统又是 GBK 时，print 抛 UnicodeEncodeError → 脚本以非零码退出，
看起来像"变异测试失败"，实际是编码问题。
后果比崩溃更坏：`verify_all.py` 会报一项假失败，人就会开始怀疑检查本身。

现在：所有审计/检查入口先调 `setup_utf8_console()`，回显外部输出用 `safe_print()`。
"""
from __future__ import annotations

import sys

REPLACEMENT = "?"


def setup_utf8_console() -> None:
    """把 stdout/stderr 切成 UTF-8 + errors='replace'，任何控制台都不再崩。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def sanitize(text: str) -> str:
    """去掉解码失败留下的替换字符，避免它把输出再搞崩一次。"""
    return text.replace("\ufffd", REPLACEMENT)


def safe_print(*args, **kwargs) -> None:
    """print 的保险版：即使输出流编码不支持某些字符也不抛异常。"""
    text = sanitize(" ".join(str(a) for a in args))
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), **kwargs)
