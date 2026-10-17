"""测试用的当前主协议加载器：自动解析 status: main，不写死文件名。

测试里的窗口/步长/窗数/划分一律从这里取（硬规则第 2 条：协议是唯一真源）。
写死过参数的测试已经踩过坑：协议从 1600/800 升到 2048/1024 后，
test_split_invariants 仍按 1600/800 断言 916/195/195，看上去"全绿"其实测的不是当前协议。
"""
from __future__ import annotations

from pathlib import Path

from config.protocol import load_protocol, window_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main_protocol_path() -> Path:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def main_protocol() -> dict:
    return load_protocol(main_protocol_path())


def main_window() -> tuple[int, int]:
    return window_cfg(main_protocol())
