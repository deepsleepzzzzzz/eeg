"""守卫测试：verify_all.py 的"结果格式化函数"必须健壮。

起因（2026-10-16）：给新检查写的格式化函数里正则少了捕获组，`_cnt()` 里 `m.group(1)`
抛 IndexError，verify_all 跑到第 5 项就崩了 —— 而这件事对 pre-commit **不可见**
（pre-commit 只跑 pytest，不跑 verify_all），所以只在手动运行时才炸。
本测试直接导入 verify_all 并把每个格式化函数喂几种输入，把这类问题挡在提交前。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_verify_all():
    spec = importlib.util.spec_from_file_location("verify_all", PROJECT_ROOT / "verify_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_check_formatter_handles_any_output():
    mod = _load_verify_all()
    samples = [
        "",
        "PASS=44\nFAIL=2\n检查项：34 通过 / 0 失败\n变异检出: 4/4\n还原校验: 通过\noverall: PASS",
        "乱码 abc 123\n[OK  ] 单元测试 73 项",
    ]
    for name, _cmd, fmt in mod.CHECKS:
        for s in samples:
            fmt(s)      # 抛异常即失败


def test_verify_all_checks_are_well_formed():
    mod = _load_verify_all()
    assert mod.CHECKS, "verify_all 里没有任何检查项"
    for name, cmd, fmt in mod.CHECKS:
        assert isinstance(name, str) and name
        assert isinstance(cmd, (list, tuple)) and cmd, name
        assert callable(fmt), name
