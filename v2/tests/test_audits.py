"""把两个审计脚本和变异测试纳入 pytest —— 这样 'pytest tests' 就是全部检查。

为什么要这样：审计脚本此前是独立脚本，只在我记得手动运行时才跑。
包进 pytest 后，pre-commit 钩子跑 pytest 就等于跑了全部，无法遗漏。

审计脚本以【子进程】方式运行并检查退出码（它们用 SystemExit 表示失败，
直接在 pytest 里 import 会打断整个测试进程）。
"""
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

PY = sys.executable


def _run(script: str, encoding: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if encoding:
        # 故意用 GBK：脚本必须自己把输出切成 UTF-8，
        # 否则回显外部输出时会抛 UnicodeEncodeError 被误判成检查失败。
        env["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [PY, str(PROJECT_ROOT / "src" / script)],
        capture_output=True, cwd=str(PROJECT_ROOT), env=env,
    )


def _tail(res: subprocess.CompletedProcess, n: int = 12) -> str:
    out = (res.stdout + res.stderr).decode("utf-8", errors="replace")
    return "\n".join(out.splitlines()[-n:])


def test_audit_main_passes():
    res = _run("audit_main.py")
    assert res.returncode == 0, "端到端审计未通过：\n" + _tail(res)


def test_audit_clean_passes():
    res = _run("audit_clean.py")
    assert res.returncode == 0, "干净性审计未通过：\n" + _tail(res)


def test_audit_detects_injected_faults():
    """变异测试：审计必须能抓到注入的错误，且必须把缓存还原干净。

    这是唯一能证明'审计本身还有效'的检查（历史上审计失过效三次）。
    """
    res = _run("audit_mutation.py", encoding="gbk")
    out = (res.stdout + res.stderr).decode("utf-8", errors="replace")
    assert res.returncode == 0, "变异测试未通过（审计有盲区或缓存未还原）：\n" + _tail(res)
    assert "变异检出: 4/4" in out, "变异未被全部检出：\n" + _tail(res)
    assert "还原校验: 通过" in out, "变异测试未把缓存还原干净：\n" + _tail(res)
