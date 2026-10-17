"""跑 run_tuning 的轻量包装：先限制 torch 线程数，再按同样命令行跑 src/tuning/run_tuning.py。

用途：速度基准测试（configs/tuning/speed_*.yaml）——GPU/CPU 线程数不同时的对比。
    python scripts/speed_wrap.py configs/tuning/speed_gpu.yaml configs/protocols/main_70_15_15_v5.yaml

2026-10-16：本文件原在项目根目录、并把 __file__ 当项目根用；移进 scripts/ 后
改为以**脚本上一级目录**为项目根，位置变了也不会静默读错路径。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SEU_THREADS", "4")))

sys.path.insert(0, str(ROOT / "src"))
sys.argv = ["run_tuning.py", "--grid", sys.argv[1], "--protocol", sys.argv[2]]

target = ROOT / "src" / "tuning" / "run_tuning.py"
src = target.read_text(encoding="utf-8")
ns = {"__name__": "__main__", "__file__": str(target)}
exec(compile(src, "run_tuning.py", "exec"), ns)