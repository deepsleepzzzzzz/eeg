"""变异测试：往主协议的数据缓存里注入错误，验证审计能否抓到。

为什么需要它：审计脚本本身可能已经失效（历史上就发生过——审计写死了旧协议、
又对"内容级泄漏"和"标签错乱"完全无感）。只有往数据里真注入错误、看审计是否
报警，才能证明审计还有效。

安全设计：
  1. 备份全部缓存文件（含 y_train/y_val/meta），任何异常都在 finally 里还原
  2. 结束时用 SHA256 比对，确认缓存被完整还原；没还原就大声报错
  3. 只动主协议下 gear/30_2 的一份缓存，且只在测试期间
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.console import safe_print, setup_utf8_console

setup_utf8_console()
PY = sys.executable
FILES = ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "meta.json")


def _main_protocol():
    from config.protocol import load_protocol
    for f in sorted((ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def _cache_dir():
    from config.protocol import dataset_dir
    proto = _main_protocol()
    from config.protocol import load_protocol
    return dataset_dir(load_protocol(proto), "gear", "30_2")


def digests(d: Path) -> dict:
    return {f: (hashlib.sha256((d / f).read_bytes()).hexdigest() if (d / f).exists() else "-") for f in FILES}


def run_audit() -> str:
    r = subprocess.run([PY, str(ROOT / "src" / "audit_main.py")], capture_output=True, cwd=str(ROOT))
    txt = r.stdout.decode("utf-8", errors="replace")
    ln = [l.strip() for l in txt.splitlines() if "PASS=" in l]
    return ln[-1] if ln else "??? (审计未输出结果)"


def mutate_dup(d: Path):
    t = np.load(d / "X_train.npy"); v = np.load(d / "X_val.npy"); v[:10] = t[:10]
    np.save(d / "X_val.npy", v)


def mutate_labels(d: Path):
    y = np.load(d / "y_train.npy"); np.random.default_rng(0).shuffle(y)
    np.save(d / "y_train.npy", y)


def mutate_value(d: Path):
    X = np.load(d / "X_train.npy"); X[0, 0, 0, 0] = 0.123456
    np.save(d / "X_train.npy", X)


def mutate_edges(d: Path):
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    m["bin_edges"][0][2] += 1.0
    (d / "meta.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


MUTATIONS = [
    ("验证集混入训练样本(内容级泄漏)", mutate_dup),
    ("打乱训练标签", mutate_labels),
    ("篡改 X_train 一个数值", mutate_value),
    ("篡改 meta 分位边界", mutate_edges),
]


def main() -> None:
    d = _cache_dir()
    # 备份目录放在系统临时目录，绝不落在 git 工作区里（否则会被误提交）
    bak = Path(tempfile.mkdtemp(prefix="seu_mutation_bak_"))
    before = digests(d)
    print(f"主协议 : {_main_protocol().name}")
    print(f"目标缓存: {d}")
    safe_print(f"基线    : {run_audit()}")
    print()
    caught = 0
    try:
        for name, fn in MUTATIONS:
            for f in FILES:
                shutil.copy2(d / f, bak / f)
            try:
                fn(d)
                res = run_audit()
            finally:
                for f in FILES:
                    shutil.copy2(bak / f, d / f)
            ok = "FAIL=0" not in res
            caught += ok
            safe_print(f"  [{'抓到' if ok else '漏掉'}] {name:32s} -> {res}")
    finally:
        shutil.rmtree(bak, ignore_errors=True)

    after = digests(d)
    restored = before == after
    print()
    print(f"缓存还原校验: {'通过（逐文件 SHA256 一致）' if restored else '*** 失败，缓存未还原 ***'}")
    print(f"变异检出: {caught}/{len(MUTATIONS)}")
    bad = (not restored) or caught != len(MUTATIONS)
    print("=" * 60)
    print(f"结果: {'PASS' if not bad else 'FAIL'}")
    print("=" * 60)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
