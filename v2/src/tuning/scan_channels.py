"""通道组合对比：建转移矩阵表示 -> 测可分性 -> 可选训练。

只读训练段, 不碰测试集。
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import dataset_dir, load_protocol  # noqa: E402
from data.build_dataset import build_condition  # noqa: E402
from data.seu_dds import class_names  # noqa: E402

def _main_protocol():
    """自动解析 status: main 的主协议（**不写死文件名**，硬规则第 2 条）。

    历史问题：这里曾写死 main_70_15_15_v2.yaml，升级到 v5 后通道扫描仍在用旧协议，
    得到的是旧窗长/旧滞后下的结论。
    """
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


BASE = load_protocol(_main_protocol())
TASK, COND = "gear", "30_2"
COMBOS = {
    "cur_123": [1, 2, 3],
    "add0_0123": [0, 1, 2, 3],
    "par_123456": [1, 2, 3, 4, 5, 6],
    "c0_0123456": [0, 1, 2, 3, 4, 5, 6],
    "all_01234567": [0, 1, 2, 3, 4, 5, 6, 7],
}


def cv_acc(X, y, reps=3):
    X = X.reshape(len(X), -1).astype(np.float64)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, reps, size=len(X))
    c = 0
    for k in range(reps):
        tr, te = fold != k, fold == k
        cen = np.stack([X[tr & (y == i)].mean(0) for i in range(5)])
        d = np.linalg.norm(X[te][:, None, :] - cen[None, :, :], axis=2)
        c += int((d.argmin(1) == y[te]).sum())
    return c / len(X)


def main():
    print(f"{'组合':>14}{'通道':>22}{'特征形状':>18}{'CV 准确率':>12}")
    out = {}
    for tag, chs in COMBOS.items():
        proto = deepcopy(BASE)
        proto["dataset"]["channel_indices"] = chs
        proto["dataset_cache_id"] = f"chprobe_{tag}"
        d = dataset_dir(proto, TASK, COND)
        if not (d / "X_train.npy").exists():
            print(f"\n>>> 构建 {tag} channels={chs}", flush=True)
            build_condition(proto, TASK, COND, force=True)
        X = np.load(d / "X_train.npy"); y = np.load(d / "y_train.npy")
        acc = cv_acc(X, y)
        out[tag] = {"channels": chs, "shape": list(X.shape), "cv_acc": acc}
        print(f"{tag:>14}{str(chs):>22}{str(X.shape):>18}{acc:>12.4f}", flush=True)
    (PROJECT_ROOT / "reports" / "scan_channels.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写盘 reports/scan_channels.json")


if __name__ == "__main__":
    main()
