"""表示层可分性扫描：不训练，只判断某个表示参数能否把类别在特征空间里分开。

用法示例：
    python src/tuning/scan_representation.py --protocol configs/protocols/<当前主协议>.yaml \
        --task gear --condition 30_2 --param preprocessing.bins --values 6,8,10,12
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.protocol import dataset_dir, load_protocol  # noqa: E402
from data.build_dataset import build_condition  # noqa: E402
from data.seu_dds import class_names  # noqa: E402


def set_nested(cfg: dict, dotted: str, value) -> None:
    section, name = dotted.split(".", 1)
    cfg[section][name] = value


def separability(X: np.ndarray, y: np.ndarray, n_classes: int):
    """返回 (类间距/类内半径 矩阵, 类内半径, 最近类中心 3 折 CV 准确率)。"""
    Xf = X.reshape(len(X), -1).astype(np.float64)
    means = np.stack([Xf[y == i].mean(0) for i in range(n_classes)])
    within = np.array([np.linalg.norm(Xf[y == i] - means[i], axis=1).mean() for i in range(n_classes)])
    ratio = np.full((n_classes, n_classes), np.nan)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            d = np.linalg.norm(means[i] - means[j])
            ratio[i, j] = ratio[j, i] = d / (0.5 * (within[i] + within[j]))

    # 最近类中心分类，3 折交叉验证
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 3, size=len(Xf))
    correct = 0
    for k in range(3):
        tr, te = fold != k, fold == k
        cen = np.stack([Xf[tr & (y == i)].mean(0) for i in range(n_classes)])
        dist = np.linalg.norm(Xf[te][:, None, :] - cen[None, :, :], axis=2)
        correct += int((dist.argmin(1) == y[te]).sum())
    return ratio, within, correct / len(Xf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--param", required=True, help="如 preprocessing.bins")
    p.add_argument("--values", required=True, help="逗号分隔")
    p.add_argument("--tag", default=None, help="缓存目录前缀，默认 probe")
    p.add_argument("--reuse", action="store_true", help="缓存已存在则不重建")
    args = p.parse_args()

    base = load_protocol(Path(args.protocol))
    names = list(class_names(args.task))
    n = len(names)
    values = [float(x) if "." in x else int(x) for x in args.values.split(",")]
    tag = args.tag or f"probe_{args.param.split('.')[-1]}"

    results = []
    for v in values:
        proto = deepcopy(base)
        set_nested(proto, args.param, v)
        proto["dataset_cache_id"] = f"{tag}_{v}"
        out = dataset_dir(proto, args.task, args.condition)
        if not (args.reuse and (out / "X_train.npy").exists()):
            print(f"\n>>> 重建缓存 {args.param}={v} -> {proto['dataset_cache_id']}", flush=True)
            build_condition(proto, args.task, args.condition, force=True)
        X = np.load(out / "X_train.npy")
        y = np.load(out / "y_train.npy")
        ratio, within, cv_acc = separability(X, y, n)
        results.append({"value": v, "shape": list(X.shape), "ratio": ratio.tolist(),
                        "within": within.tolist(), "cv_acc": cv_acc})
        print(f"\n===== {args.param} = {v}   特征形状 {X.shape} =====")
        print(f"  最近类中心 3 折 CV 准确率: {cv_acc:.4f}")
        print("  类间距/类内半径:")
        print("            " + "".join(f"{nm[:8]:>10}" for nm in names))
        for i, nm in enumerate(names):
            row = "".join("         -" if i == j else f"{ratio[i, j]:>10.3f}" for j in range(n))
            print(f"  {nm[:8]:>10}" + row)
        hard = sorted(((ratio[i, j], names[i], names[j]) for i in range(n) for j in range(i + 1, n)))
        print("  最难分三对: " + ", ".join(f"{a}-{b} {r:.3f}" for r, a, b in hard[:3]))

    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    hard_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # 固定报告 Miss/Root/Surface 这类"最难三对"
    ref = sorted(range(n), key=lambda i: names[i])
    print(f"{args.param:>20}{'CV acc':>10}{'最难对1':>10}{'最难对2':>10}{'最难对3':>10}")
    for r in results:
        rt = np.array(r["ratio"])
        hard = sorted((rt[i, j], names[i], names[j]) for i, j in hard_idx)[:3]
        print(f"{str(r['value']):>20}{r['cv_acc']:>10.4f}"
              + "".join(f"{h[0]:>10.3f}" for h in hard))
    out_json = PROJECT_ROOT / "reports" / f"scan_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"param": args.param, "task": args.task,
                                    "condition": args.condition, "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写盘: {out_json}")


if __name__ == "__main__":
    main()
