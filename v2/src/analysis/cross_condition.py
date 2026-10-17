"""跨工况迁移：20 Hz-0 V 训练 -> 30 Hz-2 V 验证（齿轮任务）。

为什么值得跑：符号化把绝对幅值映射成 6 个相对档位，理论上对"负载/转速漂移"更鲁棒。
若本文方法跨工况掉得比原始波形基线慢，"符号表示对工况漂移更鲁棒"就是一个可辩护的主张；
若两边一起崩，则说明模型严重依赖幅值信息 —— 与本项目发现的"噪声口径捷径"互相印证。

实现：把 20-0 的 train 与 30-2 的 val 拼成一份合成缓存
      artifacts/cross_condition/dataset/<task>/20_0/，再用与主实验完全相同的
      train_one() 训练（同优化器/早停/预算）。唯一变化是验证集来自另一个工况。

用法：python src/analysis/cross_condition.py --task gear --direction 20_0_to_30_2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from config.protocol import load_protocol  # noqa: E402
from training.train import train_one  # noqa: E402

RAW_MODELS = {
    "models.raw_cnn": {"arch": "raw_cnn", "conv_channels": [32, 64, 128], "kernel_size": 7, "dropout": 0.2},
    "models.wdcnn": {"arch": "wdcnn", "dropout": 0.2},
    "models.resnet1d": {"arch": "resnet1d", "dropout": 0.2},
}


def find_main_protocol() -> Path:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def find_raw_cache(task: str, condition: str) -> Path | None:
    """从 clean 基线变体的 manifest 里反查原始波形数据缓存目录。"""
    root = PROJECT_ROOT / "artifacts" / "variant_scan" / "clean"
    if not root.exists():
        return None
    for run in sorted(root.iterdir()):
        mf = run / "manifest.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text(encoding="utf-8"))
        ds = meta.get("dataset_meta")
        if not ds:
            continue
        p = Path(ds)
        if p.exists() and f"\\{task}\\" in str(p) and f"\\{condition}\\" in str(p):
            return p.parent
    return None


def synth_cache(src_train: Path, src_eval: Path, dest: Path, note: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name, src in (("X_train", src_train), ("y_train", src_train), ("X_val", src_eval), ("y_val", src_eval)):
        np.save(dest / f"{name}.npy", np.load(src / f"{name}.npy"))
    meta = json.loads((src_train / "meta.json").read_text(encoding="utf-8"))
    meta["cross_condition"] = note
    meta["eval_from"] = str(src_eval)
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gear")
    ap.add_argument("--direction", default="20_0_to_30_2")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--models", default="cnn,gcn,raw_cnn,wdcnn,resnet1d")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src_cond, eval_cond = args.direction.split("_to_")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    main_proto_path = find_main_protocol()
    base_proto = load_protocol(main_proto_path)
    safe_print(f"[cross] 基准协议 {main_proto_path.name}  方向 {args.direction}")

    # 两套来源缓存
    trans_train = PROJECT_ROOT / "artifacts" / "main_70_15_15_v5" / "dataset" / args.task / src_cond
    trans_eval = PROJECT_ROOT / "artifacts" / "main_70_15_15_v5" / "dataset" / args.task / eval_cond
    raw_train = find_raw_cache(args.task, src_cond)
    raw_eval = find_raw_cache(args.task, eval_cond)
    for lab, d in (("纯转移 train", trans_train), ("纯转移 eval", trans_eval), ("原始波形 train", raw_train), ("原始波形 eval", raw_eval)):
        safe_print(f"  {lab:14s} {'OK  ' if d and Path(d).exists() else 'MISS'} {d}")

    rows = []
    for model in models:
        is_raw = model.startswith("raw_") or model in ("wdcnn", "resnet1d")
        proto = json.loads(json.dumps(base_proto))          # 深拷贝
        proto["dataset_cache_id"] = "cross_condition"
        if is_raw:
            proto["representation"] = dict(proto["representation"], kind="raw")
            proto.setdefault("models", {})
            for k, v in RAW_MODELS.items():
                proto["models"][k.split(".", 1)[1]] = v
        src_tr, src_ev = (raw_train, raw_eval) if is_raw else (trans_train, trans_eval)
        if not (src_tr and Path(src_tr).exists() and src_ev and Path(src_ev).exists()):
            safe_print(f"  [skip] {model}: 缓存缺失")
            continue
        for seed in seeds:
            cache = PROJECT_ROOT / "artifacts" / "cross_condition" / "dataset" / args.task / src_cond
            synth_cache(Path(src_tr), Path(src_ev), cache,
                        f"{args.direction} rep={'raw' if is_raw else 'transition'}")
            exp = {"experiment_id": f"{args.task}_{model}_{src_cond}to{eval_cond}_seed{seed}",
                   "protocol": "cross_condition", "comparison_group": "cross_condition",
                   "task": args.task, "condition": src_cond, "model": model, "seed": seed}
            out_root = PROJECT_ROOT / "artifacts" / "cross_condition" / model
            try:
                r = train_one(exp, proto, evaluate_test=False, confirm_test=False,
                              output_root=out_root, device=args.device)
                acc = 100 * float(r["best_val_acc"])
                rows.append({"model": model, "seed": seed, "acc": acc})
                safe_print(f"  {model:10s} seed{seed}  跨工况验证准确率 {acc:6.2f}%")
            except Exception as exc:  # noqa: BLE001
                safe_print(f"  [FAIL] {model} seed{seed}: {type(exc).__name__}: {str(exc)[:140]}")

    out = PROJECT_ROOT / "reports" / f"cross_condition_{args.task}_{args.direction}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"\n-> {out}")


if __name__ == "__main__":
    main()
