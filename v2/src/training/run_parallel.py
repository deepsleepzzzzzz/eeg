"""CPU 多进程并行跑主实验。

为什么用 CPU：模型只有 2 万参数左右（输入 56x6x6），GPU 完全被启动开销主导。
实测单组耗时 GPU 16 秒、CPU 单线程 18 秒 —— 几乎一样。
但 CPU 可以开 16 个进程并行，40 组约 1 分钟跑完，而串行 GPU 要 20+ 分钟。

一致性：本脚本把 CUDA 关掉（CUDA_VISIBLE_DEVICES=""），全部跑在 CPU 上，
并固定 torch 线程数为 1，保证同一次实验内所有 run 走完全相同的计算路径。
（实测 CPU 与 GPU 的 val_predictions.csv 逐字节相同，只是 epoch 记账略有差异。）
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from config.protocol import artifact_root  # noqa: E402
from training.train import load_yaml, train_one  # noqa: E402

HEADER = [
    "experiment_id", "protocol", "comparison_group", "task", "condition", "model", "seed",
    "best_epoch", "best_val_acc", "plateau_acc", "plateau_std", "train_val_gap",
    "epochs_run", "final_lr", "metrics_path", "history_path", "prediction_path", "status",
]


_INTEROP_SET = False


def _pin_threads() -> None:
    """CPU 模式下每个进程只设一次 interop 线程数（torch 不允许重复设置）。"""
    global _INTEROP_SET
    if str(os.environ.get("SEU_DEVICE", "cpu")).lower() != "cpu":
        return
    torch.set_num_threads(1)          # 可重复调用
    if not _INTEROP_SET:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass                       # 已经设过，忽略
        _INTEROP_SET = True


def run_job(job: dict) -> dict:
    _pin_threads()
    device = str(os.environ.get("SEU_DEVICE", "cpu")).lower()
    if device == "cuda":
        assert torch.cuda.is_available(), (
            "要求 device=cuda 但 torch.cuda.is_available() 为 False"
        )
    protocol_cfg = load_yaml(Path(job["protocol"]))
    experiment_cfg = {
        "experiment_id": job["experiment_id"],
        "protocol": job["protocol_name"],
        "comparison_group": job["group"],
        "task": job["task"],
        "condition": job["condition"],
        "model": job["model"],
        "seed": job["seed"],
    }
    result = train_one(experiment_cfg, protocol_cfg, evaluate_test=False, confirm_test=False,
                       device="cpu")
    st = result.get("stability", {}) or {}
    run_dir = artifact_root(protocol_cfg) / job["protocol_name"] / job["experiment_id"]
    return {
        "experiment_id": job["experiment_id"],
        "protocol": job["protocol_name"],
        "comparison_group": job["group"],
        "task": job["task"],
        "condition": job["condition"],
        "model": job["model"],
        "seed": job["seed"],
        "best_epoch": result["best_epoch"],
        "best_val_acc": round(float(result["best_val_acc"]), 6),
        "plateau_acc": round(float(st.get("last20_val_acc_mean", 0.0)), 6),
        "plateau_std": round(float(st.get("last20_val_acc_std", 0.0)), 6),
        "train_val_gap": round(float(st.get("gap_train_minus_val_acc", 0.0)), 6),
        "epochs_run": st.get("epochs_run"),
        "final_lr": st.get("final_lr"),
        "metrics_path": str(run_dir / "metrics.json"),
        "history_path": str(run_dir / "history.json"),
        "prediction_path": str(run_dir / "val_predictions.csv"),
        "status": "done",
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--tasks", default="bearing,gear")
    p.add_argument("--conditions", default="20_0,30_2")
    p.add_argument("--models", default="cnn,gcn")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--group", default="main_clean_compare")
    p.add_argument("--registry", default=None)
    p.add_argument("--device", default=None, help="cpu / cuda")
    args = p.parse_args()

    if args.device:
        os.environ["SEU_DEVICE"] = args.device
    protocol_cfg = load_yaml(Path(args.protocol))
    protocol_name = str(protocol_cfg["name"])
    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]

    jobs = []
    for task in split(args.tasks):
        for cond in split(args.conditions):
            for model in split(args.models):
                for seed in [int(s) for s in split(args.seeds)]:
                    jobs.append({
                        "experiment_id": f"{task}_{model}_{cond}_seed{seed}",
                        "protocol": str(Path(args.protocol).resolve()),
                        "protocol_name": protocol_name,
                        "group": args.group,
                        "task": task, "condition": cond, "model": model, "seed": seed,
                    })

    registry = Path(args.registry) if args.registry else (PROJECT_ROOT / "registry.csv")
    print(f"[parallel] {len(jobs)} 个任务, {args.workers} 个进程, 设备={os.environ.get('SEU_DEVICE','cpu')}")
    t0 = time.time()
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                done += 1
                print(f"[{done}/{len(jobs)}] {row['experiment_id']} "
                      f"best={row['best_val_acc']:.4f} plateau={row['plateau_acc']:.4f}", flush=True)
            except Exception as exc:
                print(f"[FAIL] {j['experiment_id']}: {type(exc).__name__}: {exc}", flush=True)

    order = {j["experiment_id"]: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[r["experiment_id"]])
    with registry.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"[parallel] 完成 {len(rows)}/{len(jobs)}, 用时 {time.time()-t0:.0f} 秒")
    print(f"[parallel] registry -> {registry}")


if __name__ == "__main__":
    main()
