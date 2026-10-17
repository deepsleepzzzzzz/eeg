"""变体扫描：在当前主协议（自动解析 status: main）基线上试各种「已实现但未启用」的开关。

每个变体：
  1. 深拷贝主协议，按 dotted key 覆盖参数
  2. 若变体改动了「数据相关」的配置（dataset/split/window/preprocessing/representation），
     就用内容哈希新建一个数据缓存；否则复用主协议的缓存
  3. 并行训练，只读 train/val，不碰测试集

用法：
    python src/tuning/scan_variants.py --workers 16 --seeds 42,43
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from config.protocol import artifact_root, load_protocol  # noqa: E402
from data.build_dataset import build_condition  # noqa: E402
from training.train import train_one  # noqa: E402

def find_main_protocol() -> Path:
    """自动解析当前主协议（configs/protocols/main_*.yaml 里 status: main 的那个）。

    绝不写死文件名 —— 否则升级协议（v2 -> v4）后脚本会悄悄用旧协议。
    """
    cands = []
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            cands.append((f, cfg.get("version", "?")))
    if len(cands) != 1:
        raise RuntimeError(f"必须且只能有一个 status=main 的主协议，实际找到 {[c[0].name for c in cands]}")
    return cands[0][0]


BASE_PROTOCOL = find_main_protocol()
DATA_KEYS = ("dataset", "split", "window", "preprocessing", "representation", "noise")

# ---- 变体定义：dotted key -> 值 -------------------------------------------------
def load_variants(path: Path) -> dict:
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True) or {}
    return dict(cfg.get("variants") or {})


VARIANTS: dict[str, dict] = {}          # 由 --variants 载入
OUT_NAME = "variant_scan"               # 由 --out-name 覆盖


def set_nested(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value


# 这几项只影响模型或只是文档，不影响数据内容，不计入缓存指纹
_NON_DATA_REP_KEYS = ("attention", "channels", "input_shape")


def data_id(protocol: dict) -> str:
    payload = {k: protocol.get(k) for k in DATA_KEYS}
    rep = dict(payload.get("representation") or {})
    for k in _NON_DATA_REP_KEYS:
        rep.pop(k, None)
    payload["representation"] = rep
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "var_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# 历史说明：v2 协议的 models.gcn.gcn_steps=3 与它的 representation.lags(7) 不符（语义错误，见
# legacy/README.md）。当时这里用 BASE_FIXES 强行改成 7。**现在不再需要**：当前主协议已在 yaml 里
# 显式声明 gcn_steps，且下面的 build_variant_protocol 会按 lags 重新推导；写死 7 反而会在换协议时
# 静默改变 GCN 语义（违反硬规则第 1 条「协议是唯一真源」）。


def build_variant_protocol(name: str, overrides: dict) -> dict:
    proto = deepcopy(load_protocol(BASE_PROTOCOL))
    for k, v in overrides.items():
        set_nested(proto, k, v)
    # GCN 的 relational 布局要求 gcn_steps == 滞后数，改了 lags 就自动跟随
    lags = proto.get("representation", {}).get("lags")
    if lags:
        set_nested(proto, "models.gcn.gcn_steps", len(lags))
    proto["dataset_cache_id"] = data_id(proto)
    return proto


def _build_one_cache(job: dict) -> str:
    proto = build_variant_protocol(job["name"], job["overrides"])
    build_condition(proto, job["task"], job["condition"], force=False)
    return f'{job["name"]:16s} cache={job["cid"]} ok'


def cv_worker(job: dict) -> dict:
    device = str(os.environ.get("SEU_DEVICE", "cpu")).lower()
    if device == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    else:
        # 显式校验：声称用 GPU 就必须真的用上 GPU，否则早失败
        assert torch.cuda.is_available(), (
            "要求 device=cuda 但 torch.cuda.is_available() 为 False —— "
            "检查是否被 CUDA_VISIBLE_DEVICES 屏蔽了"
        )
    proto = build_variant_protocol(job["variant"], job["overrides"])
    exp = {
        "experiment_id": f"{job['task']}_{job['model']}_{job['condition']}_{job['variant']}_seed{job['seed']}",
        # 基准协议名。此处曾写死 "main_70_15_15_v2"。
        # **更正（2026-10-16 复核）**：该字段实际**没有被 train.py 使用** —— train_one() 只取
        # task/condition/model/seed/experiment_id（见 train.py:87-91），写进 metrics/manifest 的
        # protocol 一律来自 protocol_cfg["name"]（train.py:92,226,254）。所以已有的 822 个
        # variant_scan run 的 protocol 字段是**正确的**（main_70_15_15_v5），不是"被标成 v2"。
        # 仍然要改掉写死的名字：它一旦被下游用到就会误导（硬规则第 2 条）。
        "protocol": str(proto["name"]),
        "comparison_group": "variant_scan",
        "task": job["task"], "condition": job["condition"],
        "model": job["model"], "seed": job["seed"],
    }
    out_root = artifact_root(proto) / job.get("out_name", "variant_scan") / job["variant"]
    r = train_one(exp, proto, evaluate_test=False, confirm_test=False,
                  output_root=out_root, device=device)
    st = r.get("stability", {}) or {}
    return {
        "variant": job["variant"], "model": job["model"], "seed": job["seed"],
        "best_val_acc": float(r["best_val_acc"]),
        "plateau_acc": float(st.get("last20_val_acc_mean", float("nan"))),
        "plateau_std": float(st.get("last20_val_acc_std", float("nan"))),
        "epochs_run": st.get("epochs_run"),
        "overrides": json.dumps(job["overrides"], sort_keys=True),
    }


def archive_foreign_variants(out_root: Path, keep: set) -> list:
    """把本次配置文件【没有定义】的变体目录先归档，再开跑。

    踩过的坑（2026-10-15）：scan_variants 会把结果写进同一个
    artifacts/variant_scan/ 下，后来的扫描会把之前不同变体的结果
    就地覆盖 —— 窗长消融（win1600/win3200）就是这样丢的，
    只能从 git 历史里捞回来。

    归档位置：artifacts/variant_scan_archived/<变体名>/（不会被任何扫描写入）。
    同名目录已存在时不覆盖，改名为 <变体名>_<时间戳>。
    """
    archive_root = out_root.parent / "variant_scan_archived"
    moved = []
    if not out_root.exists():
        return moved
    for d in sorted(out_root.iterdir()):
        if not (d.is_dir() and d.name not in keep):
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / d.name
        if dest.exists():
            dest = archive_root / f"{d.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(d), str(dest))
        moved.append(f"{d.name} -> {dest}")
    return moved


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--models", default="cnn,gcn")
    p.add_argument("--task", default="gear")
    p.add_argument("--condition", default="30_2")
    p.add_argument("--variants", required=True, help="变体定义 yaml（configs/scan/*.yaml）")
    p.add_argument("--only", default=None, help="只跑名字含该子串的变体")
    p.add_argument("--device", default=None, help="cpu / cuda；不给则按 SEU_DEVICE 环境变量")
    p.add_argument("--out-name", default="variant_scan",
                   help="结果子目录名（默认 variant_scan）。给不同名字可让多组扫描结果共存，"
                        "避免触发归档或互相覆盖")
    args = p.parse_args()

    if args.device:
        os.environ["SEU_DEVICE"] = args.device
    print(f"[scan] device={os.environ.get('SEU_DEVICE','cpu')}")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    global VARIANTS, OUT_NAME
    VARIANTS = load_variants(Path(args.variants))
    OUT_NAME = args.out_name
    print(f"[scan] 变体定义: {args.variants}  ({len(VARIANTS)} 个)")
    print(f"[scan] 基准协议: {BASE_PROTOCOL.name}  (自动解析 status=main)")
    keys = [k.strip() for k in args.only.split(",")] if args.only else None
    names = [n for n in VARIANTS if (keys is None or any(k in n for k in keys))]

    jobs = []
    for name in names:
        for model in models:
            for seed in seeds:
                jobs.append({"variant": name, "overrides": VARIANTS[name],
                             "task": args.task, "condition": args.condition,
                             "model": model, "seed": seed,
                             "out_name": args.out_name})

    print(f"[scan] {len(names)} 个变体 x {len(models)} 模型 x {len(seeds)} seed = {len(jobs)} 次训练")
    print(f"[scan] 结果目录: artifacts/{args.out_name}/")

    # 先归档本次配置没定义的变体，避免覆盖（见 archive_foreign_variants 注释）
    out_root = artifact_root(build_variant_protocol(names[0], VARIANTS[names[0]])) / args.out_name
    moved = archive_foreign_variants(out_root, keep=set(VARIANTS.keys()))
    if moved:
        print(f"[scan] 已归档 {len(moved)} 个本次未定义的变体目录（防止覆盖）：", flush=True)
        for m in moved:
            print(f"        {m}", flush=True)
    print(f"[scan] 设备={os.environ.get('SEU_DEVICE','cpu')}, {args.workers} 进程")

    # ---- 阶段 1：并行预建数据缓存（少量并发，控制内存峰值）
    t0 = time.time()
    todo, seen = [], set()
    for name in names:
        proto = build_variant_protocol(name, VARIANTS[name])
        cid = proto["dataset_cache_id"]
        if cid in seen:
            continue
        seen.add(cid)
        todo.append({"name": name, "cid": cid, "overrides": VARIANTS[name],
                     "task": args.task, "condition": args.condition})
    print(f"[scan] 阶段 1/2：预建 {len(todo)} 份数据缓存（4 进程并行）", flush=True)
    with ProcessPoolExecutor(max_workers=min(4, max(1, len(todo)))) as ex:
        for r in ex.map(_build_one_cache, todo):
            print(f"   {r}", flush=True)
    print(f"[scan] 缓存就绪（{len(seen)} 份），进入训练阶段", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cv_worker, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"[FAIL] {j['variant']}/{j['model']}/s{j['seed']}: {type(exc).__name__}: {str(exc)[:120]}")
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    # 汇总 CSV 按 out-name 分开写：历史上扫描汇总一律写 reports/variant_scan.csv，
    # 跑一次新扫描就把上一次的汇总盖掉了（2026-10-16 修）。
    csv_path = out_dir / f"{OUT_NAME}.csv"   # out_dir = reports/

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[scan] {len(rows)}/{len(jobs)} 完成, {time.time()-t0:.0f} 秒 -> {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
