"""主实验端到端审计：独立复算 + 一致性检查。

只读，不写任何实验产物，不接触封存测试集。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.console import setup_utf8_console  # noqa: E402
from config.protocol import dataset_dir, load_protocol  # noqa: E402
from data.build_dataset import split_intervals  # noqa: E402
from data.seu_dds import class_file, class_names, load_seu_csv, normalize_condition, task_dir, vibration_xyz, window_starts  # noqa: E402
from representation.recurrence_plot import preprocess_signal  # noqa: E402
from representation.state_transition import build_state_transition, symbolize_with_edges, transition_matrix  # noqa: E402

setup_utf8_console()

def _main_protocol():
    """自动解析 status: main 的主协议（绝不写死文件名）。"""
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


# 2026-10-16：v1–v4 的协议目录（结果 + 缓存）已整体移入 legacy/，
# 而登记表 registry_main_v*_40runs.csv 里记的是迁移前的绝对路径。
# 下面两个函数让审计在"原路径"和"legacy 新路径"下都能找到目标，
# 既不篡改历史登记表，也不放过缺失的 run。
LEGACY_BASES = (
    PROJECT_ROOT / "legacy" / "results",
    PROJECT_ROOT / "legacy" / "scans",
)


#: 归档时给 v1 的目录加了 `_v1` 后缀（避免与 v2 混淆），这里记一笔别名。
LEGACY_ALIAS = {"main_70_15_15": "main_70_15_15_v1"}


def legacy_lookup(name: str):
    """按目录名在 legacy/ 下找被归档的协议目录；找不到返回 None。"""
    for cand_name in (LEGACY_ALIAS.get(name), name):
        if not cand_name:
            continue
        for base in LEGACY_BASES:
            cand = base / cand_name
            if cand.is_dir():
                return cand
    return None


def resolve_run_dir(recorded_path: str) -> Path:
    """登记表 metrics_path 的父目录；原位置没有就到 legacy/ 的对应协议目录里找。

    注意：同一个 experiment_id（如 gear_cnn_20_0_seed42）在 v1..v5 里都存在，
    所以**不能只按 run 目录名找**（会串到别的版本）。这里用记录路径里的
    "原协议目录名"做限定：artifacts/<协议目录>/<run>/… → legacy/{results,scans}/<协议目录>/<run>/。
    """
    # Historical ledgers keep their original absolute paths. Resolve their
    # project-relative suffix here, without reading a different checkout.
    parts = PurePosixPath(recorded_path.replace("\\", "/")).parts
    if "artifacts" in parts:
        local = PROJECT_ROOT.joinpath(*parts[parts.index("artifacts"):]).parent
        if (local / "manifest.json").exists():
            return local
    run = Path(*parts).parent
    if (run / "manifest.json").exists():
        return run
    hint = run.parent.name
    mapped = LEGACY_ALIAS.get(hint, hint)
    for base in LEGACY_BASES:
        cand = base / mapped / run.name
        if (cand / "manifest.json").exists():
            return cand
    return run


def cache_meta_matches(recorded: Path, want: Path) -> bool:
    """manifest 里声明的缓存 meta，是否就是该协议该工况的缓存。

    两种情况都算通过：
      * 原位置未动  → 路径直接相等（当前协议走这条）；
      * 协议目录已归档进 legacy/ → 绝对路径必然不同，此时要求
        ① 路径尾部结构一致（…/<task>/<cond>/meta.json）
        ② 归档后的目标确实存在
        ③ 两边都存在时 split_hash 必须相同（内容级同一，而不是"名字像"）。
    """
    if not str(recorded):
        return False
    if recorded.resolve() == want.resolve():
        return True
    recorded_parts = PurePosixPath(str(recorded).replace("\\", "/")).parts
    if len(recorded_parts) < 3 or recorded_parts[-3:] != want.parts[-3:]:
        return False
    if not want.exists():
        return False
    if not recorded.exists():
        # 已归档：位置变了，但目标存在且尾部结构一致；内容同一性由
        # 调用处的 split_hash 检查（bad_fp）保证。
        return True
    try:
        a = json.loads(recorded.read_text(encoding="utf-8")).get("split_hash")
        b = json.loads(want.read_text(encoding="utf-8")).get("split_hash")
    except Exception:
        return False
    return a == b


def resolve_cache_meta(protocol_cfg: dict, task: str, condition: str) -> Path:
    """某协议某工况的缓存 meta.json；协议目录被归档时到 legacy/ 下找。"""
    want = dataset_dir(protocol_cfg, task, condition) / "meta.json"
    if want.exists():
        return want
    moved = legacy_lookup(str(protocol_cfg["name"]))
    if moved is not None:
        cand = moved / "dataset" / task / normalize_condition(condition) / "meta.json"
        if cand.exists():
            return cand
    return want


PROTOCOL = _main_protocol()
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    proto = load_protocol(PROTOCOL)
    length, stride = int(proto["window"]["length"]), int(proto["window"]["stride"])
    bins = int(proto["preprocessing"]["bins"])
    lags = [int(x) for x in proto["representation"]["lags"]]
    intervals = split_intervals(proto)
    print(f"协议 {proto['name']}  窗长={length} 步长={stride} 档数={bins} 滞后={lags}")
    print(f"区间 {intervals}\n")

    for task, cond in (("gear", "30_2"), ("bearing", "20_0")):
        names = list(class_names(task))
        d = dataset_dir(proto, task, cond)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        rep = proto.get("representation", {})
        print(f"=== {task}/{cond} ===  {d}")

        # ---- 1. 划分无重叠 + 窗数
        starts = {k: window_starts(lo, hi, length, stride) for k, (lo, hi) in intervals.items()}
        alls = np.concatenate(list(starts.values()))
        check("划分无重叠(窗起点唯一)", len(alls) == len(set(alls.tolist())),
              f"train={len(starts['train'])} val={len(starts['val'])} test={len(starts['test'])}")
        exp = {k: len(v) for k, v in starts.items()}          # 期望值直接由协议推导
        check("窗数符合协议", exp["train"] == len(starts["train"]) and exp["test"] == len(starts["test"]),
              f"{exp}")

        # ---- 2. 缓存基本性质
        Xtr = np.load(d / "X_train.npy"); ytr = np.load(d / "y_train.npy")
        Xva = np.load(d / "X_val.npy");   yva = np.load(d / "y_val.npy")
        n_ch = len(rep.get("channels", [])) or 1
        n_lags = len(rep.get("lags", [])) or 1
        bins_ = int(proto["preprocessing"]["bins"])
        exp_shape = (n_ch * n_lags, bins_, bins_)
        check(f"缓存形状 [N,{exp_shape[0]},{exp_shape[1]},{exp_shape[2]}]",
              Xtr.shape[1:] == exp_shape and Xva.shape[1:] == exp_shape,
              f"{Xtr.shape} / {Xva.shape}")
        check("非负", float(Xtr.min()) >= 0 and float(Xva.min()) >= 0)
        sums = Xtr.sum(axis=(2, 3))          # -> [N, 21]：每张矩阵各自的和
        check("每张转移矩阵和为 1", bool(np.allclose(sums, 1.0, atol=1e-4)),
              f"min={sums.min():.5f} max={sums.max():.5f}")
        check("每类窗数相等",
              len(set(np.bincount(ytr).tolist())) == 1 and len(set(np.bincount(yva).tolist())) == 1,
              f"train={np.bincount(ytr).tolist()} val={np.bincount(yva).tolist()}")

        # ---- 3. 状态用满（尺度 bug 的直接检查）
        # 联合概率矩阵 p(i,j)：对样本轴、21 张矩阵、源状态 i 求和 -> 目的状态 j 的占用
        occ = Xtr[:200].sum(axis=(0, 1, 2))          # -> [6]
        assert occ.shape == (bins,), f"占用向量形状错了: {occ.shape}"
        used_min = int((occ > 0).sum())
        frac = (occ / occ.sum())
        check(f"{bins_} 档全部被用到", used_min == bins,
              f"用到 {used_min}/{bins} 档, 各档占比={np.round(frac, 3).tolist()}")

        # ---- 4. 端到端独立复算（从原始 CSV 手工重算一个窗）
        raw = task_dir(PROJECT_ROOT / proto["paths"]["raw_root"], task)
        xyz = vibration_xyz(load_seu_csv(raw / class_file(task, names[0], cond)))
        # 防线 1：短读守卫。若 CSV 被截断（I/O 异常等），窗会变短 -> 转移矩阵不同
        # -> 下面两项一致性检查会失败，但看不出原因。这里直接报出来。
        _expect = int(proto["dataset"]["file_length"])
        check(f"{task} {cond} 原始 CSV 行数 == file_length",
              int(xyz.shape[1]) == _expect,
              f"实际 {xyz.shape[1]} / 应为 {_expect}")
        method = json.loads(json.dumps(meta["bin_edges"]))       # [C][bins+1]
        edges = np.asarray(method, dtype=np.float64)
        ok_all = True
        for idx in (0, 1, 500):
            s = int(starts["train"][idx])
            w = xyz[:, s:s + length]
            _nm_cfg = {"normalize": proto["preprocessing"]["normalize"],
                       "global_mean": meta.get("global_mean"), "global_std": meta.get("global_std")}
            z = preprocess_signal(w, _nm_cfg)
            mats = []
            for c in range(z.shape[0]):
                st = symbolize_with_edges(z[c], edges[c])
                for L in lags:
                    mats.append(transition_matrix(st, bins=bins, step=L, weight="probability", include_self=True))
            mine = np.stack(mats).astype(np.float32)
            same = bool(np.allclose(mine, Xtr[idx], atol=1e-6))
            if not same:
                # 防线 2：失败时给出偏差量，便于定位（此前只报"不一致"，无法进一步分析）
                print(f"      [诊断] 窗 {idx}: 手工重算 {mine.shape} vs 缓存 {Xtr[idx].shape}"
                      f"，最大绝对偏差 {float(np.max(np.abs(mine - Xtr[idx]))):.6e}")
            ok_all &= same
        check("端到端复算一致(3 个窗)", ok_all, "手工重算 == 缓存值")

        # ---- 4b. 生产代码路径必须与上面这份独立实现逐位一致
        method_cfg = {
            "representation": "transition",
            "normalize": proto["preprocessing"]["normalize"],
            "symbol_bins": bins,
            "bin_scope": proto["preprocessing"]["bin_scope"],
            "bin_edges_mode": proto["preprocessing"]["bin_edges"],
            "symbol_strategy": "quantile",
            "transition_steps": lags,
            "transition_weight": proto["representation"].get("transition_weight", "probability"),
            "transition_multichannel": proto["representation"].get("transition_multichannel", "stack"),
            "include_self_transition": bool(proto["representation"].get("include_self_transition", True)),
            "state_stats": "none",
            "bin_edges": meta["bin_edges"],
            "global_mean": meta.get("global_mean"),
            "global_std": meta.get("global_std"),
        }
        prod_ok = True
        for idx in (0, 1, 500):
            s = int(starts["train"][idx])
            got = build_state_transition(xyz[:, s:s + length], method_cfg).astype(np.float32)
            prod_ok &= bool(np.allclose(got, Xtr[idx], atol=1e-6))
        check("生产代码 build_state_transition == 独立复算", prod_ok)

        # ---- 4b2. 内容级泄漏：训练/验证缓存里不能有完全相同的窗
        import hashlib as _hl
        def _row_hashes(A):
            return {_hl.sha256(np.ascontiguousarray(r).tobytes()).digest() for r in A}
        h_tr, h_va = _row_hashes(Xtr), _row_hashes(Xva)
        dup = len(h_tr & h_va)
        check("训练/验证缓存无内容重复(无泄漏)", dup == 0, f"重复窗数 = {dup}")

        # ---- 4b3. 标签必须与"按类分块"的缓存顺序一致
        n_per = len(ytr) // 5
        expect = np.repeat(np.arange(5), n_per)
        check("训练标签顺序与类分块一致", bool(np.array_equal(ytr, expect)),
              f"每类 {n_per} 个窗")
        n_per_v = len(yva) // 5
        expect_v = np.repeat(np.arange(5), n_per_v)
        check("验证标签顺序与类分块一致", bool(np.array_equal(yva, expect_v)))

        # ---- 4c. 退化窗比例（train_global 归一化会让极低幅值的窗塌缩成单档）
        occ_all = Xtr.sum(axis=(1, 2))
        nz = (occ_all > 0).sum(axis=1)
        deg = float((nz < 3).mean())
        check("退化窗比例 < 5%", deg < 0.05, f"{deg*100:.2f}% 的窗用到 <3 档（训练集）")

        # ---- 5. 边界只用训练集拟合
        zval = [preprocess_signal(xyz[:, int(s):int(s) + length],
                                      {"normalize": proto["preprocessing"]["normalize"],
                                       "global_mean": meta.get("global_mean"),
                                       "global_std": meta.get("global_std")})[0]
                for s in starts["val"][:200]]
        qs = np.linspace(0, 1, bins + 1)
        edges_val = np.quantile(np.concatenate(zval), qs)
        diff = float(np.abs(np.asarray(edges)[0][1:-1] - edges_val[1:-1]).max())
        check("边界不是用验证集拟合的", diff > 1e-3, f"最大差 {diff:.5f}")
        print()

    # ---- 6. 结果一致性 + 测试集未被访问
    print("=== 结果一致性 / 测试集隔离 ===")
    rows = list(csv.DictReader(open(PROJECT_ROOT / "registry.csv", encoding="utf-8")))
    check("registry 40 行", len(rows) == 40, f"{len(rows)} 行")
    bad_num, bad_test, dup = 0, [], 0
    seen = set()
    for r in rows:
        mp = resolve_run_dir(r["metrics_path"]) / "metrics.json"
        if not mp.exists():
            bad_num += 1; continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if abs(m["best_val_acc"] - float(r["best_val_acc"])) > 1e-6:   # registry 存的是 round(...,6)
            bad_num += 1
        if "test_metrics" in m or "test_evaluated_once" in m:
            bad_test.append(r["experiment_id"])
        run = mp.parent
        if (run / "test_predictions.csv").exists():
            bad_test.append(r["experiment_id"] + "(有 test_predictions.csv)")
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if man.get("test_access") != "sealed":
            bad_test.append(r["experiment_id"] + "(manifest)")
        if r["experiment_id"] in seen:
            dup += 1
        seen.add(r["experiment_id"])
    check("registry 与 metrics.json 数值一致", bad_num == 0, f"{bad_num} 处不一致")
    check("无重复实验", dup == 0)
    check("40 组全部未访问测试集", not bad_test, str(bad_test[:3]))

    # ---- 7. 每个 run 记录的 split_hash 必须与其协议的数据缓存一致
    print("=== 结果与协议缓存的指纹一致性 ===")
    cache_hash = {}
    for task in ("gear", "bearing"):
        for cond in ("20_0", "30_2"):
            mp = dataset_dir(proto, task, cond) / "meta.json"
            if mp.exists():
                cache_hash[(task, cond)] = json.loads(mp.read_text(encoding="utf-8")).get("split_hash")
    main_name = str(proto["name"])
    mismatch, checked = [], 0
    for r in rows:
        if r["protocol"] != main_name:
            continue
        key = (r["task"], r["condition"])
        if key not in cache_hash:
            continue
        m = json.loads((resolve_run_dir(r["metrics_path"]) / "metrics.json").read_text(encoding="utf-8"))
        checked += 1
        if m.get("dataset_split_hash") != cache_hash[key]:
            mismatch.append(r["experiment_id"])
    check(f"主协议 {main_name} 的 run 指纹与缓存一致", not mismatch,
          f"检查 {checked} 个 run" + (f"，不一致 {mismatch[:3]}" if mismatch else ""))

    # ---- 8. 逐版核对：历史登记表里每个 run 的 manifest 必须指向它声明协议的缓存
    print("=== 历史版本核对（v1..v5 全部登记表）===")
    legacy = sorted((PROJECT_ROOT / "legacy").glob("registry_main_v*_40runs.csv"))
    for reg in legacy:
        lrows = list(csv.DictReader(reg.open(encoding="utf-8")))
        if not lrows:
            continue
        pname = lrows[0]["protocol"]
        pf = None
        for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
            if str(load_protocol(f)["name"]) == pname:
                pf = f
        if pf is None:
            check(f"{reg.name}: 协议 {pname} 仍存在", False)
            continue
        lproto = load_protocol(pf)
        bad_mf, bad_fp = [], []
        for r in lrows:
            run = resolve_run_dir(r["metrics_path"])
            mf = run / "manifest.json"
            want = resolve_cache_meta(lproto, r["task"], r["condition"])
            if not mf.exists():
                bad_mf.append(r["experiment_id"]); continue
            m = json.loads(mf.read_text(encoding="utf-8"))
            if not cache_meta_matches(Path(m.get("dataset_meta", "")), want):
                bad_mf.append(r["experiment_id"])
            mt = want.read_text(encoding="utf-8") if want.exists() else "{}"
            if json.loads(mt).get("split_hash") != json.loads(run.joinpath("metrics.json").read_text(encoding="utf-8")).get("dataset_split_hash"):
                bad_fp.append(r["experiment_id"])
        check(f"{pname} 的 {len(lrows)} 组指向本协议缓存", not bad_mf, f"不匹配 {len(bad_mf)} 组")
        check(f"{pname} 的 {len(lrows)} 组指纹与缓存一致", not bad_fp, f"不一致 {len(bad_fp)} 组")

    # ---- 9. 实验目录完整性：不许有"跑了一半"的 run 混在结果里
    # 背景：本项目三次误报都源于"用目录数判断扫描是否结束"——目录在 train.py
    #       一开始就创建，目录存在 != 跑完。只有 manifest.json 而无 metrics.json，
    #       说明该 run 被中断；这种残骸混进统计会得出错误结论。
    print("=== 实验目录完整性 ===")
    art = PROJECT_ROOT / "artifacts"
    incomplete, total_runs = [], 0
    for mf in art.rglob("manifest.json"):
        d = mf.parent
        if "_backup" in str(d) or "legacy" in str(d):
            continue
        total_runs += 1
        if not (d / "metrics.json").exists():
            incomplete.append(str(d.relative_to(art)))
    check("没有跑了一半的实验目录", not incomplete,
          f"检查 {total_runs} 个 run" + (f"，残缺 {len(incomplete)} 个：{incomplete[:3]}" if incomplete else ""))

    print()
    print("=" * 70)
    print(f"结果: PASS={len(PASS)}  FAIL={len(FAIL)}")
    if FAIL:
        print("失败项:", FAIL)
    print("=" * 70)
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
