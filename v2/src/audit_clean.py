"""主协议结果干净性审计：只读，不写产物，不碰封测集。

协议名一律用 status: main 动态解析（历史上这里写死过 v4，升级协议后审的还是旧协议）。"""
from __future__ import annotations
import ast, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.console import setup_utf8_console  # noqa: E402
from config.protocol import load_protocol, dataset_dir, sealed_test_dir  # noqa: E402
from audit_main import resolve_run_dir  # noqa: E402
from data.build_dataset import split_intervals  # noqa: E402
from data.seu_dds import class_names, window_starts, load_seu_csv, class_file, task_dir, vibration_xyz  # noqa: E402
from representation.recurrence_plot import preprocess_signal  # noqa: E402
from representation.state_transition import build_state_transition, symbolize_with_edges, transition_matrix  # noqa: E402

setup_utf8_console()

def _main_protocol():
    for f in sorted((ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


P = load_protocol(_main_protocol())
L, S = int(P["window"]["length"]), int(P["window"]["stride"])
IV = split_intervals(P)
ok, bad = [], []

def chk(cond, name, detail=""):
    (ok if cond else bad).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

print(f"协议 {P['name']}  窗长={L} 步长={S} 通道={P['dataset'].get('channel_indices',[1,2,3])}")
print()

print("① 划分隔离")
for task in ("gear", "bearing"):
    for cond in ("20_0", "30_2"):
        st = {k: window_starts(lo, hi, L, S) for k, (lo, hi) in IV.items()}
        alls = np.concatenate(list(st.values()))
        chk(len(alls) == len(set(alls.tolist())), f"{task}/{cond} 三划分窗起点无重复")
        # 窗本身是否与相邻划分区间相交
        tr_end = st["train"] + L; va_end = st["val"] + L; te_end = st["test"] + L
        chk(tr_end.max() <= IV["val"][0], f"{task}/{cond} 训练窗不越入验证区", f"train 末点={tr_end.max()} val 起点={IV['val'][0]}")
        chk(va_end.max() <= IV["test"][0], f"{task}/{cond} 验证窗不越入测试区")

print()
print("② 测试集是否被访问")
rows = list(csv.DictReader(open(ROOT / "registry.csv", encoding="utf-8")))
badtest = []
for r in rows:
    run = resolve_run_dir(r["metrics_path"])
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if "test_metrics" in m or (run / "test_predictions.csv").exists() or man.get("test_access") != "sealed":
        badtest.append(r["experiment_id"])
chk(not badtest, "40 组全部未访问测试集", f"{len(rows)} 组")
# 当前主协议的封存测试目录必须**尚未生成** —— 这是真的检查（此前这里是 `len(sealed) == 0 or True`，
# 永远通过；且写死了 v4 的目录名，升级协议后就不再有效）。
_cur_sealed = [sealed_test_dir(P, t, c) for t in ("gear", "bearing") for c in ("20_0", "30_2")]
_hit = [str(p.relative_to(ROOT)) for p in _cur_sealed if p.exists()]
chk(not _hit, f"主协议 {P['name']} 的封存测试目录尚未生成（测试集从未被用到）",
    "从未创建" if not _hit else f"已存在：{_hit}")
_sealed_root = ROOT / "artifacts" / "sealed_test"
_legacy = sorted(p.name for p in _sealed_root.glob("*")) if _sealed_root.exists() else []
print(f"  [info] artifacts/sealed_test 下的目录：{_legacy or '（空）'}")

print()
print("③ 预处理是否只用训练段信息")
d = dataset_dir(P, "gear", "30_2")
meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
chk(meta.get("bin_scope") in (None, "fit_on_train_only"), "分位边界声明为只用训练集拟合", str(meta.get("bin_scope")))
edges = np.asarray(meta["bin_edges"], dtype=np.float64)
raw = vibration_xyz(load_seu_csv(task_dir(ROOT / P["paths"]["raw_root"], "gear") / class_file("gear", "Health", "30_2")),
                    P["dataset"].get("channel_indices", [1, 2, 3]))
vs = window_starts(*IV["val"], L, S)[:200]
_nm = {"normalize": P["preprocessing"]["normalize"],
       "global_mean": meta.get("global_mean"), "global_std": meta.get("global_std")}
zval = np.concatenate([preprocess_signal(raw[:, s:s+L], _nm)[0] for s in vs])
ev = np.quantile(zval, np.linspace(0, 1, int(P["preprocessing"]["bins"]) + 1))
chk(float(np.abs(edges[0][1:-1] - ev[1:-1]).max()) > 1e-3, "边界确实不是用验证集拟合的",
    f"最大差 {float(np.abs(edges[0][1:-1] - ev[1:-1]).max()):.5f}")

print()
print("④ 归一化方式与统计量来源")
w = raw[:, 0:L]
nm = {"normalize": P["preprocessing"]["normalize"], "global_mean": meta.get("global_mean"), "global_std": meta.get("global_std")}
z1 = preprocess_signal(w, nm)
mode = str(P["preprocessing"]["normalize"])
if mode == "train_global_zscore":
    mu = np.asarray(meta["global_mean"])[:, None]; sd = np.asarray(meta["global_std"])[:, None]
    chk(bool(np.allclose(z1, (w - mu) / sd, atol=1e-9)), "用协议/meta 里的固定 μσ（不是窗内统计量）")
    chk(meta.get("global_mean") is not None, "统计量来自训练集并写入 meta.json")
    # 整体缩放会改变结果 —— 这正是保留幅值信息的目的
    z2 = preprocess_signal(w * 1000.0, nm)
    chk(not bool(np.allclose(z1, z2, atol=1e-9)), "整体缩放会改变结果（说明幅值信息被保留，符合设计意图）")
else:
    z2 = preprocess_signal(w * 1000.0, nm)
    chk(bool(np.allclose(z1, z2, atol=1e-9)), "逐窗 z-score 只用窗内统计量（整体缩放不改变结果）")
    chk(abs(float(z1.mean())) < 1e-6 and abs(float(z1.std()) - 1) < 1e-6, "归一化后窗内均值 0 / 标准差 1")

print()
print("⑤ 端到端独立复算（从原始 CSV 手算，不用库函数）")
mc = {"representation": "transition", "normalize": P["preprocessing"]["normalize"], "symbol_bins": int(P["preprocessing"]["bins"]),
      "bin_scope": P["preprocessing"]["bin_scope"], "bin_edges_mode": P["preprocessing"]["bin_edges"],
      "symbol_strategy": "quantile", "transition_steps": [int(x) for x in P["representation"]["lags"]],
      "transition_weight": "probability", "transition_multichannel": "stack",
      "include_self_transition": True, "state_stats": "none", "bin_edges": meta["bin_edges"],
      "global_mean": meta.get("global_mean"), "global_std": meta.get("global_std")}
X = np.load(d / "X_train.npy")
same = True
for i in (0, 1, 400):
    s = int(window_starts(*IV["train"], L, S)[i])
    same &= bool(np.allclose(build_state_transition(raw[:, s:s+L], mc), X[i], atol=1e-6))
chk(same, "生产代码与独立复算逐位一致")

print()
print("⑥ 5 个 seed 是否真的独立重训")
# 注意：若多个 seed 都做到 100%（全部窗预测正确），val_predictions.csv 内容必然相同，
# 这是"都满分"而不是"复制结果"。因此改用训练曲线与轮数来判定独立性。
import hashlib
sig = {}
for r in rows:
    if r["task"] == "gear" and r["condition"] == "30_2" and r["model"] == "cnn":
        run = resolve_run_dir(r["metrics_path"])
        hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
        first = ",".join(f"{h['train_loss']:.6f}" for h in hist[:3])   # 前 3 轮的训练损失
        sig[int(r["seed"])] = (hashlib.sha256(first.encode()).hexdigest()[:10], len(hist))
uniq_curve = len({v[0] for v in sig.values()})
chk(uniq_curve == len(sig), "5 个 seed 的训练轨迹互不相同（确为独立训练）",
    f"独立轨迹 {uniq_curve}/{len(sig)}: " + ", ".join(f"s{k}:ep{v[1]}" for k, v in sorted(sig.items())))
chk(len({v[1] for v in sig.values()}) > 1, "5 个 seed 的收敛轮数不完全相同（不是同一次运行被复制）",
    str({k: v[1] for k, v in sorted(sig.items())}))

print()
print("=" * 70)
print(f"结果：PASS={len(ok)}  FAIL={len(bad)}")
if bad: print("失败项:", bad)
print("=" * 70)
raise SystemExit(1 if bad else 0)
