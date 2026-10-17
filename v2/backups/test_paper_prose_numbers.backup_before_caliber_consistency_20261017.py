"""论文正确性测试（三）：**正文里不在表里的数字**，全部从 artifacts 复算。

起因（2026-10-17 用户问"现在论文的数据对吗"）：表里的数字早有逐格核对
（test_paper_integrity / test_paper_noise_tables），但正文里还散着几十个数字——
滞后 / 档数 / 窗长消融、工况漂移（1080 维）、TV 值、效率比、跨工况……此前**没有任何机器核对**。

本次逐条复算，发现并修正 2 处：
  * 跨工况"本文为 51.1%"其实是 **CNN 单模型**（GCN 为 41.7%）→ 正文已补全两个模型；
  * 结论"相差 0.2–0.4 个百分点"实际为 0.18–0.32 → 已改为 0.2–0.3。
其余数字（滞后维数、滞后取样、档数、窗长配对 t、效率图的 8 个数字、TV、
效率比、表 9 的 −12/−16 行、表 5 的两列）全部复现。

本文件把它们固定住：以后任何改动导致这些数字漂移，都会在这里失败。
"""
from __future__ import annotations

import json
import math
import re
import statistics as st

import numpy as np
import pytest

from _bootstrap import PROJECT_ROOT  # noqa: F401  （把 src 加入 sys.path）

PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"
ART = PROJECT_ROOT / "artifacts"
S3 = (42, 43, 44)
S5 = (42, 43, 44, 45, 46)


@pytest.fixture(scope="module")
def paper() -> str:
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")


def run_acc(dirname: str, tag: str, task: str, model: str, cond: str, seed: int,
            metric: str = "best") -> float:
    p = ART / dirname / tag / f"{task}_{model}_{cond}_{tag}_seed{seed}" / "metrics.json"
    assert p.exists(), f"缺 run：{p.relative_to(PROJECT_ROOT)}"
    m = json.loads(p.read_text(encoding="utf-8"))
    return 100 * (m["val_metrics"]["accuracy"] if metric == "best"
                  else m["stability"]["last20_val_acc_mean"])


def group(dirname: str, tag: str, task: str, cond: str, metric: str = "best",
          seeds=S3, models=("cnn", "gcn")) -> float:
    return st.mean([run_acc(dirname, tag, task, m, cond, s, metric) for m in models for s in seeds])


def check(label: str, want: float, got: float, tol: float = 0.06) -> None:
    assert abs(want - got) <= tol, f"{label}: 论文 {want} vs 复算 {got:.4f}（容差 {tol}）"


def check_either(label: str, want: float, dirname: str, tag: str, task: str, cond: str,
                 tol: float = 0.02) -> None:
    """正文的消融数字用的是"最佳轮"或"平台均值"之一，只要其一吻合即通过（并在失败时两个都报出来）。"""
    b = group(dirname, tag, task, cond, "best")
    p = group(dirname, tag, task, cond, "plateau")
    assert abs(want - b) <= tol or abs(want - p) <= tol, (
        f"{label}: 论文 {want} | 最佳轮 {b:.4f} | 平台 {p:.4f}（容差 {tol}）")


def in_paper(paper: str, *needles: str) -> None:
    for s in needles:
        assert s in paper, f"论文里找不到这句（说明正文被改过、需同步核对）：{s!r}"


# --------------------------------------------------------------------------- #
# 一、§4.1 表示参数的消融
# --------------------------------------------------------------------------- #
def test_lag_dimension_ablation(paper):
    """滞后维数-精度：4→98.22%、7→98.48%、10→99.02%、13→99.07%（齿轮 30-2）。

    正文称滞后取 1–13 与 1–10 "性能持平（差异仅 0.05 个百分点）"：实测 0.0504，
    所以写"仅 0.05"是对的，写"< 0.05"会偏乐观（2026-10-17 已按实测改口径）。
    """
    for tag, want in (("lag4", 98.22), ("lag7", 98.48), ("base", 99.02), ("lag13", 99.07)):
        check_either(f"滞后维数 {tag}", want, "variant_scan_rep_v5", tag, "gear", "30_2")
    base = group("variant_scan_rep_v5", "base", "gear", "30_2", "plateau")
    lag13 = group("variant_scan_rep_v5", "lag13", "gear", "30_2", "plateau")
    check("滞后 1–13 与 1–10 之差", 0.05, abs(lag13 - base), tol=0.01)
    in_paper(paper, "差异仅 0.05 个百分点")


def test_lag_spacing_contrast():
    """滞后取样：连续 1–10 为 99.023%、间隔 {1,3,…,19} 为 99.235%；5 维时 98.186% / 98.117%。"""
    check_either("间隔取样 1,3,…,19", 99.235, "variant_scan_lagspace", "lag_odd10", "gear", "30_2")
    check_either("连续 1–5", 98.186, "variant_scan_lagspace", "lag_contig5", "gear", "30_2")
    check_either("间隔 1,3,5,7,9", 98.117, "variant_scan_lagspace", "lag_odd5", "gear", "30_2")
    check_either("连续 1–10", 99.023, "variant_scan_rep_v5", "base", "gear", "30_2")


def test_binning_ablation():
    """档数：B=8 比 B=6 低 0.20 个百分点、B=10 低 0.11（这两个数出自"最佳轮"口径）。"""
    base = group("variant_scan_rep_v5", "base", "gear", "30_2", "best")
    for tag, delta in (("bins8", -0.20), ("bins10", -0.11)):
        got = group("variant_scan_rep_v5", tag, "gear", "30_2", "best") - base
        check(f"{tag} - base", delta, got, tol=0.03)


def test_window_length_paired_test():
    """窗长：1600 比 2048 平均低 0.39（t=−7.98, p<0.001）、3200 高 0.12（t=1.30）。

    t 统计量用的是**总体标准差**（ddof=0），与论文一致；配对样本 = 5 seed × 2 模型。
    """
    for tag, dm, dt in (("win1600_s1024", -0.39, -7.98), ("win3200_s1024", 0.12, 1.30)):
        diffs = [run_acc("variant_scan_win_fixed", tag, "gear", mdl, "30_2", s, "best")
                 - run_acc("variant_scan_win_fixed", "base", "gear", mdl, "30_2", s, "best")
                 for mdl in ("cnn", "gcn") for s in S5]
        m = st.mean(diffs)
        t = m / (st.pstdev(diffs) / math.sqrt(len(diffs)))
        check(f"{tag} - base 均值", dm, m, tol=0.03)
        check(f"{tag} - base t({len(diffs) - 1})", dt, t, tol=0.1)


# --------------------------------------------------------------------------- #
# 二、§4.5 / 图 13 的正文数字
# --------------------------------------------------------------------------- #
def test_figure13_text_numbers():
    """图 13 正文：逐窗 −6 dB 本文 86.2 / 86.1 vs WDCNN 82.4、ResNet1D 85.7；
    全局 −10 dB 本文 68.9 / 72.0 vs 60.6 / 66.7（齿轮 20-0）。"""
    for mdl, pw, gl in (("cnn", 86.2, 68.9), ("gcn", 86.1, 72.0)):
        check(f"本文 {mdl} 逐窗 −6 dB", pw, group("variant_scan_fill", "snrm6", "gear", "20_0", "best",
                                                models=(mdl,)), tol=0.06)
        check(f"本文 {mdl} 全局 −10 dB", gl, group("variant_scan_fill", "gsnrm10", "gear", "20_0", "best",
                                                models=(mdl,)), tol=0.06)
    for mdl, pw, gl in (("wdcnn", 82.4, 60.6), ("resnet1d", 85.7, 66.7)):
        check(f"{mdl} 逐窗 −6 dB", pw, group("variant_scan", "nm6", "gear", "20_0", "best",
                                           models=(mdl,)), tol=0.06)
        check(f"{mdl} 全局 −10 dB", gl, group("variant_scan", "gnm10", "gear", "20_0", "best",
                                            models=(mdl,)), tol=0.06)


def test_efficiency_text_numbers(paper):
    """效率：CNN 单样本 0.34 ms、1/139（ResNet1D）、1/552（DDGF-Net 15.29 M）、输入规模 5.7 倍。"""
    eff = json.loads((PROJECT_ROOT / "reports/efficiency.json").read_text(encoding="utf-8"))
    eff_all = json.loads((PROJECT_ROOT / "reports/efficiency_all.json").read_text(encoding="utf-8"))
    cnn = next(x for x in eff if x["model"] == "cnn")
    check("CNN 单样本延迟 (ms)", 0.34, cnn["ms_per_sample"], tol=0.005)
    check("ResNet1D / 本文 CNN", 139, eff_all["resnet1d"]["params"] / eff_all["cnn"]["params"], tol=0.6)
    check("DDGF-Net 15.29M / 本文 CNN", 552, 15_290_000 / eff_all["cnn"]["params"], tol=2)
    check("输入规模倍数", 5.7, 6144 / 1080, tol=0.05)
    check("ResNet1D / 本文 CNN FLOPs", 732, eff_all["resnet1d"]["flops"] / eff_all["cnn"]["flops"], tol=2)
    check("raw_cnn / 本文 CNN FLOPs", 42, eff_all["raw_cnn"]["flops"] / eff_all["cnn"]["flops"], tol=0.5)
    in_paper(paper, "1/139", "1/552", "0.0277 M", "0.0325 M", "5.7 倍",
             "FLOPs 约为其 1/732", "同组参数量最大的基线 ResNet1D", "约 42 倍的 FLOPs 下降")
    assert "1/139 的算量" not in paper, "1/139 是参数量比，不是算量（FLOPs 比是 1/732）"


# --------------------------------------------------------------------------- #
# 三、表 5 的两列网络数字（此前只核对了 LDA 列与"差距列是算术差"）
# --------------------------------------------------------------------------- #
def _tables(paper: str) -> dict:
    L = paper.split("\n")
    out, i = {}, 0
    while i < len(L):
        m = re.match(r"\*\*表 (\d+[ab]?)\s+(.*?)\*\*", L[i].strip())
        if m:
            rows, j = [], i + 1
            while j < len(L):
                if L[j].lstrip().startswith("|"):
                    rows.append([c.strip() for c in L[j].strip().strip("|").split("|")])
                    j += 1
                elif rows and not L[j].strip():
                    break
                elif not rows:
                    j += 1
                else:
                    break
            if rows:
                out[m.group(1)] = rows
            i = j
        else:
            i += 1
    return out


def _body(rows):
    return [r for r in rows if not set("".join(r)) <= set("-: ")][1:]


def _f(cell: str) -> float:
    return float(re.search(r"-?\d+\.?\d*", cell.replace("−", "-").replace("**", "")).group(0))


def test_table5_network_columns_match_artifacts():
    """表 5 的 "CNN / GCN (%)" 列必须能从 artifacts 复算出来（干净 5 seed；加噪 3 seed）。"""
    rows = _body(_tables(PAPER.read_text(encoding="utf-8").replace("\r\n", "\n"))["5"])
    main = "main_70_15_15_v5"
    cases = {
        ("干净", "齿轮 20-0"): lambda m: st.mean([
            100 * json.loads((ART / main / f"gear_{m}_20_0_seed{s}" / "metrics.json")
                             .read_text(encoding="utf-8"))["val_metrics"]["accuracy"] for s in S5]),
        ("干净", "轴承 20-0"): lambda m: st.mean([
            100 * json.loads((ART / main / f"bearing_{m}_20_0_seed{s}" / "metrics.json")
                             .read_text(encoding="utf-8"))["val_metrics"]["accuracy"] for s in S5]),
        ("逐窗参考加噪，−6 dB", "齿轮 20-0"): lambda m: group("variant_scan", "snrm6", "gear", "20_0", "best", models=(m,)),
        ("逐窗参考加噪，−10 dB", "齿轮 20-0"): lambda m: group("variant_scan", "snrm10", "gear", "20_0", "best", models=(m,)),
        ("逐窗参考加噪，−10 dB", "轴承 20-0"): lambda m: group("variant_scan", "snrm10", "bearing", "20_0", "best", models=(m,)),
    }
    seen = 0
    for r in rows:
        key = (r[0], r[1])
        if key not in cases or "/" not in r[3]:
            continue
        want = [float(x) for x in r[3].split("/")]
        got = [cases[key]("cnn"), cases[key]("gcn")]
        for i, mdl in enumerate(("CNN", "GCN")):
            check(f"表 5 {key} {mdl}", want[i], got[i], tol=0.005)
        seen += 1
    assert seen == 5, f"表 5 只核对了 {seen} 行，行数不对"


# --------------------------------------------------------------------------- #
# 四、表 9 里只在表 9 出现的行（−12 / −16 dB）与轴承行
# --------------------------------------------------------------------------- #
def test_table9_rows_not_covered_by_table8():
    """表 9 的 −12 / −16 dB 两行（表 8 只到 −10 dB）与轴承 20-0 行，逐格复算。"""
    for tag, want_pw, want_gl in (("snrm12", 84.41, None), ("snrm16", 84.54, None),
                                  ("gsnrm12", None, 61.58), ("gsnrm16", None, 43.60)):
        got = group("variant_scan", tag, "gear", "20_0", "best")
        check(f"表 9 gear 20-0 {tag}", want_pw if want_pw is not None else want_gl, got, tol=0.02)
    check("表 9 轴承 20-0 逐窗 −10", 98.05, group("variant_scan", "snrm10", "bearing", "20_0", "best"), tol=0.02)
    check("表 9 轴承 20-0 全局 −10", 33.36, group("variant_scan", "gsnrm10", "bearing", "20_0", "best"), tol=0.02)


# --------------------------------------------------------------------------- #
# 五、§4.5 的 TV 与 §4.3 的 t-SNE（都从数据缓存现算）
# --------------------------------------------------------------------------- #
def _proto():
    from config.protocol import load_protocol
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")


def test_tv_values_in_text():
    """图 10 的 TV（总变差距离）= 0.5·Σ|两类工况的类平均转移矩阵之差|，x 通道第 1 个滞后。"""
    from config.protocol import dataset_dir
    proto = _proto()
    bins = int(proto["preprocessing"]["bins"])
    lags = proto["representation"]["lags"]
    got = {}
    for task in ("gear", "bearing"):
        means = {}
        for cond in ("20_0", "30_2"):
            d = dataset_dir(proto, task, cond)
            X = np.load(d / "X_train.npy").astype(np.float32)
            y = np.load(d / "y_train.npy").astype(np.int64)
            Xr = X.reshape(len(X), 3, len(lags), bins, bins)
            means[cond] = [Xr[y == c][:, 0, 0].mean(axis=0) for c in range(5)]
        got[task] = [0.5 * float(np.abs(means["20_0"][c] - means["30_2"][c]).sum()) for c in range(5)]
    # 齿轮：齿面剥落(Chipped, 1) 最大 0.250、齿面磨损(Surface, 4) 最小 0.058
    check("齿轮 TV 最大（齿面剥落）", 0.250, max(got["gear"]), tol=0.002)
    check("齿轮 TV 最小（齿面磨损）", 0.058, min(got["gear"]), tol=0.002)
    # 轴承：外圈(outer, 3) 最大 0.294、内圈(inner, 2) 最小 0.049
    check("轴承 TV 最大（外圈）", 0.294, max(got["bearing"]), tol=0.002)
    check("轴承 TV 最小（内圈）", 0.049, min(got["bearing"]), tol=0.002)


def test_feature_drift_numbers(paper):
    """§4.3 的工况漂移：原始 1080 维空间里 漂移/类内散布 = 2.1（齿轮）/ 2.2（轴承）倍。

    2026-10-17：t-SNE 图移到补充材料后，正文不再使用 t-SNE 坐标里的 53.2/8.2/6.5，
    改为在原始 1080 维空间中直接计算（LDA / t-SNE 都会压缩或扭曲工况差异）。
    同时要求正文提到"补充材料图 S1"，保证"透明性补充"没有被悄悄删掉。
    """
    from config.protocol import dataset_dir
    proto = _proto()
    for task, ratio_want in (("gear", 2.1), ("bearing", 2.2)):
        drift, spread = [], []
        for ci in range(5):
            mus = {}
            for cond in ("20_0", "30_2"):
                d = dataset_dir(proto, task, cond)
                y = np.load(d / "y_val.npy")
                X = np.load(d / "X_val.npy").reshape(len(y), -1).astype(np.float64)
                P = X[y == ci]
                mus[cond] = P.mean(axis=0)
                spread.append(float(np.linalg.norm(P - mus[cond], axis=1).mean()))
            drift.append(float(np.linalg.norm(mus["20_0"] - mus["30_2"])))
        check(f"{task} 漂移/散布 比值", ratio_want,
              float(np.mean(drift) / np.mean(spread)), tol=0.1)
    in_paper(paper, "2.1（齿轮）/ 2.2（轴承）倍", "补充材料图 S1")


# --------------------------------------------------------------------------- #
# 六、§5.5 与结论里的比较数字
# --------------------------------------------------------------------------- #
def _cc(model: str, seed: int) -> float:
    p = ART / "cross_condition" / model / f"gear_{model}_20_0to30_2_seed{seed}" / "metrics.json"
    assert p.exists(), f"缺跨工况 run：{p.name}"
    return 100 * json.loads(p.read_text(encoding="utf-8"))["val_metrics"]["accuracy"]


def test_cross_condition_text_reports_both_models(paper):
    """§5.5(2) 的跨工况必须同时给出 CNN 与 GCN（只写 51.1% 会被读成"只挑好的那个模型"）。"""
    cnn = st.mean([_cc("cnn", s) for s in S3])
    gcn = st.mean([_cc("gcn", s) for s in S3])
    raw = st.mean([_cc("raw_cnn", s) for s in S3])
    check("跨工况 本文 CNN", 51.1, cnn, tol=0.06)
    check("跨工况 本文 GCN", 41.7, gcn, tol=0.06)
    check("跨工况 原始波形基线", 77.3, raw, tol=0.06)
    in_paper(paper, "本文 CNN 为 51.1%、GCN 为 41.7%", "原始波形基线为 77.3%")


def test_clean_condition_gap_text(paper):
    """§5.5(2)/结论：干净条件下原始波形 4 个工况全部 100.00%；
    本文齿轮两个工况 99.6%–99.8%（差 0.2–0.3 个百分点），轴承两个工况同为 100.00%。
    """
    main = "main_70_15_15_v5"
    gear_gaps, bearing_equal = [], True
    for task, cond in (("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2")):
        ours = st.mean([100 * json.loads((ART / main / f"{task}_{m}_{cond}_seed{s}" / "metrics.json")
                                         .read_text(encoding="utf-8"))["val_metrics"]["accuracy"]
                        for m in ("cnn", "gcn") for s in S5])
        raw = group("variant_scan", "clean", task, cond, "best", models=("raw_cnn",))
        assert raw == 100.0, f"原始波形 {task} {cond} 干净条件不再是 100.00：{raw}"
        if task == "gear":
            assert 99.6 <= ours <= 99.8, f"齿轮 {cond} 本文 {ours:.2f}，与论文的 99.6%–99.8% 不符"
            gear_gaps.append(100.0 - ours)
        else:
            bearing_equal = bearing_equal and abs(ours - 100.0) < 1e-9
    assert bearing_equal, "轴承任务本文不再是 100.00%，论文的口径描述要改"
    for g in gear_gaps:
        assert 0.15 <= g <= 0.35, f"齿轮任务干净条件差距 {g:.2f} 不在论文写的 0.2–0.3 附近"
    in_paper(paper, "100.00% vs 本文 99.6%–99.8%", "齿轮任务相差 0.2–0.3 个百分点",
             "轴承任务两者同为 100.00%")


def test_learned_feature_tsne_matches_source(paper):
    """图 8 的学习特征 t-SNE：轮廓系数 0.69/0.79、验证精度 99.87/100.00 必须与 feature_tsne.json 一致。

    这张图由 src/analysis/tsne_embedding.py 生成（读 train.py --save-model 落盘的 best_state.pt，
    提取 CNN 的 embed() 后做 t-SNE）。数字写进论文后不能与源文件脱节。
    """
    j = json.loads((PROJECT_ROOT / "reports/feature_tsne.json").read_text(encoding="utf-8"))
    gear = j["gear_cnn_20_0_seed42"]
    bear = j["bearing_cnn_20_0_seed42"]
    check("齿轮 嵌入轮廓系数", 0.69, gear["silhouette_embedding"], tol=0.01)
    check("轴承 嵌入轮廓系数", 0.79, bear["silhouette_embedding"], tol=0.01)
    check("齿轮 验证精度", 99.87, gear["val_acc"], tol=0.02)
    check("轴承 验证精度", 100.00, bear["val_acc"], tol=0.01)
    assert gear["embedding_dim"] == 64 and bear["embedding_dim"] == 64
    in_paper(paper, "CNN 倒数第二层特征（64 维）", "嵌入空间轮廓系数：0.69 / 0.79", "随机种子 42")


def test_transition_structure_numbers(paper):
    """§4.5 图 10 的条件转移结构数字：齿轮自转移 0.25–0.33、缺齿–齿面磨损距离 0.10、
    轴承外圈自转移 0.35 / 行熵 1.46。

    2026-10-17：原句"健康集中在低状态区、缺齿向右下偏移"与等频分位设计矛盾
    （等频分位把每一类的边缘状态分布强制成近似均匀，质量不可能集中在低状态）。
    已改为按行归一化的条件结构描述，本测试把新数字钉住。
    """
    from config.protocol import dataset_dir
    proto = _proto()

    def matrices(task, cond):
        d = dataset_dir(proto, task, cond)
        X = np.load(d / "X_val.npy")
        y = np.load(d / "y_val.npy")
        out = []
        for ci in range(5):
            M = X[y == ci][:, 2].mean(axis=0)          # 通道 0、第 3 个滞后（与图 10 一致）
            out.append(M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12))
        return out

    gear = matrices("gear", "20_0")
    diag = [float(np.mean(np.diag(P))) for P in gear]
    assert abs(min(diag) - 0.245) < 0.01 and abs(max(diag) - 0.333) < 0.01, diag
    dist = float(np.linalg.norm(gear[2] - gear[4]))   # Miss=2, Surface=4
    check("缺齿 vs 齿面磨损 转移矩阵距离", 0.10, dist, tol=0.02)

    outer = matrices("bearing", "20_0")[3]            # health, ball, inner, outer, comb
    self_outer = float(np.mean(np.diag(outer)))
    ent = float(np.mean([-(r[r > 0] * np.log(r[r > 0])).sum() for r in outer]))
    check("轴承外圈 自转移", 0.35, self_outer, tol=0.01)
    check("轴承外圈 行熵", 1.46, ent, tol=0.02)
    in_paper(paper, "0.25–0.33", "Frobenius 距离 0.10", "自转移概率最高（0.35）", "行熵最低（1.46")
