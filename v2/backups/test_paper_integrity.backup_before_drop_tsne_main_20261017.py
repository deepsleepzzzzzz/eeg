r"""论文正确性测试（一）：图表编号/引用/参考文献自洽 + 与产物逐格核对的表格。

为什么要有：论文的数字散落在正文与 9 张表里，靠肉眼核对必然出错。本次实测就抓到 2 处真问题：
  * 图 13 的参数标注用了别的基准（raw_cnn 写成 0.0891 M，项目记录是 0.0325 M）；
  * 表 8 的部分 run 已被变体扫描覆盖丢失（见 tests/test_paper_noise_tables.py 与之配套的重跑）。

数据源（全部在项目内，可被数据溯源审计复核）：
  表 1      算术自洽（$2048^2$ 与 $C\cdot K\cdot B^2$）
  表 2、表 3 reports/lag_information_{gear,bearing}_20_0.csv（符号互信息扫描）
  表 4      artifacts/main_70_15_15_v5/<task>_<model>_<cond>_seed*/metrics.json（5 个种子）
  表 5      reports/lda_check.csv（LDA 列）+ 本表自身算术（差距列）
  表 6      reports/table_lit_baselines.md（文献自跑值）
  表 7      reports/efficiency.json（本文两行）
  图 13     reports/efficiency_all.json（各方法参数量，单一真源）
表 8、表 9 依赖加噪 run，另见 tests/test_paper_noise_tables.py。
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"
TOL = 0.005          # 论文里数字四舍五入到 2 位小数


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def paper_text() -> str:
    assert PAPER.exists(), f"论文稿不存在：{PAPER}"
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def tables(paper_text: str) -> dict:
    """{表号: (题注, [[单元格...], ...])}，只含真实的 Markdown 表格。"""
    L = paper_text.split("\n")
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
                out[m.group(1)] = (m.group(2), rows)
            i = j
        else:
            i += 1
    return out


def body_rows(rows: list[list[str]]) -> list[list[str]]:
    """去掉表头与 |---|---| 分隔行。"""
    return [r for r in rows if not set("".join(r)) <= set("-: ")][1:]


def num(cell: str) -> float:
    """从 '99.82 ± 0.06' / '**84.4**' / '0.0209 nat' 里取第一个数。"""
    m = re.search(r"-?\d+\.?\d*", cell.replace("−", "-").replace(",", ""))
    assert m, f"单元格里找不到数字：{cell!r}"
    return float(m.group(0))


def mean_std(vals: list[float]) -> tuple[float, float]:
    return st.mean(vals), st.pstdev(vals)


# --------------------------------------------------------------------------- #
# 一、自洽性：编号、引用、参考文献、占位符
# --------------------------------------------------------------------------- #
def test_figure_and_table_numbers_are_contiguous(paper_text, tables):
    figs = sorted({int(x) for x in re.findall(r"\*\*图 (\d+)", paper_text)})
    tabs = sorted({int(re.sub(r"\D", "", k)) for k in tables})
    assert figs == list(range(1, max(figs) + 1)), f"图号不连续：{figs}"
    assert tabs == list(range(1, max(tabs) + 1)), f"表号不连续：{tabs}"


def test_every_figure_and_table_is_cited_in_text(paper_text, tables):
    L = [l for l in paper_text.split("\n")
         if not l.startswith("![") and not l.startswith("**图") and not l.startswith("**表")]
    body = "\n".join(L)
    missing = []
    for n in sorted({int(x) for x in re.findall(r"\*\*图 (\d+)", paper_text)}):
        if not re.search(rf"图 {n}(?!\d)", body):
            missing.append(f"图 {n}")
    for k in tables:
        if not re.search(rf"表 {re.escape(k)}(?!\d)", body):
            missing.append(f"表 {k}")
    assert not missing, f"以下图表在正文里没有被引用（孤儿）：{missing}"


def test_references_are_complete_and_cited(paper_text):
    L = paper_text.split("\n")
    start = next(i for i, l in enumerate(L) if l.strip().startswith("## 参考文献"))
    refs = [l for l in L[start:] if re.match(r"^\[\d+\]", l.strip())]
    nums = [int(re.match(r"^\[(\d+)\]", l.strip()).group(1)) for l in refs]
    assert nums == list(range(1, len(nums) + 1)), f"参考文献编号不连续：{nums}"
    body = "\n".join(L[:start])
    # 先把数学环境去掉：$[0,733{,}992)$ 这类区间不是文献引用
    body = re.sub(r"\$[^$]*\$", " ", body)
    cited = {int(x) for x in re.findall(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]", body)
             for x in re.findall(r"\d+", x)}
    over = sorted(c for c in cited if c > len(nums))
    assert not over, f"正文引用了不存在的文献编号：{over}（共 {len(nums)} 条）"
    unused = sorted(set(nums) - cited)
    assert not unused, f"以下参考文献从未被正文引用：{unused}"


def test_no_placeholders_left(paper_text):
    bad = []
    for pat in ("TODO", "TBD", "XXX", "待写", "待补", "占位", "???", "【待"):
        for m in re.finditer(re.escape(pat), paper_text):
            line = paper_text[:m.start()].count("\n") + 1
            bad.append(f"第 {line} 行含 {pat!r}")
    assert not bad, "稿子里还有占位/待办痕迹：\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# 二、表格数字 ↔ 产物
# --------------------------------------------------------------------------- #
def test_table1_arithmetic(tables):
    _, rows = tables["1"]
    txt = " ".join(" ".join(r) for r in rows)
    assert "4,194,304" in txt and "1,080" in txt
    assert 2048 ** 2 == 4_194_304
    assert 3 * 10 * 6 * 6 == 1080


def test_table2_symbolic_mutual_information(tables):
    _, rows = tables["2"]
    for r in body_rows(rows):
        task_cn = r[0]
        task = "gear" if "齿轮" in task_cn else "bearing"
        src = PROJECT_ROOT / f"reports/lag_information_{task}_20_0.csv"
        recs = [x for x in csv.DictReader(src.open(encoding="utf-8")) if x["bins"] == "6"]
        le = [float(x["I_nat"]) for x in recs if int(x["lag"]) <= 10]
        gt = [float(x["I_nat"]) for x in recs if int(x["lag"]) > 10]
        assert abs(num(r[1]) - st.mean(le)) < 1e-4, (task_cn, "k≤10 均值", r[1], st.mean(le))
        assert abs(num(r[2]) - st.mean(gt)) < 1e-4, (task_cn, "k>10 均值", r[2], st.mean(gt))
        assert abs(num(r[3]) - st.mean(le) / st.mean(gt)) < 0.06, (task_cn, "比值", r[3])


def test_table3_normalized_mutual_information(tables):
    _, rows = tables["3"]
    B = [int(num(c)) for c in rows[0][1:]]
    vals = [num(c) for c in body_rows(rows)[0][1:]]
    recs = list(csv.DictReader((PROJECT_ROOT / "reports/lag_information_gear_20_0.csv")
                               .open(encoding="utf-8")))
    src = {int(x["bins"]): float(x["I_over_H"]) for x in recs if int(x["lag"]) == 1}
    for b, v in zip(B, vals):
        assert abs(v - src[b]) < 1e-4, f"B={b}: 论文 {v} vs 源 {src[b]}"


def test_table4_main_results(tables):
    _, rows = tables["4"]
    for r in body_rows(rows):
        task = "gear" if "齿轮" in r[0] else "bearing"
        cond = r[1].replace("-", "_")
        model = r[2].lower()
        got = {k: [] for k in ("accuracy", "macro_f1", "balanced_accuracy")}
        for seed in (42, 43, 44, 45, 46):
            p = (PROJECT_ROOT / "artifacts/main_70_15_15_v5" /
                 f"{task}_{model}_{cond}_seed{seed}" / "metrics.json")
            assert p.exists(), f"缺 run：{p}"
            vm = json.loads(p.read_text(encoding="utf-8"))["val_metrics"]
            for k in got:
                got[k].append(100 * float(vm[k]))
        for col, key in ((3, "accuracy"), (4, "macro_f1"), (5, "balanced_accuracy")):
            m, s = mean_std(got[key])
            assert abs(num(r[col]) - m) < TOL, f"{task} {cond} {model} {key}: 论文 {r[col]} vs 复算 {m:.4f}"
            cell = r[col].replace("−", "-")
            std_txt = cell.split("±")[1] if "±" in cell else "0"
            assert abs(num(std_txt) - s) < TOL, f"{task} {cond} {model} {key} 标准差: {std_txt} vs {s:.4f}"


def test_table5_lda_matches_source(tables):
    _, rows = tables["5"]
    recs = list(csv.DictReader((PROJECT_ROOT / "reports/lda_check.csv").open(encoding="utf-8")))
    def lda(variant, task, cond):
        for x in recs:
            if x["variant"] == variant and x["task"] == task and x["condition"] == cond:
                return 100 * float(x["lda_acc"])
        raise AssertionError(f"lda_check.csv 里没有 {variant}/{task}/{cond}")
    for r in body_rows(rows):
        variant = {"干净": "base", "逐窗参考加噪，−6 dB": "snrm6",
                   "逐窗参考加噪，−10 dB": "snrm10"}[r[0]]
        task = "gear" if "齿轮" in r[1] else "bearing"
        cond = "20_0" if "20-0" in r[1] else "30_2"
        assert abs(num(r[2]) - lda(variant, task, cond)) < TOL, (r[1], r[0], r[2])
        # 差距列 = LDA 与 CNN/GCN 之差（表内自洽）
        cnn, gcn = [num(x) for x in r[3].split("/")]
        d1, d2 = [num(x) for x in r[4].split("/")] if "/" in r[4] else (num(r[4]), None)
        assert abs(d1 - (num(r[2]) - cnn)) < TOL, (r[1], r[0], "CNN 差距列")
        if d2 is not None:
            assert abs(d2 - (num(r[2]) - gcn)) < TOL, (r[1], r[0], "GCN 差距列")


def test_table6_literature_values_match_source(tables):
    _, rows = tables["6"]
    src = (PROJECT_ROOT / "reports/table_lit_baselines.md").read_text(encoding="utf-8")
    missing = []
    for r in body_rows(rows):
        if "This work" in r[0] or not r[0].strip() or not r[3].strip():
            continue        # 分组标题行（SEU 齿轮箱数据集 [1]）没有数值
        trip = [num(r[3]), num(r[4]), num(r[5])]
        pat = re.compile(r"\|\s*[^|]*\|\s*(?:—|[\d.,]+\s*M|[\d.]+\s*M)\s*\|\s*"
                         + r"\s*\|\s*".join(f"{t:.1f}" for t in trip))
        if not pat.search(src):
            missing.append(f"{r[0]} / {r[2]}: {trip}")
    assert not missing, "表 6 的文献值在 reports/table_lit_baselines.md 里找不到：\n" + "\n".join(missing)


def test_table7_efficiency_matches_source(tables):
    _, rows = tables["7"]
    eff = {x["model"]: x for x in json.loads(
        (PROJECT_ROOT / "reports/efficiency.json").read_text(encoding="utf-8"))}
    for r in body_rows(rows):
        if "本文 CNN" in r[0]:
            assert int(num(r[1])) == eff["cnn"]["params"], r
            assert int(num(r[2])) == eff["cnn"]["flops"], r
        elif "本文 GCN" in r[0]:
            assert int(num(r[1])) == eff["gcn"]["params"], r
            assert int(num(r[2])) == eff["gcn"]["flops"], r


def test_figure13_parameters_match_single_source(paper_text):
    """图 13 与正文里的参数量必须来自 reports/efficiency_all.json（各方法参数量的唯一真源）。"""
    eff = json.loads((PROJECT_ROOT / "reports/efficiency_all.json").read_text(encoding="utf-8"))
    for name, key in (("raw_cnn", "原始波形"), ("cnn", "本文 CNN"), ("gcn", "本文 GCN")):
        assert name in eff and eff[name]["params"] > 0
    # 论文里凡是出现 "0.xxxx M 参数" 的地方，都要能在 efficiency_all.json 里找到对应项
    for m in re.finditer(r"(0\.\d{4})\s*M", paper_text):
        val = float(m.group(1)) * 1e6
        assert any(abs(val - v["params"]) < 500 for v in eff.values()), \
            f"论文里的 {m.group(0)} 在 efficiency_all.json 里没有对应项"


#: 章节标题白名单。有意改标题时同步更新这里；否则任何标题被改动（少字/改字/调序）都会报错。
EXPECTED_HEADINGS = [
    "## 摘要",
    "## 1 引言",
    "## 2 提出的方法",
    "### 2.1 符号化",
    "### 2.2 多滞后联合转移概率张量",
    "### 2.3 分类器",
    "## 3 实验方法",
    "### 3.1 数据",
    "### 3.2 实验设置",
    "## 4 结果与讨论",
    "### 4.1 表示参数的信息论分析",
    "### 4.2 主结果",
    "### 4.3 表示的质量评估：判别力来自表示还是分类器？",
    "### 4.4 对比研究",
    "### 4.5 效率与可解释性",
    "### 4.6 抗噪性能",
    "## 5 抗噪评估的口径效应与表示的稳定性（受控分析）",
    "### 5.1 两种参考功率口径",
    "### 5.2 逐窗参考的一个性质",
    "### 5.3 对照实验",
    "### 5.4 对本文结论的影响",
    "### 5.5 适用边界与说明",
    "## 6 结论",
    "## 参考文献",
]


def test_section_headings_match_whitelist(paper_text):
    """标题白名单：标题是论文骨架，被静默改动必须立刻暴露。

    起因（2026-10-16）：`### 4.5 效率与可解释性` 被截成 `### 4.5 效率与可解释`（少一个"性"字）；
    当时已有的 16 项检查、以及肉眼通读，都没抓到，直到逐行 diff 才看出来。
    """
    got = [l.strip() for l in paper_text.split("\n")
           if l.startswith("## ") or l.startswith("### ")]
    if got == EXPECTED_HEADINGS:
        return
    diff = []
    for i in range(max(len(got), len(EXPECTED_HEADINGS))):
        a = got[i] if i < len(got) else "(缺失)"
        b = EXPECTED_HEADINGS[i] if i < len(EXPECTED_HEADINGS) else "(多余)"
        if a != b:
            diff.append(f"  实际: {a}\n  期望: {b}")
    raise AssertionError("章节标题与白名单不一致（若非有意修改，说明标题被误改）：\n"
                         + "\n".join(diff))


def test_no_latex_math_delimiters(paper_text):
    """正文里不得再出现 LaTeX 数学标记（$...$、\\times、\\frac…）。

    2026-10-17 用户要求："论文中有一些像 $ 这个符号的东西，改掉" —— 正文要进 Word 排版，
    留着 LaTeX 源码只会变成乱码。转换工具：`python src/utils/strip_latex_math.py <文件>`（改前先备份）。
    """
    bad = [f"第 {i} 行：{l.strip()[:110]}"
           for i, l in enumerate(paper_text.split("\n"), 1)
           if "$" in l or re.search(r"\\[a-zA-Z]{2,}", l)]
    assert not bad, "正文里还有 LaTeX 数学标记：\n" + "\n".join(bad)


def test_headings_are_not_glued_to_paragraphs(paper_text):
    """标题行必须与上一段之间留空行，否则 Markdown 渲染时标题不生效（Word 转换同理）。

    起因（2026-10-17）：`## 5 抗噪评估的口径效应…` 紧贴在上一段末尾，肉眼不易发现。
    """
    lines = paper_text.split("\n")
    bad = [f"第 {i + 1} 行：{l.strip()[:60]}"
           for i, l in enumerate(lines)
           if l.startswith("#") and i > 0 and lines[i - 1].strip() and not lines[i - 1].startswith("#")]
    assert not bad, "标题紧贴上一段（缺空行）：\n" + "\n".join(bad)


def test_equation3_indicator_is_defined(paper_text):
    """式 (3) 的指示函数必须写成 I[·] 并给出定义，不能写成 1[...]。

    起因（2026-10-17 用户圈出公式里的一段问"这是什么、是不是错了"）：
    `1[条件]` 原本是 LaTeX 里的 \\mathbf{1}（指示函数），转成纯文本后
    很容易被读成"数字 1 乘以方括号"，所以统一写成 I[·] 并在公式下定义。
    """
    assert "I[ s_c(t) = i ∧ s_c(t+k) = j ]" in paper_text, "式 (3) 的指示函数写法不是 I[·]"
    assert "I[·] 为**指示函数**" in paper_text, "式 (3) 后面没有给出指示函数的定义"
    assert "1[ s_c(t)" not in paper_text, "又出现了会被误读的 1[...] 写法"


def test_class_overlap_terminology_is_correct(paper_text):
    """“缺齿”与“齿面磨损”是两个不同类别 → 只能写“类间重叠”，不能写“类内重叠”。

    起因（2026-10-17 用户拿这句话上网核对别人怎么写）：文献里 intra-class 指同一类内部的
    散布/紧致度，两个不同类别之间的混淆是 inter-class overlap。原稿写成“类内重叠”是术语错误；
    而且“是该表示信息损失的体现”超出了对照实验能支撑的范围（对照只排除了幅值因素）。
    """
    assert "类内重叠" not in paper_text, "两个不同类别之间不能用“类内重叠”（应为“类间重叠”）"
    assert "类间重叠" in paper_text, "正文没有使用正确的“类间重叠”表述"
    assert "信息损失的体现" not in paper_text, "重叠 ≠ 信息损失：这是超出实验的断言"
