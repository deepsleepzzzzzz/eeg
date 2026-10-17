"""图文适配测试：题注里写的"图是怎么画的"，必须与生成图时的参数一致。

为什么要有（2026-10-17 用户问"图文是否适配有没有写入测试"）：
原来的 test_paper_figures 只检查**结构**——题注有没有图、图有没有断链、图号连不连续；
它不检查**内容**——题注说"第 3 个滞后"，图里画的是不是第 3 个？题注说"行归一化"，
图里画的是联合概率还是条件概率？这类"图文不符"能一路通过所有旧测试。

本文件做三件事：
  1. 图 10 / 图 S2：题注必须与 `reports/fig_transition_meta.json`（绘图函数写出的参数台账）
     一致——通道、滞后、归一化方式、色标、主图工况、全量工况；
  2. 图 8：题注必须与 `reports/feature_tsne.json` 一致——64 维、perplexity 50、随机种子 42；
  3. 其他图：题注必须写明其关键属性（验证集 / 种子数 / 指标 / 噪声口径），
     这些是读图的前提，缺了就是图文不适配。

改图或改题注时，若两边不一致，本文件会失败——这正是我们要的。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"
REPORTS = PROJECT_ROOT / "reports"


@pytest.fixture(scope="module")
def paper() -> str:
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")


def caption(paper: str, label: str) -> str:
    """取 `**图 <label> ...**` 整行；label 可为 '10'、'S2'。"""
    for line in paper.split("\n"):
        if re.match(rf"^\*\*图 {re.escape(label)}\s", line.strip()):
            return line.strip()
    raise AssertionError(f"找不到图 {label} 的题注")


def test_figure10_and_s2_caption_matches_transition_meta(paper):
    meta = json.loads((REPORTS / "fig_transition_meta.json").read_text(encoding="utf-8"))
    # 图的真实参数
    assert meta["channel_index"] == 0 and meta["channel"] == "x"
    assert meta["lag_index"] == 2, "图 10 用的是第 3 个滞后（0-based index 2）"
    assert meta["normalization"] == "row", "图 10/S2 是行归一化后的条件转移概率"
    assert abs(float(meta["vmax"]) - 0.55) < 1e-9
    assert meta.get("difference_row") is True, "图 10 的下排是相对类均值的差异图"
    assert abs(float(meta["vmax_diff"]) - 0.1454) < 0.01
    assert meta["main_conditions"] == ["gear/20_0"]
    assert len(meta["full_conditions"]) == 4 and meta["classes"] == 5

    c10 = caption(paper, "10")
    for token in ("齿轮 20-0", "第 3 个滞后", "行归一化", "验证集", "色标 0–0.55", "图 S2",
                  "上排", "下排", "相对类均值"):
        assert token in c10, f"图 10 题注缺少『{token}』：{c10}"

    cs2 = caption(paper, "S2")
    for token in ("四个任务-工况", "行归一化", "第 3 个滞后", "色标 0–0.55"):
        assert token in cs2, f"图 S2 题注缺少『{token}』：{cs2}"


def test_figure8_caption_matches_feature_tsne_meta(paper):
    meta = json.loads((REPORTS / "feature_tsne.json").read_text(encoding="utf-8"))
    dims = {v["embedding_dim"] for v in meta.values()}
    seeds = {v["seed"] for v in meta.values()}
    assert dims == {64}, f"图 8 的嵌入维度应为 64，实际 {dims}"
    assert seeds == {42}, f"图 8 应只用随机种子 42，实际 {seeds}"

    c8 = caption(paper, "8")
    for token in ("64 维", "t-SNE", "perplexity = 50", "随机种子 42", "验证集"):
        assert token in c8, f"图 8 题注缺少『{token}』：{c8}"


def test_other_figure_captions_declare_key_attributes(paper):
    required = {
        "7": ("混淆矩阵", "验证集", "5 个种子"),
        "9": ("参数量–精度", "3 个种子", "Pareto"),
        "12": ("参考功率口径", "验证集"),
        "13": ("AWGN", "验证集"),
        "S1": ("t-SNE", "两种工况叠加"),
    }
    for label, tokens in required.items():
        c = caption(paper, label)
        for token in tokens:
            assert token in c, f"图 {label} 题注缺少『{token}』：{c}"