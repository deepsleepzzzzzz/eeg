"""元测试：证明"论文正确性测试"真的能抓到错误，而不是永远通过。

背景（2026-10-17 用户要求："要防止测试本身就是错的"）：
项目里已经有一批论文测试（表 4/5/8、摘要数字、图号、术语……），但如果某条测试
写成了"假检查"（例如只断言字符串存在、或把论文数字直接抄成期望值），它会在
真出现错误时依然通过。本文件对论文测试做**变异测试**：

  1. 取一份正确的论文文本；
  2. 逐条注入**已知错误**（改一个表格数字、改一个摘要倍数、注入 $、把"类间"改成
     "类内"、让图号跳号……）；
  3. 断言对应的测试**必须失败**；
  4. 同时断言未变异的原文**必须通过**（防止"测试因为环境坏了而永远失败"，
     那也是另一种"测试本身有问题"）。

任何一条"注入错误后测试仍通过"都会让本文件失败，并指出是哪条测试无效。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS = PROJECT_ROOT / "tests"
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INTEGRITY = _load("test_paper_integrity")
NOISE = _load("test_paper_noise_tables")
PROSE = _load("test_paper_prose_numbers")
FIGCONS = _load("test_figure_caption_consistency")


def _parse_tables(paper_text: str) -> dict:
    """与 test_paper_integrity.tables fixture 相同的解析规则（独立实现一份，避免依赖 fixture）。"""
    lines = paper_text.split("\n")
    out, i = {}, 0
    while i < len(lines):
        m = re.match(r"\*\*表 (\d+[ab]?)\s+(.*?)\*\*", lines[i].strip())
        if m:
            rows, j = [], i + 1
            while j < len(lines):
                if lines[j].lstrip().startswith("|"):
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                    j += 1
                elif rows and not lines[j].strip():
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


@pytest.fixture(scope="module")
def original() -> str:
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")


def _mutate(text: str, old: str, new: str, count: int = 1) -> str:
    assert old in text, f"变异点不存在：{old!r}（论文被改过？请同步本测试）"
    return text.replace(old, new, count)


def _expect_fail(monkeypatch, tmp_path, module, name, text, *args):
    """把模块的 PAPER 指向变异文本，调用目标测试，断言它失败。"""
    p = tmp_path / f"mutated_{name}.md"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(module, "PAPER", p, raising=False)
    fn = getattr(module, name)
    with pytest.raises(AssertionError):
        fn(*args)


def _expect_pass(monkeypatch, module, name, text, *args):
    """把模块的 PAPER 指回原文，断言目标测试通过（正对照）。"""
    monkeypatch.setattr(module, "PAPER", PAPER, raising=False)
    getattr(module, name)(*args)


# --------------------------------------------------------------------------- #
# 一、表格数字：改一个数，逐格核对的测试必须报警
# --------------------------------------------------------------------------- #
def test_table4_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "99.82 ± 0.06", "99.92 ± 0.06")
    tables = _parse_tables(mutated)
    _expect_fail(monkeypatch, tmp_path, INTEGRITY, "test_table4_main_results", mutated, tables)
    _expect_pass(monkeypatch, INTEGRITY, "test_table4_main_results", original, _parse_tables(original))


def test_table8_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "| 干净 | 99.69 |", "| 干净 | 98.69 |")
    _expect_fail(monkeypatch, tmp_path, NOISE, "test_table8_noise_accuracy_is_reproducible",
                 mutated, "齿轮 20-0", "gear", "20_0")
    _expect_pass(monkeypatch, NOISE, "test_table8_noise_accuracy_is_reproducible",
                 original, "齿轮 20-0", "gear", "20_0")


def test_table5_network_columns_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "99.82 / 99.71", "99.72 / 99.71")
    _expect_fail(monkeypatch, tmp_path, PROSE, "test_table5_network_columns_match_artifacts", mutated)
    _expect_pass(monkeypatch, PROSE, "test_table5_network_columns_match_artifacts", original)


# --------------------------------------------------------------------------- #
# 二、正文数字 / 措辞：改倍数、注入 $、换术语
# --------------------------------------------------------------------------- #
def test_efficiency_ratio_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = original.replace("1/139", "1/129")   # 全文 3 处都要改，否则 in_paper 仍能找到"1/139"
    _expect_fail(monkeypatch, tmp_path, PROSE, "test_efficiency_text_numbers", mutated, mutated)
    _expect_pass(monkeypatch, PROSE, "test_efficiency_text_numbers", original, original)


def test_learned_tsne_number_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "嵌入空间轮廓系数：0.69 / 0.79", "嵌入空间轮廓系数：0.59 / 0.79")
    _expect_fail(monkeypatch, tmp_path, PROSE, "test_learned_feature_tsne_matches_source", mutated, mutated)
    _expect_pass(monkeypatch, PROSE, "test_learned_feature_tsne_matches_source", original, original)


def test_latex_dollar_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = original.replace("给定长度为 T 的振动窗", "给定长度为 $T$ 的振动窗", 1)
    _expect_fail(monkeypatch, tmp_path, INTEGRITY, "test_no_latex_math_delimiters", mutated, mutated)
    _expect_pass(monkeypatch, INTEGRITY, "test_no_latex_math_delimiters", original, original)


def test_overlap_terminology_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "类间重叠", "类内重叠")
    _expect_fail(monkeypatch, tmp_path, INTEGRITY, "test_class_overlap_terminology_is_correct", mutated, mutated)
    _expect_pass(monkeypatch, INTEGRITY, "test_class_overlap_terminology_is_correct", original, original)


# --------------------------------------------------------------------------- #
# 三、结构：让图号跳号，连续性测试必须报警
# --------------------------------------------------------------------------- #
def test_figure_numbering_mutation_is_caught(monkeypatch, tmp_path, original):
    mutated = _mutate(original, "**图 12", "**图 14")   # 必须改题注，改正文引用不会影响连续编号
    _expect_fail(monkeypatch, tmp_path, INTEGRITY, "test_figure_and_table_numbers_are_contiguous",
                 mutated, mutated, _parse_tables(mutated))
    _expect_pass(monkeypatch, INTEGRITY, "test_figure_and_table_numbers_are_contiguous",
                 original, original, _parse_tables(original))


def test_figure_caption_mutation_is_caught(monkeypatch, tmp_path, original):
    """图文适配测试必须能抓到"题注与图不符"：把图 10 的"第 3 个滞后"改成"第 2 个"。"""
    mutated = _mutate(original,
                     "**图 10  齿轮 20-0 各类的行归一化平均转移矩阵（上）与相对类均值的差异（下）**（x 通道，第 3 个滞后，验证集；",
                     "**图 10  齿轮 20-0 各类的行归一化平均转移矩阵（上）与相对类均值的差异（下）**（x 通道，第 2 个滞后，验证集；")
    _expect_fail(monkeypatch, tmp_path, FIGCONS, "test_figure10_and_s2_caption_matches_transition_meta",
                 mutated, mutated)
    _expect_pass(monkeypatch, FIGCONS, "test_figure10_and_s2_caption_matches_transition_meta",
                 original, original)
