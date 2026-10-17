"""论文稿完整性：图/表的引用、题注、图片文件三者必须自洽。

为什么要有这个测试：2026-10-16 删图 3(a) 时，我一次替换把"图 3(b) 的图片行"也删掉了，
剩下一个没有图片的题注——肉眼没看出来（题注还在，看着很正常）。这类错误在稿子里
极难发现，但自动化检查只要 3 行就能抓到。

检查项：
  1. 每个 `**图 N ...**` 题注的前 3 行内必须有一行 `![...](...)` —— 即"题注必须有图"；
  2. 每个被正文引用的 `图 N` 都要有对应题注（不许悬空引用）；
  3. 每个 `![...](file)` 引用的图片文件必须存在于 reports/ 下（不许断链）；
  4. 每个 `**表 N ...**` 题注必须紧跟一个 Markdown 表格（不许题注没表）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"


@pytest.fixture(scope="module")
def lines() -> list[str]:
    assert PAPER.exists(), f"论文稿不存在：{PAPER}"
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def test_every_figure_caption_has_an_image(lines):
    missing = []
    for i, l in enumerate(lines):
        if l.startswith("**图 "):
            window = "\n".join(lines[max(0, i - 3):i])
            if "![" not in window:
                missing.append(f"第 {i + 1} 行：{l[:40]}")
    assert not missing, "题注没有对应图片：\n" + "\n".join(missing)


def test_no_dangling_figure_reference(lines):
    text = "\n".join(lines)
    referenced = {int(m) for m in re.findall(r"图 (\d+)", text)}
    captioned = {int(m) for m in re.findall(r"\*\*图 (\d+)", text)}
    dangling = sorted(referenced - captioned)
    assert not dangling, f"正文引用了但没有题注的图号：{dangling}"


def test_embedded_images_exist(lines):
    missing = []
    for l in lines:
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", l)
        if m:
            f = PROJECT_ROOT / "reports" / m.group(1)
            if not f.exists():
                missing.append(m.group(1))
    assert not missing, f"图片文件不存在（断链）：{missing}"


def test_every_table_caption_is_followed_by_a_table(lines):
    bad = []
    for i, l in enumerate(lines):
        if re.match(r"\*\*表 \d", l):
            nxt = [x for x in lines[i + 1:i + 4] if x.strip()]
            if not nxt or not nxt[0].lstrip().startswith("|"):
                bad.append(f"第 {i + 1} 行：{l[:40]}")
    assert not bad, "表题后面没有表格：\n" + "\n".join(bad)

def test_every_figure_has_an_explanation(lines):
    """有图必有讲解：每张图都要在正文里有一段**实质解释**，而不是只被"见图 N"点名。

    2026-10-17 用户问"有没有这种测试"。此前只有"图被引用"的检查，没有
    "图被解释"的检查——一张图可以只出现"如图 8 所示"，正文里没有任何描述。

    判定：正文（去掉题注/图片行）里存在一句提到"图 N"的话，且这句话
    要么含解释性动词（给出/展示/显示/可以看到/对比/说明…），要么足够长（≥40 字）。
    "如图 N 所示"这类纯引用不算。
    """
    text = "\n".join(lines)
    body = "\n".join(l for l in lines
                   if not l.startswith("!") and not l.strip().startswith("**图")
                   and not l.strip().startswith("**表"))
    fig_nums = sorted({int(m) for m in re.findall(r"\*\*图 (\d+)", text)})
    verbs = ("给出", "展示", "显示", "可以看到", "可见", "表明", "说明", "对比",
             "画出", "呈现", "标出", "列出", "总结", "描述", "示意", "汇总", "进一步")
    missing = []
    for n in fig_nums:
        ok = False
        for s in re.split(r"[。；\n]", body):
            if f"图 {n}" not in s:
                continue
            if len(s.strip()) >= 40 or any(v in s for v in verbs):
                ok = True
                break
        if not ok:
            missing.append(n)
    assert not missing, f"这些图在正文里只有引用、没有实质讲解：{missing}"

