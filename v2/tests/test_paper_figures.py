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
