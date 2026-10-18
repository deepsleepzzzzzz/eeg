# -*- coding: utf-8 -*-
"""v3 → v4：在齿轮箱对照表下方标明"三个数字在本表未列"及原因。

三个数字：
  99.41%  IEEE TIM 2021 作者自身方法（30-2 工况），高于本文该工况 → 只列其对比基线
  98.50%  AMMMP 2024 作者自报结果，高于本文齿轮箱下限 → 只列其对比基线
  96.02%  Algorithms 2023（SVMD 熵 + RF，齿轮任务）样本内拟合值 → 已按该口径列入

改动清单：
  1. 表 2 的表注块末追加一行「本表未列条目说明」
  2. 「引用注意事项」中补一条 5.，与之呼应
"""
import copy
import docx
from docx.oxml.ns import qn
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / ".." / "可对比文献对照表_给老师_核校版v3.docx"
OUT = BASE / ".." / "可对比文献对照表_给老师_核校版v4.docx"

d = docx.Document(str(SRC))

NOTE_TPL = (
    "本表未列条目说明：IEEE TIM 2021 与 AMMMP 2024 两文作者所提方法自身报数分别达 "
    "99.41%（30 Hz-2 V）与 98.50%，高于或接近本文对应工况结果，故本表仅列其对比基线行，"
    "完整数值可参见原文；Algorithms 2023 无独立测试集，其齿轮任务最高 96.02% 为样本内拟合值，"
    "已按该口径列入表中。"
)


def make_par_after(anchor_par, text):
    """在 anchor_par 之后插入一个同款正文段（宋体、首行缩进 2 字、两端对齐）。"""
    new_p = copy.deepcopy(anchor_par._element)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for e in new_p.findall(qn(tag)):
            new_p.remove(e)

    pPr = new_p.find(qn("w:pPr"))
    if pPr is None:
        pPr = new_p.makeelement(qn("w:pPr"), {})
        new_p.insert(0, pPr)
    for ind in pPr.findall(qn("w:ind")):
        pPr.remove(ind)
    for jc in pPr.findall(qn("w:jc")):
        pPr.remove(jc)
    pPr.append(pPr.makeelement(qn("w:ind"), {qn("w:firstLine"): "480"}))
    pPr.append(pPr.makeelement(qn("w:jc"), {qn("w:val"): "both"}))

    r = new_p.makeelement(qn("w:r"), {})
    rpr = new_p.makeelement(qn("w:rPr"), {})
    rpr.append(new_p.makeelement(qn("w:rFonts"), {
        qn("w:ascii"): "宋体", qn("w:eastAsia"): "宋体",
        qn("w:hAnsi"): "宋体", qn("w:cs"): "宋体", qn("w:hint"): "eastAsia"}))
    r.append(rpr)
    t = new_p.makeelement(qn("w:t"), {})
    t.text = text
    r.append(t)
    new_p.append(r)

    anchor_par._element.addnext(new_p)
    return new_p


# ---------- 改动 1：表 2 表注块末追加一行 ----------
done1 = False
for p in d.paragraphs:
    t = p.text
    if t.startswith("＊ 该行原文附有引用标记") and "注：除标注外，本表各方法" in t:
        # 该段是多行注块，整段文本末尾追加换行 + 新行
        full = t.rstrip()
        new_full = full + "\n" + NOTE_TPL
        runs = p.runs
        assert runs, "表 2 注块无 run"
        runs[0].text = new_full
        for r in runs[1:]:
            r.text = ""
        done1 = True
        print("已改：表 2 表注块追加「本表未列条目说明」")
        break
assert done1, "找不到表 2 表注块"

# ---------- 改动 2：引用注意事项补第 5 条 ----------
# 插到「对照性质须在正文中说明…」这一段之后
done2 = False
for p in d.paragraphs:
    if p.text.strip().startswith("对照性质须在正文中说明"):
        make_par_after(p, NOTE_TPL)
        done2 = True
        print("已改：引用注意事项补入同一条说明")
        break
assert done2, "找不到对照性质段"

# 把「本节四条」改成「本节五条」
for p in d.paragraphs:
    if p.text.strip().startswith("本节四条为引用时必须同时写明的条件"):
        runs = p.runs
        assert runs
        full = p.text.replace("本节四条为引用时必须同时写明的条件", "本节五条为引用时必须同时写明的条件")
        runs[0].text = full
        for r in runs[1:]:
            r.text = ""
        print("已改：「本节四条」→「本节五条」")
        break
else:
    raise AssertionError("找不到本节四条段")

d.save(str(OUT))
print()
print("已保存:", OUT.resolve())
