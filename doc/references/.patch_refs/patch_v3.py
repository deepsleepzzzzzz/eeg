# -*- coding: utf-8 -*-
"""文献对照表 v2 → 补表格编号 + 修正引用 + 补口径说明（B 方案）

改动清单：
  1. 四张表加正式表注（表 1 / 表 2 / 表 3 / 表 4）
  2. 正文 5 处「表 1/表 2」引用改为正确编号
  3. 基准表前补口径说明（CNN + GCN 两架构）
  4. 引用注意事项补 v4 的规范句（验证集调参、测试集封闭）
"""
import copy
import re
import shutil
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

SRC = Path("可对比文献对照表_给老师_核校版v2.docx")
OUT = Path("可对比文献对照表_给老师_核校版v3.docx")

d = docx.Document(str(SRC))
body = d.element.body


def make_caption(template_par, text):
    """按模板段落样式造一个「表 N　xxx」表注段（居中、加粗小字号）。"""
    new_p = copy.deepcopy(template_par._element)
    # 清掉所有 run
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    # 清掉书签等
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for e in new_p.findall(qn(tag)):
            new_p.remove(e)
    # 设 pPr：居中 + 无首行缩进
    pPr = new_p.find(qn("w:pPr"))
    if pPr is None:
        pPr = new_p.makeelement(qn("w:pPr"), {})
        new_p.insert(0, pPr)
    for ind in pPr.findall(qn("w:ind")):
        pPr.remove(ind)
    for jc in pPr.findall(qn("w:jc")):
        pPr.remove(jc)
    ind = pPr.makeelement(qn("w:ind"), {qn("w:firstLine"): "0"})
    pPr.append(ind)
    jc = pPr.makeelement(qn("w:jc"), {qn("w:val"): "center"})
    pPr.append(jc)
    return new_p, text


# ---------- 准备：取一个「宋体、首行缩进」的正文段作模板 ----------
tpl = None
for p in d.paragraphs:
    if p.text.strip().startswith("本表按准确率由低到高"):
        tpl = p
        break
assert tpl is not None, "找不到模板段落"

# ---------- 改动 5/6/7/8：先改正文引用（改文字，不动结构）----------
REPL_TEXT = [
    ("本表结构与表 1 相同", "本表结构与表 2 相同"),
    # 第 3 条整句替换（表1→表2，表2→表3）
    ("本文列数值须注明出处。表 1 的 97.69 ~ 99.85% 与表 2 的 99.85 ~ 100% 分别出自本文齿轮箱、轴承实验记录，不应与其他协议下的复现值混用。",
     "本文列数值须注明出处。表 2 的 97.69 ~ 99.85% 与表 3 的 99.85 ~ 100% 分别出自本文齿轮箱、轴承实验记录，不应与其他协议下的复现值混用。"),
    ("（详见表 1、表 2 对应列）", "（详见表 2、表 3 对应列）"),
    ("（表 1、表 2）", "（表 2、表 3）"),
]

# 逐段做 run 级替换：把段落全文的替换结果写回「首个 run」，其余 run 清空
def replace_in_paragraph(par, old, new):
    full = par.text
    if old not in full:
        return False
    new_full = full.replace(old, new)
    runs = par.runs
    if not runs:
        return False
    # 保留首个 run 的格式，写入新全文；其余 run 文本清空
    runs[0].text = new_full
    for r in runs[1:]:
        r.text = ""
    return True


c5 = c6 = c7 = c8 = 0
for p in d.paragraphs:
    t = p.text
    if "本表结构与表 1 相同" in t and c5 == 0:
        assert replace_in_paragraph(p, "表 1", "表 2"), "改动5失败"
        c5 += 1
    elif t.startswith("本文列数值须注明出处。"):
        assert replace_in_paragraph(p, "表 1 的", "表 2 的") and replace_in_paragraph(p, "表 2 的 99.85", "表 3 的 99.85"), "改动6失败"
        c6 += 1
    elif "详见表 1、表 2 对应列" in t:
        assert replace_in_paragraph(p, "表 1、表 2", "表 2、表 3"), "改动7失败"
        c7 += 1
    elif t.startswith("在 SEU 数据集上，本文方法"):
        assert replace_in_paragraph(p, "（表 1、表 2）", "（表 2、表 3）"), "改动8失败"
        c8 += 1

print(f"正文引用替换：改动5={c5} 改动6={c6} 改动7={c7} 改动8={c8}")
assert (c5, c6, c7, c8) == (1, 1, 1, 1), "引用替换计数异常"

# ---------- 改动 3/4：补两段说明 ----------
# 改动3：基准表前补口径说明（放在「下表数值出自本文实验记录…」之后）
for p in d.paragraphs:
    if p.text.strip().startswith("下表数值出自本文实验记录"):
        new_p, txt = make_caption(tpl, "")
        # 该段为正文段（保留首行缩进、两端对齐、左对齐而非居中）
        pPr = new_p.find(qn("w:pPr"))
        for jc in pPr.findall(qn("w:jc")):
            pPr.remove(jc)
        for ind in pPr.findall(qn("w:ind")):
            pPr.remove(ind)
        pPr.append(pPr.makeelement(qn("w:ind"), {qn("w:firstLine"): "480"}))
        pPr.append(pPr.makeelement(qn("w:jc"), {qn("w:val"): "both"}))
        r = new_p.makeelement(qn("w:r"), {})
        rpr = new_p.makeelement(qn("w:rPr"), {})
        rf = new_p.makeelement(qn("w:rFonts"), {
            qn("w:ascii"): "宋体", qn("w:eastAsia"): "宋体",
            qn("w:hAnsi"): "宋体", qn("w:cs"): "宋体", qn("w:hint"): "eastAsia"})
        rpr.append(rf)
        r.append(rpr)
        t = new_p.makeelement(qn("w:t"), {})
        t.text = "表中「本文」行取自 CNN 与 GCN 两个模型在同一划分下的实测结果，区间上、下界分别对应其中较高与较低者。"
        r.append(t)
        new_p.append(r)
        p._element.addnext(new_p)
        print("已补：基准表口径说明")
        break
else:
    raise AssertionError("找不到基准表说明段")

# 改动4：引用注意事项开头补规范句
for p in d.paragraphs:
    if p.text.strip().startswith("本节四条为引用时必须同时写明的条件"):
        new_p = copy.deepcopy(p._element)
        for r in new_p.findall(qn("w:r")):
            new_p.remove(r)
        r = new_p.makeelement(qn("w:r"), {})
        rpr = new_p.makeelement(qn("w:rPr"), {})
        rf = new_p.makeelement(qn("w:rFonts"), {
            qn("w:ascii"): "宋体", qn("w:eastAsia"): "宋体",
            qn("w:hAnsi"): "宋体", qn("w:cs"): "宋体", qn("w:hint"): "eastAsia"})
        rpr.append(rf)
        r.append(rpr)
        t = new_p.makeelement(qn("w:t"), {})
        t.text = "本文所有结论均只在验证集上调参、测试集全程封闭（参数冻结后仅评估一次），下表所列本文数值即出自该密封测试集。"
        r.append(t)
        new_p.append(r)
        p._element.addnext(new_p)
        print("已补：测试集封闭规范句")
        break
else:
    raise AssertionError("找不到引用注意事项段")

# ---------- 改动 1/2：给 4 张表加表注 ----------
CAPS = [
    ("表 1", "本文基准结果（密封测试集，随机种子 42）"),
    ("表 2", "齿轮箱任务对照表（同库、同任务 5 分类）"),
    ("表 3", "轴承任务对照表（同库、同任务 5 分类）"),
    ("表 4", "不纳入本表的条目及原因"),
]

# 重新扫描 body，找 4 个 tbl 元素
tbls = [ch for ch in body.iterchildren() if ch.tag == qn("w:tbl")]
assert len(tbls) == 4, f"表格数 {len(tbls)} != 4"

for (num, title), tbl_el in zip(CAPS, tbls):
    new_p, _ = make_caption(tpl, "")
    r = new_p.makeelement(qn("w:r"), {})
    rpr = new_p.makeelement(qn("w:rPr"), {})
    rf = new_p.makeelement(qn("w:rFonts"), {
        qn("w:ascii"): "宋体", qn("w:eastAsia"): "宋体",
        qn("w:hAnsi"): "宋体", qn("w:cs"): "宋体", qn("w:hint"): "eastAsia"})
    rpr.append(rf)
    b = new_p.makeelement(qn("w:b"), {})
    rpr.append(b)
    r.append(rpr)
    t = new_p.makeelement(qn("w:t"), {})
    t.text = f"{num}　{title}"
    r.append(t)
    new_p.append(r)
    tbl_el.addprevious(new_p)
    print(f"已加表注：{num}　{title}")

d.save(str(OUT))
print()
print("已保存:", OUT)
