# -*- coding: utf-8 -*-
"""查 v4 表格标题的实际承载方式"""
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

d = docx.Document('../导师补充要求_图表与实验汇总_v4.docx')
items = []
for ch in d.element.body.iterchildren():
    if ch.tag.endswith('}p'):
        items.append(('P', Paragraph(ch, d)))
    elif ch.tag.endswith('}tbl'):
        items.append(('T', Table(ch, d)))

for i, (k, o) in enumerate(items):
    if k == 'P':
        t = o.text.strip()
        if t and ('表' in t or '图' in t) and len(t) < 120:
            print(f'{i:3} P {t[:110]}')
