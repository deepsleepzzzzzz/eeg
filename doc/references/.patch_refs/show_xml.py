# -*- coding: utf-8 -*-
import re
import docx

d = docx.Document('可对比文献对照表_给老师_核校版v2.docx')
keys = ['本表按准确率由低到高', '本表结构与表', '下表数值出自本文实验记录', '以下文献虽使用 SEU']
for p in d.paragraphs:
    t = p.text.strip()
    if any(t.startswith(k) for k in keys):
        x = p._element.xml
        x = re.sub(r'xmlns:\w+="[^"]*"\s*', '', x)
        print('---', t[:45], '---')
        print(x[:900])
        print()
