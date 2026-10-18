# -*- coding: utf-8 -*-
"""对 可对比文献对照表_给老师_核校版v2.docx 做最后 3 处修正（原地，先备份）：
  1. 表1 行序：GBDT 89.41% 应排在 SVM 89.7% 之前（表题声明"由低到高"）
  2. 第六节 AMMMP 出处补作者/页码/DOI
  3. 第六节 Dual-Channel 出处补作者
"""
import shutil, os, docx

BASE = r"C:/Users/lgt11/Desktop/EEG/EEG/doc/references"
SRC = os.path.join(BASE, "可对比文献对照表_给老师_核校版v2.docx")
BAK = os.path.join(BASE, ".patch_refs", "v2.backup.docx")

shutil.copy2(SRC, BAK)
print("备份 ->", BAK)

d = docx.Document(SRC)

# ---------- 1) 表1 行序修正 ----------
tbl = d.tables[1]
gbdt_tr = tbl.rows[9]._tr   # GBDT 89.41%
svm_tr = tbl.rows[8]._tr    # SVM 89.7%
assert "GBDT" in tbl.rows[9].cells[0].text, tbl.rows[9].cells[0].text
assert "SVM" in tbl.rows[8].cells[0].text, tbl.rows[8].cells[0].text
tbl._tbl.remove(gbdt_tr)
svm_tr.addprevious(gbdt_tr)
print("表1 行序已调整")

# ---------- 2)(3) 出处补全 ----------
def set_para_text(p, new_text):
    if not p.runs:
        p.add_run(new_text)
        return
    p.runs[0].text = new_text
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)

NEW = {
 "Fault Diagnosis of Gearbox Based on CNN and LSTM with Attention Mechanism":
   "Cheng K, Cheng L, Mao T, et al. Fault Diagnosis of Gearbox Based on CNN and LSTM with Attention Mechanism[C]. Highlights in Science, Engineering and Technology (AMMMP 2024), 2024, 93: 285-294. DOI: 10.54097/0s38yb63",
 "Dual-Channel Parallel Multimodal Feature Fusion for Bearing Fault Diagnosis":
   "Li W, Cai H, Yang X, et al. Dual-Channel Parallel Multimodal Feature Fusion for Bearing Fault Diagnosis[J]. Machines, 2025, 13(10): 950. DOI: 10.3390/machines13100950",
}

hit = set()
for p in d.paragraphs:
    t = p.text.strip()
    for k, v in NEW.items():
        if t.startswith(k):
            set_para_text(p, v)
            hit.add(k)
print("出处补全命中:", hit)
assert len(hit) == 2, "有出处未命中，检查段落文本"

d.save(SRC)
print("已保存:", SRC)
