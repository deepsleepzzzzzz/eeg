"""把论文 Markdown（reports/论文初稿_v1.md）转成 Word（.docx）。

只做"结构级"转换：标题/正文/加粗/表格/图片/公式段落/参考文献，不做期刊排版。
中文字体：正文宋体、标题黑体；西文 Times New Roman；正文小四（10.5pt）、1.5 倍行距。

用法：python src/utils/md_to_docx.py [输入md] [输出docx]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_run_font(run, east: str = "宋体", latin: str = "Times New Roman", size: float = 10.5,
                 bold: bool = False) -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), east)


def add_runs_with_bold(paragraph, text: str, east: str = "宋体", latin: str = "Times New Roman",
                       size: float = 10.5, base_bold: bool = False) -> None:
    """按 **加粗** 分段写 run。"""
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos:m.start()])
            set_run_font(r, east, latin, size, base_bold)
        r = paragraph.add_run(m.group(1))
        set_run_font(r, east, latin, size, True)
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        set_run_font(r, east, latin, size, base_bold)


def img_width_inches(path: Path) -> float:
    try:
        from PIL import Image
        w, h = Image.open(path).size
        max_w, max_h = 6.2, 8.0
        return min(max_w, max_h * w / h) if h else 6.2
    except Exception:
        return 6.0


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "reports" / "论文初稿_v1.md"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "reports" / "论文初稿_v1.docx"
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst

    doc = Document()
    # 默认样式：宋体 + Times New Roman，小四，1.5 倍行距
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5

    lines = src.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    base_dir = src.parent
    n_img = n_tab = 0
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # 图片
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", stripped)
        if m:
            p = Path(m.group(1))
            if not p.is_absolute():
                p = base_dir / p
            if p.exists():
                doc.add_picture(str(p), width=Inches(img_width_inches(p)))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                n_img += 1
            else:
                doc.add_paragraph(f"[缺图] {m.group(1)}")
            i += 1
            continue
        # 表格
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set("".join(c)) <= set("-: ") for c in cells):  # 跳过分隔行
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncol)
                table.style = "Table Grid"
                for ri, r in enumerate(rows):
                    cells = table.add_row().cells
                    for ci in range(ncol):
                        txt = r[ci] if ci < len(r) else ""
                        para = cells[ci].paragraphs[0]
                        add_runs_with_bold(para, txt, east="宋体", size=9, base_bold=(ri == 0))
                n_tab += 1
                doc.add_paragraph("")
            continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            h = doc.add_heading(level=level)
            add_runs_with_bold(h, m.group(2), east="黑体", latin="Times New Roman",
                              size={1: 16, 2: 14, 3: 12, 4: 11}[level], base_bold=True)
            i += 1
            continue
        # 水平线 / 引用 / 普通段落
        if stripped == "---" or set(stripped) <= {"-", "="}:
            i += 1
            continue
        if stripped.startswith(">"):
            txt = stripped.lstrip(">").strip()
        else:
            txt = stripped
        para = doc.add_paragraph()
        add_runs_with_bold(para, txt)
        if re.match(r"^\[\d+\]", txt):  # 参考文献：悬挂缩进
            para.paragraph_format.left_indent = Inches(0.3)
            para.paragraph_format.first_line_indent = Inches(-0.3)
        i += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst))
    print(f"[ok] {dst.relative_to(PROJECT_ROOT) if dst.is_relative_to(PROJECT_ROOT) else dst}")
    print(f"      图片 {n_img} 张，表格 {n_tab} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())