r"""把论文正文里的 LaTeX 数学定界符 `$ ... $` 换成可读的纯文本。

为什么：论文要放进 Word 走期刊模板，正文里留着 `$` 和 `\times`、`\frac` 这类命令，
排版时要么原样显示成乱码、要么被 Word 当成普通字符。科学论文的公式仍然要用纯文本写清楚，
而不是留一堆 LaTeX 源码。

做两件事：
  1. 行间公式块（`$$` 包起来的多行）→ 一行纯文本公式，保留式号 (1)(2)…（正文里的"式 (7)"仍然对得上）；
  2. 行内 `$...$` → 纯文本（`\times`→×、`\le`→≤、`\frac{a}{b}`→a/b、`\theta`→θ …）。

用法：
    python src/utils/strip_latex_math.py --check reports/论文初稿_v1.md      # 只报告，不写
    python src/utils/strip_latex_math.py reports/论文初稿_v1.md              # 就地改写（先备份！）

改完必须跑 `python verify_all.py`：`tests/test_paper_integrity.py` 里有一条
"正文不得再出现 $ 或 LaTeX 命令"的守卫，防止回潮。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console  # noqa: E402

#: 行间公式块：按式号给出纯文本写法（式号必须保留，正文引用"式 (N)"靠它）
DISPLAY = {
    1: "s_c(t) = min{ b : x_c(t) < θ_(b+1) },   b = 0, …, B-1",
    2: "θ_b = Q_(b/B)( { x_c^train(t) } ),   b = 1, …, B-1",
    3: "P_k^(c)(i,j) = (1/(T−k)) · Σ_(t=1..T−k) I[ s_c(t) = i ∧ s_c(t+k) = j ]",
    4: "P = [ P_k^(c) ]_(c=1..C, k∈K) ∈ R^((C·K)×B×B)",
    5: "I(k) = Σ_(i,j) P_k(i,j) · log( P_k(i,j) / (p_i · p_j) )",
    6: "I(1)/H(s),   H(s) = −Σ_i p_i · log p_i",
    7: "p_noise = p_ref / 10^(R/10)",
    8: "E[加噪窗能量] = p_sig + p_noise = p_sig · (1 + 10^(−R/10))",
}

#: 行内片段：先按这张表精确替换（项目里出现过的全部写法）
INLINE = {
    r"(C\cdot K)\times B\times B": "(C·K)×B×B",
    r"1,2,3,\dots": "1, 2, 3, …",
    r"1,3,5,\dots": "1, 3, 5, …",
    r"1\!-\!5": "1–5",
    r"1\!-\!10": "1–10",
    r"1\times10^{-4}": "1×10^-4",
    r"2\times2": "2×2",
    r"3\times3": "3×3",
    r"6\times6": "6×6",
    r"30\times6\times6": "30×6×6",
    r"5\times10^{-4}": "5×10^-4",
    r"2048^2": "2048^2",
    r"N\times N": "N×N",
    r"B\ge10": "B≥10",
    r"k\le10": "k≤10",
    r"\le 0.06": "≤0.06",
    r"k\in\mathcal{K}": "k∈K",
    r"\mathcal{K}=\{1,\dots,10\}": "K = {1, …, 10}",
    r"\mathcal{K}": "K",
    r"\{1,3,5,7,9\}": "{1, 3, 5, 7, 9}",
    r"\{1,3,\dots,19\}": "{1, 3, …, 19}",
    r"1,3,5,\dots": "1, 3, 5, …",
    r"\theta_1<\cdots<\theta_{B-1}": "θ_1 < … < θ_(B-1)",
    r"\mu_c,\sigma_c": "μ_c, σ_c",
    r"\tau": "τ",
    r"x \in \mathbb{R}^{C\times T}": "x ∈ R^(C×T)",
    r"p_{\text{ref}}=p_{\text{sig}}": "p_ref = p_sig",
    r"p_{\text{sig}}=\frac{1}{CT}\sum_{c,t}x_c^2(t)": "p_sig = (1/(C·T)) · Σ_(c,t) x_c^2(t)",
    r"p_{\text{ref}}": "p_ref",
    r"P^{(c)}_k(i,j)": "P_k^(c)(i,j)",
    r"P_k": "P_k",
    r"[0,733{,}992)": "[0, 733992)",
    r"[733{,}992,891{,}276)": "[733992, 891276)",
    r"[891{,}276,1{,}048{,}560)": "[891276, 1048560)",
    r"T=2048,B=6": "T=2048, B=6",
    r"I(1)/H(s)": "I(1)/H(s)",
}

#: 兜底替换（上面的表没覆盖到的新写法）
FALLBACK = [
    (r"\times", "×"), (r"\cdot", "·"), (r"\le", "≤"), (r"\ge", "≥"), (r"\in", "∈"),
    (r"\dots", "…"), (r"\cdots", "…"), (r"\ldots", "…"), (r"\quad", "  "), (r"\qquad", "  "),
    (r"\!", ""), (r"\,", " "), (r"\;", " "), (r"\%", "%"),
    (r"\mathbb{R}", "R"), (r"\mathbb{E}", "E"), (r"\mathcal{K}", "K"),
    (r"\theta", "θ"), (r"\mu", "μ"), (r"\sigma", "σ"), (r"\tau", "τ"), (r"\lambda", "λ"),
    (r"\sum", "Σ"), (r"\min", "min"), (r"\log", "log"),
    (r"\text{", ""), (r"\mathrm{", ""), (r"\big(", "("), (r"\big)", ")"),
    (r"\left(", "("), (r"\left[", "["), (r"\right)", ")"), (r"\right]", "]"),
    (r"\{", "{"), (r"\}", "}"), (r"\wedge", "∧"),
]


def _tidy(s: str) -> str:
    s = re.sub(r"([_^])\{([^{}]*)\}", lambda m: f"{m.group(1)}{m.group(2)}", s)
    s = s.replace("{,}", ",").replace("  ", " ")
    return s.strip()


def convert_inline(span: str) -> str:
    if span in INLINE:
        return INLINE[span]
    out = span
    for a, b in FALLBACK:
        out = out.replace(a, b)
    out = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", out)
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = out.replace("{", "").replace("}", "")
    return _tidy(out)


def convert_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "$$":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "$$":
                j += 1
            tag = 0
            for k in range(i, j):
                m = re.match(r"\\tag\{(\d+)\}", lines[k].strip())
                if m:
                    tag = int(m.group(1))
            assert tag in DISPLAY, f"第 {i + 1} 行开始的公式块没有登记写法（式号 {tag}）"
            out.append(f"{DISPLAY[tag]}        ({tag})")
            i = j + 1
            continue
        out.append(re.sub(r"\$([^$]*)\$", lambda m: convert_inline(m.group(1)), lines[i]))
        i += 1
    return "\n".join(out)


def main() -> int:
    setup_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true", help="只报告残留，不写文件")
    a = ap.parse_args()

    rc = 0
    for raw in a.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        text = p.read_text(encoding="utf-8")
        new = convert_text(text)
        left = [l for l in new.split("\n") if "$" in l or "\\" in l.replace("\\n", "")]
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        if a.check:
            if left:
                rc = 1
                print(f"[残留] {rel}：{len(left)} 行仍含 $ 或反斜杠")
                for l in left[:10]:
                    print("   ", l[:120])
            else:
                print(f"[OK  ] {rel}：没有 $，没有 LaTeX 命令")
            continue
        p.write_bytes(new.replace("\n", "\r\n").encode("utf-8"))
        print(f"[write] {rel}")
        if left:
            rc = 1
            print(f"        警告：仍有 {len(left)} 行含 $ 或反斜杠，需要手工处理：")
            for l in left[:10]:
                print("   ", l[:120])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())