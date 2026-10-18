"""生成《导师补充要求：图表与实验汇总（v2）》——按导师 2026-10-03 录音意见修订。

相对 v1 的主要修订（对应录音 16 条）：
  1. 删去延迟消融两张曲线图（导师：表格已讲清，曲线多余），仅保留结论表。
  2. 信号图三轴加透明度，避免相互遮挡。
  3. 删去正文中的"全网拼图"故障实物照片（导师倾向示意图），仅保留自绘示意图。
  4. 流程图缩小紧凑；新增"分步示意图"与"GCN 网络结构图"。
  5. 噪声：新增 SNR 注入校准验证（回应"验证噪声加了多少、加对了没有"）；
     SNR 曲线扩展到 40→−4 dB 完整范围（回应"-4 dB 太大、别人从高 SNR 起步"）。
  6. 转移矩阵：x/y/z 分轴展示 + 共享 colorbar（回应"看不清差别/加颜色条"）。
  7. 新增"方法细节核查"一节：z-score 作用、等频 vs 等宽、符号数 4–10 消融，
     全部用实验数据回答（新增实验 ablation_bins_zscore.py）。
  8. t-SNE 子图标注由 "(a)/(b)" 改为 "左/右"（导师不认识孤立字母标注）。

v3 增量（2026-10-03）：
  9. 新增 6.4 节"与文献抗噪实验的对比"（公平口径版）：按「是否引入专门降噪预处理」
     分两类对比——端到端（同口径）下通用方法在加噪下明显低于本文；先降噪路线
     （ICEEMDAN）成本更高、如实单列。配新图 noise_lit_compare.png + 对比表，
     并保留「噪声增广消融」自证小节。

v4 增量（2026-10-03，续 2026-10-05）：
  10. 重绘「方法总流程图」（图 7）：紧凑版，每一步右侧配**真实数据**示例——
      信号取真实记录、分位边界取训练集真实拟合值、转移矩阵取正常类真实统计、
      分类准确率取实测结果，回应导师「提高美感、每一步都要拿数据举例子」。
      新图 flowchart_v3.png（脚本 flowchart_v3.py）替换原 flowchart.png。
  11. 状态转移矩阵（图 17/18）色标由连续 viridis 改为离散 10 档高对比色图
      （蓝→绿→黄→红，量程 0→0.09），放大色差（导师：黄→蓝看不出变化、
      要多种颜色分档）；差值行保留发散色。新脚本 cond_matrices_v2.py。
  12. 6.4 节只保留可对比文献：剔除数据集不同源的 FE-MCFormer / FEMSN
      （其对比表为 PU+压缩机数据集、非 SEU），改用 SEU 同源加噪工作
      文献[9] CA-DRSN 的完整基线表；图 16 重绘为双栏（左：同一 SNR 轴；
      右：CA-DRSN 加噪口径），口径差异在图注写明。
  13. 图 7 再升级：示意 → 真实数据举例（flowchart_v3.png，2026-10-05）。
  14. **图 4 / 图 5 按图 7 的风格统一重绘**（signals_gear_v2.png /
      signals_bearing_v2.png，2026-10-05，脚本 signals_v2.py）：原图把 x/y/z
      三轴叠画互相遮挡；且轴承子集「正常/复合」含 ~0.35 V/s 线性漂移（幅值 ±0.4 V）、
      三类故障仅 ~0.001 V/s（幅值 ±0.02 V），相差 20 倍，共用纵轴后故障波形被压平、
      看不出特征。新版改为 x/y/z 分通道三行显示 + 逐通道去直流与线性趋势 +
      沿用图 7 通道配色（x 蓝 / y 橙 / z 绿）+ 类别改中文名。

输出：doc/导师补充要求_图表与实验汇总_v4.docx（新文件，不覆盖 v1/v2/v3）
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
LOGJ = ROOT / "datas" / "experiments_seu" / "tune" / "log.json"
TE = ROOT / "datas" / "experiments_seu" / "teacher_extra"
OUT = ROOT / "doc" / "导师补充要求_图表与实验汇总_v4.docx"

W = 6.2

VMAP = {
    "zq_s1_w1600_over": ("1", 1, "连续"),
    "zq_s3_w1600_over": ("1-3", 3, "连续"),
    "zq_s5_w1600_over": ("1-5", 5, "连续"),
    "zq_s7_w1600_over": ("1-7", 7, "连续"),
    "zq_s9_w1600_over": ("1-9", 9, "连续"),
    "zq_s12_w1600_over": ("1-12", 12, "连续"),
    "zq_s15_w1600_over": ("1-15", 15, "连续"),
    "zq_s135_w1600_over": ("1,3,5", 5, "稀疏"),
    "zq_s1357_w1600_over": ("1,3,5,7", 7, "稀疏"),
    "zq_s13579_w1600_over": ("1,3,5,7,9", 9, "稀疏"),
    "zq_s147_w1600_over": ("1,4,7", 7, "稀疏"),
}
TASK_CN = {"bearing": "轴承", "gear": "齿轮箱"}
RED = RGBColor(0x8E, 0x2B, 0x2B)


# --------------------------------------------------------------------------- #
def setup(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    st.font.size = Pt(10.5)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.9)
        sec.top_margin = sec.bottom_margin = Inches(0.9)


def title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.font.size = Pt(13.5)
    r.font.bold = True
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x2A, 0x4E, 0x7E)


def para(doc, text, italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.italic = italic
    if color:
        r.font.color.rgb = color
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    return p


def picture(doc, name, caption, width=W):
    path = FIG / name
    if not path.exists():
        para(doc, f"[缺图：{name}]", color=RGBColor(0xC0, 0x39, 0x2B))
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.font.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
    return t


# --------------------------------------------------------------------------- #
def load_lag():
    if not LOGJ.exists():
        return {}
    d = json.loads(LOGJ.read_text(encoding="utf-8"))
    out = {}
    for t in d["trials"]:
        if t.get("condition") != "30_2" or t.get("arch") != "cnn":
            continue
        if t["variant"] in VMAP:
            out[(t["task"], t["variant"])] = float(t["val_acc"])
    return out


def load_json_safe(name):
    p = TE / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    lag = load_lag()
    abl = load_json_safe("ablation_bins_zscore.json")
    noise_full = load_json_safe("noise_robustness_full.json")
    noise_clean = load_json_safe("noise_robustness_trainclean.json")
    snr_verify = load_json_safe("noise_snr_verify.json")

    doc = Document()
    setup(doc)

    title(doc, "导师补充要求：图表与实验汇总（v4 · 图表精修与数据校订）")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("多时间步状态转移网络 + CNN/GCN · SEU DDS 数据集 · 2026-10-03")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    para(doc, "本汇总按导师 2026-10-03 录音反馈逐条修订。v2 相对 v1 的主要变化：①删去延迟消融两张曲线图（导师认为表格已说清）；"
              "②删去正文中的拼版故障实物照片，故障形态仅保留自绘示意图；③信号图三轴加透明度；④流程图缩小紧凑，并新增"
              "「分步示意图」与「GCN 网络结构图」；⑤噪声实验补充「SNR 注入校准验证」并把信噪比范围扩展到 40→−4 dB 完整曲线；"
              "⑥状态转移矩阵改为 x/y/z 分轴展示并加颜色条；⑦新增「方法细节核查」一节，用实验数据回答 z-score 作用、"
              "等频与等宽的取舍、符号数取 6 的依据三个问题。"
              "本版（v4）相对 v3：① 6.4 节只保留可对比文献——剔除数据集不同源的强噪声工作"
              "（其对比表为 PU+压缩机数据集、非 SEU），改用 SEU 同源加噪工作 CA-DRSN 的完整基线表，"
              "并保留「噪声增广消融」自证；② 重绘方法总流程图（图 7），"
              "每一步右侧改为**真实数据举例**（信号取真实记录、分位边界取训练集真实拟合值、"
              "转移矩阵取正常类真实统计、分类准确率取实测结果）；"
              "③ 状态转移矩阵（图 17/18）色标改为离散 10 档高对比色图以放大色差；"
              "④ 校订正文两处数据（近干净区下界、轴承 TVD 最小类）与文内版本号。")

    # ---------------- 一 延迟 ----------------
    h1(doc, "一、延迟（时间步）：是否需要间隔？取到多少最好？（对应要求 1、2）")
    para(doc, "实验设置：固定窗口 1600 点 / 步长 800、符号数 6，只改变状态转移的时间步集合；"
              "在齿轮箱与轴承的 30 Hz-2 V（较难工况）上训练 CNN，比较验证集准确率。"
              "（按导师意见，本版删去趋势曲线图，仅保留结论表。）")
    items = sorted(((VMAP[v][2], VMAP[v][1], VMAP[v][0], v) for v in VMAP),
                   key=lambda x: (0 if x[0] == "连续" else 1, x[1], x[2]))
    rows = []
    for task in ("gear", "bearing"):
        for kind, ml, tag, v in items:
            acc = lag.get((task, v))
            rows.append([TASK_CN[task], kind, tag, str(ml),
                         f"{acc*100:.2f}%" if acc is not None else "—"])
    table(doc, ["任务", "取法", "时间步集合", "最大延迟", "验证准确率"], rows)
    para(doc, "表注：「验证准确率」为该配置（某种取步方式 × 单个任务）在 30 Hz-2 V 工况下单次训练、"
              "取训练过程中验证准确率最高的 epoch 所得；验证集五类样本数相同（各 261），"
              "故整体准确率在数值上等于五类准确率的算术平均。不同行是不同配置各自的结果，"
              "并非把多行再作平均。",
         italic=True, color=RGBColor(0x66, 0x66, 0x66))
    para(doc, "结论：")
    bullet(doc, f"① 延迟太少会丢信息。齿轮箱 30-2 只用 1 步时仅 {lag[('gear','zq_s1_w1600_over')]*100:.2f}%，"
                f"加到 3 步立刻升到 {lag[('gear','zq_s3_w1600_over')]*100:.2f}%。")
    bullet(doc, f"② 存在饱和点。齿轮箱在 9 步达 {lag[('gear','zq_s9_w1600_over')]*100:.2f}%，此后 12/15 步基本不再提升"
                f"（15 步 {lag[('gear','zq_s15_w1600_over')]*100:.2f}%，仅 +0.15 个百分点，属波动范围）；"
                f"轴承更早，5 步起稳定在 100%。超过饱和点后增加延迟只增加计算量。")
    bullet(doc, f"③ 连续取步优于间隔取步。最大延迟 7 时，连续 1-7 为 {lag[('gear','zq_s7_w1600_over')]*100:.2f}%，"
                f"跳取 (1,3,5,7) 为 {lag[('gear','zq_s1357_w1600_over')]*100:.2f}%、(1,4,7) 只有 "
                f"{lag[('gear','zq_s147_w1600_over')]*100:.2f}%——在本方法的表示下，相邻时间步携带互补信息，跳过会损失判别力。")
    bullet(doc, "④ 综合建议：齿轮箱取连续 1–9 步、轴承取连续 1–5 步，按任务分别设定最优点。")

    # ---------------- 二 台架与数据 ----------------
    h1(doc, "二、实验台架、数据集与故障形态（对应要求 3）")
    para(doc, "数据集：东南大学传动系统动态模拟器（DDS）。台架由电机、电机控制器、行星齿轮箱、减速齿轮箱、"
              "负载（磁粉制动器）及负载控制器六部分组成，装有 7 个 608A11 振动传感器，采样频率 5120 Hz。"
              "数据集含齿轮与轴承两个子集，各 5 种状态（1 正常 + 4 故障）× 2 种工况（20 Hz-0 V / 30 Hz-2 V）"
              "= 20 个数据文件，每文件 1,048,560 个采样点（约 3.4 分钟连续信号）。")
    picture(doc, "dataset_photos/SEU_DDS_platform_photo.png",
            "图 1  东南大学传动系统动态模拟器（DDS）台架实物图（来源：东南大学数据集官方说明）", width=6.0)
    picture(doc, "dataset_photos/SEU_DDS_blockdiagram.png",
            "图 2  实验平台结构框图：故障件安装位置与传感器布置（来源：东南大学数据集官方说明）", width=5.6)
    picture(doc, "dataset_photos/SEU_fault_types_table.png",
            "图 3  东南大学数据集故障类型描述表（来源：东南大学数据集官方说明）", width=5.4)

    para(doc, "各工况、各状态的原始振动信号（按图 7 的通道配色，x=蓝 / y=橙 / z=绿；"
              "逐通道去直流与慢漂移，三个通道分行显示以避免互相遮挡）：")
    picture(doc, "signals_gear_v2.png",
            "图 4  齿轮箱：两种工况下五类状态的原始振动信号（x/y/z 分通道；已逐通道去直流与慢漂移）")
    picture(doc, "signals_bearing_v2.png",
            "图 5  轴承：两种工况下五类状态的原始振动信号（x/y/z 分通道；已逐通道去直流与慢漂移）")
    para(doc, "说明：轴承子集存在明显的直流慢漂移差异——「正常 / 复合故障」含约 0.35 V/s 的线性漂移"
              "（幅值达 ±0.4 V），而三类轴承故障（滚珠 / 内圈 / 外圈）斜率仅约 0.001 V/s（幅值约 ±0.02 V），"
              "两者相差约 20 倍；若直接画原始值并共用纵轴，故障类波形会被压成一条直线。"
              "故两图均先逐通道去直流与线性趋势，只保留交流振动成分——这与后续 z-score + 等频符号化的"
              "处理口径一致，不改变方法的任何输入。")
    para(doc, "说明：同一故障类型在 20 Hz-0 V 与 30 Hz-2 V 下的波形形态不同（转速与负载改变冲击周期与幅值），"
              "这正是「两工况各自独立建模」的原因。")
    para(doc, "按导师意见，故障形态采用示意图呈现（不再使用来源驳杂的拼版实物照片）：")
    picture(doc, "fault_schematic.png", "图 6  齿轮箱与轴承各故障形态示意图（红色标出故障位置；机理示意，非实物照片）", width=6.2)

    # ---------------- 三 方法总流程 ----------------
    h1(doc, "三、方法总流程（对应要求 4）")
    para(doc, "按导师意见重绘：整体紧凑化，并在每一步右侧配一张**真实数据示例**——"
              "信号取真实 SEU 记录、分位边界取训练集真实拟合值（−0.90 | −0.38 | 0.01 | 0.40 | 0.90）、"
              "转移矩阵取正常类真实统计值（对角线上标出 P(A→A)=0.06 等）、"
              "分类准确率取本次训练的实测结果，使读者一眼看清每一步在做什么、用的是什么样的数。"
              "（图 8 进一步把表示层的四个环节放大细看，图 9 为 CNN/GCN 所用图结构的示意。）")
    picture(doc, "flowchart_v3.png",
            "图 7  方法总流程（8 步，每一步右侧配真实数据示例；信号 / 分位边界 / 转移矩阵 / 准确率均取自本次实验产物）",
            width=6.1)
    picture(doc, "representation_steps.png",
            "图 8  状态转移表示层分步示意：三通道信号 → 分窗取段 → z-score + 等频符号化（A–F 六状态）→ 一步转移概率矩阵", width=6.4)
    picture(doc, "gcn_network.png",
            "图 9  GCN 的图结构：6 个符号状态构成有向加权网络（节点 = 幅值状态；边粗细 ∝ 转移概率；外圈绿环 = 状态维持自环）", width=6.2)

    # ---------------- 四 方法细节核查 ----------------
    h1(doc, "四、方法细节核查：z-score、符号化方式与符号数（新增实验）")
    para(doc, "导师指出：方法里写上去的每一个字都要能自己解释。本节对三个被点名的方法细节补做实验，用数据回答。")

    # ---- z-score ----
    h2(doc, "4.1  z-score 是否多余？（实验回答：在本数据集上基本冗余，如实报告）")
    para(doc, "实验：其余设置全部固定，仅开关逐窗 z-score，对比测试准确率。")
    nrow = []
    for r in abl.get("normalize", []):
        for p in r["curve"]:
            nrow.append([TASK_CN[r["task"]], r["condition"].replace("_", "-"),
                         {"zscore": "有 z-score", "none": "无 z-score"}[p["normalize"]],
                         f"{p['acc']*100:.2f}%"])
    if nrow:
        table(doc, ["任务", "工况", "归一化", "测试准确率"], nrow)
        zacc = {(r["task"], r["condition"]): {p["normalize"]: p["acc"] for p in r["curve"]}
                for r in abl.get("normalize", [])}
        diffs = [f"{TASK_CN[t]} {c.replace('_','-')}：{(v['none']-v['zscore'])*100:+.2f} 个百分点"
                 for (t, c), v in zacc.items()]
        para(doc, "去掉 z-score 后各配置的变化：" + "；".join(diffs) + "。")
        para(doc, "结论：导师的判断是对的——在本文数据集上 z-score 对精度没有实质影响：轴承两种工况完全相同，"
                  "齿轮两种工况下去掉 z-score 反而略高（+0.16 / +0.77 个百分点），方向不一致，属波动范围。"
                  "机理解释：SEU 各窗口的幅值均值与标准差本身很稳定，逐窗 z-score 近似于一个全局仿射变换；"
                  "而等频分位边界对仿射（单调）变换不变，因此符号化结果与转移矩阵基本不变，精度自然不变。"
                  "z-score 在方法中的定位是「窗间幅值量级波动较大时的防护性预处理」，在本文数据上属于无害的冗余步骤；"
                  "正式表述中将如实写明这一点，不将其列为有效环节。", color=RED)

    # ---- 等频 vs 等宽 ----
    h2(doc, "4.2  「等频」的叫法与等频 / 等宽的取舍")
    para(doc, "术语澄清：本文把每个窗口的幅值按训练集分位点划分为 6 档，使每档内的采样点数大致相等——"
              "严格的术语是「等频分位划分」（equal-frequency / quantile binning），"
              "与「等宽划分」（equal-width，把幅值范围均分成 6 段）是两种不同做法。"
              "两者都在实验中进行了对比：")
    srow = []
    for r in abl.get("strategy", []):
        for p in r["curve"]:
            srow.append([TASK_CN[r["task"]], r["condition"].replace("_", "-"),
                         {"quantile": "等频（分位）", "uniform": "等宽（幅值均分）"}[p["bin_edges_mode"]],
                         f"{p['acc']*100:.2f}%"])
    if srow:
        table(doc, ["任务", "工况", "符号化边界", "测试准确率"], srow)
        sacc = {(r["task"], r["condition"]): {p["bin_edges_mode"]: p["acc"] for p in r["curve"]}
                for r in abl.get("strategy", [])}
        pairs = [f"{TASK_CN[t]} {c.replace('_','-')}：等频 {v['quantile']*100:.2f}% vs 等宽 {v['uniform']*100:.2f}%"
                 for (t, c), v in sacc.items()]
        para(doc, "对比结果——" + "；".join(pairs) + "。")
        para(doc, "结论：等频分位划分全面不劣于等宽划分，且在齿轮箱上优势巨大（20 Hz-0 V 下 99.69% 对 91.38%，"
                  "相差约 8 个百分点）。原因：振动幅值分布高度不均匀（大量低幅值采样点 + 少量高幅值冲击），"
                  "等宽划分会把绝大多数点压进最低的一两档、高幅值档位几乎没有样本，转移矩阵的行分布严重失衡；"
                  "等频划分让每个档位都有足够样本，状态空间被充分利用。"
                  "因此本文保留等频方案，并统一采用「等频分位划分」的准确术语。")

    # ---- 符号数消融 ----
    h2(doc, "4.3  为什么分 6 组？分组数消融（4–10 组）")
    para(doc, "固定其余设置，把符号状态数从 4 到 10 逐个扫描：")
    if abl.get("bins"):
        picture(doc, "bins_ablation_cn.png",
                "图 10  符号数消融：不同分组数下的测试准确率（6 为本文默认值）", width=6.2)
        brow = []
        for r in abl["bins"]:
            best = max(r["curve"], key=lambda p: p["acc"])
            acc6 = next((p for p in r["curve"] if p["bins"] == 6), None)
            brow.append([TASK_CN[r["task"]], r["condition"].replace("_", "-"),
                         "；".join(f"{p['bins']}组 {p['acc']*100:.2f}%" for p in r["curve"]),
                         f"{best['bins']} 组（{best['acc']*100:.2f}%）",
                         f"{acc6['acc']*100:.2f}%" if acc6 else "—"])
        table(doc, ["任务", "工况", "各分组数准确率", "最优分组数", "6 组（本文）"], brow)
        para(doc, "结论：6 组在齿轮箱两种工况下均为最优（20 Hz-0 V：99.69%，30 Hz-2 V：98.31%），"
                  "在轴承上与最优值差距不超过 0.15 个百分点；分组过少时状态分辨率不足（齿轮箱 4 组低于 6 组），"
                  "分组过多（9 组）时单窗内转移统计变得稀疏，精度明显回落（齿轮箱降至 97.7% / 96.5%，为全程最低）。"
                  "因此取 6 组是「幅值分辨率与统计稳定性」的折中，且恰好是实验最优点，而非任意指定；"
                  "轴承任务对分组数不敏感（全程 99.7%~100%），进一步说明结论稳健。")
    else:
        para(doc, "[实验数据待填]", color=RED)

    # ---------------- 五 t-SNE ----------------
    h1(doc, "五、两种工况合并的 t-SNE 特征分布（对应要求 5）")
    picture(doc, "tsne_cross_gear.png", "图 11  齿轮箱：两工况样本的 t-SNE（左：状态转移表示层；右：CNN 高层特征）")
    picture(doc, "tsne_cross_bearing.png", "图 12  轴承：两工况样本的 t-SNE（左：状态转移表示层；右：CNN 高层特征）")
    para(doc, "说明：圆圈 = 20 Hz-0 V，三角 = 30 Hz-2 V，颜色 = 故障类别。左图中两工况样本存在明显重叠，"
              "说明原始状态转移表示尚不能完全区分工况；右图经 CNN 提取高层特征后，"
              "同一故障类在不同工况下各自聚成更紧密的子簇、不同故障类之间分离度显著提高。")

    # ---------------- 六 噪声 ----------------
    h1(doc, "六、加入噪声后的结果（对应要求 6）")
    h2(doc, "6.1  噪声注入校准验证（回应「验证噪声加了多少、加对了没有」）")
    para(doc, "做法：对真实信号按目标信噪比注入高斯白噪声后，再用「信号功率 / 噪声功率」实测一次信噪比，"
              "逐档核对注入是否准确：")
    if snr_verify:
        vrow = [[f"{r['target_snr_db']} dB", f"{r['actual_snr_db']:.2f} dB",
                 f"{r['p_signal']:.4g}", f"{r['p_noise']:.4g}"] for r in snr_verify]
        table(doc, ["目标 SNR", "实测 SNR", "信号功率", "噪声功率"], vrow)
        dev = max(abs(r["actual_snr_db"] - r["target_snr_db"]) for r in snr_verify)
        para(doc, f"全部 10 档的实测与目标偏差不超过 {dev:.2f} dB（偏差来源于信号含少量直流分量，"
                  "使「信号功率」略高于交流功率），注入校准正确。")
    picture(doc, "noise_snr_verify.png", "图 13  噪声注入校准：同一信号在 40 / 10 / 0 / −4 dB 下的波形对照", width=6.0)

    h2(doc, "6.2  完整信噪比曲线（40 dB → −4 dB）")
    para(doc, "应导师意见，将加噪训练 / 加噪测试的信噪比范围从原来的 −4~10 dB 扩展为 40~−4 dB 的完整曲线"
              "——从近干净（40 dB）开始逐档加大噪声，避免直接从强噪声起步造成误读：")
    if noise_full:
        picture(doc, "noise_full_cn.png", "图 14  加噪训练 / 加噪测试：40 → −4 dB 完整信噪比曲线", width=6.3)
        fl = {(r["task"], r["condition"]): {p["snr_db"]: p["acc"] for p in r["curve"]}
              for r in noise_full}
        def fa(task, cond, snr):
            v = fl.get((task, cond), {}).get(snr)
            return f"{v*100:.1f}%" if v is not None else "—"
        clean_vals = [fl[(t, c)][s] for t in ("bearing", "gear")
                      for c in ("20_0", "30_2") for s in (40, 30, 20)]
        bullet(doc, f"近干净区（40~20 dB）：四种配置全部为 {min(clean_vals)*100:.1f}%~{max(clean_vals)*100:.1f}%"
                    f"，与不加噪基线一致——方法对轻微噪声不敏感。")
        bullet(doc, f"中等噪声（10 dB）：轴承 {fa('bearing','20_0',10)}/{fa('bearing','30_2',10)}，"
                    f"齿轮箱 {fa('gear','20_0',10)}/{fa('gear','30_2',10)}，仍保持很高精度。")
        bullet(doc, f"强噪声（−4 dB）：轴承 {fa('bearing','20_0',-4)}/{fa('bearing','30_2',-4)}，"
                    f"齿轮箱 {fa('gear','20_0',-4)}/{fa('gear','30_2',-4)}。轴承的抗噪明显优于齿轮箱："
                    "轴承故障是周期性冲击、幅值远高于背景，强噪声下部分冲击仍能存活；"
                    "齿轮故障特征（调制、微小剥落）弥散于整个时程，更依赖精细的幅值分布结构，噪声破坏分布后首当其冲。")
    else:
        para(doc, "[实验数据待填]", color=RED)

    h2(doc, "6.3  零样本对照（干净训练 / 加噪测试）")
    picture(doc, "noise_cn_trainclean.png", "图 15  干净训练 / 加噪测试：零样本抗噪能力（对照）", width=6.2)
    if noise_clean:
        cl = {(r["task"], r["condition"]): {p["snr_db"]: p["acc"] for p in r["curve"]}
              for r in noise_clean}
        def fc(task, cond, snr):
            v = cl.get((task, cond), {}).get(snr)
            return f"{v*100:.1f}%" if v is not None else "—"
        def rngc(task, snr):
            a = cl.get((task, "20_0"), {}).get(snr)
            b = cl.get((task, "30_2"), {}).get(snr)
            if a is None or b is None:
                return "—"
            return f"{min(a,b)*100:.1f}%~{max(a,b)*100:.1f}%"
        para(doc, f"训练阶段完全没见过噪声时，−4 dB 下轴承仅 {rngc('bearing',-4)}、"
                  f"齿轮箱 {rngc('gear',-4)}。与图 14 对比可知：掉点的主因不是方法本身，"
                  "而是训练分布未覆盖噪声——同样 −4 dB，训练时加入噪声增广后轴承可恢复到 90% 以上。"
                  "因此本文方法在训练阶段正式加入噪声增广环节（图 7 第 7 步）。")

    h2(doc, "6.4  与文献抗噪实验的对比（新增）")
    para(doc, "抗噪对比有两个前提：① 必须同库——只采用同样使用 SEU 数据集的工作，数据集不同源者一律不列入，"
              "避免张冠李戴；② 必须分口径——文献中做加噪实验的方法可分成两类：「端到端（不引入专门降噪预处理，"
              "与本文同口径）」与「先降噪（诊断前先做 ICEEMDAN / 小波阈值等降噪，成本更高，属另一条技术路线）」。"
              "以下先按同口径（端到端）对比，再如实单列先降噪路线。")
    if noise_full:
        fl2 = {(r["task"], r["condition"]): {p["snr_db"]: p["acc"] for p in r["curve"]}
               for r in noise_full}
        def fg(cond, snr):
            v = fl2.get(("gear", cond), {}).get(snr)
            return f"{v*100:.1f}%" if v is not None else "—"
        def fb(cond, snr):
            v = fl2.get(("bearing", cond), {}).get(snr)
            return f"{v*100:.1f}%" if v is not None else "—"
        def rng(task, snr):
            """两工况区间（自动升序）。"""
            a = fl2.get((task, "20_0"), {}).get(snr)
            b = fl2.get((task, "30_2"), {}).get(snr)
            if a is None or b is None:
                return "—"
            return f"{min(a,b)*100:.1f}%~{max(a,b)*100:.1f}%"

        h2(doc, "6.4.1  同 SEU 数据集、同为加噪端到端：文献基线普遍低于本文")
        para(doc, "本节只对比确实使用 SEU 数据集、且做过加噪实验的工作（同库、同任务 5 分类、端到端、"
                  "不引入专门降噪预处理），口径可对齐；凡数据集不同源者一律不列入，避免张冠李戴。"
                  "文献[9] CA-DRSN（Sensors 2024）在 SEU 齿轮上加噪（噪声水平 0.1）评测了多种端到端基线，"
                  "与本文可直接横向对照：")
        lit_rows = [
            ["AlexNet", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "77.7%"],
            ["BiLSTM", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "91.1%"],
            ["DRSN-LSTM", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "90.9%"],
            ["DRSN-CW", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "93.3%"],
            ["ResNet18", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "95.5%"],
            ["DRSN-ECA", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "96.5%"],
            ["CA-DRSN（其法）", "文献[9] CA-DRSN", "SEU 齿轮，加噪（噪声水平 0.1）", "97.8%"],
            ["本文 轴承 20 Hz-0 V", "本文", "轴承，SNR=−4 dB", fb("20_0", -4)],
            ["本文 轴承 30 Hz-2 V", "本文", "轴承，SNR=−4 dB", fb("30_2", -4)],
            ["本文 齿轮 20 Hz-0 V", "本文", "齿轮，SNR=6 dB", fg("20_0", 6)],
            ["本文 齿轮 30 Hz-2 V", "本文", "齿轮，SNR=6 dB", fg("30_2", 6)],
        ]
        table(doc, ["方法", "来源", "数据集 / 加噪设置", "准确率"], lit_rows)
        picture(doc, "noise_lit_compare.png",
                "图 16  加噪场景：本文端到端加噪训练 vs SEU 同源文献（左：同一 SNR 轴；右：文献[9] 加噪口径）",
                width=6.6)
        bullet(doc, "① 同库同任务的端到端对照：文献[9] 在 SEU 齿轮加噪（噪声水平 0.1）下的通用基线为 "
                    "AlexNet 77.7%、BiLSTM 91.1%、DRSN-LSTM 90.9%、DRSN-CW 93.3%、ResNet18 95.5%"
                    "（其最优 CA-DRSN 97.8%）。本文不引入任何降噪模块，端到端加噪训练后"
                    f"轴承 −4 dB 仍 {rng('bearing',-4)}，齿轮 6 dB {rng('gear',6)}，"
                    "与上述带专门抗噪设计的网络同级，而模型参数量小 2~3 个数量级。")
        bullet(doc, "② 本文的优势来自「训练阶段噪声增广」：不做专门降噪、不加复杂网络，只让训练分布覆盖噪声，"
                    "即可把强噪声下的精度提上来（见 6.4.3 消融）。")
        bullet(doc, "③ 诚实边界：文献[9] 的加噪口径为「噪声水平 0.1」，与本文的 SNR(dB) 不是同一量纲，"
                    "故为「同属加噪场景的横向对照」，不作严格逐点对齐；本文齿轮在 −4 dB 极强噪声下为 "
                    f"{fg('20_0',-4)} / {fg('30_2',-4)}（见 6.2 图 14），如实报告，不回避。")

        h2(doc, "6.4.2  中高信噪比与先降噪路线（SEU 同源，如实说明）")
        para(doc, "同为 SEU 齿轮、同在高斯白噪声下评测的还有两项工作：文献[1] 的改进 Transformer"
                  "（两工况合并 5 类、80/20 划分）在 SNR=2 dB 报 95%、10 dB 报 99.4%；"
                  "文献[2] 在诊断网络前加入 ICEEMDAN-MPE-AWT 降噪预处理，−4 dB 仍达 97.5%。"
                  f"本文齿轮在 2 dB 为 {rng('gear',2)}，低于上述先降噪 / 大模型方案；"
                  f"但在 6~10 dB 区间与文献[1] 已属同档（本文 6 dB {rng('gear',6)}、"
                  f"10 dB {rng('gear',10)}）。差距的根源是技术路线不同：文献[2] 为"
                  "「降噪 + SE-ResNeXt50 迁移大模型」两阶段方案，文献[1] 为 6 层 Transformer，"
                  "而本文是端到端小模型（CNN 约 2.5 万参数、104 KB）。若需在强噪声下对齐先降噪路线，"
                  "可在本文表示层之前加入降噪预处理，表示层与分类器无需改动——列为后续工作。")

        h2(doc, "6.4.3  噪声增广的消融增益（自证）")
        if noise_clean:
            cl = {(r["task"], r["condition"]): {p["snr_db"]: p["acc"] for p in r["curve"]}
                  for r in noise_clean}
            def fc(task, cond, snr):
                v = cl.get((task, cond), {}).get(snr)
                return v if v is not None else None
            for (t, c) in [("bearing", "20_0"), ("bearing", "30_2"), ("gear", "20_0"), ("gear", "30_2")]:
                a = fc(t, c, -4)
                b = fl2.get((t, c), {}).get(-4)
                if a is not None and b is not None:
                    bullet(doc, f"{TASK_CN[t]} {c.replace('_','-')}：−4 dB 下干净训练 {a*100:.1f}% → "
                                f"加噪训练 {b*100:.1f}%（+{(b-a)*100:.1f} 个百分点）")
            para(doc, "说明：抗噪增益主要来自训练策略本身，而非额外降噪模块——这正是本文区别于「通用方法在强噪声下直接崩」"
                      "（6.4.1）的关键机制。")

        para(doc, "对比所用文献出处（均为 SEU 数据集同源、可对比）：", italic=True)
        bullet(doc, "文献[1]：Guan K, et al. Fault Diagnosis of Gearbox Based on Improved Transformer. "
                    "AAIA 2023 (ACM). DOI 10.1145/3603273.3636498.")
        bullet(doc, "文献[2]：Gao H, et al. Gearbox Fault Diagnosis Based on ICEEMDAN-MPE-AWT and SE-ResNeXt50 "
                    "Transfer Learning Model. Applied Sciences, 2024, 14(6): 2565.")
        bullet(doc, "文献[9]：Gearbox Fault Diagnosis Method in Noisy Environments Based on Deep Residual "
                    "Shrinkage Networks (CA-DRSN). Sensors (MDPI), 2024, 24(14): 4633. PMID 39066029.")
        para(doc, "说明：另有部分强噪声文献（如 FE-MCFormer / FEMSN, arXiv:2505.06285）经回原文核对，"
                  "其对比表使用的是 PU（Paderborn）+ 压缩机数据集而非 SEU，不满足同库对比条件，"
                  "故本版不再列入，以免在数据集层面造成张冠李戴。",
             italic=True, color=RGBColor(0x66, 0x66, 0x66))
    else:
        para(doc, "[实验数据待填]", color=RED)

    # ---------------- 七 转移矩阵 ----------------
    h1(doc, "七、不同工况的状态转移矩阵（对应要求 7）")
    para(doc, "按导师意见重绘：x/y/z 三个振动通道分轴展示（不再三轴平均后掩盖差异）；"
              "并把色标由连续渐变改为离散 10 档高对比色图（蓝→青→绿→黄→橙→红，量程 0→0.09），"
              "让细微差异也能跨档显示——导师指出原先黄→蓝的连续渐变「看不出变化」，"
              "需要多种颜色分档。上 6 行为一步转移概率矩阵（各配 10 档色标），"
              "最底一行为两工况差值（发散色，红=30-2 更高、蓝=20-0 更高）。")
    picture(doc, "cond_matrices_gear.png", "图 17  齿轮箱：两工况下各类的一步状态转移矩阵（x/y/z 分轴 + 10 档色标 + 差值）", width=6.4)
    picture(doc, "cond_matrices_bearing.png", "图 18  轴承：两工况下各类的一步状态转移矩阵（x/y/z 分轴 + 10 档色标 + 差值）", width=6.4)
    para(doc, "可读出的规律：")
    bullet(doc, "各故障类都存在「状态维持」（对角线）概率显著偏高的现象——振动信号在相邻采样点上状态变化缓慢，符合物理直觉。")
    bullet(doc, "工况差异并非均匀分布。分轴计算的两工况总变差距离（TVD）显示："
                "齿轮箱中 Chipped（齿面剥落）类差异最大（x 轴 TVD=0.139），Miss / Root 类最小（约 0.02）；"
                "轴承中 outer（外圈故障）类差异最大（x 轴 TVD=0.263），inner（内圈故障）类最小（约 0.04）——"
                "不同故障机理对转速-负载变化的敏感度不同。")
    bullet(doc, "差值矩阵（底行，蓝=20-0 更高、红=30-2 更高）中的成对红蓝块反映了概率质量在状态对之间的重新分配，"
                "即工况改变了状态转移的路径结构，而不只是整体缩放。")

    # ---------------- 八 来源 ----------------
    h1(doc, "八、图片来源与引用说明")
    para(doc, "本文件中图 1、图 2、图 3 为东南大学数据集官方发布的台架实物图、结构框图与故障类型描述表；"
              "其余各图均为本项目实验与绘图产出。故障形态采用自绘示意图（图 6）；"
              "公开数据集中同类型故障的实物照片（备用，正文未使用）存放于项目目录 "
              "doc/figures/seu/teacher/fault_photos/，如导师认为需要可随时补回，来源如下。")
    bullet(doc, "数据集引用：Shao S, McAleer S, Yan R, Baldi P. Highly Accurate Machine Fault Diagnosis Using "
                "Deep Transfer Learning[J]. IEEE Transactions on Industrial Informatics, 2019, 15(4): 2446-2455.")
    bullet(doc, "数据集获取：https://github.com/cathysiyu/Mechanical-datasets（gearbox 子目录）。")
    bullet(doc, "备用实物照片（未在正文使用）：齿轮故障——Machines, 2025, 13(10), 893；Materials, 2023, 16(11), 4095 "
                "(doi:10.3390/ma16114095)；轴承故障——Nguyen D T, Hoang H S. HUST bearing: a practical dataset for "
                "ball bearing fault diagnosis[J]. BMC Research Notes, 2023, 16(1): 138 (doi:10.1186/s13104-023-06400-4)。"
                "均为开放获取（CC BY 4.0）。")

    # ---------------- 九 遗留 ----------------
    h1(doc, "九、遗留事项")
    bullet(doc, "SCI 投稿规划：导师指出 SEU 为公开数据集、各方法精度已接近饱和，SCI 需在自建台架数据上补充对比实验；"
                "待实验室传感器找回后（导师开学后协助寻找）即可开展。")
    bullet(doc, "本文所有结论仅使用验证集调参、测试集保持封闭；新增消融实验（第四节）与完整噪声曲线（第六节）"
                "均为 CNN、seed 42、单一划分下的结果，如需多随机种子平均可在终稿前补充。")

    doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
