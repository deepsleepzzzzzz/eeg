"""生成给导师的《补充图表与实验汇总》Word 文档。

用 python-docx 直接排版并嵌入 PNG（比走 HTML 流水线更可控）。
运行： C:/Users/lgt11/.workbuddy/binaries/python/envs/default/Scripts/python.exe
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
OUT = ROOT / "doc" / "导师补充要求_图表与实验汇总.docx"

W = 6.2  # 图片宽度（英寸）

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


def main() -> None:
    lag = load_lag()
    doc = Document()
    setup(doc)

    title(doc, "导师补充要求：图表与实验汇总")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("多时间步状态转移网络 + CNN/GCN · SEU DDS 数据集 · 2026-10-02")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    para(doc, "本文件按导师提出的 7 条要求逐条给出对应图表与实验说明。所有准确率均来自本项目实验记录；"
              "涉及超参数的实验只使用验证集（测试集保持封闭），不使用测试集调参。")

    # ---------------- 1 & 2 ----------------
    h1(doc, "一、延迟（时间步）：是否需要间隔？取到多少最好？（对应要求 1、2）")
    para(doc, "实验设置：固定窗口 1600 点 / 步长 800、符号数 6、z-score + 等频边界，只改变状态转移的时间步集合；"
              "分别在齿轮箱与轴承的 30 Hz-2 V（较难工况）上训练 CNN，比较验证集准确率。")
    items = sorted(((VMAP[v][2], VMAP[v][1], VMAP[v][0], v) for v in VMAP),
                   key=lambda x: (0 if x[0] == "连续" else 1, x[1], x[2]))
    rows = []
    for task in ("gear", "bearing"):
        for kind, ml, tag, v in items:
            acc = lag.get((task, v))
            rows.append([TASK_CN[task], kind, tag, str(ml),
                         f"{acc*100:.2f}%" if acc is not None else "—"])
    table(doc, ["任务", "取法", "时间步集合", "最大延迟", "验证准确率"], rows)
    para(doc, "")
    picture(doc, "lag_steps_curve.png", "图 1  连续取步：验证准确率随最大延迟的变化")
    picture(doc, "lag_sparse_vs_dense.png", "图 2  同一最大延迟下，稀疏取步与连续取步的对比")
    def g(task, v):
        a = lag.get((task, v))
        return f"{a*100:.2f}%" if a is not None else "—"

    para(doc, "结论：")
    bullet(doc, f"① 延迟太少会丢信息。齿轮箱 30-2 只用 1 步时仅 {g('gear','zq_s1_w1600_over')}，"
                f"加到 3 步立刻升到 {g('gear','zq_s3_w1600_over')}——说明必须保证足够长的时程覆盖。")
    bullet(doc, f"② 存在饱和点。齿轮箱在 9 步达 {g('gear','zq_s9_w1600_over')}，之后 12 步 "
                f"{g('gear','zq_s12_w1600_over')}、15 步 {g('gear','zq_s15_w1600_over')} 基本不再提升；"
                f"轴承饱和更早，3 步即 {g('bearing','zq_s3_w1600_over')}、5 步起稳定在 100%。"
                f"超过该点后增加延迟只增加计算量、不带来精度收益。")
    bullet(doc, f"③ 建议连续取步、不要间隔。同一最大延迟下，连续取步始终优于间隔取步："
                f"最大延迟 7 时，连续 1-7 为 {g('gear','zq_s7_w1600_over')}，而跳着取的 (1,3,5,7) 只有 "
                f"{g('gear','zq_s1357_w1600_over')}、(1,4,7) 只有 {g('gear','zq_s147_w1600_over')}；"
                f"最大延迟 9 时连续 1-9 为 {g('gear','zq_s9_w1600_over')}，(1,3,5,7,9) 只有 "
                f"{g('gear','zq_s13579_w1600_over')}。")
    bullet(doc, "④ 这一结果与「相邻步信息冗余、可以跳着取」的直觉相反：在本方法的表示下，相邻时间步之间仍携带互补信息，"
                "跳过它们会损失判别力；若为压缩通道数而跳取，代价约 1~3 个百分点，(1,4,7) 这类大步跳过则会损失 6 个百分点以上。")
    bullet(doc, f"⑤ 综合建议：齿轮箱取连续 1–9 步、轴承取连续 1–5 步，按任务分别设定最优点"
                f"（本结论在齿轮箱与轴承两个任务上一致，说明该规律稳定）。")

    # ---------------- 3 ----------------
    h1(doc, "二、实验台架、数据集与故障形态（对应要求 3）")
    para(doc, "数据集：东南大学传动系统动态模拟器（DDS）。台架由电机、电机控制器、行星齿轮箱、减速齿轮箱、"
              "负载（磁粉制动器）及负载控制器六部分组成，装有 7 个 608A11 振动传感器，采样频率 5120 Hz。"
              "数据集含齿轮与轴承两个子集，各 5 种状态（1 正常 + 4 故障）× 2 种工况（20 Hz-0 V / 30 Hz-2 V）"
              "= 20 个数据文件，每文件 1,048,560 个采样点（约 3.4 分钟连续信号）。")
    picture(doc, "dataset_photos/SEU_DDS_platform_photo.png",
            "图 3  东南大学传动系统动态模拟器（DDS）台架实物图（来源：东南大学数据集官方说明）", width=6.0)
    picture(doc, "dataset_photos/SEU_DDS_blockdiagram.png",
            "图 4  实验平台结构框图：故障件安装位置与传感器布置（来源：东南大学数据集官方说明）", width=5.6)
    picture(doc, "dataset_photos/SEU_fault_types_table.png",
            "图 5  东南大学数据集故障类型描述表（来源：东南大学数据集官方说明）", width=5.4)

    para(doc, "各工况、各状态的原始振动信号：")
    picture(doc, "signals_gear.png", "图 6  齿轮箱：两种工况下五类状态的三轴原始振动信号")
    picture(doc, "signals_bearing.png", "图 7  轴承：两种工况下五类状态的三轴原始振动信号")
    para(doc, "说明：同一故障类型在 20 Hz-0 V 与 30 Hz-2 V 下的波形形态不同（转速与负载改变冲击周期与幅值），"
              "这正是「两工况各自独立建模」的原因。")
    picture(doc, "fault_schematic.png", "图 8  齿轮箱与轴承各故障形态示意图")
    para(doc, "说明：图 8 为按数据集故障类型绘制的机理示意图（红色标出故障位置），用于说明故障发生部位。")

    picture(doc, "fault_photos/fault_specimens_real.png",
            "图 9  齿轮箱与滚动轴承各类故障的实物形态对照（来源见第七节）", width=6.4)
    para(doc, "说明：上排为齿轮故障的实物照片（缺齿、崩齿、齿面点蚀、齿根裂纹），"
              "下排为滚动轴承故障的实物照片（内圈、外圈、滚动体、复合，红色圆圈标出损伤位置）。"
              "东南大学数据集官方仅公开振动信号与故障类型描述，未随数据发布故障件实物照片"
              "（官方说明中明确「哪个轴承、齿轮坏了信息未提供」）。因此图 9 取自公开数据集"
              "（均为开放获取、CC BY 4.0）中同类型故障的实物照片，用于直观说明各类故障的物理形态，"
              "具体来源与许可见第七节；台架与安装位置见图 3、图 4。",
         color=RGBColor(0x8E, 0x2B, 0x2B))

    # ---------------- 4 ----------------
    h1(doc, "三、方法总流程（对应要求 4）")
    picture(doc, "flowchart.png", "图 10  方法总流程（含训练阶段噪声增广环节）", width=5.6)

    # ---------------- 5 ----------------
    h1(doc, "四、两种工况合并的 t-SNE 特征分布（对应要求 5）")
    picture(doc, "tsne_cross_gear.png", "图 11  齿轮箱：两工况样本的 t-SNE（左：状态转移表示层；右：CNN 高层特征）")
    picture(doc, "tsne_cross_bearing.png", "图 12  轴承：两工况样本的 t-SNE（左：状态转移表示层；右：CNN 高层特征）")
    para(doc, "说明：圆圈 = 20 Hz-0 V，三角 = 30 Hz-2 V，颜色 = 故障类别。左图中两工况样本存在明显重叠，"
              "说明原始状态转移表示尚不能完全区分工况；右图经 CNN 提取高层特征后，"
              "同一故障类在不同工况下各自聚成更紧密的子簇、不同故障类之间分离度显著提高，"
              "说明模型学到的是与故障机理相关的判别特征，而工况差异被压缩为一个可控的偏移。")

    # ---------------- 6 ----------------
    h1(doc, "五、加入噪声后的结果（对应要求 6）")
    picture(doc, "noise_cn.png", "图 13  加噪训练 / 加噪测试：不同信噪比下的准确率")
    picture(doc, "noise_cn_trainclean.png", "图 14  干净训练 / 加噪测试：零样本抗噪能力")
    para(doc, "结论：")
    bullet(doc, "图 13（加噪训练 / 加噪测试，与文献同协议）：轴承在 −4 dB 强噪声下仍保持 91.7%~95.7%，"
                "齿轮箱为 46.9%~67.1%；轴承的抗噪是本文方法的优势项，齿轮箱相对较弱，如实报告、不作优势论据。")
    bullet(doc, "图 14（干净训练 / 加噪测试，零样本）：训练阶段完全没见过噪声时，所有配置都大幅掉点"
                "（−4 dB 时轴承 29.8%~58.0%、齿轮箱 25.7%~29.7%）。主要原因是：① 加噪后信号总方差增大，"
                "z-score 归一化相当于把有效信号进一步缩小；② 等频符号化边界是在干净数据上估计的，"
                "噪声使幅值分布变宽、大量样本越过原边界发生符号翻转；③ 随机符号抖动使状态转移矩阵趋向均匀、"
                "对角线（状态维持）概率下降，模型学到的判别结构被破坏。齿轮故障特征（调制、微小剥落）"
                "本就比轴承冲击特征更弥散，因此掉点更早、更深。")
    bullet(doc, "图 13 与图 14 对比说明：掉点的主因不是方法本身，而是训练分布未覆盖噪声——同样 −4 dB，"
                "训练时加入噪声增广后轴承可恢复到 91.7%~95.7%。因此本文方法在训练阶段正式加入噪声增广环节"
                "（流程见图 10 第 7 步）：对训练信号注入多 SNR 高斯白噪声后按原流程编码，"
                "使模型在训练时即见过噪声分布，从而获得抗噪能力；同等效果也可通过在带噪数据上重新估计"
                "符号化边界实现。")
    bullet(doc, "两工况对比：图 13 中 20 Hz-0 V 优于 30 Hz-2 V；图 14 中轴承 30 Hz-2 V（58.0%）反而高于"
                "20 Hz-0 V（29.8%），即干净数据上更准的模型零样本抗噪未必更稳（精度-鲁棒性权衡），如实报告。")

    # ---------------- 7 ----------------
    h1(doc, "六、不同工况的状态转移矩阵（对应要求 7）")
    picture(doc, "cond_matrices_gear.png", "图 15  齿轮箱：两工况下各类的一步状态转移概率矩阵及差值", width=6.2)
    picture(doc, "cond_matrices_bearing.png", "图 16  轴承：两工况下各类的一步状态转移概率矩阵及差值", width=6.2)
    para(doc, "说明：每列自上而下为 20 Hz-0 V、30 Hz-2 V 以及两者之差（红 = 30-2 概率更高，蓝 = 20-0 更高）；"
              "TVD 为两概率矩阵的总变差距离，数值越大说明该故障类受工况影响越明显。")
    para(doc, "可读出的规律：")
    bullet(doc, "各故障类都存在「状态维持」（对角线）概率显著偏高的现象——振动信号在相邻采样点上状态变化缓慢，这符合物理直觉。")
    bullet(doc, "工况差异并非均匀分布：齿轮箱中 Chipped（齿面剥落）类的两工况差异最大，而 Miss / Root 类差异较小，"
                "说明不同故障机理对转速-负载变化的敏感度不同。")
    bullet(doc, "差值矩阵中的成对红蓝块（某处 30-2 更高、对应处 20-0 更高）反映了概率质量在状态对之间的重新分配，"
                "即工况改变了状态转移的路径结构，而不只是整体缩放。")

    # ---------------- 七 ----------------
    h1(doc, "七、图片来源与引用说明")
    para(doc, "本文件中图 3、图 4、图 5 为东南大学数据集官方发布的台架实物图、结构框图与故障类型描述表，"
              "已按数据集官方要求注明来源；图 1、图 2、图 6~图 8、图 10~图 16 为本项目实验与绘图产出；"
              "图 9（各类故障实物照片）取自公开数据集，均按开放获取许可（CC BY 4.0）使用，明细如下。")
    bullet(doc, "数据集引用：Shao S, McAleer S, Yan R, Baldi P. Highly Accurate Machine Fault Diagnosis Using "
                "Deep Transfer Learning[J]. IEEE Transactions on Industrial Informatics, 2019, 15(4): 2446-2455.")
    bullet(doc, "数据集获取：https://github.com/cathysiyu/Mechanical-datasets（gearbox 子目录）。")
    bullet(doc, "台架与故障类型说明：东南大学数据集官方说明文档（实验平台实物图、结构框图、故障类型表）。")
    bullet(doc, "图 9 齿轮「缺齿 / 崩齿」实物照片：Machines, 2025, 13(10), 893（MDPI 开放获取，CC BY 4.0）。")
    bullet(doc, "图 9 齿轮「齿面点蚀 / 齿根裂纹」实物照片：He H, Mura A, Zhang T, Liu H, Xu W. Investigation of "
                "Crack Propagation Behaviour in Thin-Rim Gears: Experimental Tests and Numerical Simulations[J]. "
                "Materials, 2023, 16(11), 4095. doi:10.3390/ma16114095（MDPI 开放获取，CC BY 4.0）。")
    bullet(doc, "图 9 轴承「内圈 / 外圈 / 滚动体 / 复合」实物照片：Nguyen D T, Hoang H S. "
                "HUST bearing: a practical dataset for ball bearing fault diagnosis[J]. BMC Research Notes, 2023, "
                "16(1): 138. doi:10.1186/s13104-023-06400-4（数据：Mendeley Data, doi:10.17632/cbv7jyx4p9.3，CC BY 4.0）。")

    h1(doc, "八、遗留事项")
    para(doc, "「各种故障的实物图」：东南大学数据集官方仅公开振动信号与故障类型描述，未随数据发布各故障件的"
              "单独实物照片（官方说明中明确「哪个轴承、齿轮坏了信息未提供」）。本项目已用官方台架实物图（图 3）、"
              "结构框图（图 4）、故障类型表（图 5）、自绘机理示意图（图 8）作为替代，并补充了"
              "公开数据集（CC BY 4.0）中同类型故障的实物照片（图 9）。")
    bullet(doc, "说明：图 9 的实物照片并非东南大学 DDS 台架上那几件故障件本身，而是公开数据集中同类型故障的真实零件照片，"
                "用于直观说明「缺齿、崩齿、齿面点蚀、齿根裂纹」与「内圈、外圈、滚动体、复合」这些故障的物理形态；引用时请保留第七节所列来源。")
    bullet(doc, "可选补充一：取自东南大学数据集原论文 Shao et al., IEEE TII 2019, 15(4): 2446-2455 的故障件插图（该图版权属 IEEE，引用需注明出处）。")
    bullet(doc, "可选补充二：若实验室有同型 DDS 台架，自行拍摄各故障件照片，最稳妥且无版权顾虑。")

    doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
