"""导师问题6 补充：加噪实验与 SEU 同源文献的对比图（v3，只留可对比工作）。

只保留**满足"同库对比"条件**的文献 —— 即确实使用 SEU(DDS) 数据集、且做过加噪实验的工作：
  - 文献[9] CA-DRSN (Sensors 2024, 24(14):4633, PMID 39066029)
      SEU 齿轮、加噪（噪声水平 0.1）；原文表 5：各端到端基线 77.7%~97.8%
  - 文献[1] 改进 Transformer (Guan et al., AAIA 2023, DOI 10.1145/3603273.3636498)
      SEU 齿轮 5 类、两工况合并、80/20；SNR=2 dB→95%，SNR=10 dB→99.4%
  - 文献[2] ICEEMDAN-MPE-AWT + SE-ResNeXt50 (Gao et al., Appl. Sci. 2024, 14(6):2565)
      SEU 齿轮 5 类、先降噪；其法 −4~6 dB 区间 97.5%~99.3%

已删除：FE-MCFormer / FEMSN (arXiv:2505.06285) —— 经回原文核对，其对比表用的是
PU(Paderborn)+压缩机数据集，**并非 SEU**，不满足"同库对比"，故不再列入图表。

布局：左栏 = SNR 轴对比（本文 4 条加噪训练曲线 vs 文献[1]/[2]）；
      右栏 = 文献[9] 的加噪口径（噪声水平 0.1，非 SNR 分贝）横条对比。
两栏口径不同，图注已注明「横向对照、不逐点对齐」。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "datas" / "experiments_seu" / "teacher_extra"
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 文献抗噪结果（均为 SEU 数据集）
LIT1 = {2: 95.0, 10: 99.4}                                        # 改进 Transformer（两工况合并）
LIT2 = {-4: 97.5, -2: 98.2, 0: 98.8, 2: 99.0, 4: 99.2, 6: 99.3}   # ICEEMDAN-MPE-AWT 其法
LIT9 = [("AlexNet", 77.7), ("BiLSTM", 91.1), ("DRSN-LSTM", 90.9), ("DRSN-CW", 93.3),
        ("ResNet18", 95.5), ("DRSN-ECA", 96.5), ("CA-DRSN（其法）", 97.8)]      # 文献[9] 表 5

OUR_STYLE = {
    ("gear", "20_0"): ("#2471a3", "^", "本文 齿轮 20 Hz-0 V"),
    ("gear", "30_2"): ("#1e8449", "D", "本文 齿轮 30 Hz-2 V"),
    ("bearing", "20_0"): ("#c0392b", "o", "本文 轴承 20 Hz-0 V"),
    ("bearing", "30_2"): ("#e67e22", "s", "本文 轴承 30 Hz-2 V"),
}


def main() -> None:
    data = json.loads((TD / "noise_robustness_full.json").read_text(encoding="utf-8"))
    fl = {(r["task"], r["condition"]): {p["snr_db"]: p["acc"] * 100 for p in r["curve"]}
          for r in data}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 6.0), gridspec_kw={"width_ratios": [1.62, 1.0]})

    # ================= 左：SNR 轴 =================
    for r in data:
        key = (r["task"], r["condition"])
        if key not in OUR_STYLE:
            continue
        color, mk, label = OUR_STYLE[key]
        pts = sorted((p["snr_db"], p["acc"] * 100) for p in r["curve"] if p["snr_db"] <= 10)
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], ls="-", marker=mk,
                 color=color, lw=2.0, ms=6, label=label)

    xs2 = sorted(LIT2)
    ax1.plot(xs2, [LIT2[x] for x in xs2], ls="--", marker="s", color="#8e44ad",
             lw=1.6, ms=6, label="文献[2] 先降噪路线（ICEEMDAN，SEU 齿轮）")

    xs1 = sorted(LIT1)
    ax1.plot(xs1, [LIT1[x] for x in xs1], ls="--", marker="*", color="#b7950b",
             lw=1.6, ms=11, label="文献[1] 改进 Transformer（SEU 齿轮）")

    ax1.axvline(-4, color="#555555", lw=0.8, ls=":", alpha=0.7)
    ax1.text(-4, 3.0, "强噪声 −4 dB", rotation=90, fontsize=8.5, color="#555555",
             ha="right", va="bottom")
    ax1.set_xlabel("信噪比 SNR (dB)", fontsize=11.5)
    ax1.set_ylabel("测试准确率 (%)", fontsize=11.5)
    ax1.set_title("(a) 同一 SNR 轴：本文加噪训练 vs SEU 同源文献", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-12, 12)
    ax1.set_xticks(range(-10, 12, 2))
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8.4, loc="lower right", framealpha=0.95)

    # ================= 右：文献[9] 加噪口径 =================
    names = [n for n, _ in LIT9]
    vals = [v for _, v in LIT9]
    y = np.arange(len(names))[::-1]
    colors = ["#1a5fa8" if n.startswith("CA-DRSN") else "#7fb3d5" for n in names]
    bars = ax2.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    for yy, vv in zip(y, vals):
        ax2.text(vv + 0.35, yy, f"{vv:.1f}", fontsize=8.5, va="center", color="#333333")
    ax2.set_xlim(70, 103.5)
    ax2.set_xticks([70, 75, 80, 85, 90, 95, 100])
    ax2.set_xlabel("准确率 (%)", fontsize=11.5)
    ax2.set_title("(b) 文献[9] CA-DRSN：SEU 齿轮 + 加噪（噪声水平 0.1）", fontsize=12)
    ax2.grid(axis="x", alpha=0.3, zorder=0)

    # 本文齿轮 6 dB 区间参考带（口径不同，仅示量级）
    lo, hi = min(fl[("gear", "20_0")][6], fl[("gear", "30_2")][6]), \
             max(fl[("gear", "20_0")][6], fl[("gear", "30_2")][6])
    ax2.axvspan(lo, hi, color="#e74c3c", alpha=0.13, zorder=1)
    ax2.axvline(lo, color="#e74c3c", lw=1.0, ls=":", zorder=2)
    ax2.axvline(hi, color="#e74c3c", lw=1.0, ls=":", zorder=2)
    ax2.text(hi + 0.3, len(names) - 1.35, f"本文齿轮 6 dB\n{lo:.1f}~{hi:.1f}%\n（口径不同）",
             fontsize=8.0, color="#c0392b", va="top")
    ax2.set_ylim(-0.7, len(names) - 0.3)

    fig.text(0.5, 0.028,
             "本文=端到端加噪训练、分工况 5 类（齿轮/轴承各两工况，仅取 ≤10 dB 段）；"
             "文献[2]=先降噪再诊断（成本更高）；文献[1]=两工况合并 5 类（80/20）。"
             "左栏为 SNR 口径、右栏为「噪声水平 0.1」口径，两者不同，仅作横向对照、不逐点对齐。",
             ha="center", fontsize=7.6, color="#555555")

    fig.tight_layout(rect=(0, 0.065, 1, 0.99))
    out = FIG / "noise_lit_compare.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    main()
