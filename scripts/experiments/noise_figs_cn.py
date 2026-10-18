"""导师问题6：加噪结果的中文图。

两套协议各出一张：
  noise_cn.png           加噪训练 / 加噪测试（与文献同协议）
  noise_cn_trainclean.png 干净训练 / 加噪测试（更难，考察零样本抗噪）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "datas" / "experiments_seu" / "teacher_extra"
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TASK_CN = {"bearing": "轴承", "gear": "齿轮箱"}
COND_CN = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}
STYLE = {
    ("bearing", "20_0"): ("#c0392b", "-", "o"),
    ("bearing", "30_2"): ("#e67e22", "-", "s"),
    ("gear", "20_0"): ("#2471a3", "-", "^"),
    ("gear", "30_2"): ("#1e8449", "-", "D"),
}


def draw(path: Path, out_name: str, suptitle: str, note: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, task in zip(axes, ("bearing", "gear")):
        for r in data:
            if r["task"] != task:
                continue
            key = (r["task"], r["condition"])
            if key not in STYLE:
                continue
            color, ls, mk = STYLE[key]
            xs = [p["snr_db"] for p in r["curve"]]
            ys = [p["acc"] * 100 for p in r["curve"]]
            ax.plot(xs, ys, marker=mk, ls=ls, color=color, lw=1.8, ms=5,
                    label=f"{TASK_CN[task]} {COND_CN[r['condition']]}")
            for x, y in zip(xs, ys):
                if x in (-4, 0, 6):
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 7), fontsize=7.5, ha="center", color=color)
        ax.set_xlabel("信噪比 SNR (dB)", fontsize=11)
        ax.set_title(f"({('a' if task == 'bearing' else 'b')}) {TASK_CN[task]}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
        ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel("测试准确率 (%)", fontsize=11)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.text(0.5, 0.012, note, ha="center", fontsize=8.5, color="#555555")
    out = FIG / out_name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    draw(TD / "noise_robustness.json", "noise_cn.png",
         "加噪后的分类准确率（加噪训练 / 加噪测试）",
         "协议：对原始信号按目标 SNR 注入高斯白噪声，分位边界在加噪训练集上重新拟合；每档 SNR 独立建模。")
    draw(TD / "noise_robustness_trainclean.json", "noise_cn_trainclean.png",
         "加噪后的分类准确率（干净训练 / 加噪测试）",
         "协议：模型仅在干净信号上训练，测试时才注入噪声；考验零样本抗噪能力（无任何噪声先验）。")
