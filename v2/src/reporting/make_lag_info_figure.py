"""画表示参数的信息论分析图（对应 Shankar et al. 的 AMI / FNN 图）。"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from reporting.make_noise_figure import setup_font  # noqa: E402
from utils.console import setup_utf8_console  # noqa: E402
setup_utf8_console()

def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
setup_font()

for ax, (task, cond, title) in zip(axes, [("gear", "20_0", "齿轮 20-0"), ("bearing", "20_0", "轴承 20-0")]):
    p = PROJECT_ROOT / "reports" / f"lag_information_{task}_{cond}.csv"
    if not p.exists():
        ax.axis("off"); continue
    rows = load(p)
    ks = [int(r["lag"]) for r in rows if int(r["bins"]) == 6]
    Is = [float(r["I_nat"]) for r in rows if int(r["bins"]) == 6]
    ax.bar(ks, Is, color="tab:blue", alpha=0.75)
    mean_in = sum(v for k, v in zip(ks, Is) if k <= 10) / 10
    mean_out = sum(v for k, v in zip(ks, Is) if k > 10) / len([k for k in ks if k > 10])
    ax.axhline(mean_in, color="tab:red", ls="--", lw=1.2, label=f"k≤10 均值 {mean_in:.4f}")
    ax.axhline(mean_out, color="tab:green", ls=":", lw=1.5, label=f"k>10 均值 {mean_out:.4f}")
    ax.axvline(10.5, color="gray", ls="-", lw=1.0, alpha=0.6)
    ax.set_title(f"{title}（B=6 档）", fontsize=11)
    ax.set_xlabel("滞后 k（采样点）")
    ax.set_ylabel("符号互信息 I(k)  (nat)")
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("本文表示的滞后阶数：符号互信息随滞后的衰减（训练集）", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out1 = PROJECT_ROOT / "reports" / "fig_lag_mi.png"
fig.savefig(out1, bbox_inches="tight")
print("[ok]", out1)

fig2, ax2 = plt.subplots(figsize=(5.6, 4.2))
setup_font()
p = PROJECT_ROOT / "reports" / "lag_information_gear_20_0.csv"
rows = [r for r in load(p) if int(r["lag"]) == 1]
bs = [int(r["bins"]) for r in rows]
ih = [float(r["I_over_H"]) for r in rows]
ax2.plot(bs, ih, "o-", color="tab:blue", lw=1.6)
for b, v in zip(bs, ih):
    ax2.annotate(f"{v:.4f}", (b, v), textcoords="offset points", xytext=(4, 6), fontsize=8)
ax2.set_xlabel("量化档数 B")
ax2.set_ylabel("归一化互信息 I(1)/H(s)")
ax2.set_title("量化档数的信息量饱和曲线（齿轮 20-0，lag=1）", fontsize=10)
ax2.set_xticks(bs)
ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = PROJECT_ROOT / "reports" / "fig_bins_saturation.png"
fig2.savefig(out2, bbox_inches="tight")
print("[ok]", out2)
