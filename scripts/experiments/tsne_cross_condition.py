"""导师问题5：把两种工况放进同一张 t-SNE 图，看特征提取后的分布。

左：状态转移表示（[21,6,6] 展平，两工况同源可比）
右：CNN 高层特征（各工况用各自训练好的模型提取，按维度做 z-score 统一尺度）
颜色 = 故障类别，标记 = 工况。只用 val（测试集封闭）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.train_runtime import get_device, load_array_pair, make_loader, resolve_path
from models.cnn import build_model

REP = ROOT / "datas" / "experiments_seu" / "representations_resplit"
FIG = ROOT / "doc" / "figures" / "seu" / "teacher"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONDS = ("20_0", "30_2")
COND_LABEL = {"20_0": "20 Hz-0 V", "30_2": "30 Hz-2 V"}
MARKER = {"20_0": "o", "30_2": "^"}
COLORS = plt.get_cmap("tab10").colors


def _ckpt_root(task: str) -> Path:
    return ROOT / "datas" / "experiments_seu" / f"checkpoints_{task}_cnn_resplit"


def load_cond(task: str, cond: str):
    d = REP / task / cond
    X, y = load_array_pair(d, "val")
    names = json.loads((d / "meta.json").read_text(encoding="utf-8"))["class_names"]
    return X.astype(np.float32), y.astype(np.int64), list(names)


def embed_cond(task: str, cond: str, X: np.ndarray, y: np.ndarray, n_cls: int) -> np.ndarray:
    cfg = OmegaConf.load(resolve_path(f"config/seu_{task}_cnn.yaml"))
    run_dir = _ckpt_root(task) / task / cond / "seed_42"
    device = get_device(cfg)
    model = build_model(n_cls, cfg, in_channels=int(X.shape[1]), image_size=int(X.shape[-1])).to(device)
    ckpt = torch.load(run_dir / f"best_seu_{task}_cnn.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = make_loader(X, y, 256, 0, False)
    vecs = []
    with torch.no_grad():
        for xb, _ in loader:
            vecs.append(model.embed(xb.to(device)).cpu().numpy())
    return np.concatenate(vecs, axis=0)


def tsne2d(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    n = int(Z.shape[0])
    perp = float(min(30.0, max(5.0, (n - 1) / 4.0)))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(Z)


def fig_task(task: str) -> None:
    X20, y20, names = load_cond(task, "20_0")
    X30, y30, _ = load_cond(task, "30_2")
    n_cls = len(names)

    # --- 左：状态转移表示（两工况同一算法生成，天然同源） ---
    Z_rep = np.vstack([X20.reshape(len(X20), -1), X30.reshape(len(X30), -1)])
    lab = np.concatenate([y20, y30])
    grp = np.array(["20_0"] * len(X20) + ["30_2"] * len(X30))
    xy_rep = tsne2d(Z_rep)

    # --- 右：CNN 高层特征（各工况用各自模型，按维度 z-score 归一后合并） ---
    E20 = embed_cond(task, "20_0", X20, y20, n_cls)
    E30 = embed_cond(task, "30_2", X30, y30, n_cls)
    def _z(A):
        return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-8)
    xy_emb = tsne2d(np.vstack([_z(E20), _z(E30)]))

    pretty = {"gear": "齿轮箱", "bearing": "轴承"}[task]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, xy, title in (
        (axes[0], xy_rep, "左：多时间步状态转移表示层"),
        (axes[1], xy_emb, "右：CNN 高层特征层"),
    ):
        for c, name in enumerate(names):
            for cond in CONDS:
                m = (lab == c) & (grp == cond)
                if not m.any():
                    continue
                ax.scatter(xy[m, 0], xy[m, 1], s=9 if cond == "20_0" else 13,
                           c=[COLORS[c % 10]], marker=MARKER[cond],
                           alpha=0.55 if cond == "20_0" else 0.8, linewidths=0,
                           label=name if cond == "20_0" else None)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.15)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[c % 10], label=name)
               for c, name in enumerate(names)]
    handles += [
        Line2D([], [], marker=MARKER["20_0"], ls="", color="gray", label="20 Hz-0 V"),
        Line2D([], [], marker=MARKER["30_2"], ls="", color="gray", label="30 Hz-2 V"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9.5, framealpha=0.9)
    fig.suptitle(f"SEU DDS {pretty}：两种工况样本的 t-SNE 特征分布（验证集）", fontsize=14)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    out = FIG / f"tsne_cross_{task}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out.name, flush=True)


if __name__ == "__main__":
    for t in ("gear", "bearing"):
        fig_task(t)
