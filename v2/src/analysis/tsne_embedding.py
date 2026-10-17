"""网络学习特征的 t-SNE（文献里的标准画法）——与"原始 1080 维表示"的 t-SNE 区分开。

为什么单独做这件事（2026-10-17 用户问"有没有人看起来可分的 t-SNE"）：
同类论文（DSMC-ECA / YConvFormer / GhostNet 等）画的都是**训练后网络倒数第二层的特征**，
不是手工输入表示；分类器被训练来分开这些类，所以图看上去干净。本文原来的图 8 画的是
原始 1080 维转移概率张量、还把两种工况叠在一起（10 个簇），属于最不利的画法。

本脚本：
  1. 读取 `train.py --save-model` 落盘的最佳权重（best_state.pt）；
  2. 用模型自带的 `embed()` 提取验证集特征（CNN：全局池化向量；GCN：分类头之前的图向量）；
  3. 按类别做 t-SNE（perplexity=50、random_state=42），画 1σ 椭圆 + 类心；
  4. 输出一张多面板图 + 一个 JSON（各类轮廓系数、验证精度）。

用法：
    python src/analysis/tsne_embedding.py \
      --pairs "gear:20_0:cnn:42,bearing:20_0:cnn:42" \
      --state-root artifacts/embed_runs \
      --out reports/fig_feature_tsne.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import dataset_dir, load_protocol  # noqa: E402
from data.seu_dds import class_names  # noqa: E402
from models.cnn import build_model  # noqa: E402
from reporting.class_labels import cn_name  # noqa: E402
from reporting.make_noise_figure import setup_font  # noqa: E402
from training.train import build_cfg, load_arrays  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
TASK_CN = {"gear": "齿轮", "bearing": "轴承"}


def main_protocol() -> dict:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        cfg = load_protocol(f)
        if str(cfg.get("status")) == "main":
            return cfg
    raise RuntimeError("找不到 status=main 的主协议")


def embed_split(model, X: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(device)
            out.append(model.embed(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="task:condition:model:seed[,...]")
    ap.add_argument("--state-root", default="artifacts/embed_runs")
    ap.add_argument("--out", default="reports/fig_feature_tsne.png")
    ap.add_argument("--perplexity", type=float, default=50.0)
    a = ap.parse_args()

    setup_font()
    proto = main_protocol()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_root = PROJECT_ROOT / a.state_root

    pairs = []
    for item in a.pairs.split(","):
        task, cond, model, seed = item.split(":")
        pairs.append((task, cond, model, int(seed)))

    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2 * len(pairs), 6.6))
    if len(pairs) == 1:
        axes = [axes]
    stats = {}
    for ax, (task, cond, model_name, seed) in zip(axes, pairs):
        exp_id = f"{task}_{model_name}_{cond}_seed{seed}"
        run_dir = state_root / exp_id
        state_path = run_dir / "best_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"缺权重 {state_path}（先跑 train.py --save-model --output-root {a.state_root}）")
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        d = dataset_dir(proto, task, cond)
        X, y = load_arrays(d, "val")
        cfg = build_cfg(proto, model_name, device=str(device))
        net = build_model(len(class_names(task)), cfg, in_channels=X.shape[1], image_size=X.shape[-1]).to(device)
        net.load_state_dict(blob["state"])
        F = embed_split(net, X, device)
        Z = TSNE(n_components=2, perplexity=a.perplexity, init="pca", learning_rate="auto",
                 random_state=42, max_iter=1500).fit_transform(StandardScaler().fit_transform(F))

        names = list(class_names(task))
        for ci in range(5):
            m = y == ci
            pts = Z[m]
            mu = pts.mean(axis=0)
            ax.scatter(pts[:, 0], pts[:, 1], s=9, color=COLORS[ci], alpha=0.55,
                       edgecolors="none", zorder=2)
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
            w, h = 2 * 1.0 * np.sqrt(np.maximum(vals, 1e-12))
            ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=to_rgba(COLORS[ci], 0.10),
                                 edgecolor=to_rgba(COLORS[ci], 0.85), lw=1.2, zorder=3))
            ax.scatter(*mu, s=110, marker="D", facecolor="white", edgecolor=COLORS[ci],
                       lw=1.5, zorder=5)
        metrics_path = run_dir / "metrics.json"
        acc = None
        if metrics_path.exists():
            acc = 100 * float(json.loads(metrics_path.read_text(encoding="utf-8"))["val_metrics"]["accuracy"])
        title = f"{TASK_CN[task]}任务（{model_name.upper()} 倒数第二层特征）"
        if acc is not None:
            title += f"  验证精度 {acc:.2f}%"
        ax.set_title(title, fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc"); sp.set_linewidth(0.8)
        handles = [Line2D([], [], marker="o", ls="", color=COLORS[i], markersize=7) for i in range(5)]
        ax.legend(handles, [cn_name(task, n) for n in names], fontsize=8.5, loc="best", framealpha=0.92)
        sil = float(silhouette_score(StandardScaler().fit_transform(F), y, sample_size=min(1500, len(y)),
                                     random_state=42))
        stats[exp_id] = {"task": task, "condition": cond, "model": model_name, "seed": seed,
                         "val_acc": acc, "silhouette_embedding": sil,
                         "embedding_dim": int(F.shape[1])}
        print(f"[ok] {exp_id}: embed_dim={F.shape[1]}  val_acc={acc}  silhouette={sil:.3f}")

    fig.suptitle("网络倒数第二层特征的 t-SNE（训练后的 CNN/GCN，验证集）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PROJECT_ROOT / a.out
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (PROJECT_ROOT / "reports/feature_tsne.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())