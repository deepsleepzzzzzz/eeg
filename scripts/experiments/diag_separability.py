"""Check separability between the hardest SEU class pairs via transition-matrix distance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK = sys.argv[1] if len(sys.argv) > 1 else "gear"
COND = sys.argv[2] if len(sys.argv) > 2 else "30_2"

rep_dirs = sorted(Path("datas/experiments_seu/tune").glob("*"))
cands = [d for d in rep_dirs if (d / TASK / COND / "X_val.npy").exists()]
if not cands:
    print("no cached representation found")
    sys.exit(1)
# pick the most recent cache
cache = cands[-1]
print(f"使用缓存: {cache}")
X_val = np.load(cache / TASK / COND / "X_val.npy")
y_val = np.load(cache / TASK / COND / "y_val.npy")
X_tr = np.load(cache / TASK / COND / "X_train.npy")
y_tr = np.load(cache / TASK / COND / "y_train.npy")
print(f"train={X_tr.shape} val={X_val.shape}")

n_cls = int(y_tr.max()) + 1
# class centroids from train, then pairwise cosine distance
cent = []
for c in range(n_cls):
    v = X_tr[y_tr == c].reshape(len(X_tr[y_tr == c]), -1).mean(axis=0)
    cent.append(v)
cent = np.stack(cent)
norm = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
sim = norm @ norm.T

print("\n训练集类中心余弦相似度（越高越难分）:")
hdr = "        " + "".join(f"{i:>8d}" for i in range(n_cls))
print(hdr)
pairs = []
for i in range(n_cls):
    row = "".join(f"{sim[i, j]:8.3f}" for j in range(n_cls))
    print(f"  cls{i:>4d}{row}")
    for j in range(i + 1, n_cls):
        pairs.append((sim[i, j], i, j))
pairs.sort(reverse=True)
print("\n最相似类对（按余弦相似度）:")
for s, i, j in pairs[:4]:
    mi = X_tr[y_tr == i].mean(axis=0).ravel()
    mj = X_tr[y_tr == j].mean(axis=0).ravel()
    d = float(np.linalg.norm(mi - mj))
    rel = d / (float(np.linalg.norm(mi)) + 1e-12)
    print(f"  cls{i} vs cls{j}: 余弦={s:.4f} 中心L2距离={d:.5f} 相对差异={rel:.4f}")

# per-sample: how confusable, using nearest-centroid on val
val_flat = X_val.reshape(len(X_val), -1)
dist = ((val_flat[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
pred = dist.argmin(axis=1)
acc = float((pred == y_val).mean())
print(f"\n最近类中心分类器 val_acc={acc:.4f}（不用训练，纯表示可分性上界参考）")
cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(y_val, pred):
    cm[t, p] += 1
print("混淆矩阵（行真值 列预测）:")
for i in range(n_cls):
    print("  " + "".join(f"{v:6d}" for v in cm[i]))
