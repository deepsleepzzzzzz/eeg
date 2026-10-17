"""判定性测试：抗噪曲线是否其实是"能量分类"造成的假象。

背景：add_noise() 里 p_noise = p_sig / 10**(snr/10)，p_sig 是【该窗自己】的均方，
      因此加噪窗能量 ≈ 干净窗能量 × 常数。于是"仅凭窗能量"可能在任何 SNR 下都能分类。

【协议必须匹配】分类器在加噪的 train 上拟合，在有同样 SNR 的 val 上评估，
否则会得到假结论（第一版就踩了这个坑：用干净 train 拟合、噪声 val 评估，
噪声在 log 空间对"响亮类"和"安静类"的抬升幅度不同，导致最近类中心全部错位）。

特征：① 每通道对数均方（3 维）  ② 池化对数均方（1 维）
分类器：最近类中心（与训练代码完全无关）

用法：python src/analysis/energy_only.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from config.protocol import load_protocol, window_cfg, channel_indices, raw_root as protocol_raw_root  # noqa: E402
from data.build_dataset import split_intervals  # noqa: E402
from data.seu_dds import add_noise, class_file, class_names, cut_windows, load_seu_csv, task_dir, vibration_xyz, window_starts  # noqa: E402

CONDITIONS = (("gear", "20_0"), ("gear", "30_2"), ("bearing", "20_0"), ("bearing", "30_2"))


def find_main_protocol() -> Path:
    from config.protocol import load_protocol as _lp

    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(_lp(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def load_split(protocol: dict, task: str, condition: str, split: str):
    length, stride = window_cfg(protocol)
    lo, hi = split_intervals(protocol)[split]
    chans = channel_indices(protocol)
    folder = task_dir(protocol_raw_root(protocol), task)
    xs, ys = [], []
    for label, name in enumerate(class_names(task)):
        raw = load_seu_csv(folder / class_file(task, name, condition))
        xyz = vibration_xyz(raw, chans)
        n = int(xyz.shape[1])
        starts = window_starts(lo, min(int(hi), n), length, stride)
        for w in cut_windows(xyz, starts, length):
            xs.append(np.asarray(w, dtype=np.float64))
            ys.append(label)
    return xs, np.asarray(ys, dtype=np.int64)


def feat_per_channel(windows) -> np.ndarray:
    return np.stack([np.log(np.mean(np.asarray(w, dtype=np.float64) ** 2, axis=1) + 1e-30) for w in windows])


def feat_pooled(windows) -> np.ndarray:
    return np.log(np.array([np.mean(np.asarray(w, dtype=np.float64) ** 2) for w in windows]) + 1e-30)[:, None]


def nearest_centroid(train_x, train_y, test_x):
    classes = sorted(set(train_y.tolist()))
    cent = np.stack([train_x[train_y == c].mean(axis=0) for c in classes])
    d = ((test_x[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(classes)[d.argmin(axis=1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--snr", default="40,4,0,-4,-6,-8,-10,-12,-14,-16,-20")
    p.add_argument("--reference", default="per_window", choices=["per_window", "global"],
                   help="噪声参考功率口径，与 noise.reference 对应")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    snrs = [float(s) for s in args.snr.split(",")]

    proto_path = find_main_protocol()
    protocol = load_protocol(proto_path)
    safe_print(f"[energy_only] 基准协议: {proto_path.name}（匹配加噪：train 与 val 同 SNR）")
    safe_print(f"[energy_only] 噪声参考口径: {args.reference}")
    safe_print("格式 = SNR: 三通道对数能量准确率 / 池化对数能量准确率（仅窗口能量，无任何学习）\n")

    out_rows = []
    for task, condition in CONDITIONS:
        tr_x, tr_y = load_split(protocol, task, condition, "train")
        va_x, va_y = load_split(protocol, task, condition, "val")
        p_ref = None
        if args.reference == "global":
            import numpy as _np
            p_ref = float(_np.mean(_np.concatenate([_np.asarray(w, dtype=_np.float64).ravel() for w in tr_x]) ** 2))
        parts = []
        for snr in snrs:
            a3, a1 = [], []
            for sd in seeds:
                ntr = add_noise(tr_x, snr, sd, p_ref=p_ref)
                nva = add_noise(va_x, snr, sd + 100, p_ref=p_ref)
                a3.append(float((nearest_centroid(feat_per_channel(ntr), tr_y, feat_per_channel(nva)) == va_y).mean()))
                a1.append(float((nearest_centroid(feat_pooled(ntr), tr_y, feat_pooled(nva)) == va_y).mean()))
            parts.append(f"{snr:+4.0f}:{100*np.mean(a3):5.1f}/{100*np.mean(a1):5.1f}")
            out_rows.append({"task": task, "condition": condition, "reference": args.reference,
                             "snr_db": snr, "acc_3ch": float(np.mean(a3)), "acc_pooled": float(np.mean(a1))})
        safe_print(f"{task:8s} {condition} : " + "  ".join(parts))

    import csv as _csv
    out = PROJECT_ROOT / f"reports/energy_only_{args.reference}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    safe_print(f"\n-> {out}")


if __name__ == "__main__":
    main()
