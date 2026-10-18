"""Verify noise injection SNR correctness + visualize clean vs noisy (SEU).

Advisor asked: "跑完加噪实验后自己验证一下噪声到底加了多少、加对了没有".

This script, for each target SNR, injects AWGN into a real SEU window and
measures the *actual* SNR = 10*log10(P_signal / P_noise) to confirm the
injection matches the label. Pure numpy (no torch), so it runs immediately.

Outputs: teacher_extra/noise_snr_verify.json + doc/figures/seu/noise_snr_verify.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.seu_dds import class_file, cut_windows, load_seu_csv, task_dir, vibration_xyz, window_starts

OUT_DIR = ROOT / "datas/experiments_seu/teacher_extra"
FIG_DIR = ROOT / "doc" / "figures" / "seu"

SNRS = [40, 30, 20, 15, 10, 6, 2, 0, -2, -4]


def add_noise(w, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=np.float64)
    p_sig = float(np.mean(w ** 2))
    p_noise = p_sig / (10 ** (snr_db / 10.0)) if p_sig > 0 else 0.0
    noise = np.sqrt(p_noise) * rng.standard_normal(w.shape)
    return w + noise, noise


def measure_snr(w_clean, noise):
    p_sig = float(np.mean(w_clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    return 10.0 * np.log10(p_sig / p_noise) if p_noise > 0 else float("inf")


def main():
    # use a real bearing "health 20_0" window as the probe signal
    folder = task_dir(Path("datas/seu"), "bearing")
    raw = load_seu_csv(folder / class_file("bearing", "health", "20_0"))
    xyz = vibration_xyz(raw)
    starts = window_starts(0, 1600, 1600, 1600)
    win = cut_windows(xyz, starts, 1600)[0]  # [3, 1600]
    x = win[0]  # use channel 0

    rows = []
    for snr in SNRS:
        noisy, noise = add_noise(x, snr, seed=0)
        actual = measure_snr(x, noise)
        rows.append({"target_snr_db": snr, "actual_snr_db": round(actual, 3),
                     "p_signal": round(float(np.mean(x ** 2)), 6),
                     "p_noise": round(float(np.mean(noise ** 2)), 6)})
        print(f"target={snr:>4d} dB  actual={actual:>8.3f} dB  "
              f"P_sig={np.mean(x**2):.6f}  P_noise={np.mean(noise**2):.6f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "noise_snr_verify.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # visual: clean + a few SNR levels
    sel = [40, 10, 0, -4]
    fig, axes = plt.subplots(len(sel), 1, figsize=(9, 2.2 * len(sel)), sharex=True)
    for ax, snr in zip(axes, sel):
        noisy, noise = add_noise(x, snr, seed=0)
        actual = measure_snr(x, noise)
        ax.plot(x[:400], lw=0.6, color="tab:blue", label="clean")
        ax.plot(noisy[:400], lw=0.4, color="tab:red", alpha=0.8,
                label=f"SNR {snr} dB (actual {actual:.2f} dB)")
        ax.set_ylabel(f"{snr} dB")
        ax.legend(loc="upper right", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Noise injection verification: clean vs noisy (bearing health 20_0, ch0)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_snr_verify.png", dpi=130)
    plt.close(fig)
    print(f"wrote noise_snr_verify.json + PNG")


if __name__ == "__main__":
    main()
