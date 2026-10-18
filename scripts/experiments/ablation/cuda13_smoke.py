"""CUDA 13 smoke: ONE baseline run (bins 6, lags 1-3, bearing, CNN, 20_0, seed 42) on the existing b6_L123 cache.
Writes only to outputs/ablation/cuda13_smoke/. Test split not evaluated; no paper figures."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.build_seu_dataset import build_condition
from scripts.experiments.train_seu import train_one

OUT = ROOT / "outputs" / "ablation" / "cuda13_smoke"
cfg = OmegaConf.load(ROOT / "config" / "seu_bearing_cnn.yaml")
OmegaConf.set_struct(cfg, False)
cfg.method.symbol_bins = 6
cfg.method.transition_steps = [1, 2, 3]
cfg.data.output_root = str(ROOT / "outputs" / "ablation" / "representations" / "b6_L123")  # existing cache, reused read-only
cfg.train.lr = 5e-4; cfg.train.dropout = 0.1
cfg.train.output_dir = str(OUT / "run")
cfg.train.evaluate_test = False
info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
t = time.perf_counter(); build_condition("bearing", "20_0", cfg, force=False); info["build_or_skip_s"] = time.perf_counter() - t
t = time.perf_counter()
r = train_one("bearing", "20_0", 42, cfg, evaluate_test=False, confirm_test=False, write_paper_figures=False)
dt = time.perf_counter() - t
info.update({"train_s": dt, "epochs_ran": r["epochs_ran"], "best_epoch": r["best_epoch"], "best_val_acc": r["best_val_acc"],
             "val_macro_f1": r["val_macro_f1"], "input_shape": r["input_shape"],
             "reference_seed42_val_acc_pre_cuda13": 0.6867})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "smoke_result.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
print("SMOKE", json.dumps(info, default=str))
