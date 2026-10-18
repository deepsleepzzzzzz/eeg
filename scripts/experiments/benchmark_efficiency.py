"""Efficiency benchmark for SEU CNN / GCN classifiers.

Reports, per (task, arch):
  - parameters (trainable / total)
  - model size on disk (fp32 state_dict bytes)
  - FLOPs (Conv2d + Linear via forward hooks; GCN graph matmuls estimated)
  - forward+backward time per training step (batch)
  - inference latency + throughput (samples/sec) over the val split
  - peak CUDA memory during training steps

Output -> datas/experiments_seu/teacher_extra/efficiency.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.train_runtime import get_device, load_array_pair, make_loader, resolve_path
from models.cnn import build_model

OUT_DIR = Path("datas/experiments_seu/teacher_extra")
CONFIGS = [
    "config/seu_bearing_cnn.yaml",
    "config/seu_bearing_gcn.yaml",
    "config/seu_gear_cnn.yaml",
    "config/seu_gear_gcn.yaml",
]


def count_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Count MACs for Conv2d + Linear via hooks (2*MACs = FLOPs)."""
    total = [0]

    def conv_hook(m, inp, out):
        o = out[0]
        k = m.weight
        # output spatial * kernel volume * in_channels * out_channels
        macs = o.numel() * (k.shape[1] * k.shape[2] * k.shape[3])
        total[0] += macs

    def linear_hook(m, inp, out):
        o = out[0]
        macs = o.numel() * m.weight.shape[1]
        total[0] += macs

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    gcn_matmul_macs = 0
    arch = getattr(model, "_arch_hint", "cnn")
    if arch == "gcn":
        # RelLayer: torch.matmul(mats[:,:,si] [B,C,N,N], wh [B,C,N,F]) per step
        b = x.shape[0]
        n = x.shape[-1]
        num_graphs = x.shape[1]
        steps = model.steps
        hidden = model.hidden
        c = num_graphs // steps
        # per-step matmul MACs = B*C*N*N*hidden ; summed over steps and 2 layers
        gcn_matmul_macs = 2 * (b * c * n * n * hidden * steps)
    return total[0] + gcn_matmul_macs


def benchmark_one(config_path: str) -> dict:
    cfg = OmegaConf.load(resolve_path(config_path))
    task = str(cfg.data.task)
    condition = "30_2"  # hardest condition, representative
    arch = str(cfg.model.arch).lower()
    device = get_device(cfg)
    data_dir = resolve_path(cfg.data.output_root) / task / condition
    X_train, y_train = load_array_pair(data_dir, "train")
    X_val, y_val = load_array_pair(data_dir, "val")
    in_channels = int(X_train.shape[1])
    image_size = int(X_train.shape[-1])
    names_n = 5

    model = build_model(names_n, cfg, in_channels=in_channels, image_size=image_size).to(device)
    model._arch_hint = arch

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model size (fp32 bytes) via state_dict
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.tell()

    # FLOPs
    x_probe = torch.zeros(1, in_channels, image_size, image_size, device=device)
    flops = count_flops(model, x_probe)

    # training step timing + peak memory
    batch_size = int(cfg.train.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    criterion = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(device)
    xb = torch.from_numpy(X_train[:batch_size]).to(device)
    yb = torch.from_numpy(y_train[:batch_size]).to(device)

    # warmup
    for _ in range(3):
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    n_steps = 20
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    step_time = (time.perf_counter() - t0) / n_steps
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # inference latency + throughput over val
    val_loader = make_loader(X_val, y_val, batch_size, 0, False)
    model.eval()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for vx, vy in val_loader:
            model(vx.to(device))
    torch.cuda.synchronize(device)
    inf_time = time.perf_counter() - t0
    n_val = int(len(y_val))
    throughput = n_val / inf_time if inf_time > 0 else 0.0
    latency_per_batch = inf_time / max(1, (n_val // batch_size))

    return {
        "task": task,
        "arch": arch,
        "input_shape": [in_channels, image_size, image_size],
        "params": int(n_params),
        "trainable_params": int(n_trainable),
        "model_size_bytes": int(model_bytes),
        "model_size_kb": round(model_bytes / 1024, 2),
        "flops_per_sample": int(flops),
        "train_step_ms": round(step_time * 1000, 3),
        "peak_cuda_memory_mb": round(peak_mem, 2),
        "inference_val_samples": n_val,
        "inference_time_s": round(inf_time, 4),
        "throughput_samples_per_s": round(throughput, 1),
        "latency_per_batch_ms": round(latency_per_batch * 1000, 3),
        "device": str(device),
        "batch_size": batch_size,
    }


def main():
    results = []
    for c in CONFIGS:
        r = benchmark_one(c)
        results.append(r)
        print(f"{r['task']:8s} {r['arch']:4s}  params={r['params']:,}  "
              f"size={r['model_size_kb']}KB  FLOPs={r['flops_per_sample']:,}  "
              f"step={r['train_step_ms']}ms  thr={r['throughput_samples_per_s']:.0f}/s  "
              f"mem={r['peak_cuda_memory_mb']}MB", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / OUT_DIR / "efficiency.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'efficiency.json'}")


if __name__ == "__main__":
    main()
