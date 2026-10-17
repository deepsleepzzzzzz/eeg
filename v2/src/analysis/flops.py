"""参数量 / FLOPs / 推理时间 —— 用 PyTorch 官方计数器，不再用来源不明的旧值。

背景：表里此前的 "0.384 MFLOPs / 0.558 MFLOPs" 来源不明，且与手算不符
      （当前主协议 v5 的输入是 30 通道；文中旧值属历史记录）。

口径说明：
  - FLOPs 用 torch.utils.flop_counter.FlopCounterMode，按官方定义（乘加算 2 次）；
  - 单样本、推理模式、batch=1；
  - 推理时间 = GPU 上 200 次前向的均值（warmup 50 次），需 CUDA。

用法：python src/analysis/flops.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.console import setup_utf8_console, safe_print  # noqa: E402

setup_utf8_console()

from config.protocol import (  # noqa: E402
    load_protocol,
    build_train_omega,
    method_cfg_from_protocol,
)
from models.cnn import build_model  # noqa: E402


def find_main_protocol() -> Path:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


def measure(model, shape, device, n_warmup=50, n_iter=200):
    x = torch.randn(*shape, device=device)
    model = model.to(device).eval()
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    flops = int(fc.get_total_flops())
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iter * 1e3
    return flops, ms


def main() -> None:
    proto_path = find_main_protocol()
    proto = load_protocol(proto_path)
    lags = [int(x) for x in proto["representation"]["lags"]]
    chans = proto["representation"]["channels"]
    nodes = int(proto["preprocessing"]["bins"])
    ncls = int(proto["dataset"]["num_classes"])
    in_ch = len(lags) * len(chans)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"协议: {proto_path.name}")
    safe_print(f"输入: 转移矩阵 [{in_ch}, {nodes}, {nodes}]  （= {len(chans)} 通道 × {len(lags)} 滞后）")
    safe_print(f"设备: {device}\n")
    safe_print(f"{'模型':>6} | {'参数量':>10} | {'FLOPs':>14} | {'推理(ms/样本)':>14}")
    safe_print("-" * 56)

    rows = []
    for name in ("cnn", "gcn"):
        cfg = build_train_omega(proto, name, device=str(device))
        model = build_model(ncls, cfg, in_channels=in_ch, image_size=nodes)
        nparam = sum(p.numel() for p in model.parameters())
        flops, ms = measure(model, (1, in_ch, nodes, nodes), device)
        rows.append({"model": name, "params": nparam, "flops": flops,
                     "ms_per_sample": ms, "device": str(device)})
        safe_print(f"{name.upper():>6} | {nparam:>10,d} | {flops:>11,d} ({flops/1e6:.3f} M) | {ms:>11.3f}")

    # 对文献：DSMC-ECA 报 0.204 M 参数 / 10.037 MFLOPs（其输入为 1024 点单通道）
    safe_print("\n对照（文献报告值，输入口径不同，仅供量级参照）：")
    safe_print("  DSMC-ECA (Sensors 2026) : 204,000 参数 / 10,037,000 FLOPs / 1024 点单通道")
    safe_print("  DDGF-Net (Machines 2026): 15,290,000 参数 / 61.2 MB (FP32) / 1024 点单通道")

    out = PROJECT_ROOT / "reports" / "efficiency.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"\n-> {out}")


if __name__ == "__main__":
    main()
