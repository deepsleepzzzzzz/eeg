"""守卫：原始波形基线模型与 representation.kind 的接线。

为什么需要（2026-10-15）：
  为了回答"本文方法到底比谁强"，新增了原始波形基线（1D-CNN / WDCNN / ResNet1D）。
  这条路径引入了新的分派逻辑（model_cfg_from_protocol 的任意模型名分支、
  build_model 的 raw 分派、encode_windows 的 raw 表示），必须有测试锁住，
  否则静默回落到 CNN 会导致"基线其实是 CNN"这种致命错误。
"""
import sys

import numpy as np
import pytest
import torch

from _bootstrap import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.protocol import build_train_omega, load_protocol, model_cfg_from_protocol  # noqa: E402
from models.cnn import build_model  # noqa: E402


def _main_protocol():
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return load_protocol(f)
    raise RuntimeError("找不到主协议")


@pytest.mark.parametrize("arch", ["raw_cnn", "wdcnn", "resnet1d"])
def test_raw_baselines_build_and_forward(arch):
    """三个基线必须真的建出来、能前向、且【不是】CNN。"""
    from models.cnn import RPCNN
    from models.raw_models import RAW_ARCHS

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"method": {"attention": "none"},
                            "model": {"arch": arch},
                            "train": {"dropout": 0.2}})
    model = build_model(5, cfg, in_channels=3, image_size=2048)
    assert isinstance(model, RAW_ARCHS[arch]), f"{arch} 回落到了别的架构"
    assert not isinstance(model, RPCNN), f"{arch} 静默回落成了 RPCNN"
    y = model(torch.randn(2, 3, 2048))
    assert tuple(y.shape) == (2, 5)


def test_unknown_model_name_raises():
    """协议里没有的模型名必须报错，不能静默回落到 CNN。"""
    proto = {"models": {"cnn": {"conv_channels": [32], "kernel_size": 3, "pool_size": 2}}}
    with pytest.raises(KeyError, match="没有模型配置"):
        model_cfg_from_protocol(proto, "definitely_not_a_model")


def test_representation_kind_defaults_to_transition():
    """主协议没写 kind 时必须仍是 transition（不能影响已有结果）。"""
    proto = _main_protocol()
    cfg = build_train_omega(proto, "cnn", device="cpu")
    assert cfg["method"]["representation"] == "transition"


def test_raw_representation_shape():
    """raw 表示输出 [C, T]，与符号转移的 [K, B, B] 明显不同。"""
    from data.build_dataset import encode_windows

    cfg = {"representation": "raw", "normalize": "zscore"}
    X = encode_windows([np.random.randn(3, 256).astype(np.float64)], cfg)
    assert X.shape == (1, 3, 256), f"raw 表示输出形状错误: {X.shape}"
