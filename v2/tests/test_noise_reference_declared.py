"""守卫：噪声参考功率口径必须在 yaml 里显式声明，不许靠代码默认值。

背景（2026-10-15）：
  `noise.reference` 曾是一个静默默认值（per_window）。逐窗参考会让
  加噪窗能量 = 干净窗能量 × 常数，"窗能量"因此成为与表示无关的捷径。
  实测：齿轮 20-0 在 -10 dB 虚高 13.9 点，轴承 20-0 虚高 64.7 点。

本测试锁死两件事：
  1. 所有启用噪声的 yaml 必须显式写 noise.reference
  2. build_dataset 在缺该字段时必须报错（不能再有静默默认）
"""
import sys

import pytest
import yaml

from _bootstrap import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

YAML_DIRS = [PROJECT_ROOT / "configs" / "scan", PROJECT_ROOT / "configs" / "protocols"]


def _noise_variants():
    """(文件, 变体名, 噪声配置) —— 覆盖扫描文件里的每个噪声变体。"""
    out = []
    for d in YAML_DIRS:
        for f in sorted(d.glob("*.yaml")):
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            variants = raw.get("variants")
            if isinstance(variants, dict):
                for name, over in variants.items():
                    if isinstance(over, dict):
                        ref = over.get("noise.reference")
                        enabled = over.get("noise.enabled")
                        if enabled:
                            out.append((f.name, name, ref))
            else:
                noise = raw.get("noise")
                if isinstance(noise, dict) and noise.get("reference") is None:
                    out.append((f.name, "<文件级 noise>", None))
    return out


def test_every_noise_variant_declares_reference():
    missing = [(f, v) for f, v, ref in _noise_variants() if ref is None]
    assert not missing, f"这些噪声变体没显式声明 noise.reference：{missing}"


def test_build_dataset_refuses_missing_reference():
    """缺字段时必须报错 —— 防止静默默认回归。"""
    from data.build_dataset import noise_reference_mode

    cfg = {"enabled": True, "type": "awgn", "snr_db": -10, "splits": "train,val", "seed": 0}
    with pytest.raises(ValueError, match="noise.reference"):
        noise_reference_mode(cfg, "unit_test")
    # 显式声明后必须原样返回，不能改写大小写之外的内容
    assert noise_reference_mode({**cfg, "reference": "global"}, "unit_test") == "global"
    assert noise_reference_mode({**cfg, "reference": "per_window"}, "unit_test") == "per_window"
