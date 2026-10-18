"""资产完整性：参考文献格式、图片文件、实验/扫描/调参配置、数据清单。

与"论文数字"那几层互补：这层保证**论文引用和依赖的实体文件/配置本身是好的**。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"
REPORTS = PROJECT_ROOT / "reports"
CONFIGS = PROJECT_ROOT / "configs"
PROTOCOLS = CONFIGS / "protocols"


# --------------------------------------------------------------------------- #
# 一、参考文献：25 条逐条格式校验
# --------------------------------------------------------------------------- #
def _references() -> list[tuple[int, str]]:
    lines = PAPER.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("## 参考文献"))
    out = []
    for l in lines[start + 1:]:
        m = re.match(r"^\[(\d+)\]\s+(.*)$", l.strip())
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


REFS = _references()


@pytest.mark.parametrize("num,text", REFS, ids=[f"ref{n}" for n, _ in REFS])
def test_reference_is_well_formed(num, text):
    """编号连续 + 有作者 + 有年份 + 有刊物（斜体）+ DOI（若有）格式正确。"""
    assert num == REFS[num - 1][0], f"参考文献编号不连续：{num}"
    assert re.search(r"[A-Z][a-z]+", text), f"[{num}] 缺作者"
    assert re.search(r"\b(19|20)\d{2}\b", text), f"[{num}] 缺年份"
    assert "*" in text, f"[{num}] 刊物名没有用斜体"
    if "10." in text:
        assert re.search(r"10\.\d{4,9}/", text), f"[{num}] 的 DOI 格式可疑：{text[-60:]}"


# --------------------------------------------------------------------------- #
# 二、图片文件：16 张逐一校验
# --------------------------------------------------------------------------- #
FIG_FILES = [
    "fig_pipeline_style_a.png", "fig_condition_signals.png", "fig_fault_modes.png",
    "fig_lag_mi.png", "fig_lag_spacing.png", "fig_bins_saturation.png",
    "fig_confusion.png", "fig_feature_tsne.png", "fig_efficiency_scatter.png",
    "fig_transition.png", "fig_cond_matrices_gear.png", "fig_cond_matrices_bearing.png",
    "fig_noise_protocol.png", "fig_noise_robustness.png", "fig_tsne.png",
    "fig_transition_full.png",
]


@pytest.mark.parametrize("fname", FIG_FILES)
def test_figure_file_is_valid_png(fname):
    p = REPORTS / fname
    assert p.exists(), f"图片不存在：{fname}"
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{fname} 不是 PNG"
    assert len(data) > 5_000, f"{fname} 太小（{len(data)} 字节），疑似空白图"
    # 用 PIL 校验尺寸
    from PIL import Image
    w, h = Image.open(p).size
    assert w >= 400 and h >= 200, f"{fname} 尺寸异常：{w}x{h}"


# --------------------------------------------------------------------------- #
# 三、实验配置：protocol 字段必须指向真实存在的协议文件
# --------------------------------------------------------------------------- #
EXPERIMENT_CFGS = sorted(p for p in (CONFIGS / "experiments").rglob("*.yaml"))
PROTOCOL_NAMES = {p.stem for p in PROTOCOLS.glob("*.yaml")}


@pytest.mark.parametrize("cfg", EXPERIMENT_CFGS, ids=[p.name for p in EXPERIMENT_CFGS])
def test_experiment_config_protocol_is_declared_and_valid(cfg):
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    proto = data.get("protocol")
    assert proto, f"{cfg.name} 缺 protocol 字段"
    assert proto in PROTOCOL_NAMES, f"{cfg.name} 的 protocol={proto} 不在 configs/protocols/ 里"
    assert data.get("experiment_id"), f"{cfg.name} 缺 experiment_id"
    assert data.get("task"), f"{cfg.name} 缺 task"
    # 不同用途的配置字段不同：baseline 用 seed_list，supp 用 condition=all，
    # external 用 dataset；不强求统一的 model/condition/seed。


# --------------------------------------------------------------------------- #
# 四、扫描/调参配置：能解析，且扫描配置有 variants
# --------------------------------------------------------------------------- #
SCAN_CFGS = sorted((CONFIGS / "scan").glob("*.yaml"))
TUNING_CFGS = sorted((CONFIGS / "tuning").glob("*.yaml"))


@pytest.mark.parametrize("cfg", SCAN_CFGS, ids=[p.name for p in SCAN_CFGS])
def test_scan_config_parses_and_has_variants(cfg):
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    assert "variants" in data, f"{cfg.name} 缺 variants"


@pytest.mark.parametrize("cfg", TUNING_CFGS, ids=[p.name for p in TUNING_CFGS])
def test_tuning_config_parses(cfg):
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{cfg.name} 不是合法 YAML 映射"


# --------------------------------------------------------------------------- #
# 五、数据清单：_files.json 与 manifest sha256 一致
# --------------------------------------------------------------------------- #
def test_data_manifest_consistency():
    files_json = json.loads((PROJECT_ROOT / "data" / "seu" / "_files.json").read_text(encoding="utf-8"))
    manifest = PROJECT_ROOT / "data" / "seu" / "data_manifest_sha256.csv"
    assert manifest.exists(), "data_manifest_sha256.csv 不存在"
    if isinstance(files_json, list):
        names = {str(x.get("path")) for x in files_json if x.get("path")}
    else:
        names = set(files_json.keys())
    assert names, "_files.json 解析不出文件名"
    # manifest 里的文件名应覆盖 _files.json 的文件名
    manifest_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        parts = [c.strip() for c in line.split(",")]
        if parts and parts[0]:
            manifest_names.add(parts[0])
    assert names <= manifest_names, (
        f"_files.json 里有 {len(names - manifest_names)} 个文件不在 sha256 清单里："
        f"{sorted(names - manifest_names)[:5]}")