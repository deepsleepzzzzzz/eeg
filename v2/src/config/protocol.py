from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_protocol(path: str | Path) -> dict:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)


DEFAULT_CHANNEL_INDICES = [1, 2, 3]   # 行星齿轮箱 x/y/z


def channel_indices(protocol: dict) -> list[int]:
    """0-based raw CSV columns to use."""
    return channel_indices_with_source(protocol)[0]


def channel_indices_with_source(protocol: dict) -> tuple[list[int], str]:
    """返回 (通道列表, 来源)。来源恒为 'protocol'。

    通道索引必须显式写在协议里。历史上它靠 DEFAULT_CHANNEL_INDICES 兜底，
    违反硬规则第 1 条 —— 一旦默认值被改动，所有结果会静默改变。
    现在缺了直接报错（与 noise.reference 同样的处理）。
    """
    ch = protocol.get("dataset", {}).get("channel_indices")
    if not ch:
        raise ValueError(
            "协议 %r 未显式声明 dataset.channel_indices。\n"
            "  通道索引决定用哪几路物理信号，必须写死在协议里。\n"
            "  例如行星齿轮箱 x/y/z 写：dataset.channel_indices: [1, 2, 3]"
            % (protocol.get("name"),)
        )
    return [int(c) for c in ch], "protocol"


def window_cfg(protocol: dict) -> tuple[int, int]:
    window = protocol["window"]
    return int(window["length"]), int(window["stride"])


def method_cfg_from_protocol(protocol: dict) -> dict:
    prep = protocol["preprocessing"]
    rep = protocol["representation"]
    return {
        # kind=transition（默认，符号状态转移）或 raw（原始波形，用于基线）
        "representation": rep.get("kind", "transition"),
        "attention": rep.get("attention", "none"),
        "normalize": prep["normalize"],
        "global_mean": prep.get("global_mean"),
        "global_std": prep.get("global_std"),
        "prefilter": prep.get("prefilter", "none"),
        "prefilter_k": int(prep.get("prefilter_k", 20)),
        "prefilter_fs": float(prep.get("prefilter_fs", 5120.0)),
        "prefilter_lo": float(prep.get("prefilter_lo", 300.0)),
        "prefilter_hi": float(prep.get("prefilter_hi", 2200.0)),
        "prefilter_taps": int(prep.get("prefilter_taps", 101)),
        "prefilter_wavelet": str(prep.get("prefilter_wavelet", "db4")),
        "prefilter_level": int(prep.get("prefilter_level", 3)),
        "prefilter_mode": str(prep.get("prefilter_mode", "soft")),
        "symbol_bins": int(prep["bins"]),
        "bin_scope": prep["bin_scope"],
        "bin_edges_mode": prep["bin_edges"],
        "symbol_strategy": "quantile",
        "transition_steps": [int(x) for x in rep["lags"]],
        "transition_weight": rep.get("transition_weight", "probability"),
        "transition_multichannel": rep.get("transition_multichannel", "stack"),
        "include_self_transition": bool(rep.get("include_self_transition", True)),
        "state_stats": rep.get("state_stats", "none"),
        # 混合表示：在转移张量后追加每通道 raw_fuse 个原始采样点的 6x6 区块。
        # None = 纯转移张量（默认，主协议不加此键，故不影响已有结果）
        "raw_fuse": rep.get("raw_fuse", None),
        # 幅值补充（补回等频符号化丢掉的能量信息）：None=关闭 / "rms" / "state"
        # 与 raw_fuse 同理：主协议不写此键 => 现有结果不受影响；实验变体在 scan yaml 里显式声明。
        "energy_stats": rep.get("energy_stats", None),
    }


def model_cfg_from_protocol(protocol: dict, model_name: str) -> dict:
    models = protocol["models"]
    name = str(model_name).lower()
    if name == "gcn":
        g = models["gcn"]
        return {
            "arch": "gcn",
            "gcn_layout": g.get("layout", "relational"),
            "gcn_steps": int(g.get("gcn_steps", 3)),
            "gcn_hidden": int(g.get("hidden", 32)),
            "gcn_layers": int(g.get("layers", 2)),
            "gcn_pool": g.get("gcn_pool", "flat"),
            "gcn_attention": bool(g.get("gcn_attention", False)),
        }
    if name != "cnn":
        # 任意基线模型（raw_cnn / wdcnn / resnet1d 等）：配置直接写在 models.<name> 下
        extra = models.get(name)
        if extra is None:
            raise KeyError(f"协议里没有模型配置: {name}（已有 {sorted(models)}）")
        cfg = dict(extra)
        cfg.setdefault("arch", name)
        return cfg
    c = models["cnn"]
    return {
        "arch": "cnn",
        "conv_channels": [int(x) for x in c["conv_channels"]],
        "kernel_size": int(c["kernel_size"]),
        "pool_size": int(c["pool_size"]),
        "fc_hidden": [int(x) for x in c.get("fc_hidden", [])],
    }


def train_cfg_from_protocol(protocol: dict, device: str | None = None) -> dict:
    tr = protocol["training"]
    return {
        "device": device or "cuda",
        "require_gpu": False,
        "deterministic": bool(tr.get("deterministic", True)),
        "num_workers": int(tr.get("num_workers", 0)),
        "batch_size": int(tr["batch_size"]),
        "epochs": int(tr["epochs"]),
        "lr": float(tr["lr"]),
        "weight_decay": float(tr["weight_decay"]),
        "dropout": float(tr["dropout"]),
        "optimizer": str(tr.get("optimizer", "adam")),
        "label_smoothing": float(tr.get("label_smoothing", 0.0)),
        "stop_on": str(tr.get("stop_on", "val_acc")),
        "early_stop_patience": int(tr["early_stop_patience"]),
        "lr_scheduler": dict(tr.get("lr_scheduler") or {}),
        "seed": 42,
    }


def build_train_omega(protocol: dict, model_name: str, device: str | None = None):
    return OmegaConf.create({
        "method": method_cfg_from_protocol(protocol),
        "model": model_cfg_from_protocol(protocol, model_name),
        "train": train_cfg_from_protocol(protocol, device=device),
    })


DEFAULT_PATHS = {
    "raw_root": "data/seu",
    "artifact_root": "artifacts",
    "sealed_test_root": "artifacts/sealed_test",
}


def resolve_project_path(protocol: dict, key: str) -> Path:
    paths = protocol.get("paths", DEFAULT_PATHS)
    value = paths.get(key, DEFAULT_PATHS[key])
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def raw_root(protocol: dict) -> Path:
    return resolve_project_path(protocol, "raw_root")


def artifact_root(protocol: dict) -> Path:
    return resolve_project_path(protocol, "artifact_root")


def sealed_test_root(protocol: dict) -> Path:
    return resolve_project_path(protocol, "sealed_test_root")


def dataset_cache_id(protocol: dict) -> str:
    """Directory name of the data cache.

    The cache depends only on the data-related sections (dataset/split/window/
    preprocessing/representation), not on the training hyper-parameters.  A new
    training-protocol version therefore declares ``dataset_cache_id`` and keeps
    reusing the exact same cache instead of silently rebuilding a different one.
    """
    return str(protocol.get("dataset_cache_id") or protocol["name"])


def dataset_dir(protocol: dict, task: str, condition: str) -> Path:
    return artifact_root(protocol) / dataset_cache_id(protocol) / "dataset" / task / condition


def sealed_test_dir(protocol: dict, task: str, condition: str) -> Path:
    return sealed_test_root(protocol) / dataset_cache_id(protocol) / task / condition
