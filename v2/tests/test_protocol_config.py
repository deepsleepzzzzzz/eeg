"""协议加载测试：yaml 里写的值必须真的被读出来（不许被代码默认值兜底）。

协议文件**自动解析 status: main**（硬规则第 2 条）。本测试同时把当前主协议的
关键参数**钉死**：换协议时这里会失败，逼你确认"改协议是有意的"，而不是让它悄悄生效。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from config.protocol import (
    load_protocol,
    method_cfg_from_protocol,
    model_cfg_from_protocol,
    train_cfg_from_protocol,
    window_cfg,
)
from data.build_dataset import split_intervals

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _main_protocol_path() -> Path:
    for f in sorted((PROJECT_ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            return f
    raise RuntimeError("找不到 status=main 的主协议")


PROTOCOL = _main_protocol_path()


def test_main_protocol_is_v5():
    protocol = load_protocol(PROTOCOL)
    assert protocol["status"] == "main"
    assert protocol["name"] == "main_70_15_15_v5"


def test_protocol_values_are_loaded_from_yaml():
    protocol = load_protocol(PROTOCOL)
    # 主协议 v5：窗 2048 / 步 1024、滞后 1-10、时间有序 70/15/15
    assert window_cfg(protocol) == (2048, 1024)
    assert split_intervals(protocol) == {
        "train": [0, 733992],
        "val": [733992, 891276],
        "test": [891276, 1048560],
    }
    method = method_cfg_from_protocol(protocol)
    assert method["symbol_bins"] == 6
    assert method["transition_steps"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert method["bin_scope"] == "fit_on_train_only"
    assert method["bin_edges_mode"] == "equal_frequency_quantile"
    cnn = model_cfg_from_protocol(protocol, "cnn")
    gcn = model_cfg_from_protocol(protocol, "gcn")
    assert cnn["conv_channels"] == [32, 64]
    assert gcn["gcn_layers"] == 2 and gcn["gcn_hidden"] == 32
    # GCN 的 relational 布局要求 gcn_steps == 滞后数
    assert int(protocol["models"]["gcn"]["gcn_steps"]) == len(method["transition_steps"])
    train = train_cfg_from_protocol(protocol)
    assert train["lr"] == 0.0005
    assert train["dropout"] == 0.2
    assert train["batch_size"] == 32
    assert train["epochs"] == 200
    assert train["early_stop_patience"] == 30


def test_loader_follows_protocol_changes():
    protocol = load_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["preprocessing"]["bins"] = 7
    changed["training"]["lr"] = 0.0001
    assert method_cfg_from_protocol(changed)["symbol_bins"] == 7
    assert train_cfg_from_protocol(changed)["lr"] == 0.0001
