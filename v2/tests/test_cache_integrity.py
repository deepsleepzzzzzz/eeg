"""Regression tests for fresh clones, stale caches and interrupted rebuilds.

Only temporary synthetic train/validation data is generated; no sealed test is read.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from _protocol import main_protocol
from config.protocol import dataset_dir
from data import build_dataset as bd
from data.seu_dds import class_file, class_names, task_dir


@pytest.fixture
def cache(tmp_path):
    cfg = copy.deepcopy(main_protocol())
    cfg["name"] = "synthetic_cache_review"
    cfg["paths"] = {"raw_root": str(tmp_path / "raw"),
                    "artifact_root": str(tmp_path / "artifacts"),
                    "sealed_test_root": str(tmp_path / "sealed")}
    cfg["dataset"]["file_length"] = 800
    cfg["window"].update(length=40, stride=40)
    raw = task_dir(Path(cfg["paths"]["raw_root"]), "gear")
    raw.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for name in class_names("gear"):
        np.savetxt(raw / class_file("gear", name, "20_0"), rng.normal(size=(800, 8)), delimiter=",")
    bd.build_condition(cfg, "gear", "20_0")
    return cfg, dataset_dir(cfg, "gear", "20_0")


def test_complete_cache_is_reused_without_encoding(cache, monkeypatch):
    cfg, directory = cache
    before = (directory / "meta.json").read_bytes()
    monkeypatch.setattr(bd, "encode_windows", lambda *args: pytest.fail("reused cache was rebuilt"))
    bd.build_condition(cfg, "gear", "20_0")
    assert (directory / "meta.json").read_bytes() == before


def test_fresh_clone_metadata_is_rebuilt_without_force(cache):
    cfg, directory = cache
    meta = json.loads((directory / "meta.json").read_text())
    for key in ("cache_integrity", "cache_fingerprint"):
        meta.pop(key)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for array in directory.glob("*.npy"):
        array.unlink()
    bd.build_condition(cfg, "gear", "20_0")
    assert (directory / "X_train.npy").exists()
    bd.validate_cache(cfg, "gear", "20_0")


@pytest.mark.parametrize("change", ["split", "representation", "preprocessing"])
def test_data_configuration_changes_are_rejected(cache, change):
    cfg, _ = cache
    cfg = copy.deepcopy(cfg)
    if change == "split":
        cfg["split"].update(val_ratio=.20, test_ratio=.10)
    elif change == "representation":
        cfg["representation"]["lags"] = [1]
    else:
        cfg["preprocessing"]["normalize"] = "zscore"
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        bd.build_condition(cfg, "gear", "20_0")


def test_training_settings_do_not_invalidate_representation(cache):
    cfg, _ = cache
    cfg["training"]["lr"] *= 2
    cfg["models"]["cnn"]["conv_channels"] = [8, 16]
    bd.build_condition(cfg, "gear", "20_0")


@pytest.mark.parametrize("damage", ["missing", "changed", "all_missing", "legacy"])
def test_incomplete_or_tampered_cache_is_rejected(cache, damage):
    cfg, directory = cache
    if damage == "missing":
        (directory / "X_val.npy").unlink()
    elif damage == "all_missing":
        for array in directory.glob("*.npy"):
            array.unlink()
    elif damage == "changed":
        array = np.load(directory / "X_val.npy")
        array.flat[0] += 1
        np.save(directory / "X_val.npy", array)
    else:
        meta = json.loads((directory / "meta.json").read_text())
        meta.pop("cache_integrity")
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="--force"):
        bd.build_condition(cfg, "gear", "20_0")


def test_raw_csv_changes_are_rejected(cache):
    cfg, _ = cache
    source = task_dir(Path(cfg["paths"]["raw_root"]), "gear") / class_file("gear", "Health", "20_0")
    with source.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(RuntimeError, match="raw CSV contents changed"):
        bd.build_condition(cfg, "gear", "20_0")


def test_fitted_metadata_changes_are_rejected(cache):
    cfg, directory = cache
    meta = json.loads((directory / "meta.json").read_text())
    meta["bin_edges"][0][2] += 1
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fitted metadata changed"):
        bd.validate_cache(cfg, "gear", "20_0")


def test_interrupted_rebuild_revokes_completion_and_rejects_mixed_arrays(cache, monkeypatch):
    cfg, directory = cache
    old_train = bd.file_sha256(directory / "X_train.npy")
    old_val = bd.file_sha256(directory / "X_val.npy")
    source = task_dir(Path(cfg["paths"]["raw_root"]), "gear") / class_file("gear", "Health", "20_0")
    raw = np.loadtxt(source, delimiter=",")
    raw[:, 1:4] *= 3
    np.savetxt(source, raw, delimiter=",")
    encode = bd.encode_windows
    calls = []

    def interrupt(windows, method):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("injected interruption")
        return encode(windows, method)

    with monkeypatch.context() as patch:
        patch.setattr(bd, "encode_windows", interrupt)
        with pytest.raises(RuntimeError, match="injected interruption"):
            bd.build_condition(cfg, "gear", "20_0", force=True)
    assert bd.file_sha256(directory / "X_train.npy") != old_train
    assert bd.file_sha256(directory / "X_val.npy") == old_val
    assert not (directory / "meta.json").exists()
    with pytest.raises(RuntimeError, match="completion metadata"):
        bd.build_condition(cfg, "gear", "20_0")
    bd.build_condition(cfg, "gear", "20_0", force=True)
    bd.validate_cache(cfg, "gear", "20_0")


def test_default_validation_does_not_read_sealed_test(cache, monkeypatch):
    cfg, _ = cache
    digest = bd.file_sha256

    def guarded(path):
        assert not path.name.endswith("_test.npy")
        return digest(path)

    monkeypatch.setattr(bd, "file_sha256", guarded)
    bd.validate_cache(cfg, "gear", "20_0")
    assert not Path(cfg["paths"]["sealed_test_root"]).exists()


def test_training_rejects_corrupt_cache_before_loading_arrays(cache, monkeypatch):
    from training import train

    cfg, directory = cache
    (directory / "X_val.npy").unlink()
    monkeypatch.setattr(train, "load_arrays", lambda *args: pytest.fail("training loaded invalid arrays"))
    experiment = {"task": "gear", "condition": "20_0", "model": "cnn",
                  "seed": 42, "experiment_id": "synthetic_only"}
    with pytest.raises(RuntimeError, match="X_val.npy"):
        train.train_one(experiment, cfg, False, False)


def test_training_records_cache_identity_in_all_outputs(cache, tmp_path):
    import torch
    from training import train

    cfg, directory = cache
    experiment = {"task": "gear", "condition": "20_0", "model": "cnn",
                  "seed": 42, "experiment_id": "synthetic_only"}
    output = tmp_path / "runs"
    result = train.train_one(experiment, cfg, False, False, epochs_override=1,
                             device="cpu", output_root=output, save_model=True)
    expected = json.loads((directory / "meta.json").read_text())["cache_fingerprint"]
    run = output / experiment["experiment_id"]
    assert result["cache_fingerprint"] == expected
    for filename in ("metrics.json", "manifest.json"):
        assert json.loads((run / filename).read_text())["cache_fingerprint"] == expected
    assert torch.load(run / "best_state.pt", map_location="cpu", weights_only=True)["cache_fingerprint"] == expected
    assert "test_metrics" not in result
    assert not Path(cfg["paths"]["sealed_test_root"]).exists()
