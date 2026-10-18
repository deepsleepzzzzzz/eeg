"""Review regressions: missing official evidence and machine-specific entry points."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import audit_main
import audit_mutation
from audit_provenance import audit_official_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("manifest", [None, [], [{"path": "unrelated.csv", "sha": "abc"}],
                                     [{"path": "gearbox/gearset/Health_20_0.csv", "sha": "wrong"}]])
def test_missing_or_mismatched_official_evidence_fails(tmp_path, manifest):
    if manifest is not None:
        (tmp_path / "_files.json").write_text(json.dumps(manifest), encoding="utf-8")
    sources = {"Health": {"file": "Health_20_0.csv", "git_blob_sha1": "abc"}}
    assert not audit_official_files(tmp_path, "gear", sources)["pass"]


def test_matching_official_hash_passes(tmp_path):
    (tmp_path / "_files.json").write_text(json.dumps([
        {"path": "gearbox/gearset/Health_20_0.csv", "sha": "abc"}]), encoding="utf-8")
    sources = {"Health": {"file": "Health_20_0.csv", "git_blob_sha1": "abc"}}
    assert audit_official_files(tmp_path, "gear", sources)["pass"]


@pytest.mark.parametrize("filename", ["_files.json", "data_manifest_sha256.csv"])
def test_official_manifests_are_not_ignored(tmp_path, filename):
    if shutil.which("git") is None:
        pytest.skip("Git is required to check ignore rules")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    (tmp_path / "data/seu").mkdir(parents=True)
    (tmp_path / "data/seu" / filename).touch()
    result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", "--", f"data/seu/{filename}"], cwd=tmp_path)
    assert result.returncode == 1
    assert subprocess.run(["git", "check-ignore", "--quiet", "--", "data/seu/gearbox/raw.csv"], cwd=tmp_path).returncode == 0


def test_verification_and_mutation_use_the_invoking_interpreter():
    spec = importlib.util.spec_from_file_location("review_verify_all", ROOT / "verify_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.PY).samefile(sys.executable)
    assert Path(audit_mutation.PY).samefile(sys.executable)
    assert all(Path(command[0]).samefile(sys.executable) for _, command, _ in module.CHECKS)


@pytest.mark.parametrize("recorded", [r"D:\original\artifacts\protocol\run\metrics.json",
                                      "/original/artifacts/protocol/run/metrics.json"])
def test_historical_registry_paths_resolve_inside_this_checkout(tmp_path, monkeypatch, recorded):
    directory = tmp_path / "artifacts/protocol/run"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit_main, "PROJECT_ROOT", tmp_path)
    assert audit_main.resolve_run_dir(recorded) == directory
