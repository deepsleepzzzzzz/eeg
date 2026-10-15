"""Verify review fixes in isolation, or reproduce defects in the original EEG.zip.

Default: test checkout fixes. --target package tests the repair overlay against
the archived dependencies. --mode reproduce --target package tests the old bugs.
Real recordings, weights and test arrays are never extracted or evaluated.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

REPAIR_FILES = (
    "data/seu_cache.py", "scripts/experiments/build_seu_dataset.py",
    "scripts/experiments/train_seu.py", "scripts/experiments/plot_seu_val.py",
    "scripts/experiments/seu_figures.py",
)


def run_reproduction_worker(archive: Path) -> int:
    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    from omegaconf import OmegaConf
    from scripts.experiments import build_seu_dataset as builder
    from scripts.experiments.seu_figures import paper_stem

    if not Path(builder.__file__).resolve().is_relative_to(Path.cwd().resolve()):
        raise RuntimeError("Tests must import the extracted package, not the checkout")

    class PackageReviewTests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="case_", dir=Path.cwd())
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.cfg = OmegaConf.load("config/seu_smoke_bearing_cnn.yaml")
            self.cfg.data.raw_root = str(self.root / "raw")
            self.cfg.data.output_root = str(self.root / "cache")
            self.cfg.data.epoch_samples = 80
            self.cfg.data.window_stride = 80
            self.cfg.data.max_windows_per_class = None
            folder = self.root / "raw" / "gearbox" / "bearingset"
            folder.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for name in ("health", "ball", "inner", "outer", "comb"):
                np.savetxt(folder / f"{name}_20_0.csv", rng.normal(size=(8000, 8)), delimiter=",")

        def build(self, force=False):
            with contextlib.redirect_stdout(io.StringIO()):
                return builder.build_condition("bearing", "20_0", self.cfg, force)

        def metadata(self):
            return json.loads((Path(self.cfg.data.output_root) / "bearing/20_0/meta.json").read_text(encoding="utf-8"))

        def test_01_changed_split_reuses_stale_cache(self):
            """Changing val=.15 to .20 leaves the old cache; --force is a control."""
            original = self.build()
            self.assertEqual(original["val_ratio"], 0.15)
            self.assertEqual(original["arrays"]["val"]["n"], 75)
            self.cfg.data.val_ratio = 0.20
            self.cfg.data.test_ratio = 0.10
            result = self.build()
            stale = self.metadata()
            self.assertTrue(result["skipped"])
            self.assertEqual(stale["val_ratio"], 0.15)
            self.assertEqual(stale["arrays"]["val"]["n"], 75)
            rebuilt = self.build(force=True)
            self.assertEqual(rebuilt["val_ratio"], 0.20)
            self.assertEqual(rebuilt["arrays"]["val"]["n"], 100)
            print("EVIDENCE cache: changed val_ratio=.20 still has 75 windows (.15); --force produces 100", flush=True)

        def test_02_different_sample_sets_share_fingerprint(self):
            """Synthetic rebuilds and the saved full/smoke sets both collide."""
            full = self.build()
            self.cfg.data.output_root = str(self.root / "limited_cache")
            self.cfg.data.max_windows_per_class = 2
            limited = self.build()
            self.assertEqual(full["arrays"]["train"]["n"], 350)
            self.assertEqual(limited["arrays"]["train"]["n"], 10)
            self.assertEqual(full["split_sample_id_fingerprint"], limited["split_sample_id_fingerprint"])
            with zipfile.ZipFile(archive) as z:
                saved_full = json.loads(z.read("datas/experiments_seu/representations/bearing/20_0/meta.json"))
                saved_smoke = json.loads(z.read("datas/experiments_seu/smoke/bearing/20_0/meta.json"))
            self.assertNotEqual(saved_full["arrays"]["train"]["n"], saved_smoke["arrays"]["train"]["n"])
            self.assertEqual(saved_full["split_sample_id_fingerprint"], saved_smoke["split_sample_id_fingerprint"])
            print("EVIDENCE fingerprint: synthetic 350 vs 10 and packaged "
                  f"{saved_full['arrays']['train']['n']} vs {saved_smoke['arrays']['train']['n']} share fingerprints", flush=True)

        def test_03_attention_figures_overwrite_baseline(self):
            """Use package filenames and real copy2 writes to prove replacement."""
            paper_dir = self.root / "paper"
            paper_dir.mkdir()
            pairs = (("transition_none_cnn", "transition_self_cnn"),
                     ("transition_gcn", "transition_gcn_attn"))
            for baseline, attention in pairs:
                for figure in ("confusion", "tsne"):
                    with self.subTest(baseline=baseline, figure=figure):
                        base_stem = paper_stem("bearing", "20_0", baseline)
                        attn_stem = paper_stem("bearing", "20_0", attention)
                        self.assertEqual(base_stem, attn_stem)
                        source = self.root / f"val_{figure}.png"
                        target = paper_dir / f"{base_stem}_{figure}.png"
                        source.write_bytes(b"baseline figure marker")
                        shutil.copy2(source, target)
                        self.assertEqual(target.read_bytes(), b"baseline figure marker")
                        source.write_bytes(b"attention figure marker")
                        shutil.copy2(source, paper_dir / f"{attn_stem}_{figure}.png")
                        self.assertEqual(target.read_bytes(), b"attention figure marker")
            self.assertNotEqual(paper_stem("bearing", "20_0", pairs[0][0]), paper_stem("bearing", "20_0", pairs[1][0]))
            self.assertNotEqual(paper_stem("bearing", "20_0", pairs[0][0]), paper_stem("bearing", "30_2", pairs[0][0]))
            print("EVIDENCE figures: CNN/self and GCN/attention replace baseline confusion and t-SNE files", flush=True)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PackageReviewTests)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    print("Interpretation: OK means all reviewed defects reproduced; no production fix was applied.")
    return 0 if result.wasSuccessful() else 1


def run_regression_worker() -> int:
    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from data.seu_cache import validate_cache
    from scripts.experiments import build_seu_dataset as builder
    from scripts.experiments import train_seu as training
    from scripts.experiments import plot_seu_val as plotting
    from scripts.experiments.seu_figures import paper_stem

    for module in (builder, training, plotting):
        if not Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve()):
            raise RuntimeError("Unexpected checkout imports in isolated test")
    torch.set_num_threads(1)

    class RegressionTests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="case_", dir=Path.cwd())
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.cfg = OmegaConf.load("config/seu_smoke_bearing_cnn.yaml")
            self.cfg.data.raw_root = str(self.root / "raw")
            self.cfg.data.output_root = str(self.root / "cache")
            self.cfg.data.epoch_samples = 80
            self.cfg.data.window_stride = 80
            self.cfg.data.max_windows_per_class = None
            self.cfg.train.device = "cpu"
            self.cfg.train.epochs = 1
            self.cfg.train.output_dir = str(self.root / "runs")
            self.cache_dir = Path(self.cfg.data.output_root) / "bearing/20_0"
            self.raw = self.root / "raw/gearbox/bearingset"
            self.raw.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for name in ("health", "ball", "inner", "outer", "comb"):
                np.savetxt(self.raw / f"{name}_20_0.csv", rng.normal(size=(8000, 8)), delimiter=",")

        def build(self, force=False):
            with contextlib.redirect_stdout(io.StringIO()):
                return builder.build_condition("bearing", "20_0", self.cfg, force)

        def validate(self, meta):
            validate_cache(meta, self.cfg, "bearing", "20_0", self.cache_dir)

        def test_01_configuration_mismatch_is_rejected(self):
            old = self.build()
            self.assertTrue(self.build()["skipped"])
            self.cfg.data.val_ratio = .20
            self.cfg.data.test_ratio = .10
            with self.assertRaisesRegex(RuntimeError, "configuration.*--force"):
                self.build()
            # Training must reject before reading any arrays or creating a model.
            with patch.object(training, "load_array_pair", side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(RuntimeError, "configuration"):
                    training.train_one("bearing", "20_0", 42, self.cfg, False, False, False)
            new = self.build(force=True)
            self.assertEqual(old["arrays"]["val"]["n"], 75)
            self.assertEqual(new["arrays"]["val"]["n"], 100)
            self.assertNotEqual(old["split_sample_id_fingerprint"], new["split_sample_id_fingerprint"])
            self.validate(new)

        def test_02_fingerprint_tracks_actual_samples_and_source(self):
            full = self.build()
            repeated = self.build(force=True)
            self.assertEqual(full["split_sample_id_fingerprint"], repeated["split_sample_id_fingerprint"])
            self.cfg.data.max_windows_per_class = 2
            limited = self.build(force=True)
            self.assertEqual(limited["arrays"]["train"]["n"], 10)
            self.assertNotEqual(full["split_sample_id_fingerprint"], limited["split_sample_id_fingerprint"])
            path = self.raw / "health_20_0.csv"
            content = path.read_bytes()
            # Change data without changing file length or row count.
            offset = next(i for i, value in enumerate(content) if 48 <= value <= 56)
            path.write_bytes(content[:offset] + bytes([content[offset] + 1]) + content[offset + 1:])
            with self.assertRaisesRegex(RuntimeError, "raw source files changed"):
                self.build()
            changed = self.build(force=True)
            self.assertNotEqual(limited["split_sample_id_fingerprint"], changed["split_sample_id_fingerprint"])

        def test_03_figure_names_keep_methods_and_seeds_distinct(self):
            methods = ("transition_none_cnn", "transition_self_cnn", "transition_gcn", "transition_gcn_attn")
            paths = set()
            for method in methods:
                for seed in (42, 43):
                    for kind in ("confusion", "tsne"):
                        target = self.root / f"{paper_stem('bearing', '20_0', method, seed)}_{kind}.png"
                        self.assertNotIn(target, paths)
                        paths.add(target)
                        marker = f"{method}:{seed}:{kind}".encode()
                        target.write_bytes(marker)
            self.assertEqual(len(paths), 16)
            for method in methods:
                for seed in (42, 43):
                    for kind in ("confusion", "tsne"):
                        target = self.root / f"{paper_stem('bearing', '20_0', method, seed)}_{kind}.png"
                        self.assertEqual(target.read_bytes(), f"{method}:{seed}:{kind}".encode())
            self.assertEqual(paper_stem("bearing", "20_0", methods[0]), "bearing_cnn_20_0")

        def test_04_representation_changes_invalidate_but_classifier_changes_do_not(self):
            meta = self.build()
            self.cfg.method.attention = "self"
            self.cfg.model.arch = "gcn"
            self.cfg.train.lr = .001
            self.validate(meta)
            self.assertTrue(self.build()["skipped"])
            original_root = self.cfg.data.raw_root
            self.cfg.data.raw_root = str(self.root / "other_raw")
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                self.validate(meta)
            self.cfg.data.raw_root = original_root
            for key, value in (("transition_steps", [1, 3, 5]), ("normalize", "zscore"), ("symbol_bins", 8)):
                with self.subTest(key=key):
                    original = self.cfg.method[key]
                    self.cfg.method[key] = value
                    with self.assertRaisesRegex(RuntimeError, "configuration"):
                        self.validate(meta)
                    self.cfg.method[key] = original

        def test_05_legacy_missing_and_corrupt_arrays_are_rejected(self):
            meta = self.build()
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                self.validate({"arrays": meta["arrays"]})
            path = self.cache_dir / "X_val.npy"
            saved = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                self.validate(meta)
            path.write_bytes(saved[:-1] + bytes([saved[-1] ^ 1]))
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                self.validate(meta)
            path.write_bytes(saved)
            # Sealed-test validation must not read test data; builder checks all splits.
            (self.cache_dir / "X_test.npy").unlink()
            self.validate(meta)
            with self.assertRaisesRegex(RuntimeError, "X_test.npy"):
                self.build()

        def test_06_failed_rebuild_has_no_valid_completion_marker(self):
            self.build()
            with patch.object(builder, "encode_windows", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    self.build(force=True)
            self.assertFalse((self.cache_dir / "meta.json").exists())

        def test_07_training_and_plotting_use_identity_and_unique_destinations(self):
            self.build()
            def fake_figures(model, loader, device, names, confusion, out_dir, title, seed=42):
                files = {}
                for kind in ("confusion", "tsne"):
                    path = Path(out_dir) / f"val_{kind}.png"
                    path.write_bytes(title.encode())
                    files[f"val_{kind}"] = path
                return files
            for attention in ("none", "self"):
                self.cfg.method.attention = attention
                self.cfg.train.output_dir = str(self.root / "runs" / attention)
                with patch.object(training, "ROOT", self.root), patch.object(training, "save_val_figures", side_effect=fake_figures), contextlib.redirect_stdout(io.StringIO()):
                    result = training.train_one("bearing", "20_0", 42, self.cfg, False, False)
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(result["cache_fingerprint"], json.loads((self.cache_dir / "meta.json").read_text())["cache_fingerprint"])
            figures = self.root / "doc/figures/seu"
            self.assertEqual(len(list(figures.glob("*.png"))), 4)
            config_path = self.root / "plot.yaml"
            OmegaConf.save(self.cfg, config_path)
            with patch.object(plotting, "ROOT", self.root), patch.object(plotting, "RUNS", ((str(config_path), "bearing", "20_0"),)), patch.object(plotting, "save_val_figures", side_effect=fake_figures), contextlib.redirect_stdout(io.StringIO()):
                plotting.main()
                # Change raw data and rebuild with the same configuration: old weights must fail.
                with (self.raw / "health_20_0.csv").open("a") as stream:
                    stream.write("0,0,0,0,0,0,0,0\n")
                self.build(force=True)
                with self.assertRaisesRegex(RuntimeError, "checkpoint/metrics cache identity"):
                    plotting.main()

        def test_08_invalid_ratios_and_overlap_metadata(self):
            self.cfg.data.test_ratio = .5
            with self.assertRaisesRegex(ValueError, "sum to one"):
                self.build()
            self.cfg.data.test_ratio = .15
            self.cfg.data.window_stride = 40
            self.assertTrue(self.build()["overlap"])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RegressionTests)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    checkout = Path(__file__).resolve().parents[2]
    default_archive = checkout / "EEG_review_snapshot.zip"
    if not default_archive.exists():
        default_archive = checkout / "EEG.zip"
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target", choices=("checkout", "package"), default="checkout")
    parser.add_argument("--mode", choices=("verify", "reproduce"), default="verify")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if args.mode == "reproduce" and args.target != "package":
        parser.error("--mode reproduce requires --target package")
    if args.worker:
        return run_reproduction_worker(archive) if args.mode == "reproduce" else run_regression_worker()
    # Allow only project source/config roots, excluding archived Git data,
    # embedded old projects, data arrays and checkpoints.
    with tempfile.TemporaryDirectory(prefix="eeg_package_review_") as temp:
        root = Path(temp).resolve()
        if args.target == "checkout":
            for folder in ("models", "representation", "scripts", "data", "experiments", "utils", "config"):
                for path in (checkout / folder).rglob("*"):
                    if path.suffix not in (".py", ".yaml"):
                        continue
                    destination = root / path.relative_to(checkout)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
        else:
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    name = item.filename
                    source = name.startswith(("models/", "representation/", "scripts/", "data/", "experiments/", "utils/")) and name.endswith(".py")
                    config = name.startswith("config/") and name.endswith(".yaml")
                    if not (source or config):
                        continue
                    destination = (root / name).resolve()
                    if not destination.is_relative_to(root):
                        raise ValueError(f"Unsafe archive path: {name}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(z.read(item))
                for name in ("scripts/experiments/build_seu_dataset.py", "scripts/experiments/seu_figures.py"):
                    print(f"SOURCE {name} sha256={hashlib.sha256(z.read(name)).hexdigest()}", flush=True)
            if args.mode == "verify":
                for name in REPAIR_FILES:
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = (checkout / name).read_text(encoding="utf-8")
                    if name.endswith("build_seu_dataset.py"):
                        # Keep the archive's equal-width fitting algorithm.
                        source = source.replace("import fit_train_edges, uses_train_edges", "import fit_train_amplitude_edges as fit_train_edges, uses_train_edges")
                        source = source.replace('method.get("bin_edges_mode", "quantile")', 'method.get("bin_edges_mode", "uniform")')
                    destination.write_text(source, encoding="utf-8")
                print("Testing repair overlay on package dependencies (original equal-width bin fitting).", flush=True)
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--archive", str(archive),
             "--target", args.target, "--mode", args.mode],
            cwd=root,
            check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
