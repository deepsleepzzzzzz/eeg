"""Reproduce the three EEG.zip review findings without changing production data.

Passing tests mean the reviewed defects are reproduced, not fixed. The package's
Python/config files run in an isolated temporary directory; real recordings,
weights and test arrays are never extracted or evaluated.
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


def run_worker(archive: Path) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[2] / "EEG.zip")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    archive = args.archive.resolve()
    if args.worker:
        return run_worker(archive)
    # Allow only project source/config roots, excluding archived Git data,
    # embedded old projects, data arrays and checkpoints.
    with tempfile.TemporaryDirectory(prefix="eeg_package_review_") as temp:
        root = Path(temp).resolve()
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
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--archive", str(archive)],
            cwd=root,
            check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
