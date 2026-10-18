"""Draw val confusion and t-SNE from saved SEU checkpoints. Does not train or open test."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_runtime import get_device, load_array_pair, make_loader, method_name, resolve_path
from data.seu_cache import validate_cache
from models.cnn import build_model
from scripts.experiments.seu_figures import paper_stem, paper_title, save_val_figures
from utils.io_utils import ensure_dir, load_json

RUNS = (
    ("config/seu_bearing_cnn.yaml", "bearing", "20_0"),
    ("config/seu_bearing_cnn.yaml", "bearing", "30_2"),
    ("config/seu_bearing_gcn.yaml", "bearing", "20_0"),
    ("config/seu_bearing_gcn.yaml", "bearing", "30_2"),
    ("config/seu_gear_cnn.yaml", "gear", "20_0"),
    ("config/seu_gear_cnn.yaml", "gear", "30_2"),
    ("config/seu_gear_gcn.yaml", "gear", "20_0"),
    ("config/seu_gear_gcn.yaml", "gear", "30_2"),
)


def main() -> None:
    paper_dir = Path(ensure_dir(str(ROOT / "doc" / "figures" / "seu")))
    seed = 42
    for config_path, task, condition in RUNS:
        cfg = OmegaConf.load(resolve_path(config_path))
        data_dir = resolve_path(cfg.data.output_root) / task / condition
        run_dir = resolve_path(cfg.train.output_dir) / task / condition / f"seed_{seed}"
        metrics = load_json(str(run_dir / "metrics.json"))
        meta = load_json(str(data_dir / "meta.json"))
        validate_cache(meta, cfg, task, condition, data_dir, splits=("val",))
        names = list(metrics["class_names"])
        X_val, y_val = load_array_pair(data_dir, "val")
        device = get_device(cfg)
        model = build_model(
            len(names),
            cfg,
            in_channels=int(X_val.shape[1]),
            image_size=int(X_val.shape[-1]),
        ).to(device)
        ckpt = torch.load(
            run_dir / str(cfg.train.get("checkpoint_name", "best.pt")),
            map_location=device,
            weights_only=False,
        )
        if ckpt.get("cache_fingerprint") != meta["cache_fingerprint"] or metrics.get("cache_fingerprint") != meta["cache_fingerprint"]:
            raise RuntimeError("checkpoint/metrics cache identity differs or is missing; regenerate training results before plotting")
        model.load_state_dict(ckpt["model"])
        loader = make_loader(X_val, y_val, int(cfg.train.batch_size), 0, False)
        title = paper_title(task, condition, method_name(cfg))
        figs = save_val_figures(
            model,
            loader,
            device,
            names,
            metrics["val_confusion_matrix"],
            run_dir,
            title,
            seed=seed,
        )
        stem = paper_stem(task, condition, method_name(cfg), seed=seed)
        shutil.copy2(figs["val_confusion"], paper_dir / f"{stem}_confusion.png")
        shutil.copy2(figs["val_tsne"], paper_dir / f"{stem}_tsne.png")
        print(f"wrote {stem}", flush=True)


if __name__ == "__main__":
    main()
