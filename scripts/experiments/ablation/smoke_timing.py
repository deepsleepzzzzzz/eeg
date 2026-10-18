"""Scoping smoke: build one SEU condition with given bins/lags into outputs/ablation_scoping, time build + train.
Writes nothing outside outputs/ablation_scoping. Test split never evaluated."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments.build_seu_dataset import build_condition
from scripts.experiments.train_seu import train_one

OUT = ROOT / "outputs" / "ablation_scoping"
BEST = {("bearing","cnn"):(5e-4,0.1),("bearing","gcn"):(5e-4,0.3),("gear","cnn"):(5e-4,0.1),("gear","gcn"):(1e-3,0.3)}

def make_cfg(task, arch, bins, lags):
    cfg = OmegaConf.load(ROOT / "config" / f"seu_{task}_{arch}.yaml")
    OmegaConf.set_struct(cfg, False)
    tag = f"b{bins}_L{''.join(map(str,lags))}"
    cfg.method.symbol_bins = int(bins)
    cfg.method.transition_steps = list(lags)
    if arch == "gcn":
        cfg.model.gcn_steps = len(lags)
    cfg.data.output_root = str(OUT / "representations" / tag)
    lr, dr = BEST[(task, arch)]
    cfg.train.lr = lr; cfg.train.dropout = dr
    cfg.train.output_dir = str(OUT / "runs" / tag / arch)
    cfg.train.evaluate_test = False
    return cfg, tag

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bearing"); p.add_argument("--condition", default="20_0")
    p.add_argument("--bins", type=int, default=6); p.add_argument("--lags", default="1,2,3")
    p.add_argument("--archs", default="cnn,gcn"); p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    lags = [int(x) for x in a.lags.split(",")]
    timing = {"task": a.task, "condition": a.condition, "bins": a.bins, "lags": lags}
    cfg, tag = make_cfg(a.task, "cnn", a.bins, lags)
    t = time.perf_counter(); build_condition(a.task, a.condition, cfg, force=False)
    timing["build_s"] = time.perf_counter() - t
    if not a.build_only:
        for arch in a.archs.split(","):
            cfg, _ = make_cfg(a.task, arch, a.bins, lags)
            if a.epochs: cfg.train.epochs = a.epochs
            t = time.perf_counter()
            r = train_one(a.task, a.condition, 42, cfg, evaluate_test=False, confirm_test=False, write_paper_figures=False)
            dt = time.perf_counter() - t
            timing[arch] = {"train_s": dt, "epochs_ran": r["epochs_ran"], "s_per_epoch": dt / r["epochs_ran"],
                            "best_epoch": r["best_epoch"], "best_val_acc": r["best_val_acc"], "val_macro_f1": r["val_macro_f1"],
                            "input_shape": r["input_shape"]}
    out = OUT / "timing"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.task}_{a.condition}_{tag}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("TIMING", json.dumps(timing))

if __name__ == "__main__":
    main()
