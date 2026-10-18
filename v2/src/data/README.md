Shared data loading, split, and windowing code. Do not duplicate per experiment.

## Cache integrity

`build_dataset.py` owns the representation-cache contract. Reuse validates all
data-affecting protocol settings, raw CSV SHA256 values, fitted metadata and
train/validation array SHA256 values. Training checks the same settings, metadata
and arrays before loading them; changing only model/training settings preserves
reuse. New checkpoints, metrics and manifests record the cache fingerprint.
Changes to representation semantics must increment `CACHE_FORMAT_VERSION`.

A fresh clone has metadata but no arrays, so the documented build command creates
the missing train/validation cache without `--force`. A partial, changed or legacy
cache that still contains arrays is rejected with an explicit `--force` rebuild
instruction. Old metrics and checkpoints are left unchanged; their historical
results are not retroactively assigned a new cache identity.

Rebuilding removes completion metadata before overwriting arrays and publishes
new metadata atomically only after every array is written. Interrupted rebuilds
cannot be reused. Test arrays are neither built nor validated by default; the
existing explicit final-evaluation flags remain required.
