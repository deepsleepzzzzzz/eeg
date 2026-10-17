# SEU Fault Diagnosis v2

This is the clean restart project. The old project must not be modified or deleted.

## Main protocol

- Main protocol: `configs/protocols/main_70_15_15.yaml`
- Compatibility protocol: `configs/protocols/compat_80_20_random.yaml`
- All internal experiments must use the main protocol unless explicitly marked as compatibility or external validation.

## Rules

1. Protocol first.
2. Test set is sealed until the model is frozen.
3. Training set: fit model and preprocessing parameters.
4. Validation set: hyperparameter tuning and early stopping.
5. Test set: one-time final evaluation only.
6. Every experiment has its own config, output directory, and registry row.
7. Paper tables and figures are generated from artifacts, never edited by hand.
8. Old results go to `artifacts/golden/` and are used only as regression references.
# probe
# probe
