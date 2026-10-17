# Experiment configs

Each experiment config must reference a protocol in `configs/protocols/`.
Do not put split/window/preprocessing rules in experiment configs unless the protocol explicitly marks them as variable.

Required fields:
- experiment_id
- protocol
- comparison_group
- task
- condition
- model
- seed or seed_list
