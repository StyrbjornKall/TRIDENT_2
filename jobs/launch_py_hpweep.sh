#!/usr/bin/env bash
source .env

# Local mode (default): expands the sweep YAML into a grid and runs every
# combination sequentially, no W&B account or manually-created sweep needed.
uv run /home/skall/TRIDENT_2/scripts/kfold_cross_validation_sweep.py \
    --config_file_path ./wandb_configs/cross_validations/interpolation_100ep.yaml

# Online mode: run against a sweep you already created on the W&B website.
# uv run /home/skall/TRIDENT_2/scripts/kfold_cross_validation_sweep.py \
#     --use_wandb \
#     --wandb_entity_name StyrbjornKall \
#     --wandb_project_name tox_across_species_hpsweep \
#     --wandb_sweep_id ar7xu6mo