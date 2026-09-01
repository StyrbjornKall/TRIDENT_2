#!/usr/bin/env bash
source .env

# Local mode (default): expands the sweep YAML into a grid and runs every
# combination (e.g. all fold_id values) sequentially, no W&B account needed.
uv run ./scripts/kfold_cross_validation_sweep.py \
    --config_file_path ./wandb_configs/cross_validations/unknown_chemicals_butina02_100ep.yaml

# Online mode: run against a sweep you already created on the W&B website.
# uv run ./scripts/kfold_cross_validation_sweep.py \
#     --use_wandb \
#     --wandb_entity_name $WANDB_ENTITY_NAME \
#     --wandb_project_name $WANDB_PROJECT_NAME \
#     --wandb_sweep_id 8el09yfq \ # Fill in this after agent has been launched
#     --count 0