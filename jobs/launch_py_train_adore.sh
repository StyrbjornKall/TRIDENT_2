#!/usr/bin/env bash
source .env

# Local mode (default): expands the sweep YAML into a grid and runs every
# combination sequentially, no W&B account or manually-created sweep needed.
uv run ./scripts/train_adore.py \
    --config_file_path ./wandb_configs/benchmarking/posthuma_bench_adore.yaml

# Online mode: run against a sweep you already created on the W&B website.
# uv run ./scripts/train_adore.py \
#     --use_wandb \
#     --wandb_entity_name $WANDB_ENTITY_NAME \
#     --wandb_project_name $WANDB_PROJECT_NAME \
#     --wandb_sweep_id tyyebtz9