#!/usr/bin/env bash
source .env

# Local mode (default): no W&B account needed, nothing is sent over the network.
uv run ./scripts/run_experiment.py \
    --config_file_path ./wandb_configs/experiments/experiment_example_local.json

# Online mode: log this run to a real W&B account instead.
# uv run ./scripts/run_experiment.py \
#     --use_wandb \
#     --wandb_entity_name $WANDB_ENTITY_NAME \
#     --wandb_project_name $WANDB_PROJECT_NAME \
#     --config_file_path ./wandb_configs/experiments/experiment_example_wandb.json