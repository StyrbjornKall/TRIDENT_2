#!/usr/bin/env bash
source .env

# Local mode (default): no W&B account needed, nothing is sent over the network.
uv run ./scripts/train_final_model.py \
    --config_file_path "./wandb_configs/final_model/final_model_100ep.json"

# Online mode: log this run to a real W&B account instead.
# uv run ./scripts/train_final_model.py \
#     --use_wandb \
#     --wandb_entity_name StyrbjornKall \
#     --wandb_project_name tox_across_species \
#     --config_file_path "./wandb_configs/final_model/final_model_100ep.json"