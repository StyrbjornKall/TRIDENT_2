#!/usr/bin/env bash
source .env

# Local mode (default): evaluate a run from its locally saved sweep_config.json
# (written next to the run's results/models under save_dir), no W&B account needed.
for config_path in run_outputs/sweep_local/sweep_config.json;
    do
    uv run ./scripts/run_posthoc_evaluation.py \
        --config_file_path "$config_path"
done

# Online mode: fetch the run config from a real W&B account instead.
# for id in fz52v4b8;
#     do
#     uv run ./scripts/run_posthoc_evaluation.py \
#     --wandb_entity_name $WANDB_ENTITY_NAME \
#     --wandb_project_name $WANDB_PROJECT_NAME \
#         --wandb_run_id $id \
#         --resume_run True
# done