#!/usr/bin/env bash
source .env
uv run ./scripts/train_posthuma_bench.py \
    --config_file_path ./wandb_configs/benchmarking/train_posthuma_model_adore.json