#!/usr/bin/env bash
source .env
uv run ./scripts/train_leo_bench.py \
    --config_file_path ./wandb_configs/benchmarking/leo_bench_adore.json