import os
import numpy as np
import pandas as pd
import json
import wandb
import argparse
from loguru import logger
from datetime import datetime
from trident2.logger.setup_logger import setup_logger
from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.train_utils import seed_all, run_one_fold_training, Dict2Class


def trainer(config=None, wandb_run_dir=None):

    # Set random seeds and deterministic pytorch for reproducibility
    seed_all(config.seed)

    # Load dataframe
    if config.data_dir.endswith("pkl.zip"):
        data = pd.read_pickle(config.data_dir, compression="zip")
    else:
        data = pd.read_csv(config.data_dir)
    data = preprocess_data(data, config)

    logger.success("Successfully loaded data")

    # Run one fold
    logger.info(f"\n Running fold {config.fold_id} using seed {config.seed}")

    _, _, _, _, _ = run_one_fold_training(data, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log this run to a real Weights & Biases account (requires "
        "--wandb_entity_name/--wandb_project_name and being logged in). "
        "Omit this flag to run entirely locally, no W&B account needed.",
    )
    parser.add_argument(
        "--wandb_entity_name", required=False, help="W&B entity name", default=None
    )
    parser.add_argument(
        "--wandb_project_name", required=False, help="W&B project name", default=None
    )
    parser.add_argument(
        "--wandb_run_dir",
        default="./wandb",
        help="Directory to save wandb runs",
        required=False,
    )
    parser.add_argument(
        "--config_file_path", required=True, help="Path to config file, JSON or YAML"
    )
    args = parser.parse_args()
    if args.use_wandb and (
        args.wandb_entity_name is None or args.wandb_project_name is None
    ):
        parser.error(
            "--wandb_entity_name and --wandb_project_name are required when "
            "--use_wandb is set."
        )

    setup_logger(
        log_file=f"./logs/run_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(args.config_file_path) as f:
        config = json.load(f)
    config = Dict2Class(config)

    os.makedirs(args.wandb_run_dir, exist_ok=True)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        wandb.login()
    wandb.init(
        entity=args.wandb_entity_name,
        project=args.wandb_project_name,
        notes="",
        dir=args.wandb_run_dir,
        mode="online" if args.use_wandb else "disabled",
    )
    wandb.config.update(config)

    wandb.config.update(
        {
            "wandb_run_name": wandb.run.id,
            "results_save_pth": f"{config.save_dir}/results",
            "model_save_pth": f"{config.save_dir}/models",
        }
    )

    for pth in [wandb.config.results_save_pth, wandb.config.model_save_pth]:
        os.makedirs(pth, exist_ok=True)

    trainer(config=wandb.config, wandb_run_dir=args.wandb_run_dir)
