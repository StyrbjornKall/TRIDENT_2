import os

os.chdir(os.path.dirname(os.path.abspath("__file__")))

import numpy as np
import pandas as pd
import json
import wandb
import argparse
from loguru import logger
from datetime import datetime
from trident2.logger.setup_logger import setup_logger
from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.train_utils import seed_all, train_final_model, Dict2Class


def trainer(config=None, wandb_run_dir=None):
    config = wandb.config

    # Set random seeds and deterministic pytorch for reproducibility
    seed_all(config.seed)

    config.wandb_run_name = wandb.run.id
    config.sweep_save_pth = f"{config.save_dir}/final_model_{wandb.run.id}"
    config.results_save_pth = f"{config.sweep_save_pth}/results"
    config.model_save_pth = f"{config.sweep_save_pth}/models"

    setup_logger(
        log_file=f"{config.sweep_save_pth}/logs/{wandb.run.id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"
    )

    for pth in [config.results_save_pth, config.model_save_pth]:
        os.makedirs(pth, exist_ok=True)
        logger.info(f"Created directory: {pth}")

    # Save sweep config to json file
    config_save_pth = f"{config.sweep_save_pth}/sweep_config.json"
    with open(config_save_pth, "w") as f:
        json.dump(config.as_dict(), f, indent=4)
    logger.success(f"Saved sweep config to {config_save_pth}")

    # Load dataframe
    if config.data_dir.endswith("pkl.zip"):
        data = pd.read_pickle(config.data_dir, compression="zip")
    else:
        data = pd.read_csv(config.data_dir)
    data = preprocess_data(data, config)

    logger.success("Successfully loaded data")

    _ = train_final_model(data, config)


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
        "--wandb_entity_name", required=False, default=None, help="W&B entity name"
    )
    parser.add_argument(
        "--wandb_project_name", required=False, default=None, help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_dir",
        default="/home/skall/TRIDENT_2/wandb",
        help="Directory to save wandb runs",
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
    os.makedirs(args.wandb_run_dir, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(args.config_file_path) as f:
        config = json.load(f)
    config = Dict2Class(config)

    if args.use_wandb:
        wandb.login()
    wandb.init(
        entity=args.wandb_entity_name,
        project=args.wandb_project_name,
        notes="",
        dir=args.wandb_run_dir,
        mode="online" if args.use_wandb else "disabled",
    )

    wandb.config.update(config)

    trainer(config=wandb.config, wandb_run_dir=args.wandb_run_dir)
