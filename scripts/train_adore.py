import os
import functools
import numpy as np
import pandas as pd
import wandb
import argparse
from loguru import logger
from datetime import datetime
from trident2.logger.setup_logger import setup_logger
from trident2.training.train_utils import seed_all, run_one_fold_training_adore
from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.sweep_utils import run_local_sweep


def trainer(config=None, wandb_run_dir=None, use_wandb=True, sweep_name=None):

    # Initialize a new wandb run (mode="disabled" makes this a fully local no-op)
    with wandb.init(
        config=config, dir=wandb_run_dir, mode="online" if use_wandb else "disabled"
    ):
        config = wandb.config

        # Set random seeds and deterministic pytorch for reproducibility
        seed_all(config.seed)

        # If called by wandb.agent, this config will be set by the Sweep Controller
        config = wandb.config

        # Load dataframe
        if config.data_dir.endswith("pkl.zip"):
            data = pd.read_pickle(config.data_dir, compression="zip")
        else:
            data = pd.read_csv(config.data_dir)
        # Concentration already log transformed
        data["duration"] = np.log10(data["duration"])
        data["effect"] = data["result_effect"].copy()
        data["organism_lifestage_categorized"] = None
        data["administration_route_categorized"] = None
        data = data.rename(columns={"chem_name": "chemical_name"})
        data = preprocess_data(data, config)

        logger.success("Successfully loaded data")

        # Run one fold
        logger.info(f"\n Running fold {config.fold_id} using seed {config.seed}")

        config.wandb_run_name = wandb.run.id
        config.results_save_pth = f"{config.save_dir}/results"
        config.model_save_pth = f"{config.save_dir}/models"

        for pth in [f"{config.save_dir}/results", f"{config.save_dir}/models"]:
            os.makedirs(pth, exist_ok=True)

        _, _, _, _, _ = run_one_fold_training_adore(data, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Run this sweep against a real Weights & Biases sweep (requires "
        "--wandb_entity_name/--wandb_project_name/--wandb_sweep_id and being "
        "logged in). Omit this flag to expand and run the sweep YAML locally, "
        "no W&B account needed.",
    )
    parser.add_argument(
        "--wandb_entity_name", required=False, default=None, help="W&B entity name"
    )
    parser.add_argument(
        "--wandb_project_name", required=False, default=None, help="W&B project name"
    )
    parser.add_argument(
        "--wandb_sweep_id", required=False, default=None, help="W&B sweep ID"
    )
    parser.add_argument(
        "--config_file_path",
        required=False,
        default=None,
        help="Path to a local sweep YAML config (used when --use_wandb is not set)",
    )
    parser.add_argument(
        "--wandb_run_dir",
        default="/wandb",
        help="Directory to save wandb runs",
    )
    args = parser.parse_args()
    setup_logger(
        log_file=f"./logs/train_adore_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.use_wandb:
        if (
            args.wandb_entity_name is None
            or args.wandb_project_name is None
            or args.wandb_sweep_id is None
        ):
            parser.error(
                "--wandb_entity_name, --wandb_project_name and --wandb_sweep_id "
                "are required when --use_wandb is set."
            )
        wandb.login()
        trainer_partial = functools.partial(
            trainer, wandb_run_dir=args.wandb_run_dir, use_wandb=True
        )

        # Run wandb agent (runs the script)
        wandb.agent(
            f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.wandb_sweep_id}",
            trainer_partial,
        )
    else:
        if args.config_file_path is None:
            parser.error("--config_file_path is required when --use_wandb is not set.")
        run_local_sweep(
            args.config_file_path, trainer, wandb_run_dir=args.wandb_run_dir
        )
