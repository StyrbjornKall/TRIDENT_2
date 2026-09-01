import os
import glob
import functools
import numpy as np
import pandas as pd
import wandb
import argparse
import json
from loguru import logger
from datetime import datetime
from trident2.logger.setup_logger import setup_logger
from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.train_utils import seed_all, run_one_fold_training
from trident2.training.sweep_utils import run_local_sweep


def trainer(
    config=None,
    wandb_run_dir=None,
    debug=False,
    use_wandb=True,
    sweep_name=None,
):

    # Initialize a new wandb run (mode="disabled" makes this a fully local no-op)
    with wandb.init(
        config=config, dir=wandb_run_dir, mode="online" if use_wandb else "disabled"
    ):
        # If called by wandb.agent, this config will be set by the Sweep Controller
        config = wandb.config

        # Set random seeds and deterministic pytorch for reproducibility
        seed_all(config.seed)

        config.wandb_run_name = wandb.run.id
        sweep_id = wandb.run.sweep_id or sweep_name or "local"
        config.sweep_save_pth = f"{config.save_dir}/sweep_{sweep_id}"
        config.results_save_pth = f"{config.sweep_save_pth}/results"
        config.model_save_pth = f"{config.sweep_save_pth}/models"
        config.debug_mode = debug

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

        if config.debug_mode:
            logger.warning(
                "Debug mode is ON. Using a small subset of the data for quick testing."
            )
            data = data.sample(n=10000, random_state=config.seed).reset_index(drop=True)

        logger.success("Successfully loaded data")

        # Run one fold
        logger.info(f"\n Running fold {config.fold_id} using seed {config.seed}")

        _, _, _, _, _ = run_one_fold_training(data, config)

        if getattr(config, "delete_models_after_run", False):
            logger.warning("Deleting saved models after run.")
            model_files = glob.glob(
                os.path.join(config.model_save_pth, f"{config.wandb_run_name}_*")
            )
            for model_file in model_files:
                os.remove(model_file)
                logger.info(f"Deleted {model_file}")
            logger.success(
                f"Deleted {len(model_files)} model file(s) for run "
                f"{config.wandb_run_name}"
            )


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
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of runs to execute for the sweep (default: 1, only used "
        "with --use_wandb)",
    )
    parser.add_argument(
        "--wandb_run_dir",
        default="/home/skall/TRIDENT_2/wandb",
        help="Directory to save wandb runs",
    )
    args = parser.parse_args()
    os.makedirs(args.wandb_run_dir, exist_ok=True)

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
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()
        trainer_partial = functools.partial(
            trainer,
            wandb_run_dir=args.wandb_run_dir,
            debug=args.debug,
            use_wandb=True,
        )

        # Run wandb agent (runs the script)
        wandb.agent(
            f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.wandb_sweep_id}",
            trainer_partial,
            count=args.count if args.count > 0 else None,
        )
    else:
        if args.config_file_path is None:
            parser.error("--config_file_path is required when --use_wandb is not set.")
        run_local_sweep(
            args.config_file_path,
            functools.partial(trainer, debug=args.debug),
            wandb_run_dir=args.wandb_run_dir,
        )
