import os

os.chdir(os.path.dirname(os.path.abspath("__file__")))

import numpy as np
import pandas as pd
import json
import wandb
import argparse
from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.train_utils import seed_all, run_post_hoc_evaluation, Dict2Class
import glob
import os


def evaluator_from_wandb_run(
    config=None, resume_run=False, wandb_entity_name=None, wandb_project_name=None
):
    if resume_run:
        if wandb_entity_name is None or wandb_project_name is None:
            raise ValueError(
                "wandb_entity_name and wandb_project_name must be provided when resuming a run."
            )
        wandb.init(
            entity=wandb_entity_name,
            project=wandb_project_name,
            id=config.wandb_run_name,
            resume="must",
        )

    # Set random seeds and deterministic pytorch for reproducibility
    seed_all(config.seed)

    # Load dataframe
    if config.data_dir.endswith("pkl.zip"):
        data = pd.read_pickle(config.data_dir, compression="zip")
    else:
        data = pd.read_csv(config.data_dir)
    data = preprocess_data(data, config)

    # Try to load train/val ids from config
    train_ids = glob.glob(
        f"{config.results_save_pth}/{config.wandb_run_name}*_train_ids.pkl.zip"
    )[0]
    val_ids = glob.glob(
        f"{config.results_save_pth}/{config.wandb_run_name}*_val_ids.pkl.zip"
    )[0]
    train_ids = pd.read_pickle(train_ids, compression="zip")["SK_unique_id"].tolist()
    val_ids = pd.read_pickle(val_ids, compression="zip")["SK_unique_id"].tolist()

    print("Successfully loaded data")

    # Run one fold
    print(f"\n Running post-hoc evaluation using seed {config.seed}")

    _ = run_post_hoc_evaluation(config, data, train_ids=train_ids, val_ids=val_ids)

    if resume_run:
        wandb.finish()


def evaluator_from_config_file(config=None):
    """Run post-hoc evaluation from a locally saved run config (e.g. the
    ``sweep_config.json`` written by run_experiment.py / train_final_model.py
    / the sweep scripts), with no W&B account or network access required."""
    evaluator_from_wandb_run(config, resume_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_entity_name", required=False, help="W&B entity name", default=None
    )
    parser.add_argument(
        "--wandb_project_name", required=False, help="W&B project name", default=None
    )
    parser.add_argument(
        "--wandb_sweep_id", required=False, help="W&B sweep ID", default=None
    )
    parser.add_argument(
        "--wandb_run_id", required=False, help="W&B run ID", default=None
    )
    parser.add_argument(
        "--wandb_run_dir",
        default="/home/skall/TRIDENT_2/wandb",
        help="Directory to save wandb runs",
        required=False,
    )
    parser.add_argument(
        "--config_file_path",
        required=False,
        help="Path to config file, JSON or YAML",
        default=None,
    )
    parser.add_argument(
        "--resume_run",
        help="Whether to resume the W&B run when evaluating",
        default=False,
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # If a config file path is provided, load config from that file
    if args.config_file_path is not None:
        with open(args.config_file_path) as f:
            config = json.load(f)
        config = Dict2Class(config)
        evaluator_from_config_file(config)

    # If a wandb run id is provided, load config from that run
    if args.wandb_run_id is not None:
        wandb.login()
        api = wandb.Api()
        run = api.run(
            f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.wandb_run_id}"
        )
        config = run.config
        config = Dict2Class(config)
        print(f"\nEvaluating W&B run ID: {args.wandb_run_id}\n")
        evaluator_from_wandb_run(
            config,
            resume_run=args.resume_run,
            wandb_entity_name=args.wandb_entity_name,
            wandb_project_name=args.wandb_project_name,
        )
    # If a wandb sweep id is provided, all runs in that sweep will be evaluated if they do not have status "finished"
    if args.wandb_sweep_id is not None:
        wandb.login()
        api = wandb.Api()
        sweep = api.sweep(
            f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.wandb_sweep_id}"
        )
        run_ids = [run.id for run in sweep.runs if run.state != "finished"]
        configs = [run.config for run in sweep.runs if run.state != "finished"]
        for run_id, config in zip(run_ids, configs):
            config = Dict2Class(config)
            print(
                f"\nEvaluating W&B run ID: {run_id} for sweep {args.wandb_sweep_id}\n"
            )
            evaluator_from_wandb_run(
                config,
                resume_run=args.resume_run,
                wandb_entity_name=args.wandb_entity_name,
                wandb_project_name=args.wandb_project_name,
            )
