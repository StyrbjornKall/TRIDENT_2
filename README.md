# TRIDENT-2

TRIDENT-2 is a multimodal deep-learning framework for predicting chemical
toxicity endpoints (EC50, EC10, NOEC, LOEC) and effects across a wide range of
species. It is based on a multimodal transformer architecture that,
together with auxiliary modalities (taxonomic embeddings, exposure metadata,
duration, concentration units) and a multi-task regression head, predicts the four endpoint simultaneously.

The model is trained to be aware of species taxonomy through a
taxonomic-ranking auxiliary loss that exploits the taxonomic tree of species to
regularise predictions across related species. Similarly, it is trained to be
toxicity aware by the addition of a toxicity endpoint ordering loss.

The repository contains the full data-preprocessing, training, hyperparameter
sweep, k-fold cross-validation, post-hoc evaluation and inference pipeline
used to build and benchmark TRIDENT-2.

## Final model availability

The trained TRIDENT-2 model is available through our TRIDENT website: https://trident.serve.scilifelab.se/

Here you can interact with the model without coding experience to predict chemical toxicity of any chemical towards thousands of species.

## Repository layout

```
.
├── data/                          # Pre-processed toxicology data, taxonomy dicts, Butina cluster CSVs
├── jobs/                          # launch shell scripts
├── scripts/                       
│   ├── run_experiment.py          # Single-fold training run
│   ├── kfold_cross_validation_sweep.py  # W&B agent for k-fold CV sweeps
│   ├── train_final_model.py       # Train the final production model on all data
│   ├── train_adore.py             # Generate kfold validation results the ADORE dataset
│   ├── train_leo_bench.py         # LibFM-based Posthuma et al 2025 benchmark on the ADORE dataset
│   └── run_posthoc_evaluation.py  # Re-evaluate a finished run
├── src/
│   └── trident2/                  # Library code (importable as `trident2.*`)
│       ├── preprocessing/         # `preprocess_data.py`, `preprocess_taxonomy.py`
│       ├── torch_data_utils/      # Datasets, collators, splitters (KFold/GroupKFold/TwoWayKFold)
│       ├── model/                 # Model architectures, losses, tokenizer extension
│       ├── training/              # Training/eval loops and metric helpers
│       ├── inference/             # Inference utilities and exposure-setting builders
│       └── logger/                # `loguru`-based logger setup
├── wandb_configs/                 # YAML/JSON configs for W&B sweeps and model training
├── setup.sh                       # Setup script for environment
├── pyproject.toml                 # Project metadata + dependencies (uv-managed)
├── ruff.toml
└── README.md
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

The simplest approach is to run the `setup.sh <uv|pip>` script

```bash
# install with uv (recommended)
./setup.sh uv
# install with pip
./setup.sh pip
```

## Data

Training data is available in this repository under [data/](data/) as a zipped csv.
Required columns are documented in
[`PreprocessData.filter`](src/trident2/preprocessing/preprocess_data.py#L22):
`SMILES`, `endpoint`, `effect`, `conc`, `duration`, `species_group_corrected`,
`conc_sign`, `conc_unit`, `chemical_name`, `administration_route`,
`organism_lifestage`, `NCBI_last_known_rank`, plus the taxonomic rank columns
(`NCBI_rank_superkingdom` … `NCBI_rank_species`).

Auxiliary files also required for training:

- `taxid2parent.json`, `taxid2rank.json`, `taxid2sciname.json`,
  `taxid2spgroup.json` – NCBI taxonomy lookups.
- `taxonomy_tree_SK_*.nwk` – filled Newick tree.
- `butina_clusters_1024_3.csv` – ECFP4 / Butina cluster IDs used for clustering similar chemicals together to prevent data leakage.

Auxiliary files required for training downloadable from [zenodo](#TODO):

- `taxid2embedding_*.json` – MDS-derived taxonomic embeddings from the newick tree.

## Running experiments

Most entry points are configured by **JSON configs** or **sweep YAMLs** under
[wandb_configs/](wandb_configs/) and require `PYTHONPATH=src` so the
`trident2` package is importable (should be importable if user used `setup.sh`).

By default every entry point below runs **entirely locally** — no Weights &
Biases account, login, or internet connection required. Pass `--use_wandb`
(plus `--wandb_entity_name`/`--wandb_project_name`, and `--wandb_sweep_id` for
the sweep scripts) to log to a real W&B account instead. See
[jobs/](jobs/) for both variants of each command.

### Reproduce final model

```bash
uv run python scripts/train_final_model.py \
    --config_file_path wandb_configs/final_model/final_model_100ep.json

# Or, logging to a real W&B account:
uv run python scripts/train_final_model.py \
    --use_wandb --wandb_entity_name <entity> --wandb_project_name <project> \
    --config_file_path wandb_configs/final_model/final_model_100ep.json
```

### Single-fold training run

```bash
uv run python scripts/run_experiment.py \
    --config_file_path wandb_configs/experiments/experiment_example_local.json

# Or, logging to a real W&B account:
uv run python scripts/run_experiment.py \
    --use_wandb --wandb_entity_name <entity> --wandb_project_name <project> \
    --config_file_path wandb_configs/experiments/experiment_example_wandb.json
```

### K-fold cross-validation sweep

```bash
uv run python scripts/kfold_cross_validation_sweep.py \
    --config_file_path wandb_configs/cross_validations/interpolation_100ep.yaml

# Or, against a sweep already created on the W&B website:
uv run python scripts/kfold_cross_validation_sweep.py \
    --use_wandb --wandb_entity_name <entity> --wandb_project_name <project> \
    --wandb_sweep_id <sweep_id>
```

Local mode expands the sweep YAML's `method: grid` `parameters` into every
combination (e.g. all `fold_id` values) and runs them sequentially in one
process, no manual "create a sweep on the W&B website" step needed. Only
`method: grid` sweeps are supported locally.

The sweep config controls splitting (`kfold_identifier`,
`kfold_stratification_identifier`), augmentation (`shuffle_smiles_prob`,
`drop_metadata_prob`, `lower_taxonomic_rank_prob`), training hyperparameters,
and which auxiliary losses to use (`apply_taxonomic_ranking_loss`,
`constrain_endpoint_order`, `lambda_hier`).

### Post-hoc evaluation of a finished run

```bash
# From a locally saved run (the sweep_config.json written next to that
# run's results/models under its save_dir):
uv run python scripts/run_posthoc_evaluation.py \
    --config_file_path <save_dir>/sweep_<id>/sweep_config.json

# Or, fetching the run config from a real W&B account:
uv run python scripts/run_posthoc_evaluation.py \
    --wandb_entity_name <entity> --wandb_project_name <project> \
    --wandb_run_id <run_id> --resume_run True
```

### Baselines

- LibFM (Leo benchmark): [scripts/train_leo_bench.py](src/train_leo_bench.py)
  — requires a compiled `libFM` binary (`STORAGE` env var must point to its parent dir).
- ADORE benchmark: [scripts/train_adore.py](src/train_adore.py).

## Citation
#TODO