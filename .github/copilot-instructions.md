# Copilot / AI-agent instructions for TRIDENT-2

These notes capture the conventions and non-obvious wiring of this repo so an
AI agent can be productive quickly. Read [README.md](../README.md) first for the
high-level picture.

## What this project is

A multimodal deep-learning pipeline for **ecotoxicity regression** across
species. A ChemBERTa/RoBERTa-style transformer encodes SMILES + token-level
metadata; an MDS-derived taxonomic embedding, a duration scalar and one-hot
encoded units are fused; a multi-task head predicts `EC50`, `EC10`, `NOEC`,
`LOEC`. Training adds an **endpoint-ordering penalty** and a **taxonomic
hierarchical-consistency loss** built on an `ete3` NCBI tree.

## Architecture & important modules

The library lives in [src/trident2/](../src/trident2/) and is imported as
`trident2.*`. Entry-point scripts in [src/](../src/) all assume
`PYTHONPATH=src`.

- Models — [src/trident2/model/model_utils.py](../src/trident2/model/model_utils.py)
  - Top-level fusion model: `TRIDENT2` (and legacy `TRIDENT`).
  - Encoders: `MultimodalRoBERTa`
  - Heads: `MultiTaskRegressionModule`, `RegressionModule`
  - Taxonomic side: `TaxonomicEmbedder`, `TaxonomicRankingLoss`.
  - Endpoint ordering: `endpoint_constraint_loss`, helper `collect_preds`.
- Datasets / collators — [src/trident2/torch_data_utils/torch_data_utils.py](../src/trident2/torch_data_utils/torch_data_utils.py)
  - `MultiModalDataset` + `MultiModalCollator` are the canonical pair used everywhere.
  - Splitting: `MakeTrainTestSplit`, `GroupKFolds`
  - `build_dataloader` handle samplers
    (`WeightedRandomSampler`, `SequentialSampler`, length buckets).
- Preprocessing — [src/trident2/preprocessing/preprocess_data.py](../src/trident2/preprocessing/preprocess_data.py)
  - Single entry point used by every training script:
    `preprocess_data(data, config)`. Drops/keeps a fixed list of columns; do
    not rely on extra columns surviving this call.
  - SMILES augmentation: `enumerate_smiles`. Taxonomy filling:
    `fill_missing_taxonomy` (and the more elaborate version in
    [preprocess_taxonomy.py](../src/trident2/preprocessing/preprocess_taxonomy.py)).
- Training loops — [src/trident2/training/train_utils.py](../src/trident2/training/train_utils.py)
  - `seed_all`, `Dict2Class`, `save_ckp` / `load_ckp` (saves DNN and encoder
    weights to two separate `.pt` files: `*_dnn_saved_weights.pt` and
    `*_transformer_encoder_weights.pt`).
  - One-fold training: `run_one_fold_training` (and ADORE variant
    `run_one_fold_training_adore`). Final-model training:
    `train_final_model`. Post-hoc eval: `run_post_hoc_evaluation`.
  - Inner loops: `train`, `evaluate`. Metric helpers in
    [performance_calculations.py](../src/trident2/training/performance_calculations.py)
    (`calculate_median_prediction_and_label`, `calculate_weighted_avg`).
  - Default base model string: `StyrbjornKall/Multimodal-ChemBERTa-1`.

## Configuration model (very important)

All scripts are driven by a flat config object accessed by attribute (`config.lr`,
`config.epochs`, …). It is created either from a JSON/YAML file via
[`Dict2Class`](../src/trident2/training/train_utils.py#L74) or as a `wandb.config`
when a W&B sweep is active. **Do not** assume dict-style access — code uses
attribute access throughout. When extending configs, also update the
representative YAML/JSON examples under
[wandb_configs/](../wandb_configs/) (see
[wandb_configs/final_model/final_model_100ep.json](../wandb_configs/final_model/final_model_100ep.json)
for the canonical key set).

Notable config keys to know:

- Data: `data_dir` (zip-pickled DataFrame), `taxonomic_embedding_dict`,
  `taxonomic_parent_dict`, `taxonomic_tree`, `butina_dir`,
  `endpoints`, `effects`, `species_groups`, `lifestages`,
  `administration_routes`, `butina_cutoff`.
- Splitting: `k_folds`, `fold_id`, `kfold_identifier`,
  `kfold_stratification_identifier`.
- Columns: `smiles_column`, `taxid_column`, `token_metadata_columns`,
  `embedding_columns`, `duration_column`, `onehot_column`, `label_column`,
  `endpoint_column`, `columns_to_onehot_encode`.
- Augmentation: `shuffle_smiles_prob`, `drop_metadata_prob`,
  `lower_taxonomic_rank_prob`.
- Model: `base_model`, `fusion_network_config`, `regression_task_config`
  (per-endpoint head config).
- Loss: `loss_function` (string name from `torch.nn`), `constrain_endpoint_order`,
  `apply_taxonomic_ranking_loss`, `lambda_hier`.
- W&B / saving: `save_dir` (results go to `<save_dir>/results`, models to
  `<save_dir>/models`), `wandb_run_name`, `manual_save_at_epochs`,
  `save_final_epoch`.

## Developer workflows

- Environment: `uv sync` (pinned via `uv.lock`). Python 3.11–3.13. Hard-pinned:
  `torch==2.1.2`, `transformers==4.39.3`, `huggingface_hub==0.21.4`,
  `rdkit==2025.3.2`, `ete3==3.1.3`. Do not bump these casually — checkpoints
  and HF Hub revisions assume them.
- Always run with `PYTHONPATH=src` (or use `uv run` from repo root after
  `uv pip install -e .`). The launch scripts in [jobs/](../jobs/) export this
  explicitly.
- Lint/format: `uv run ruff check .`, `uv run ruff format .` — config in
  [ruff.toml](../ruff.toml) (line length 88, double quotes, target py312,
  selecting only `E4 E7 E9 F`). Don’t add unrelated cleanups to PRs.
- Logging: prefer `from loguru import logger` (used throughout preprocessing).
  Existing prints in training loops are intentional for SLURM stdout.

## Project-specific conventions

- **Two-file checkpoints.** Models are split into encoder + DNN weights when
  saved (`save_ckp`). Loading must use `load_ckp` against an identically built
  model — see [run_inference.py](../src/run_inference.py) for the canonical
  rebuild path.
- **DataFrame-centric pipeline.** Most functions accept and return pandas
  DataFrames. Filtering happens once via `PreprocessData.filter`, which drops
  every column not in its explicit allow-list — add new columns to that list
  if you need them downstream.
- **Endpoints as task keys.** `preds` are `dict[str, Tensor]` keyed by
  `"EC50" | "EC10" | "NOEC" | "LOEC"`. New endpoints require updates to
  `MultiTaskRegressionModule`, `endpoint_constraint_loss`,
  `inference_utils.predict`, and the per-endpoint `regression_task_config`
  in every config file.
- **Taxonomic embeddings are MDS vectors** (default dim 768) keyed by NCBI
  taxid; `taxid_column` in configs is `NCBI_last_known_rank`. The taxonomy
  tree (`taxonomy_tree_SK_*.nwk`) carries a custom `rank` attribute on each
  node, used by `TaxonomicRankingLoss` to filter species-level leaves.
- **Splitting helpers** support both single-column and multi-column group
  identifiers. For unknown-chemicals experiments, `kfold_identifier` is set
  to a Butina-cluster column produced by joining `butina_dir`.
- **Absolute paths in configs** When running locally, override `save_dir`, `data_dir`,
  taxonomy paths and `butina_dir` to local equivalents — do not commit local
  paths.
- **Reproducibility.** Every training script calls
  `seed_all(config.seed)` and sets `torch.backends.cudnn.deterministic = True`
  and `os.environ["TOKENIZERS_PARALLELISM"] = "false"`. Preserve this when
  adding new entry points.
- **W&B is optional but assumed.** `run_experiment.py` works without W&B
  (entity/project args are optional); the sweep scripts (`hyperparameter_sweep.py`,
  `kfold_cross_validation_sweep.py`, `train_adore.py`) require a running
  sweep ID. `train_final_model.py` always logs to W&B.

## When making changes

- Editing models: update both training (`run_one_fold_training`,
  `train_final_model`) and inference (`build_model_and_tokenizer`, `predict`)
  paths so checkpoints stay loadable.
- Editing the dataset/collator: keep the `(inputs_dict, labels, taxids,
  endpoint, ...)` shape produced by `MultiModalCollator` consistent — the
  training loop unpacks by key.
- Adding a config key: thread it through `Dict2Class`-built configs **and**
  add it to one example YAML and one example JSON under [wandb_configs/](../wandb_configs/).
- New auxiliary losses go next to `endpoint_constraint_loss` /
  `TaxonomicRankingLoss` and should be gated by a boolean config flag.
- Avoid hardcoding Alvis paths; read them from the config object.
