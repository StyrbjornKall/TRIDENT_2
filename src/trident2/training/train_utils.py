from datetime import datetime
import json
from loguru import logger
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import random
import numpy as np
import time
import seaborn as sns
import os
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Union
import string
from ete3 import Tree

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from trident2.torch_data_utils.torch_data_utils import (
    build_dataloader,
    GroupKFolds,
    MultiModalCollator,
    MultiModalDataset,
)
from trident2.training.performance_calculations import (
    calculate_weighted_avg,
    calculate_median_prediction_and_label,
)
from trident2.model.model_utils import (
    TRIDENT2,
    MultiTaskRegressionModule,
    MultimodalRoBERTa,
    load_taxonomic_embedding_dict,
)
from trident2.model.model_utils import (
    collect_preds,
    endpoint_constraint_loss,
    TaxonomicRankingLoss,
)


def print_gpu_info(device):
    logger.info(f"- GPUs on node: {torch.cuda.get_device_name()}")
    logger.info(f"- Number of GPUs available: {torch.cuda.device_count()}")
    logger.info(f"- Using {device} device")
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t - r - a
    logger.info(f"- {np.round(f / 1000000000, 2)} Gb free on CUDA")


def seed_all(seed):
    torch.manual_seed(seed)  # pytorch random seed
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def __str__(self):
        # Generate a string representation of all attributes
        attrs = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self):
        # Use the same as __str__ for a more descriptive representation
        return self.__str__()


# Function to save fine-tuned ChemBERTa and DNN-module
def save_ckp(model, checkpoint_dir):
    if isinstance(model, nn.DataParallel):
        torch.save(
            model.module.dnn.state_dict(), checkpoint_dir + "_dnn_saved_weights.pt"
        )
        torch.save(
            model.module.transformer_encoder.state_dict(),
            checkpoint_dir + "_transformer_encoder_weights.pt",
        )
        logger.success(f"Saved DNN to {checkpoint_dir + '_dnn_saved_weights.pt'}")
        logger.success(
            f"Saved Transformer Encoder to {checkpoint_dir + '_transformer_encoder_weights.pt'}"
        )
    else:
        torch.save(model.dnn.state_dict(), checkpoint_dir + "_dnn_saved_weights.pt")
        torch.save(
            model.transformer_encoder.state_dict(),
            checkpoint_dir + "_transformer_encoder_weights.pt",
        )
        logger.success(f"Saved DNN to {checkpoint_dir + '_dnn_saved_weights.pt'}")
        logger.success(
            f"Saved Transformer Encoder to {checkpoint_dir + '_transformer_encoder_weights.pt'}"
        )


def export_hf_trident_checkpoint(
    model,
    tokenizer,
    config,
    checkpoint_dir,
    taxonomic_embedding_dict=None,
):
    """
    Export a full TRIDENT2 checkpoint directory that can be uploaded to the
    Hugging Face Hub and loaded with AutoModel.from_pretrained(..., trust_remote_code=True)
    and AutoTokenizer.from_pretrained(...).
    """
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    export_dir = checkpoint_dir + "_hf_trident2"
    os.makedirs(export_dir, exist_ok=True)

    # Copy remote-code files used by AutoModel/AutoConfig.
    trident_root = os.path.dirname(os.path.dirname(__file__))
    hf_src_dir = os.path.join(trident_root, "hf")
    shutil.copy2(
        os.path.join(hf_src_dir, "configuration_trident2.py"),
        os.path.join(export_dir, "configuration_trident2.py"),
    )
    shutil.copy2(
        os.path.join(hf_src_dir, "modeling_trident2.py"),
        os.path.join(export_dir, "modeling_trident2.py"),
    )

    embedding_dim = getattr(config, "embedding_dim", 768)
    with_duration_neuron = getattr(config, "duration_column", None) is not None
    onehot_length = max(
        0,
        int(core_model.dnn.input_layer_size)
        - int(embedding_dim)
        - (1 if with_duration_neuron else 0),
    )

    # Save full TRIDENT2 state dict in Hugging Face standard filename.
    full_state_dict = core_model.state_dict()

    # Embed taxonomic lookup vectors into model weights for a self-contained checkpoint.
    taxid_to_index = {}
    num_taxa = 0
    taxonomic_embed_dim = None
    if taxonomic_embedding_dict is not None:
        num_taxa = len(taxonomic_embedding_dict)
        taxonomic_embed_dim = len(next(iter(taxonomic_embedding_dict.values())))
        taxonomic_table = torch.zeros(
            (num_taxa, taxonomic_embed_dim), dtype=torch.float32
        )
        for taxid, vector in taxonomic_embedding_dict.items():
            taxid_to_index[taxid] = len(taxid_to_index)
            idx = taxid_to_index.get(taxid, None)
            if idx is None:
                continue
            vector_tensor = torch.tensor(vector, dtype=torch.float32)
            if vector_tensor.numel() != taxonomic_embed_dim:
                logger.warning(
                    f"Taxonomic embedding for taxid {taxid} has incorrect dimension "
                    f"({vector_tensor.numel()} != {taxonomic_embed_dim}), skipping."
                )
                continue
            taxonomic_table[idx] = vector_tensor

        full_state_dict[
            "transformer_encoder.taxonomic_embedder.taxonomic_table.weight"
        ] = taxonomic_table

    torch.save(full_state_dict, os.path.join(export_dir, "pytorch_model.bin"))

    # Save tokenizer for AutoTokenizer.from_pretrained.
    tokenizer.save_pretrained(export_dir)

    # Keep a copy of the training config for reproducibility.
    with open(
        os.path.join(export_dir, "trident2_training_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(vars(config), f, indent=2, default=str)

    model_config = {
        "model_type": "trident2",
        "architectures": ["TRIDENT2Model"],
        "auto_map": {
            "AutoConfig": "configuration_trident2.TRIDENT2Config",
            "AutoModel": "modeling_trident2.TRIDENT2Model",
        },
        "base_model_name_or_path": getattr(config, "base_model", None),
        "backbone_config": core_model.transformer_encoder.base_model.config.to_dict(),
        "regression_task_config": getattr(config, "regression_task_config", {}),
        "fusion_network_config": getattr(config, "fusion_network_config", None),
        "embedding_dim": embedding_dim,
        "onehot_length": onehot_length,
        "with_duration_neuron": with_duration_neuron,
        "prepend_external": getattr(
            core_model.transformer_encoder, "prepend_external", True
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "taxid_to_index": taxid_to_index,
        "num_taxa": num_taxa,
    }

    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    logger.success(f"Exported full Hugging Face TRIDENT2 checkpoint to {export_dir}")


def load_ckp(checkpoint_dir, model, device="cuda:0"):
    checkpoint_dnn = torch.load(
        checkpoint_dir + "_dnn_saved_weights.pt", map_location=device
    )
    checkpoint_roberta = torch.load(
        checkpoint_dir + "_transformer_encoder_weights.pt", map_location=device
    )
    logger.success(f"Loaded DNN from {checkpoint_dir + '_dnn_saved_weights.pt'}")
    logger.success(
        f"Loaded Transformer Encoder from {checkpoint_dir + '_transformer_encoder_weights.pt'}"
    )
    if isinstance(model, nn.DataParallel):
        model.module.dnn.load_state_dict(checkpoint_dnn)
        model.module.transformer_encoder.load_state_dict(checkpoint_roberta)
        return model
    else:
        model.dnn.load_state_dict(checkpoint_dnn)
        model.transformer_encoder.load_state_dict(checkpoint_roberta)
        return model


# function to train the model on epoch
def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fun,
    batch_num: Union[bool, None] = None,
    epoch: Union[int, None] = None,
    global_step: Union[bool, None] = None,
    device: str = "cuda:0",
    log_to_wandb: bool = True,
    scaler=None,
    mixed_precision: bool = True,
    constrain_endpoint_order: bool = False,
    taxonomic_ranking_loss: nn.Module = None,
):
    model.train()
    logger.info("\nTraining...")
    total_loss = 0.0
    total_preds = []
    total_labels = []
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    if (scaler is None) and mixed_precision:
        raise (
            Exception(
                "Trying to use mixed precision training without scaler or vice versa. This is not possible."
            )
        )

    for step, batch in enumerate(tqdm(dataloader)):
        # Data transfer to GPU
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in ["labels", "endpoint", "taxids"]
        }
        endpoints = batch.get("endpoint", None)
        taxids = batch.get("taxids", None)
        labels = batch["labels"].to(device)

        # Forward pass with autocast
        if mixed_precision:
            with torch.autocast(device_type=device_type):
                loss = 0.0
                preds, _ = model(**inputs, taxids=taxids)
                # We employ an endpoint ordering constraint loss if specified that penalises if e.g. NOEC higher than LOEC
                if constrain_endpoint_order:
                    ep_constraint_loss = endpoint_constraint_loss(
                        preds, wrong_ordering_penalty_weight=1.0, margin=1e-3
                    )
                    loss += ep_constraint_loss
                # We employ an additional taxonomic ranking loss if specified that enforces lower taxonomic ranks to converge towardds the average of their children
                if taxonomic_ranking_loss is not None:
                    taxrank_loss = taxonomic_ranking_loss(
                        model=model,
                        inputs=inputs,
                        taxids=taxids,
                    )
                    loss += taxrank_loss

                preds = collect_preds(preds=preds, endpoints=endpoints, device=device)
                loss += loss_fun(preds, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            loss = 0.0
            preds, _ = model(**inputs, taxids=taxids)
            # We employ an endpoint ordering constraint loss if specified that penalises if e.g. NOEC higher than LOEC
            if constrain_endpoint_order:
                ep_constraint_loss = endpoint_constraint_loss(
                    preds, wrong_ordering_penalty_weight=1.0, margin=1e-3
                )
                loss += ep_constraint_loss

            # We employ an additional taxonomic ranking loss if specified that enforces lower taxonomic ranks to converge towardds the average of their children
            if taxonomic_ranking_loss is not None:
                taxrank_loss = taxonomic_ranking_loss(
                    model=model,
                    inputs=inputs,
                    taxids=taxids,
                )
                loss += taxrank_loss

            preds = collect_preds(preds=preds, endpoints=endpoints, device=device)
            loss += loss_fun(preds, labels)

            # Backward pass and optimization
            loss.backward()
            # Clip gradient to prevent exploding gradients and update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Accumulate loss and predictions
        total_loss += loss.detach().cpu()
        total_preds.append(preds.detach().cpu())
        total_labels.append(labels.detach().cpu())

        # Log metrics (every 10 steps)
        if log_to_wandb and step % 10 == 0:
            wandb.log(
                {
                    "Training Batch Loss": loss.item(),
                    "Training Batch Ep constraint Loss": ep_constraint_loss.item()
                    if constrain_endpoint_order
                    else np.nan,
                    "Training Batch Taxonomic ranking Loss": taxrank_loss.item()
                    if taxonomic_ranking_loss is not None
                    else np.nan,
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                }
            )
        if batch_num is not None:
            batch_num[1] += 1

    # Compute final metrics
    total_preds = torch.cat(total_preds).numpy()
    total_labels = torch.cat(total_labels).numpy()
    avg_loss = total_loss.item() / len(dataloader)
    median_loss = np.median(abs(total_preds - total_labels))

    if log_to_wandb:
        wandb.log(
            {
                "Training Loss function": avg_loss,
                "Training Mean Loss": np.mean(abs(total_preds - total_labels)),
                "Training Epoch": epoch,
                "Training Median Loss": median_loss,
                "Training RMSE Loss": np.sqrt(
                    np.mean((total_labels - total_preds) ** 2)
                ),
                "global_step": global_step,
            }
        )

    return avg_loss, median_loss, total_preds, total_labels, batch_num


# function to validate the model on epoch
def evaluate(
    model,
    dataloader,
    dataset,
    loss_fun,
    batch_num: Union[int, None] = None,
    epoch: Union[int, None] = None,
    global_step: Union[int, None] = None,
    device: str = "cuda:0",
    mixed_precision: bool = False,
    log_to_wandb: bool = True,
):

    logger.info("\nEvaluating...")
    model.eval()
    total_preds = []
    total_labels = []
    total_loss = 0
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    # Initialize validation array in which to log results
    val_results = dataset.copy()
    cls_embeddings = []

    # iterate over batches
    for step, batch in enumerate(tqdm(dataloader)):
        # Extract batch samples
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in ["labels", "endpoint", "taxids"]
        }
        endpoints = batch.get("endpoint", None)
        taxids = batch.get("taxids", None)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=mixed_precision):
                # Predict batch
                preds, roberta_output = model(**inputs, taxids=taxids)
                preds = collect_preds(preds=preds, endpoints=endpoints, device=device)

                # Calculate batch loss
                loss = loss_fun(preds, labels)
            total_loss += loss.item()

            # Cast to float32 before leaving GPU so downstream numpy ops are always fp32
            cls_embeddings.append(roberta_output.detach().float().cpu().numpy())
            preds = preds.detach().float().cpu().numpy()
            labels = labels.detach().float().cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels)
        if batch_num is not None:
            batch_num[1] += 1

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    val_results["CLS_embeddings"] = np.concatenate(cls_embeddings, axis=0).tolist()
    val_results["labels"] = total_labels
    val_results["preds"] = total_preds
    val_results["residuals"] = val_results.labels - val_results.preds
    val_results["absolute_error"] = abs(total_labels - total_preds)
    median_loss = val_results.absolute_error.median()
    val_results_normalized = calculate_weighted_avg(
        calculate_median_prediction_and_label(val_results)
    )
    median_loss_norm = abs(val_results_normalized.residuals).median()
    avg_loss_norm = abs(val_results_normalized.residuals).mean()
    if log_to_wandb:
        wandb.log(
            {
                "Validation Loss function": avg_loss,
                "Validation Mean Loss": val_results.absolute_error.mean(),
                "Validation Median Loss": median_loss,
                "Validation Loss Normalized": median_loss_norm,
                "Validation Mean Loss Normalized": avg_loss_norm,
                "Validation RMSE Loss Normalized": np.sqrt(
                    (
                        (val_results_normalized.labels - val_results_normalized.preds)
                        ** 2
                    ).mean()
                ),
                "Validation Epoch": epoch,
                "global_step": global_step,
            }
        )

    return (
        avg_loss,
        avg_loss_norm,
        median_loss,
        median_loss_norm,
        total_preds,
        batch_num,
        val_results,
    )


# Function to create scatter plot with trendline, metrics, and violin plots
def plot_with_metrics(data, title, color, xlim=4, ylim=4, x="labels", y="preds"):
    # Plot scatter
    X = data[x].to_numpy()
    Y = data[y].to_numpy()

    lr = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    x_lr = np.array([-xlim, xlim])
    y_lr = lr.predict(x_lr.reshape(-1, 1)).reshape(1, -1)[0]

    r2 = (np.corrcoef(X, Y) ** 2)[0, 1]
    rmse = np.sqrt(mean_squared_error(X, Y))
    mae = mean_absolute_error(X, Y)

    g = sns.jointplot(
        data=data,
        x=x,
        y=y,
        xlim=(-xlim, xlim),
        ylim=(-ylim, ylim),
        space=-0.2,
        alpha=0,
        marginal_kws={"alpha": 0.0, "fill": False},
    ).plot_joint(
        sns.scatterplot,
        alpha=0.6,
        linewidths=0.5,
        edgecolors="black",
        s=20,
        color=color,
    )

    sns.lineplot(
        x=x_lr, y=y_lr, linewidth=2, linestyle="--", color="black", ax=g.ax_joint
    )

    g.plot_marginals(sns.violinplot, alpha=0.5, color=color)

    text = "\n".join(
        (f"RMSE: {rmse:.3f}", f"MAE: {mae:.3f}", f"R²: {r2:.3f}", f"n={len(X)}")
    )

    g.ax_joint.text(
        0.05,
        0.92,
        text,
        transform=g.ax_joint.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Customize plot
    for ax in [g.ax_marg_x, g.ax_marg_y]:
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(bottom=False, left=False, top=False, right=False)
    g.ax_marg_x.set_xlabel("Actual", fontsize=12)
    g.ax_marg_y.set_ylabel("Predicted", fontsize=12)
    g.ax_marg_x.set_title(title, fontsize=14)
    plt.close(g.fig)

    return g


def run_one_fold_training(data, config):
    ######## DataLoading ##################################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    # Build folds of categorical identifiers (like SMILES and species)
    folds = GroupKFolds(n_splits=config.k_folds, seed=config.seed, shuffle=True).split(
        df=data,
        group_cols=config.kfold_identifier,
        stratify_col=getattr(config, "kfold_stratification_identifier", None),
    )

    # Build train and validation sets for current fold
    train_idx = folds[f"fold_{config.fold_id}"]["train_idx"]
    val_idx = folds[f"fold_{config.fold_id}"]["val_idx"]

    train_set = data.iloc[train_idx]
    val_set = data.iloc[val_idx]

    # Save fold assignment
    train_set[["SK_unique_id"]].to_pickle(
        f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_train_ids.pkl.zip",
        compression="zip",
    )
    val_set[["SK_unique_id"]].to_pickle(
        f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_ids.pkl.zip",
        compression="zip",
    )

    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    # Check number of overlapping identifiers
    overlap = train_set[config.kfold_identifier].merge(
        val_set[config.kfold_identifier], on=config.kfold_identifier
    )
    logger.info(f"Number of overlapping identifiers: {len(overlap)}")

    # Load taxonomic embedding dictionary if specified
    if getattr(config, "taxonomic_embedding_dict", None) is not None:
        taxonomic_embedding_dict = load_taxonomic_embedding_dict(
            config.taxonomic_embedding_dict
        )
        logger.success(
            f"Loaded taxonomic embedding dictionary with {len(taxonomic_embedding_dict)} entries"
        )
    else:
        logger.warning("No taxonomic embedding dictionary provided")
        taxonomic_embedding_dict = None

    # Load parent dictionary if specified
    if getattr(config, "taxonomic_parent_dict", None) is not None:
        with open(config.taxonomic_parent_dict, "r", encoding="utf-8") as f:
            taxonomic_parent_dict = json.load(f)
        logger.success(
            f"Loaded parent dictionary with {len(taxonomic_parent_dict)} entries"
        )
    else:
        logger.warning("No parent dictionary provided")
        taxonomic_parent_dict = None

    train_set = MultiModalDataset(
        df=train_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=getattr(config, "shuffle_smiles_prob", 0.0),
        drop_metadata_prob=getattr(config, "drop_metadata_prob", 0.0),
        lower_taxonomic_rank_prob=getattr(config, "lower_taxonomic_rank_prob", 0.0),
        parent_dict=taxonomic_parent_dict,
    )
    val_set = MultiModalDataset(
        df=val_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=0.0,
        drop_metadata_prob=0.0,
        lower_taxonomic_rank_prob=0.0,
        parent_dict=taxonomic_parent_dict,
    )

    collate_fn = MultiModalCollator(
        tokenizer=tokenizer,
        max_len=getattr(config, "max_len", 512),
        padding=getattr(config, "padding", "longest"),
        truncation=getattr(config, "truncation", True),
        padding_idx=tokenizer.pad_token_id,
    )
    train_dataloader = build_dataloader(
        dataset=train_set,
        sampler=getattr(config, "training_sampler", "WeightedRandomSampler"),
        stratification_method=getattr(config, "stratification_method", None),
        sampler_weight_args=getattr(config, "sampler_weight_args", None),
        batch_size=getattr(config, "train_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(config, "pin_memory", True),
    )
    val_dataloader = build_dataloader(
        dataset=val_set,
        sampler=getattr(config, "validation_sampler", "SequentialSampler"),
        stratification_method=getattr(config, "stratification_method", None),
        sampler_weight_args=getattr(config, "sampler_weight_args", None),
        batch_size=getattr(config, "val_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(config, "pin_memory", True),
    )

    logger.success("Successfully built dataloader")
    logger.warning(
        f"SMILES overlap train/validation: {len(set(train_set.df.SMILES_Canonical_RDKit.tolist()) & set(val_set.df.SMILES_Canonical_RDKit.tolist()))}"
    )

    ######## MODEL ##################################################################################

    transformer = AutoModel.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    dnn = MultiTaskRegressionModule(
        input_layer_size=(
            getattr(config, "embedding_dim", 768)
            + (
                len(train_set[0]["one_hot_encoding"])
                if getattr(config, "onehot_column", None) is not None
                else 0
            )
            + (1 if getattr(config, "duration_column", None) is not None else 0)
        ),
        regression_task_configs=config.regression_task_config,
        fusion_network_config=getattr(config, "fusion_network_config", None),
    )

    if transformer.config.model_type == "roberta":
        mmtransformer = MultimodalRoBERTa(
            base_model=transformer,
            padding_idx=tokenizer.pad_token_id,
            prepend_external=True,
            taxonomic_embedding_dict=taxonomic_embedding_dict,
        )

    else:
        raise NotImplementedError("Model not implemented")
    model = TRIDENT2(transformer_encoder=mmtransformer, dnn=dnn).to(
        getattr(config, "device", "cuda:0"),
    )

    del mmtransformer, dnn, transformer
    logger.success("Successfully built model\n")

    ######## TRAINING CONFIG ##################################################################################

    model_parameters = model.parameters()

    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=config.lr,
        betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)),
        eps=1e-08,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * config.epochs * len(train_dataloader),
        num_training_steps=config.epochs * len(train_dataloader),
    )
    if getattr(config, "mixed_precision_training", True):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    logger.success("Successfully built optimizer")

    if config.loss_function == "MSELoss":
        loss_function = nn.MSELoss()
    elif config.loss_function == "L1Loss":
        loss_function = nn.L1Loss()
    else:
        logger.warning("Loss function not matching, defaulting to L1Loss.")
        loss_function = nn.L1Loss()
    if (
        getattr(config, "apply_taxonomic_ranking_loss", False)
        and getattr(config, "taxonomic_tree", None) is not None
    ):
        logger.info("Applying taxonomic ranking loss")
        taxonomy_tree = Tree(
            config.taxonomic_tree,
            format=1,
        )
        taxonomic_ranking_loss = TaxonomicRankingLoss(
            lambda_hier=getattr(config, "lambda_hier", 0.01),
            taxonomic_tree=taxonomy_tree,
            rank_weights={
                "genus": 62,
                "family": 15,
                "order": 10,
                "class": 5,
                "phylum": 5,
                "kingdom": 2,
                "superkingdom": 1,
            },
            training_taxids=train_set.df[getattr(config, "taxid_column", None)]
            .unique()
            .tolist(),
            detach_species=getattr(
                config, "detach_species_in_taxonomic_ranking_loss", True
            ),
            sample_n_inputs=getattr(config, "taxonomic_ranking_loss_n_samples", 10),
        )
    else:
        taxonomic_ranking_loss = None
    logger.success("Successfully built loss function")

    best_val_loss = np.inf
    best_val_loss_norm = np.inf
    batch_num = [0, 0]
    global_step = 0

    # Log initial validation loss
    (
        avg_loss,
        avg_loss_norm,
        median_loss,
        median_loss_norm,
        _,
        batch_num,
        val_results,
    ) = evaluate(
        model=model,
        dataloader=val_dataloader,
        dataset=val_set.df,
        loss_fun=loss_function,
        batch_num=batch_num,
        epoch=-1,
        global_step=global_step - 1,
        device=getattr(config, "device", "cuda:0"),
        mixed_precision=getattr(config, "mixed_precision_training", True),
        log_to_wandb=getattr(config, "log_to_wandb", True),
    )

    if median_loss < best_val_loss:
        best_val_loss = median_loss

    get_running_stats(
        df=val_results,
        global_step=global_step - 1,
        log_to_wandb=getattr(config, "log_to_wandb", True),
    )

    ######## RUN TRAINING ##################################################################################
    logger.info("\nRunning epochs...")
    start_time = time.time()

    for epoch in tqdm(range(config.epochs)):
        avg_loss, median_loss, total_preds, total_labels, batch_num = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            loss_fun=loss_function,
            batch_num=batch_num,
            epoch=epoch,
            global_step=global_step,
            device=getattr(config, "device", "cuda:0"),
            mixed_precision=getattr(config, "mixed_precision_training", True),
            log_to_wandb=getattr(config, "log_to_wandb", True),
            constrain_endpoint_order=getattr(config, "constrain_endpoint_order", False),
            taxonomic_ranking_loss=taxonomic_ranking_loss,
        )

        (
            avg_loss,
            avg_loss_norm,
            median_loss,
            median_loss_norm,
            _,
            batch_num,
            val_results,
        ) = evaluate(
            model=model,
            dataloader=val_dataloader,
            dataset=val_set.df,
            loss_fun=loss_function,
            batch_num=batch_num,
            epoch=epoch,
            global_step=global_step,
            device=getattr(config, "device", "cuda:0"),
            mixed_precision=getattr(config, "mixed_precision_training", True),
            log_to_wandb=getattr(config, "log_to_wandb", True),
        )

        get_running_stats(
            df=val_results,
            global_step=global_step,
            log_to_wandb=getattr(config, "log_to_wandb", True),
        )

        # Update and log epoch results
        if median_loss_norm < best_val_loss_norm:
            best_val_loss = median_loss
            best_val_loss_norm = median_loss_norm
            best_validation_results = val_results
            best_validation_mean_norm_loss = avg_loss_norm
            best_epoch = epoch
            if getattr(config, "save_model", False) or getattr(
                config, "save_best_model", False
            ):
                save_ckp(
                    model=model,
                    checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_best_model",
                )

        # Manual save points
        if getattr(config, "manual_save_at_epochs", False) and epoch in getattr(
            config, "manual_save_at_epochs", []
        ):
            save_ckp(
                model=model,
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_epoch_{epoch}",
            )

        wandb.log(
            {
                "Best Validation Median Loss": best_val_loss,
                "Best Validation Median Loss Normalized": best_val_loss_norm,
                "Best Validation Mean Loss Normalized": best_validation_mean_norm_loss,
                "global_step": global_step,
            }
        )

        global_step += 1

    train_time = (time.time() - start_time) / 60

    wandb.log(
        {
            "Total train time (min)": train_time,
            "epoch time (s)": train_time / config.epochs * 60,
            "best epoch": best_epoch,
        }
    )

    logger.success(
        f"{'Metric':<15}{'Value':<10}\n"
        f"{'-' * 25}\n"
        f"{'Epochs: ':<15}{epoch:<10}\n"
        f"{'Best Median Loss: ':<15}{best_val_loss:<10.2f}\n"
        f"{'Best Median Loss Normalized: ':<15}{best_val_loss_norm:<10.2f}\n"
        f"{'Best Mean Loss Normalized: ':<15}{best_validation_mean_norm_loss:<10.2f}\n"
        f"{'Training Time (min): ':<15}{train_time:<10.2f}\n"
    )

    if getattr(config, "save_final_epoch", False):
        save_ckp(
            model,
            f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_final_epoch",
        )

    # Save results locally and plot predicted vs actual
    if getattr(config, "save_results", False):
        if getattr(config, "save_final_epoch", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_final_epoch",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_final_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config,
                train_results,
                val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if getattr(config, "save_best_model", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_best_model",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_best_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config,
                train_results,
                val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if len(getattr(config, "manual_save_at_epochs", [])) > 0:
            for epoch in getattr(config, "manual_save_at_epochs", []):
                model = load_ckp(
                    model=model,
                    device=getattr(config, "device", "cuda:0"),
                    checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_epoch_{epoch}",
                )
                val_results = get_eval_stats(
                    config, model, val_set, collate_fn, loss_function
                )
                train_results = get_eval_stats(
                    config, model, train_set, collate_fn, loss_function
                )
                val_results.to_pickle(
                    f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_epoch_{epoch}.pkl.zip",
                    compression="zip",
                )
                plot_predicted_vs_actual(
                    config,
                    train_results,
                    val_results,
                    normalize=True,
                    x="labels",
                    y="preds",
                )
        else:
            logger.warning(
                "No results saved locally as no save options were selected.\n"
            )

    return (
        best_val_loss,
        best_val_loss_norm,
        best_validation_mean_norm_loss,
        global_step,
        best_validation_results,
    )


def run_one_fold_training_adore(data, config):
    ######## DataLoading ##################################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    # Build train, validation, and test sets for current fold
    train_set = data[
        (data["split_occurrence"] != str(config.fold_id))
        & (data["split_occurrence"] != "test")
    ]
    val_set = data[(data["split_occurrence"] == str(config.fold_id))]

    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    # Check number of overlapping identifiers
    overlap = train_set[config.kfold_identifier].merge(
        val_set[config.kfold_identifier], on=config.kfold_identifier
    )
    logger.info(f"Number of overlapping identifiers: {len(overlap)}")

    # Load taxonomic embedding dictionary if specified
    if getattr(config, "taxonomic_embedding_dict", None) is not None:
        taxonomic_embedding_dict = load_taxonomic_embedding_dict(
            config.taxonomic_embedding_dict
        )
        logger.success(
            f"Loaded taxonomic embedding dictionary with {len(taxonomic_embedding_dict)} entries"
        )
    else:
        logger.info("No taxonomic embedding dictionary provided")
        taxonomic_embedding_dict = None

    # Load parent dictionary if specified
    if getattr(config, "taxonomic_parent_dict", None) is not None:
        with open(config.taxonomic_parent_dict, "r", encoding="utf-8") as f:
            taxonomic_parent_dict = json.load(f)
        logger.success(
            f"Loaded parent dictionary with {len(taxonomic_parent_dict)} entries"
        )
    else:
        logger.info("No parent dictionary provided")
        taxonomic_parent_dict = None

    train_set = MultiModalDataset(
        df=train_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=getattr(config, "shuffle_smiles_prob", 0.0),
        drop_metadata_prob=getattr(config, "drop_metadata_prob", 0.0),
        lower_taxonomic_rank_prob=getattr(config, "lower_taxonomic_rank_prob", 0.0),
        parent_dict=taxonomic_parent_dict,
    )
    val_set = MultiModalDataset(
        df=val_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=0.0,
        drop_metadata_prob=0.0,
        lower_taxonomic_rank_prob=0.0,
        parent_dict=taxonomic_parent_dict,
    )

    collate_fn = MultiModalCollator(
        tokenizer=tokenizer,
        max_len=getattr(config, "max_len", 512),
        padding=getattr(config, "padding", "longest"),
        truncation=getattr(config, "truncation", True),
        padding_idx=tokenizer.pad_token_id,
    )
    train_dataloader = build_dataloader(
        dataset=train_set,
        sampler=getattr(config, "training_sampler", "WeightedRandomSampler"),
        stratification_method=getattr(config, "stratification_method", None),
        sampler_weight_args=getattr(config, "sampler_weight_args", None),
        batch_size=getattr(config, "train_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(config, "pin_memory", True),
    )
    val_dataloader = build_dataloader(
        dataset=val_set,
        sampler=getattr(config, "validation_sampler", "SequentialSampler"),
        stratification_method=getattr(config, "stratification_method", None),
        sampler_weight_args=getattr(config, "sampler_weight_args", None),
        batch_size=getattr(config, "val_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(config, "pin_memory", True),
    )

    logger.success("Successfully built dataloader")
    logger.warning(
        f"SMILES overlap train/validation: {len(set(train_set.df.SMILES_Canonical_RDKit.tolist()) & set(val_set.df.SMILES_Canonical_RDKit.tolist()))}"
    )

    ######## MODEL ##################################################################################
    transformer = AutoModel.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    dnn = MultiTaskRegressionModule(
        input_layer_size=(
            getattr(config, "embedding_dim", 768)
            + (
                len(train_set[0]["one_hot_encoding"])
                if getattr(config, "onehot_column", None) is not None
                else 0
            )
            + (1 if getattr(config, "duration_column", None) is not None else 0)
        ),
        regression_task_configs=config.regression_task_config,
        fusion_network_config=getattr(config, "fusion_network_config", None),
    )

    mmtransformer = MultimodalRoBERTa(
        base_model=transformer,
        padding_idx=tokenizer.pad_token_id,
        prepend_external=True,
        taxonomic_embedding_dict=taxonomic_embedding_dict,
    )
    model = TRIDENT2(transformer_encoder=mmtransformer, dnn=dnn).to(
        getattr(config, "device", "cuda:0"),
    )

    del mmtransformer, dnn, transformer
    logger.success("Successfully built model\n")

    ######## TRAINING CONFIG ##################################################################################
    model_parameters = model.parameters()

    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=config.lr,
        betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)),
        eps=1e-08,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * config.epochs * len(train_dataloader),
        num_training_steps=config.epochs * len(train_dataloader),
    )
    if getattr(config, "mixed_precision_training", True):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    logger.success("Successfully built optimizer")

    if config.loss_function == "MSELoss":
        loss_function = nn.MSELoss()
    elif config.loss_function == "L1Loss":
        loss_function = nn.L1Loss()
    else:
        logger.warning("Loss function not matching, defaulting to L1Loss.")
        loss_function = nn.L1Loss()
    if (
        getattr(config, "apply_taxonomic_ranking_loss", False)
        and getattr(config, "taxonomic_tree", None) is not None
    ):
        logger.info("Applying taxonomic ranking loss")
        taxonomy_tree = Tree(
            config.taxonomic_tree,
            format=1,
        )
        taxonomic_ranking_loss = TaxonomicRankingLoss(
            lambda_hier=getattr(config, "lambda_hier", 0.01),
            taxonomic_tree=taxonomy_tree,
            rank_weights={
                "genus": 62,
                "family": 15,
                "order": 10,
                "class": 5,
                "phylum": 5,
                "kingdom": 2,
                "superkingdom": 1,
            },
            training_taxids=train_set.df[getattr(config, "taxid_column", None)]
            .unique()
            .tolist(),
            detach_species=getattr(
                config, "detach_species_in_taxonomic_ranking_loss", True
            ),
            sample_n_inputs=getattr(config, "taxonomic_ranking_loss_n_samples", 10),
        )
    else:
        taxonomic_ranking_loss = None
    logger.success("Successfully built loss function")

    best_val_loss = np.inf
    best_val_loss_norm = np.inf
    batch_num = [0, 0]
    global_step = 0

    # Log initial validation loss
    (
        avg_loss,
        avg_loss_norm,
        median_loss,
        median_loss_norm,
        _,
        batch_num,
        val_results,
    ) = evaluate(
        model=model,
        dataloader=val_dataloader,
        dataset=val_set.df,
        loss_fun=loss_function,
        batch_num=batch_num,
        epoch=-1,
        global_step=global_step - 1,
        device=getattr(config, "device", "cuda:0"),
        mixed_precision=getattr(config, "mixed_precision_training", True),
        log_to_wandb=getattr(config, "log_to_wandb", True),
    )

    if median_loss < best_val_loss:
        best_val_loss = median_loss

    get_running_stats(
        df=val_results,
        global_step=global_step - 1,
        log_to_wandb=getattr(config, "log_to_wandb", True),
    )

    ######## RUN TRAINING ##################################################################################
    logger.info("\nRunning epochs...")
    start_time = time.time()

    for epoch in tqdm(range(config.epochs)):
        avg_loss, median_loss, total_preds, total_labels, batch_num = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            loss_fun=loss_function,
            batch_num=batch_num,
            epoch=epoch,
            global_step=global_step,
            device=getattr(config, "device", "cuda:0"),
            mixed_precision=getattr(config, "mixed_precision_training", True),
            log_to_wandb=getattr(config, "log_to_wandb", True),
            constrain_endpoint_order=getattr(config, "constrain_endpoint_order", False),
            taxonomic_ranking_loss=taxonomic_ranking_loss,
        )

        (
            avg_loss,
            avg_loss_norm,
            median_loss,
            median_loss_norm,
            _,
            batch_num,
            val_results,
        ) = evaluate(
            model=model,
            dataloader=val_dataloader,
            dataset=val_set.df,
            loss_fun=loss_function,
            batch_num=batch_num,
            epoch=epoch,
            global_step=global_step,
            device=getattr(config, "device", "cuda:0"),
            mixed_precision=getattr(config, "mixed_precision_training", True),
            log_to_wandb=getattr(config, "log_to_wandb", True),
        )

        get_running_stats(
            df=val_results,
            global_step=global_step,
            log_to_wandb=getattr(config, "log_to_wandb", True),
        )

        # Update and log epoch results
        if median_loss_norm < best_val_loss_norm:
            best_val_loss = median_loss
            best_val_loss_norm = median_loss_norm
            best_validation_results = val_results
            best_validation_mean_norm_loss = avg_loss_norm
            best_epoch = epoch
            if getattr(config, "save_model", False) or getattr(
                config, "save_best_model", False
            ):
                save_ckp(
                    model=model,
                    checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_best_model",
                )

        # Manual save points
        if getattr(config, "manual_save_at_epochs", False) and epoch in getattr(
            config, "manual_save_at_epochs", []
        ):
            save_ckp(
                model=model,
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_epoch_{epoch}",
            )

        wandb.log(
            {
                "Best Validation Median Loss": best_val_loss,
                "Best Validation Median Loss Normalized": best_val_loss_norm,
                "Best Validation Mean Loss Normalized": best_validation_mean_norm_loss,
                "global_step": global_step,
            }
        )

        global_step += 1

    train_time = (time.time() - start_time) / 60

    wandb.log(
        {
            "Total train time (min)": train_time,
            "epoch time (s)": train_time / config.epochs * 60,
            "best epoch": best_epoch,
        }
    )

    logger.success(
        f"{'Metric':<15}{'Value':<10}\n"
        f"{'-' * 25}\n"
        f"{'Epochs: ':<15}{epoch:<10}\n"
        f"{'Best Median Loss: ':<15}{best_val_loss:<10.2f}\n"
        f"{'Best Median Loss Normalized: ':<15}{best_val_loss_norm:<10.2f}\n"
        f"{'Best Mean Loss Normalized: ':<15}{best_validation_mean_norm_loss:<10.2f}\n"
        f"{'Training Time (min): ':<15}{train_time:<10.2f}\n"
    )

    if getattr(config, "save_final_epoch", False):
        save_ckp(
            model,
            f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_final_epoch",
        )

    # Save results locally and plot predicted vs actual
    if getattr(config, "save_results", False):
        if getattr(config, "save_final_epoch", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_final_epoch",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_final_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config,
                train_results,
                val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if getattr(config, "save_best_model", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_best_model",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_best_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config,
                train_results,
                val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if len(getattr(config, "manual_save_at_epochs", [])) > 0:
            for epoch in getattr(config, "manual_save_at_epochs", []):
                model = load_ckp(
                    model=model,
                    device=getattr(config, "device", "cuda:0"),
                    checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_epoch_{epoch}",
                )
                val_results = get_eval_stats(
                    config, model, val_set, collate_fn, loss_function
                )
                train_results = get_eval_stats(
                    config, model, train_set, collate_fn, loss_function
                )
                val_results.to_pickle(
                    f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_epoch_{epoch}.pkl.zip",
                    compression="zip",
                )
                plot_predicted_vs_actual(
                    config,
                    train_results,
                    val_results,
                    normalize=True,
                    x="labels",
                    y="preds",
                )
        else:
            logger.warning(
                "No results saved locally as no save options were selected.\n"
            )

    return (
        best_val_loss,
        best_val_loss_norm,
        best_validation_mean_norm_loss,
        global_step,
        best_validation_results,
    )


def train_final_model(data, config):
    ######## DataLoading ##################################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    train_set = data

    logger.info(f"Train set size: {len(train_set)}")

    # Load taxonomic embedding dictionary if specified
    if getattr(config, "taxonomic_embedding_dict", None) is not None:
        taxonomic_embedding_dict = load_taxonomic_embedding_dict(
            config.taxonomic_embedding_dict
        )
        logger.success(
            f"Loaded taxonomic embedding dictionary with {len(taxonomic_embedding_dict)} entries"
        )
    else:
        logger.info("No taxonomic embedding dictionary provided")
        taxonomic_embedding_dict = None

    # Load parent dictionary if specified
    if getattr(config, "taxonomic_parent_dict", None) is not None:
        with open(config.taxonomic_parent_dict, "r", encoding="utf-8") as f:
            taxonomic_parent_dict = json.load(f)
        logger.success(
            f"Loaded parent dictionary with {len(taxonomic_parent_dict)} entries"
        )
    else:
        logger.info("No parent dictionary provided")
        taxonomic_parent_dict = None

    train_set = MultiModalDataset(
        df=train_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=getattr(config, "shuffle_smiles_prob", 0.0),
        drop_metadata_prob=getattr(config, "drop_metadata_prob", 0.0),
        lower_taxonomic_rank_prob=getattr(config, "lower_taxonomic_rank_prob", 0.0),
        parent_dict=taxonomic_parent_dict,
    )

    collate_fn = MultiModalCollator(
        tokenizer=tokenizer,
        max_len=getattr(config, "max_len", 512),
        padding=getattr(config, "padding", "longest"),
        truncation=getattr(config, "truncation", True),
        padding_idx=tokenizer.pad_token_id,
    )
    train_dataloader = build_dataloader(
        dataset=train_set,
        sampler=getattr(config, "training_sampler", "WeightedRandomSampler"),
        stratification_method=getattr(config, "stratification_method", None),
        sampler_weight_args=getattr(config, "sampler_weight_args", None),
        batch_size=getattr(config, "train_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(config, "pin_memory", True),
    )

    logger.success("Successfully built dataloader")

    ######## MODEL ##################################################################################

    transformer = AutoModel.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    dnn = MultiTaskRegressionModule(
        input_layer_size=(
            getattr(config, "embedding_dim", 768)
            + (
                len(train_set[0]["one_hot_encoding"])
                if getattr(config, "onehot_column", None) is not None
                else 0
            )
            + (1 if getattr(config, "duration_column", None) is not None else 0)
        ),
        regression_task_configs=config.regression_task_config,
        fusion_network_config=getattr(config, "fusion_network_config", None),
    )

    if transformer.config.model_type == "roberta":
        mmtransformer = MultimodalRoBERTa(
            base_model=transformer,
            padding_idx=tokenizer.pad_token_id,
            prepend_external=True,
            taxonomic_embedding_dict=taxonomic_embedding_dict,
        )
    else:
        raise NotImplementedError("Model not implemented")
    model = TRIDENT2(transformer_encoder=mmtransformer, dnn=dnn).to(
        getattr(config, "device", "cuda:0"),
    )

    del mmtransformer, dnn, transformer
    logger.success("Successfully built model\n")

    ######## TRAINING CONFIG ##################################################################################
    model_parameters = model.parameters()

    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=config.lr,
        betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)),
        eps=1e-08,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * config.epochs * len(train_dataloader),
        num_training_steps=config.epochs * len(train_dataloader),
    )
    if getattr(config, "mixed_precision_training", True):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    logger.success("Successfully built optimizer")

    if config.loss_function == "MSELoss":
        loss_function = nn.MSELoss()
    elif config.loss_function == "L1Loss":
        loss_function = nn.L1Loss()
    else:
        logger.warning("Loss function not matching, defaulting to L1Loss.")
        loss_function = nn.L1Loss()
    if (
        getattr(config, "apply_taxonomic_ranking_loss", False)
        and getattr(config, "taxonomic_tree", None) is not None
    ):
        logger.info("Applying taxonomic ranking loss")
        taxonomy_tree = Tree(
            config.taxonomic_tree,
            format=1,
        )
        taxonomic_ranking_loss = TaxonomicRankingLoss(
            lambda_hier=getattr(config, "lambda_hier", 0.01),
            taxonomic_tree=taxonomy_tree,
            rank_weights={
                "genus": 62,
                "family": 15,
                "order": 10,
                "class": 5,
                "phylum": 5,
                "kingdom": 2,
                "superkingdom": 1,
            },
            training_taxids=train_set.df[getattr(config, "taxid_column", None)]
            .unique()
            .tolist(),
            detach_species=getattr(
                config, "detach_species_in_taxonomic_ranking_loss", True
            ),
            sample_n_inputs=getattr(config, "taxonomic_ranking_loss_n_samples", 10),
        )
    else:
        taxonomic_ranking_loss = None
    logger.success("Successfully built loss function")

    batch_num = [0, 0]
    global_step = 0

    ######## RUN TRAINING ##################################################################################
    logger.info("\nRunning epochs...")
    start_time = time.time()

    for epoch in tqdm(range(config.epochs)):
        avg_loss, median_loss, total_preds, total_labels, batch_num = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            loss_fun=loss_function,
            batch_num=batch_num,
            epoch=epoch,
            global_step=global_step,
            device=getattr(config, "device", "cuda:0"),
            mixed_precision=getattr(config, "mixed_precision_training", True),
            log_to_wandb=getattr(config, "log_to_wandb", True),
            constrain_endpoint_order=getattr(config, "constrain_endpoint_order", False),
            taxonomic_ranking_loss=taxonomic_ranking_loss,
        )

        # Manual save points
        if getattr(config, "manual_save_at_epochs", False) and epoch in getattr(
            config, "manual_save_at_epochs", []
        ):
            checkpoint_dir = f"{config.model_save_pth}/{config.wandb_run_name}_final_model_epoch_{epoch}"
            save_ckp(
                model=model,
                checkpoint_dir=checkpoint_dir,
            )
            export_hf_trident_checkpoint(
                model=model,
                tokenizer=tokenizer,
                config=config,
                checkpoint_dir=checkpoint_dir,
                taxonomic_embedding_dict=taxonomic_embedding_dict,
            )

        global_step += 1

    train_time = (time.time() - start_time) / 60

    wandb.log(
        {
            "Total train time (min)": train_time,
            "epoch time (s)": train_time / config.epochs * 60,
        }
    )

    logger.info(
        f"{'Metric':<15}{'Value':<10}\n"
        f"{'-' * 25}\n"
        f"{'Epochs: ':<15}{epoch:<10}\n"
        f"{'Training Time (min): ':<15}{train_time:<10.2f}\n"
    )

    final_checkpoint_dir = (
        f"{config.model_save_pth}/{config.wandb_run_name}_final_model_final_epoch"
    )
    save_ckp(model, final_checkpoint_dir)
    export_hf_trident_checkpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=final_checkpoint_dir,
        taxonomic_embedding_dict=taxonomic_embedding_dict,
    )

    model = load_ckp(
        model=model,
        device=getattr(config, "device", "cuda:0"),
        checkpoint_dir=final_checkpoint_dir,
    )

    train_results = get_eval_stats(config, model, train_set, collate_fn, loss_function)

    # Save results locally
    if getattr(config, "save_results", False):
        train_results.to_pickle(
            f"{config.results_save_pth}/{config.wandb_run_name}_final_model_final_epoch_results.pkl.zip",
            compression="zip",
        )

    return global_step


def get_running_stats(df, global_step, log_to_wandb=True, verbose=False):
    metrics = []
    for endpoint in df.endpoint.unique():
        for species_group in df.species_group_corrected.unique():
            df_norm = calculate_weighted_avg(
                calculate_median_prediction_and_label(
                    df[
                        (df.species_group_corrected == species_group)
                        & (df.endpoint == endpoint)
                    ]
                )
            )

            if verbose:
                logger.info(
                    f"\033[1mStats for endpoint: {endpoint}_{species_group}\033[0m\n"
                    f"{'Metric':<15}{'Value':<10}\n"
                    f"{'-' * 25}\n"
                    f"{'Median residual: ':<15}{(df_norm.labels - df_norm.preds).median():<10.2f}\n"
                    f"{'Median AE: ':<15}{(df_norm.labels - df_norm.preds).abs().median():<10.2f}\n"
                    f"{'Mean AE: ':<15}{(df_norm.labels - df_norm.preds).abs().mean():<10.2f}\n"
                )

            if log_to_wandb:
                wandb.log(
                    {
                        f"{endpoint}/{endpoint} {species_group} median AE": (
                            df_norm.labels - df_norm.preds
                        )
                        .abs()
                        .median(),
                        "global_step": global_step,
                    }
                )

            metrics.append((df_norm.labels - df_norm.preds).abs().median())
    if log_to_wandb:
        metrics = np.array(metrics)
        metrics = metrics[~np.isnan(metrics)]
        mean_performance, std_performance = np.mean(metrics), np.std(metrics)
        wandb.log(
            {
                "mean_spgroup_val_median_abs_error": mean_performance,
                "std_spgroup_val_median_abs_error_std": std_performance,
                "global_step": global_step,
            }
        )


def get_eval_stats(
    config, model, data_set, collate_fn, loss_function, return_cls_embeddings=False
):

    # Build new pytorch dataset and dataloader manually
    dataloader = build_dataloader(
        dataset=data_set,
        sampler="SequentialSampler",
        stratification_method=None,
        sampler_weight_args=None,
        batch_size=getattr(config, "val_batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True),
        collate_fn=collate_fn,
    )

    _, _, _, _, _, _, results = evaluate(
        model=model,
        dataloader=dataloader,
        dataset=data_set.df,
        loss_fun=loss_function,
        batch_num=None,
        epoch=None,
        global_step=None,
        device=getattr(config, "device", "cuda:0"),
        mixed_precision=getattr(config, "mixed_precision_training", True),
        log_to_wandb=False,
    )

    if "CLS_embeddings" in results.columns and not return_cls_embeddings:
        results = results.drop(columns=["CLS_embeddings"])

    return results


def run_post_hoc_evaluation(config, data, train_ids: list = None, val_ids: list = None):
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    if (train_ids is not None) and (val_ids is not None):
        logger.info("Using provided train and validation identifiers to build datasets")
        train_set = data[data["SK_unique_id"].isin(train_ids)]
        val_set = data[data["SK_unique_id"].isin(val_ids)]
    else:
        logger.info(
            "No train and validation identifiers provided, building folds using k-fold split"
        )
        # Build folds of categorical identifiers (like SMILES and species)
        folds = GroupKFolds(
            n_splits=config.k_folds, seed=config.seed, shuffle=True
        ).split(
            df=data,
            group_cols=config.kfold_identifier,
            stratify_col=getattr(config, "kfold_stratification_identifier", None),
        )

        # Build train and validation sets for current fold
        train_idx = folds[f"fold_{config.fold_id}"]["train_idx"]
        val_idx = folds[f"fold_{config.fold_id}"]["val_idx"]

        train_set = data.iloc[train_idx]
        val_set = data.iloc[val_idx]

    # Load taxonomic embedding dictionary if specified
    if getattr(config, "taxonomic_embedding_dict", None) is not None:
        taxonomic_embedding_dict = load_taxonomic_embedding_dict(
            config.taxonomic_embedding_dict
        )
        logger.success(
            f"Loaded taxonomic embedding dictionary with {len(taxonomic_embedding_dict)} entries"
        )
    else:
        logger.info("No taxonomic embedding dictionary provided")
        taxonomic_embedding_dict = None

    # Load parent dictionary if specified
    if getattr(config, "taxonomic_parent_dict", None) is not None:
        with open(config.taxonomic_parent_dict, "r", encoding="utf-8") as f:
            taxonomic_parent_dict = json.load(f)
        logger.success(
            f"Loaded parent dictionary with {len(taxonomic_parent_dict)} entries"
        )
    else:
        logger.info("No parent dictionary provided")
        taxonomic_parent_dict = None

    train_set = MultiModalDataset(
        df=train_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=0.0,
        drop_metadata_prob=0.0,
        lower_taxonomic_rank_prob=0.0,
        parent_dict=taxonomic_parent_dict,
    )
    val_set = MultiModalDataset(
        df=val_set,
        smiles_column=getattr(config, "smiles_column", None),
        token_metadata_columns=getattr(config, "token_metadata_columns", None),
        taxid_column=getattr(config, "taxid_column", None),
        embedding_columns=getattr(config, "embedding_columns", None),
        duration_column=getattr(config, "duration_column", None),
        onehot_column=getattr(config, "onehot_column", None),
        label_column=getattr(config, "label_column", None),
        feature_column=getattr(config, "feature_column", None),
        endpoint_column=getattr(config, "endpoint_column", None),
        sep_token=tokenizer.special_tokens_map["sep_token"],
        shuffle_smiles_prob=0.0,
        drop_metadata_prob=0.0,
        lower_taxonomic_rank_prob=0.0,
        parent_dict=taxonomic_parent_dict,
    )

    collate_fn = MultiModalCollator(
        tokenizer=tokenizer,
        max_len=getattr(config, "max_len", 512),
        padding=getattr(config, "padding", "longest"),
        truncation=getattr(config, "truncation", True),
        padding_idx=tokenizer.pad_token_id,
    )

    transformer = AutoModel.from_pretrained(
        config.base_model,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True,
    )

    dnn = MultiTaskRegressionModule(
        input_layer_size=(
            getattr(config, "embedding_dim", 768)
            + (
                len(train_set[0]["one_hot_encoding"])
                if getattr(config, "onehot_column", None) is not None
                else 0
            )
            + (1 if getattr(config, "duration_column", None) is not None else 0)
        ),
        regression_task_configs=config.regression_task_config,
        fusion_network_config=getattr(config, "fusion_network_config", None),
    )

    if transformer.config.model_type == "roberta":
        mmtransformer = MultimodalRoBERTa(
            base_model=transformer,
            padding_idx=tokenizer.pad_token_id,
            prepend_external=True,
            taxonomic_embedding_dict=taxonomic_embedding_dict,
        )

    else:
        raise NotImplementedError("Model not implemented")
    model = TRIDENT2(transformer_encoder=mmtransformer, dnn=dnn).to(
        getattr(config, "device", "cuda:0"),
    )

    del mmtransformer, dnn, transformer
    logger.success("Successfully built model\n")

    if config.loss_function == "MSELoss":
        loss_function = nn.MSELoss()
    elif config.loss_function == "L1Loss":
        loss_function = nn.L1Loss()
    else:
        logger.warning("Loss function not matching, defaulting to L1Loss.")
        loss_function = nn.L1Loss()

    # Run post hoc evaluation
    logger.info("Running post-hoc evaluation...\n")
    # Save results locally and plot predicted vs actual
    if getattr(config, "save_results", False):
        if getattr(config, "save_final_epoch", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_final_epoch",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_final_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config=config,
                train_results=train_results,
                val_results=val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if getattr(config, "save_best_model", False):
            model = load_ckp(
                model=model,
                device=getattr(config, "device", "cuda:0"),
                checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_best_model",
            )
            val_results = get_eval_stats(
                config,
                model,
                val_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            train_results = get_eval_stats(
                config,
                model,
                train_set,
                collate_fn,
                loss_function,
                return_cls_embeddings=False,
            )
            val_results.to_pickle(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_best_epoch.pkl.zip",
                compression="zip",
            )
            plot_predicted_vs_actual(
                config=config,
                train_results=train_results,
                val_results=val_results,
                normalize=True,
                x="labels",
                y="preds",
            )
        if len(getattr(config, "manual_save_at_epochs", [])) > 0:
            for epoch in getattr(config, "manual_save_at_epochs", []):
                model = load_ckp(
                    model=model,
                    device=getattr(config, "device", "cuda:0"),
                    checkpoint_dir=f"{config.model_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_epoch_{epoch}",
                )
                val_results = get_eval_stats(
                    config, model, val_set, collate_fn, loss_function
                )
                train_results = get_eval_stats(
                    config, model, train_set, collate_fn, loss_function
                )
                val_results.to_pickle(
                    f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_val_data_epoch_{epoch}.pkl.zip",
                    compression="zip",
                )
                plot_predicted_vs_actual(
                    config=config,
                    train_results=train_results,
                    val_results=val_results,
                    normalize=True,
                    x="labels",
                    y="preds",
                )
        else:
            logger.warning(
                "No results saved locally as no save options were selected.\n"
            )


def plot_predicted_vs_actual(
    config=None,
    train_results=None,
    val_results=None,
    normalize=True,
    x="labels",
    y="preds",
):

    if normalize:
        if train_results is not None:
            train_results = calculate_weighted_avg(
                calculate_median_prediction_and_label(train_results)
            )
        if val_results is not None:
            val_results = calculate_weighted_avg(
                calculate_median_prediction_and_label(val_results)
            )

    # Create a figure for subplots
    fig, axes = plt.subplots(1, 2, figsize=(25, 16))
    rnm_name = (
        "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S%f")
    )
    try:
        if train_results is not None:
            g0 = plot_with_metrics(
                train_results.copy(),
                "Train: Predicted vs. Actual",
                color="teal",
                xlim=6,
                ylim=6,
                x=x,
                y=y,
            )
            g0.savefig(f"{rnm_name}_g0.png", dpi=300)
            axes[0].imshow(mpimg.imread(f"{rnm_name}_g0.png"))
        if val_results is not None:
            g1 = plot_with_metrics(
                val_results.copy(),
                "Validation: Predicted vs. Actual",
                color="coral",
                xlim=6,
                ylim=6,
                x=x,
                y=y,
            )
            g1.savefig(f"{rnm_name}_g1.png", dpi=300)
            axes[1].imshow(mpimg.imread(f"{rnm_name}_g1.png"))

        # turn off x and y axis
        [ax.set_axis_off() for ax in axes.ravel()]
        plt.tight_layout()
        plt.show()

        # Save results locally
        if getattr(config, "save_results", False):
            fig.savefig(
                f"{config.results_save_pth}/{config.wandb_run_name}_nfolds_{config.k_folds}_foldid_{config.fold_id}_pred_vs_actual.png",
                dpi=300,
            )

        if getattr(config, "log_to_wandb", True):
            # Log the figure to WandB
            wandb.log({"Prediction vs Actual (one per chemical)": wandb.Image(fig)})

        os.remove(f"{rnm_name}_g0.png")
        os.remove(f"{rnm_name}_g1.png")
    except Exception as e:
        logger.error(f"Error plotting predicted vs actual: {e}")
