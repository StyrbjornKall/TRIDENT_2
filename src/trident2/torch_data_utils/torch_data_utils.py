import tqdm as tqdm
import pandas as pd
import numpy as np
import random
from typing import List, TypeVar, Union, Dict, Tuple, Optional
import warnings
from loguru import logger

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    SequentialSampler,
    WeightedRandomSampler,
    RandomSampler,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold

from collections import Counter

from trident2.preprocessing.preprocess_data import enumerate_smiles

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class MakeTrainTestSplit:
    """
    Builds split (similar to folds) based on splitting all unique SMILES with train/val ratio.
    """

    def __init__(self):
        pass

    def split(
        self, smiles: pd.DataFrame, test_size: float = 0.2, seed: int = 42
    ) -> pd.DataFrame:

        unique_smiles = smiles.unique().tolist()

        split = pd.DataFrame(
            data=np.transpose(np.array([unique_smiles])), columns=["SMILES"]
        )

        if test_size != 0:
            train, val = train_test_split(
                unique_smiles, test_size=test_size, random_state=seed
            )

            split["train"] = split.SMILES.isin(train)
            split["val"] = split.SMILES.isin(val)
        else:
            split["train"] = True
            split["val"] = False

        return split


class GroupKFolds:
    """
    Wrapper for StratifiedGroupKFold and GroupKFold that works with one or more grouping columns.
    Returns a dictionary of folds instead of a generator.
    Each fold contains 'train_idx' and 'val_idx' (indices of the *original* dataframe).
    """

    def __init__(self, n_splits: int = 5, seed: int = 42, shuffle: bool = True):
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle

    def split(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        stratify_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split the dataframe into stratified group folds.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing the data.
        stratify_col : Optional[str]
            Column to use for stratification. If None, no stratification is done (uses GroupKFold).
        group_cols : str or list of str
            Column(s) to use for grouping.

        Returns
        -------
        folds : dict
            Dictionary of folds, each with keys 'train_idx' and 'val_idx' (original indices).
        """
        df = df.copy()  # avoid mutating caller’s dataframe
        df["__orig_idx"] = np.arange(len(df))  # track original indices

        # If no stratification, shuffle manually
        if self.shuffle:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Build groups *after* shuffling
        if isinstance(group_cols, list):
            groups = df[group_cols].astype(str).agg("_".join, axis=1)
        else:
            groups = df[group_cols].astype(str)

        y = df[stratify_col] if stratify_col else None

        # Choose the splitter
        if stratify_col:
            splitter = StratifiedGroupKFold(n_splits=self.n_splits)
        else:
            splitter = GroupKFold(n_splits=self.n_splits)

        folds = {}
        for i, (train_idx, val_idx) in enumerate(
            splitter.split(X=df, y=y, groups=groups)
        ):
            # Map back to original indices
            folds[f"fold_{i + 1}"] = {
                "train_idx": df.loc[train_idx, "__orig_idx"].values,
                "val_idx": df.loc[val_idx, "__orig_idx"].values,
            }

        return folds


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    sampler: str = "SequentialSampler",
    stratification_method: Union[str, None] = None,
    sampler_weight_args: Union[List[str], None] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory=True,
) -> torch.utils.data.DataLoader:
    """
    - dataset: PyTorch dataset.
    - sampler: supports ['RandomSampler', 'WeightedRandomSampler', 'SequentialSampler']
    - stratification_method: supports 'sqrt' or None. Defaults to None which leads to 1/weights being used if weighted sampler is chosen. If 'sqrt' sampler will use 1/sqrt(weight).
    - sampler_weight_args: list of columns to use for sampler weight defined in order [SMILES_col_name, effect_col_name, endpoint_col_name]
    - num_workers: number of workers to use in DataLoader
    """
    df = dataset.df.copy()

    if sampler == "SequentialSampler":
        logger.info(
            "SequentialSampler was chosen, disregarding any provided args for weighted sampler..."
        )
        sampler = SequentialSampler(dataset)
    elif sampler == "RandomSampler":
        logger.info(
            "RandomSampler was chosen, disregarding any provided args for weighted sampler..."
        )
        sampler = RandomSampler(dataset)
    elif sampler == "WeightedRandomSampler":
        if sampler_weight_args is None:
            raise (
                Exception("Specified WeightedRandomSampler but no args for weights.")
            )
        else:
            counts = Counter(zip(*(df[arg].tolist() for arg in sampler_weight_args)))
            weights = 1 / np.array(
                [
                    counts[i]
                    for i in zip(*(df[arg].tolist() for arg in sampler_weight_args))
                ]
            )

        if stratification_method == "sqrt":
            logger.info(
                f"Uses raw sqrt(1/weights) for weighted sampling based on {sampler_weight_args}."
            )
            weights = np.sqrt(weights)
        else:
            logger.info(
                f"Uses raw 1/weights for weighted sampling based on {sampler_weight_args}."
            )

        samples_weight = torch.from_numpy(np.array(weights))
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )
    else:
        raise (
            Exception(
                "Choose one of sampler ['RandomSampler', 'WeightedRandomSampler', 'SequentialSampler']"
            )
        )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    logger.info(f"Built dataloader with {len(dataset)} samples.")
    return dataloader


class MultiModalDataset(Dataset):
    # Default mapping from known token_metadata_columns entries to the missingness token
    # that preprocess_data() inserts for genuinely missing values (see preprocess_data.py).
    # Used so that the drop_metadata_prob augmentation substitutes the *correct* missingness
    # token for the dropped column instead of erasing it from the input text entirely.
    DEFAULT_METADATA_MISSING_TOKENS = {
        "effect": "<missing_effect>",
        "administration_route_categorized": "<missing_administration_route>",
        "organism_lifestage_categorized": "<missing_lifestage>",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_column: Optional[str] = None,
        token_metadata_columns: Optional[List[str]] = None,
        taxid_column: Optional[str] = None,
        embedding_columns: Optional[List[str]] = None,
        duration_column: Optional[str] = None,
        onehot_column: Optional[str] = None,
        label_column: Optional[str] = None,
        feature_column=None,
        endpoint_column: Optional[str] = None,
        sep_token: str = "</s>",
        shuffle_smiles_prob: float = 0.0,
        drop_metadata_prob: float = 0.0,
        lower_taxonomic_rank_prob: float = 0.0,
        parent_dict: Optional[Dict[str, str]] = None,
        metadata_missing_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): The dataframe containing all the data.
            smiles_column: input smiles.
            token_metadata_columns: List of columns for token metadata (optional).
            taxid_column: Name of the taxid column (optional).
            embedding_columns: List of columns for embeddings (optional).
            duration_column: Name of the duration column (optional).
            onehot_column: Name of the one-hot encoding column (optional).
            label_column: Name of the label column.
            feature_column: Name of the feature column (optional).
            endpoint_column: Name of the endpoint column (optional).
            sep_token (str): Token used to separate concatenated text columns.
            shuffle_smiles_prob (float): Probability of shuffling SMILES strings.
            drop_metadata_prob (float): Probability of dropping metadata columns.
            lower_taxonomic_rank_prob (float): Probability of replacing taxid with a higher rank.
            parent_dict (dict): Dictionary mapping taxid to its parent taxid (required if lower_taxonomic_rank_prob > 0).
            metadata_missing_tokens (dict): Mapping from token_metadata_columns entries to the
                missingness token that should be substituted in when that column's value is
                dropped via drop_metadata_prob augmentation (e.g. {"effect": "<missing_effect>"}).
                Falls back to DEFAULT_METADATA_MISSING_TOKENS for any column not provided.
        """
        self.df = df.copy().reset_index(drop=True)
        self.sep_token = sep_token

        # Augmentation settings
        self.shuffle_smiles_prob = shuffle_smiles_prob
        self.drop_metadata_prob = drop_metadata_prob
        self.lower_taxonomic_rank_prob = lower_taxonomic_rank_prob
        self.parent_dict = parent_dict

        # Column names
        self.smiles_column = smiles_column
        self.token_metadata_columns = token_metadata_columns
        self.taxid_column = taxid_column
        self.embedding_columns = embedding_columns
        self.duration_column = duration_column
        self.onehot_column = onehot_column
        self.label_column = label_column
        self.feature_column = feature_column
        self.endpoint_column = endpoint_column

        # Build per-column missingness-token lookup for the drop_metadata_prob augmentation,
        # combining any user-provided overrides with the defaults.
        self.metadata_missing_tokens = {
            **self.DEFAULT_METADATA_MISSING_TOKENS,
            **(metadata_missing_tokens or {}),
        }
        if self.token_metadata_columns:
            missing_mapping = [
                c
                for c in self.token_metadata_columns
                if c not in self.metadata_missing_tokens
            ]
            if missing_mapping:
                logger.warning(
                    f"No missingness token mapping found for token_metadata_columns {missing_mapping}. "
                    "Dropped metadata for these columns will be omitted from the input text instead of "
                    "replaced with a missingness token. Pass `metadata_missing_tokens` to avoid this."
                )

        # Pre-extract columns into arrays
        self.smiles = df[smiles_column].values if smiles_column else None
        self.metadata_arrays = (
            [df[c].values for c in self.token_metadata_columns]
            if self.token_metadata_columns
            else None
        )
        self.taxids = df[taxid_column].values if taxid_column else None
        self.embeddings_arrays = (
            [df[c].values for c in embedding_columns] if embedding_columns else None
        )
        self.durations = df[duration_column].values if duration_column else None
        self.onehots = df[onehot_column].values if onehot_column else None
        self.features = df[feature_column].values if feature_column else None
        self.endpoints = df[endpoint_column].values if endpoint_column else None
        self.labels = df[label_column].values if label_column else None

        # Logic checks
        # If lower_taxonomic_rank_prob is more than 0, parent_dict must also be provided
        if self.lower_taxonomic_rank_prob > 0.0 and self.parent_dict is None:
            raise ValueError(
                "parent_dict must be provided if lower_taxonomic_rank_prob is greater than 0."
            )
        # If parent_dict is provided, notify user if lower_taxonomic_rank_prob is 0
        if self.parent_dict is not None and self.lower_taxonomic_rank_prob == 0.0:
            logger.warning(
                "Warning: parent_dict is provided but lower_taxonomic_rank_prob is 0. Taxonomic rank lowering will not be applied."
            )

        # If no metadata is specified, the dropout probability is set to 0
        if (
            self.token_metadata_columns is None or len(self.token_metadata_columns) <= 1
        ) and self.drop_metadata_prob > 0.0:
            logger.warning(
                "No metadata columns specified, setting drop_metadata_prob to 0.0"
            )
            self.drop_metadata_prob = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Build text input
        texts = []

        if self.metadata_arrays:
            for col, arr in zip(self.token_metadata_columns, self.metadata_arrays):
                metadata = arr[idx]
                if np.random.rand() < self.drop_metadata_prob:
                    # Substitute the missingness token specific to this modality/column so the
                    # tokenizer/model sees the same signal it would for genuinely missing data,
                    # instead of silently dropping the value from the input text.
                    metadata = self.metadata_missing_tokens.get(col)
                if metadata is not None and not pd.isna(metadata):
                    texts.append(metadata)
            # Add a sep token after this to show the model that this was a separate modality
            texts.append(self.sep_token)

        # Append SMILES last to avoid training adaptive positional embeddings for metadata
        if self.smiles is not None:
            smiles = self.smiles[idx]
            if np.random.rand() < self.shuffle_smiles_prob:
                smiles = enumerate_smiles(smiles)  # keep this function same
            texts.append(smiles)

        input_text = "".join([t for t in texts if t])

        # External embeddings (kept as numpy arrays)
        if self.embeddings_arrays:
            embeds = [
                arr[idx] if arr[idx] is not None else None
                for arr in self.embeddings_arrays
            ]

        # Taxonomic ID (with optional augmentation to lower rank)
        if self.taxids is not None:
            taxid = self.taxids[idx]
            # Apply taxonomic rank augmentation: move down the taxonomy tree
            if np.random.rand() < self.lower_taxonomic_rank_prob:
                move_down_n_steps = random.choices(
                    [1, 2, 3, 4, 5, 6, 7], weights=[62, 15, 10, 5, 5, 2, 1], k=1
                )[0]
                for _ in range(move_down_n_steps):
                    taxid = self.parent_dict.get(taxid, taxid)

        item = {
            "input_text": input_text,
            "external_embeds": embeds if self.embeddings_arrays is not None else None,
            "taxid": taxid if self.taxids is not None else None,
            "duration": [self.durations[idx]]
            if self.durations is not None
            else [np.nan],
            "one_hot_encoding": self.onehots[idx] if self.onehots is not None else None,
            "featurized_inputs": self.features[idx]
            if self.features is not None
            else None,
            "endpoint": self.endpoints[idx] if self.endpoints is not None else None,
            "labels": self.labels[idx] if self.labels is not None else np.nan,
        }
        return item


class MultiModalCollator:
    def __init__(
        self,
        tokenizer,
        max_len: int = 512,
        padding: str = "longest",
        truncation: bool = True,
        padding_idx: int = 1,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Hugging Face tokenizer for text tokenization.
            max_len (int): Maximum token length.
            padding (str): Padding strategy for tokenization.
            truncation (bool): Whether to truncate sequences.
            embedding_dim (int): Dimensionality of the external embeddings.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        self.padding_idx = padding_idx

    def __call__(self, batch):
        # Labels - return tensor with possible NaN values
        labels = [item["labels"] for item in batch]
        labels = torch.tensor(labels, dtype=torch.float32)

        # External embeddings - replace None with zero tensors if not all are None
        external_embeds = None
        if all(item["external_embeds"] is not None for item in batch):
            # each item['external_embeds'] is a list of numpy arrays
            emb_list = [
                np.stack(item["external_embeds"]).astype(np.float32) for item in batch
            ]
            external_embeds = torch.tensor(np.stack(emb_list))

        # One-hot
        one_hot_encoding = [item["one_hot_encoding"] for item in batch]
        one_hot_encoding = (
            torch.tensor(np.stack(one_hot_encoding), dtype=torch.float32)
            if None not in one_hot_encoding
            else None
        )

        # Durations - return tensor with possible NaN values
        durations = [item["duration"] for item in batch]
        durations = torch.tensor(durations, dtype=torch.float32)

        # Endpoints - return list including None values, set to None only if all are None
        endpoints = [item["endpoint"] for item in batch]
        endpoints = endpoints if any(endpoints) else None

        # Taxids, only set to None if all taxids are None
        taxids = [item["taxid"] for item in batch]
        taxids = taxids if any(taxids) else None

        # Tokenize
        texts = [item["input_text"] for item in batch]

        encodings = self.tokenizer.batch_encode_plus(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_len,
            return_tensors="pt",
        )
        encodings["attention_mask"][encodings["input_ids"] == self.padding_idx] = 0

        # Build the batch dictionary
        batch_output = {}
        if encodings is not None:
            batch_output["input_ids"] = encodings["input_ids"]
            batch_output["attention_mask"] = encodings["attention_mask"]
        if taxids is not None:
            batch_output["taxids"] = taxids
        if external_embeds is not None:
            batch_output["external_embeds"] = external_embeds
        if one_hot_encoding is not None:
            batch_output["one_hot_encoding"] = one_hot_encoding
        if durations is not None:
            batch_output["duration"] = durations
        if labels is not None:
            batch_output["labels"] = labels
        if endpoints is not None:
            batch_output["endpoint"] = endpoints

        return batch_output
