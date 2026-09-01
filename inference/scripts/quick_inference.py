"""
Command-line interface for TRIDENT-2.

This module provides the main entry point for running TRIDENT-2
inference from the command line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import types
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from dotenv import load_dotenv

from trident2_inference.inference.torch_dataset import (
    MultiModalDataset,
    MultiModalCollator,
)
from trident2_inference.inference.torch_loaders import build_dataloader
from trident2_inference.inference.model import build_model_and_tokenizer
from trident2_inference.inference.preprocess_input import (
    generate_inference_metadata,
    validate_inference_effect,
    validate_administration_route,
    validate_organism_lifestage,
)
from trident2_inference.inference.model import predict
from trident2_inference.inference.taxonomy import load_taxonomic_information
from trident2_inference.utils.chem_utils import (
    canonicalize_smiles,
    calculate_rdkit_descriptors,
)
from trident2_inference.utils.setup_logger import setup_logger, log_memory_usage

load_dotenv()  # Load environment variables from .env file

TAXID_TO_NAMES = os.getenv("TAXA_NAMES", "data/taxid2sciname.json")
TAXID_TO_SPECIES_GROUP = os.getenv("TAXA_GROUPS", "data/taxid2spgroup.json")
STANDARD_INFERENCE_SETTINGS = os.getenv(
    "MAPPING", "data/taxa_assigned_units_effects_durations_2026-04-17.csv"
)
MODEL = os.getenv("MODEL", "StyrbjornKall/TRIDENT-2")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with TRIDENT transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Table mode: single CSV/Excel/TXT file with SMILES and taxid columns
    quick_inference.py --data_pth data.csv --output_pth results.csv

    # List mode: separate SMILES and taxid files (cartesian product)
    quick_inference.py --smiles_file smiles.txt --taxid_file taxids.txt --output_pth results.csv

    # Using a config file (all flags can also come from JSON)
    quick_inference.py --config_file_path config.json --mixed_precision
        """,
    )

    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--config_file_path",
        type=str,
        help="Path to config file (JSON). Command-line args override config values.",
    )
    io_group.add_argument(
        "--data_pth",
        type=str,
        help="Path to input data (CSV, Excel, TXT) with SMILES and taxid columns.",
    )
    io_group.add_argument(
        "--smiles_file",
        type=str,
        help="Plain-text file with one SMILES per line. "
        "Combined with --taxid_file as a cartesian product.",
    )
    io_group.add_argument(
        "--taxid_file",
        type=str,
        help="Plain-text file with one NCBI taxid per line. "
        "Combined with --smiles_file as a cartesian product.",
    )
    io_group.add_argument(
        "--output_pth", type=str, help="Path to save predictions (CSV, Excel, TXT)"
    )

    # Taxonomic data arguments
    tax_group = parser.add_argument_group("Taxonomic Information")
    tax_group.add_argument(
        "--taxid2speciesname_dict_path",
        type=str,
        default=TAXID_TO_NAMES,
        help="JSON file mapping taxids to latin names",
    )
    tax_group.add_argument(
        "--taxid2speciesgroup_dict_path",
        type=str,
        default=TAXID_TO_SPECIES_GROUP,
        help="JSON file mapping taxids to species groups",
    )
    tax_group.add_argument(
        "--inference_mapping_df_path",
        type=str,
        default=STANDARD_INFERENCE_SETTINGS,
        help="CSV/pickle with taxa to effect/duration/unit mappings",
    )

    # Column specification arguments
    col_group = parser.add_argument_group("Column Specifications")
    col_group.add_argument(
        "--ncbi_taxid_column",
        type=str,
        help="Column containing taxids (default: ncbi_taxid)",
    )
    col_group.add_argument(
        "--SMILES_column",
        type=str,
        help="Column containing SMILES strings (default: SMILES)",
    )
    col_group.add_argument(
        "--species_column",
        type=str,
        help="Column containing species names (alternative to taxids)",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model identifier (default: StyrbjornKall/TRIDENT-2)",
    )
    model_group.add_argument(
        "--batch_size", type=int, help="Batch size for inference (default: 64)"
    )
    model_group.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision (fp16) for faster inference",
    )
    model_group.add_argument(
        "--return_CLS_embeddings",
        action="store_true",
        help="Return CLS embeddings along with predictions",
    )

    # Inference metadata arguments
    # Each of these accepts one or more values. When more than one value is
    # given (across any of the five fields, in any combination), the input
    # data is expanded via a cartesian (cross) join so that predictions are
    # generated for every combination, e.g. --inference_effect MOR ITX
    # produces predictions for both effects for every SMILES x taxid pair.
    meta_group = parser.add_argument_group("Inference Metadata")
    meta_group.add_argument(
        "--inference_administration_route",
        type=str,
        nargs="+",
        help="Administration route(s)",
    )
    meta_group.add_argument(
        "--inference_organism_lifestage_categorized",
        type=str,
        nargs="+",
        help="Organism lifestage(s)",
    )
    meta_group.add_argument(
        "--inference_effect", type=str, nargs="+", help="Effect type(s)"
    )
    meta_group.add_argument(
        "--inference_duration", type=float, nargs="+", help="Exposure duration(s)"
    )
    meta_group.add_argument(
        "--inference_conc_unit", type=str, nargs="+", help="Concentration unit(s)"
    )

    # Runtime arguments
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited data (100 samples)",
    )
    runtime_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    return args


def validate_data_schema(df: pd.DataFrame, config) -> None:
    """Validate that data has required columns."""
    logger.info("Validating data schema...")

    has_taxid = config.ncbi_taxid_column in df.columns
    _species_col = getattr(config, "species_column", None)
    has_species = _species_col and _species_col in df.columns

    if not has_taxid and not has_species:
        raise ValueError(
            f"Data must contain either '{config.ncbi_taxid_column}' or species column"
        )

    if config.SMILES_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"SMILES column '{config.SMILES_column}' not found. Available: {available}"
        )

    logger.info("Data schema validated successfully")


def load_data(config) -> pd.DataFrame:
    """Load and validate input data from a table file (CSV, Excel, or TXT)."""
    logger.info(f"Loading data from {config.data_pth}")

    if config.data_pth.endswith(".csv"):
        df = pd.read_csv(config.data_pth)
    elif config.data_pth.endswith(".xlsx") or config.data_pth.endswith(".xls"):
        df = pd.read_excel(config.data_pth)
    elif config.data_pth.endswith(".txt"):
        df = pd.read_csv(config.data_pth, delimiter="\t")
    else:
        raise ValueError("Data path must be a CSV, Excel, or TXT file")

    logger.info(f"Loaded {len(df)} samples")
    log_memory_usage(logger, "after data loading")

    if config.debug:
        df = df.iloc[:100].reset_index(drop=True)
        logger.info("DEBUG MODE: Limited to 100 samples")

    validate_data_schema(df, config)
    return df


def load_data_from_lists(config) -> pd.DataFrame:
    """Build an input DataFrame from separate SMILES and taxid list files.

    Reads one SMILES per line from ``config.smiles_file`` and one NCBI taxid
    per line from ``config.taxid_file``, then expands them into a full
    cartesian product (every SMILES × every taxid).

    Args:
        config: Inference config namespace with ``smiles_file``, ``taxid_file``,
            and optionally ``debug`` attributes.

    Returns:
        DataFrame with columns ``SMILES`` and ``ncbi_taxid``.
    """
    with open(config.smiles_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    with open(config.taxid_file) as f:
        taxid_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(smiles_list)} SMILES from {config.smiles_file}")
    logger.info(f"Loaded {len(taxid_list)} taxids from {config.taxid_file}")

    combinations = list(itertools.product(smiles_list, taxid_list))
    logger.info(
        f"Cartesian product: {len(smiles_list)} × {len(taxid_list)} = "
        f"{len(combinations):,} combinations"
    )

    df = pd.DataFrame(combinations, columns=["SMILES", "ncbi_taxid"])

    if getattr(config, "debug", False):
        df = df.iloc[:100].reset_index(drop=True)
        logger.info("DEBUG MODE: Limited to 100 combinations")

    return df


def build_metadata_combinations(config) -> pd.DataFrame:
    """Build the cartesian product of all supplied --inference_* metadata values.

    Each of ``inference_administration_route``, ``inference_organism_lifestage_categorized``,
    ``inference_effect``, ``inference_duration``, and ``inference_conc_unit`` may be a
    single value, a list of values, or None (in config/CLI). This expands every
    combination across all five fields into one row per combination, so the
    result can be cross-joined with the input data.

    Args:
        config: Inference config namespace, whose metadata attributes may be
            scalars, lists, or None.

    Returns:
        DataFrame with one row per combination and columns
        ``inference_administration_route_categorized``,
        ``inference_organism_lifestage_categorized``, ``inference_effect``,
        ``inference_duration``, ``inference_conc_unit``.
    """
    fields = {
        "inference_administration_route_categorized": getattr(
            config, "inference_administration_route", None
        ),
        "inference_organism_lifestage_categorized": getattr(
            config, "inference_organism_lifestage_categorized", None
        ),
        "inference_effect": getattr(config, "inference_effect", None),
        "inference_duration": getattr(config, "inference_duration", None),
        "inference_conc_unit": getattr(config, "inference_conc_unit", None),
    }

    # Normalize each field to a non-empty list (None stays as [None])
    normalized = {
        key: (value if isinstance(value, list) and len(value) > 0 else [value])
        for key, value in fields.items()
    }

    n_combos = 1
    for values in normalized.values():
        n_combos *= len(values)
    if n_combos > 1:
        logger.info(
            f"Multiple values supplied for inference metadata: "
            f"expanding into {n_combos} combinations (cartesian product)"
        )

    combos = list(itertools.product(*normalized.values()))
    return pd.DataFrame(combos, columns=list(normalized.keys()))


def cross_join(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Cross (cartesian) join a data DataFrame with a metadata combinations DataFrame.

    Args:
        df: Input data (one row per SMILES x taxid combination).
        metadata_df: One row per inference metadata combination, as produced
            by ``build_metadata_combinations``.

    Returns:
        DataFrame with ``len(df) * len(metadata_df)`` rows.
    """
    df = df.copy()
    metadata_df = metadata_df.copy()
    df["_cross_key"] = 1
    metadata_df["_cross_key"] = 1
    merged = df.merge(metadata_df, on="_cross_key").drop(columns=["_cross_key"])
    return merged.reset_index(drop=True)


def save_outputs(
    df_out: pd.DataFrame,
    cls_mapping: Optional[pd.DataFrame],
    config,
) -> None:
    """Save predictions and embeddings to disk."""
    logger.info("Saving outputs...")

    output_dir = os.path.dirname(config.output_pth)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Save main predictions
    if config.output_pth.endswith(".csv") or config.output_pth.endswith(".csv.zip"):
        compression = "zip" if config.output_pth.endswith(".zip") else None
        df_out.to_csv(config.output_pth, index=False, compression=compression)
    elif config.output_pth.endswith(".xlsx") or config.output_pth.endswith(".xls"):
        df_out.to_excel(config.output_pth, index=False)
    elif config.output_pth.endswith(".txt") or config.output_pth.endswith(".txt.zip"):
        compression = "zip" if config.output_pth.endswith(".zip") else None
        df_out.to_csv(config.output_pth, index=False, sep="\t", compression=compression)
    else:
        raise ValueError("Output path must be CSV, Excel, TXT file")

    logger.info(f"Output shape: {df_out.shape}")
    logger.success(f"Saved predictions to {config.output_pth}")

    # Log prediction statistics
    logger.info("Prediction Summary:")
    for endpoint in ["EC50", "EC10", "NOEC", "LOEC"]:
        if endpoint in df_out.columns:
            values = df_out[endpoint].dropna()
            if len(values) > 0:
                logger.info(
                    f"  {endpoint}: min={values.min():.3f}, max={values.max():.3f}, "
                    f"mean={values.mean():.3f}, median={values.median():.3f}"
                )

    # Save CLS embeddings if applicable
    if getattr(config, "return_CLS_embeddings", False) and cls_mapping is not None:
        cls_path = (
            config.output_pth.replace(".csv", "")
            .replace(".pkl", "")
            .replace(".zip", "")
        )
        cls_path += "_CLS_embedding_mapping.pkl.zip"
        cls_mapping.to_pickle(cls_path, compression="zip")
        logger.success(f"Saved CLS embedding mapping to {cls_path}")


def main() -> None:
    """Main entry point for TRIDENT-2 inference."""
    args = parse_arguments()

    # Load config file if provided
    config_dict = {}
    if args.config_file_path:
        with open(args.config_file_path) as f:
            config_dict = json.load(f)

    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config_file_path":
            config_dict[key] = value

    # Create config object from merged dict
    config = types.SimpleNamespace(**config_dict)

    # Apply defaults for attributes not supplied by config file or CLI
    if not getattr(config, "ncbi_taxid_column", None):
        config.ncbi_taxid_column = "ncbi_taxid"
    if not getattr(config, "SMILES_column", None):
        config.SMILES_column = "SMILES"
    if not getattr(config, "batch_size", None):
        config.batch_size = 64
    if not getattr(config, "model", None):
        config.model = (
            "/home/skall/trident2_inference/models/"
            "sxno3jhi_final_model_epoch_70_hf_trident2"
        )
    if not getattr(config, "debug", None):
        config.debug = False
    if not getattr(config, "output_pth", None):
        if getattr(config, "data_pth", None):
            config.output_pth = (
                config.data_pth.replace(".csv", "_predictions.csv.zip")
                .replace(".pkl", "_predictions.pkl")
                .replace(".xlsx", "_predictions.xlsx")
                .replace(".xls", "_predictions.xls")
            )
        elif getattr(config, "smiles_file", None):
            base = os.path.splitext(config.smiles_file)[0]
            config.output_pth = f"{base}_predictions.csv"

    # Validate that at least one input mode is specified
    has_table = getattr(config, "data_pth", None)
    has_lists = getattr(config, "smiles_file", None) and getattr(
        config, "taxid_file", None
    )
    if not has_table and not has_lists:
        raise ValueError(
            "Specify either --data_pth (table mode) or both "
            "--smiles_file and --taxid_file (list mode)."
        )

    # Validate --inference_effect / --inference_administration_route /
    # --inference_organism_lifestage_categorized values
    validate_inference_effect(getattr(config, "inference_effect", None))
    validate_administration_route(
        getattr(config, "inference_administration_route", None)
    )
    validate_organism_lifestage(
        getattr(config, "inference_organism_lifestage_categorized", None)
    )

    # Setup logging
    logger = setup_logger(level="DEBUG" if getattr(config, "debug", False) else "INFO")
    logger.info("=" * 60)
    logger.info("Starting TRIDENT-2 inference pipeline")
    logger.info("=" * 60)

    try:
        # Setup device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if device == "cpu":
            logger.warning("Running on CPU - this will be slow")
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load data
        start_time = time.time()
        if getattr(config, "smiles_file", None) and getattr(config, "taxid_file", None):
            df = load_data_from_lists(config)
            # List-mode always uses the standard column names
            config.SMILES_column = "SMILES"
            config.ncbi_taxid_column = "ncbi_taxid"
        else:
            df = load_data(config)

        # Add metadata columns (cross-joined so every --inference_* combination
        # is predicted, e.g. --inference_effect MOR ITX doubles the data)
        logger.info("Adding metadata columns")
        metadata_df = build_metadata_combinations(config)
        df = cross_join(df, metadata_df)
        logger.info(f"Dataset size after metadata expansion: {len(df)} samples")

        # Load taxonomic information
        logger.info("Loading taxonomic information")
        (taxid2speciesgroup_dict, taxid2speciesname_dict, taxid2lineage_df) = (
            load_taxonomic_information(
                taxid2speciesgroup_dict_path=getattr(
                    config, "taxid2speciesgroup_dict_path", None
                ),
                taxid2speciesname_dict_path=getattr(
                    config, "taxid2speciesname_dict_path", None
                ),
                taxid2lineage_csv_path=getattr(config, "taxid2lineage_csv_path", None),
            )
        )

        # Process taxid column
        ncbi_taxid_column = config.ncbi_taxid_column
        if ncbi_taxid_column in df.columns:
            df[ncbi_taxid_column] = df[ncbi_taxid_column].astype(str)
            df["inference_ncbi_taxid"] = df[ncbi_taxid_column].copy()
            logger.info(f"Using taxid column: {ncbi_taxid_column}")
        elif taxid2speciesname_dict is not None and getattr(
            config, "species_column", None
        ):
            speciesname2taxid = {v: k for k, v in taxid2speciesname_dict.items()}
            df["inference_ncbi_taxid"] = df[config.species_column].map(
                lambda x: speciesname2taxid.get(x)
            )
            missing = df["inference_ncbi_taxid"].isna().sum()
            if missing > 0:
                logger.warning(f"Dropping {missing} samples with missing taxids")
                df = df[~df["inference_ncbi_taxid"].isna()].reset_index(drop=True)
        else:
            raise ValueError("TaxID column not found and cannot map species names")

        # Add species metadata
        if taxid2speciesgroup_dict:
            df["inference_species_group"] = df["inference_ncbi_taxid"].map(
                lambda x: taxid2speciesgroup_dict.get(x, "unknown")
            )
        if taxid2speciesname_dict:
            df["inference_species_name"] = df["inference_ncbi_taxid"].map(
                lambda x: taxid2speciesname_dict.get(x, "unknown")
            )

        # Merge lineage information
        if taxid2lineage_df is not None:
            df = df.merge(
                taxid2lineage_df,
                how="left",
                left_on="inference_ncbi_taxid",
                right_on="NCBI_rank_species",
            )

        # Load inference mapping
        inference_mapping_df = None
        if getattr(config, "inference_mapping_df_path", None):
            logger.info(
                f"Loading inference mapping from {config.inference_mapping_df_path}"
            )
            inference_mapping_df = pd.read_csv(config.inference_mapping_df_path)

        # Generate inference metadata
        logger.info("Generating inference metadata")
        df = generate_inference_metadata(
            df=df, inference_mapping_df=inference_mapping_df
        )

        # Canonicalize SMILES
        logger.info("Canonicalizing SMILES strings")
        smiles_column = getattr(config, "SMILES_column", "SMILES")
        df["inference_smiles"] = canonicalize_smiles(
            df[smiles_column].tolist(),
            canonical=True,
            isomericSmiles=False,
            drop_erroneous_smiles=True,
        )

        invalid_count = df["inference_smiles"].isna().sum()
        if invalid_count > 0:
            logger.warning(f"Dropping {invalid_count} samples with invalid SMILES")
            df = df[~df["inference_smiles"].isna()].reset_index(drop=True)

        logger.info(f"Final dataset size: {len(df)} samples")

        # Build model and tokenizer
        logger.info("Building model and tokenizer")
        model, tokenizer = build_model_and_tokenizer(
            model_name=config.model,
            device=device,
        )
        logger.success(f"Model and tokenizer loaded successfully from {config.model}")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        # Build dataset and dataloader
        logger.info("Building dataset and dataloader")
        mm_dataset = MultiModalDataset(
            df=df,
            smiles_column="inference_smiles",
            token_metadata_columns=[
                "inference_effect",
                "inference_administration_route_categorized",
                "inference_organism_lifestage_categorized",
            ],
            taxid_column="inference_ncbi_taxid",
            duration_column="inference_duration",
            onehot_column="inference_onehot",
            sep_token=tokenizer.special_tokens_map["sep_token"],
        )

        collate_fn = MultiModalCollator(
            tokenizer=tokenizer,
            max_len=200,
            padding="longest",
            truncation=True,
            padding_idx=tokenizer.pad_token_id,
        )

        dataloader = build_dataloader(
            dataset=mm_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        logger.success(f"Created dataloader with {len(dataloader)} batches")

        # Run inference
        logger.info(
            f"Running inference: batch_size={config.batch_size}, "
            f"mixed_precision={getattr(config, 'mixed_precision', False)}"
        )
        inference_start = time.time()

        preds, cls_embeddings = predict(
            model=model,
            dataloader=dataloader,
            device=device,
            mixed_precision=getattr(config, "mixed_precision", False),
            return_cls_embeddings=getattr(config, "return_CLS_embeddings", False),
        )

        inference_time = time.time() - inference_start
        samples_per_sec = len(df) / inference_time

        logger.success("Inference completed")
        logger.info(f"Inference time: {inference_time:.2f} seconds")
        logger.info(f"Inference speed: {samples_per_sec:.2f} samples/second")

        # Prepare output
        logger.info("Preparing output files")
        df_preds = pd.DataFrame(preds)
        df_out = pd.concat([df.reset_index(drop=True), df_preds], axis=1)

        # Convert duration back from log10(hours) to plain integer hours.
        # Rows where duration was originally missing (flagged via the
        # duration-missing one-hot) are restored to NaN rather than the
        # 1e-6 sentinel.
        logger.info("Converting inference_duration from log10 to integer hours")
        missing_duration_mask = df_out["inference_onehot_duration_missing"].apply(
            lambda x: x[0] == 1
        )
        df_out["inference_duration"] = (
            (10 ** df_out["inference_duration"]).round().astype("Int64")
        )
        df_out.loc[missing_duration_mask, "inference_duration"] = pd.NA

        # Strip the "<...>" tokenizer wrapping from the effect column
        df_out["inference_effect"] = (
            df_out["inference_effect"]
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
        )

        # Drop one-hot encoding columns - not needed in the saved output
        df_out = df_out.drop(
            columns=[
                "inference_onehot_conc_unit",
                "inference_onehot_duration_missing",
                "inference_onehot",
            ]
        )

        cls_mapping = None
        if getattr(config, "return_CLS_embeddings", False):
            df_out["CLS_embedding"] = cls_embeddings.tolist()
            cls_mapping = df_out[
                [
                    "inference_smiles",
                    "inference_ncbi_taxid",
                    "inference_effect",
                    "inference_administration_route_categorized",
                    "inference_organism_lifestage_categorized",
                    "CLS_embedding",
                ]
            ].drop_duplicates(
                subset=[
                    "inference_smiles",
                    "inference_ncbi_taxid",
                    "inference_effect",
                    "inference_administration_route_categorized",
                    "inference_organism_lifestage_categorized",
                ]
            )
            df_out = df_out.drop(columns=["CLS_embedding"])

        # Add InChI / InChIKey columns derived from the canonicalised SMILES
        logger.info("Computing InChI and InChIKey for output SMILES…")
        valid_smiles = [
            s for s in df_out["inference_smiles"].dropna().unique().tolist()
        ]
        descriptors = calculate_rdkit_descriptors(valid_smiles)
        descriptors.pop("smiles_input", None)
        df_out["inchi"] = df_out["inference_smiles"].map(
            lambda s: descriptors.get(s, {}).get("inchi") if pd.notna(s) else None
        )
        df_out["inchikey"] = df_out["inference_smiles"].map(
            lambda s: descriptors.get(s, {}).get("inchikey") if pd.notna(s) else None
        )

        save_outputs(df_out, cls_mapping, config)

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Inference pipeline completed successfully")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("ERROR: Inference pipeline failed")
        logger.error("=" * 60)
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
