import pandas as pd
import numpy as np
import argparse
import os
import duckdb
from typing import Optional
from loguru import logger
from setup_logger import setup_logger


def preprocess(
    df: pd.DataFrame,
    concentration_thresh: Optional[float] = np.inf,
    duration_thresh: Optional[float] = np.inf,
    log_data: bool = True,
    turn_duration_outliers_to_nan: bool = False,
    turn_duration_units_other_than_hours_to_nan: bool = True,
    upper_endpoint_effect: bool = True,
) -> pd.DataFrame:

    if turn_duration_units_other_than_hours_to_nan:
        logger.info(
            f"  Turning {(df.duration_unit != 'h').sum()} duration values with units other than hours to NaN"
        )
        df.loc[df.duration_unit != "h", "duration"] = np.nan

    logger.info(
        f"  Turning {(df.duration <= 0).sum()} duration values to NaN because they are <= 0"
    )
    df.loc[df.duration <= 0, "duration"] = np.nan
    logger.info(
        "  Adding 'duration_missing' column to help neural network learn that these values were originally missing"
    )
    df["duration_missing"] = (
        df["duration"].isna().astype(int).astype(str).replace({"0": pd.NA})
    )
    logger.info(
        f"  Filling {df.duration.isna().sum()} missing duration values with a small value (1e-6) to avoid issues with log-transforming"
    )
    df["duration"] = df["duration"].fillna(1e-6)
    logger.info(
        f"  Dropping {(df.duration > duration_thresh).sum()} rows where duration > {duration_thresh}"
    )
    df = df[df.duration < duration_thresh]
    logger.info(f"  Dropping {(df.conc <= 0).sum()} rows where conc <= 0")
    df = df[df.conc > 0]
    logger.info(
        f"  Dropping {(df.conc > concentration_thresh).sum()} rows where conc > {concentration_thresh}"
    )
    df = df[df.conc < concentration_thresh]

    if log_data:
        logger.info("  Log-transforming concentration and duration values")
        df.conc = np.log10(df.conc)
        df.duration = np.log10(df.duration)

    if turn_duration_outliers_to_nan:
        logger.info("  Turning duration outliers to NaN")

        # Takes the IQR and turns other durations to None
        Q1 = df.duration.quantile(0.2)
        Q3 = df.duration.quantile(0.8)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.duration = df.duration.apply(
            lambda x: np.nan if x < lower_bound or x > upper_bound else x
        )

    if upper_endpoint_effect:
        logger.info("  Capitalizing endpoint and effect values")
        df.endpoint = df.endpoint.str.upper()
        df.effect = df.effect.str.upper()

    return df


def plot_data_description(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df.conc, bins=100, ax=axes[0], kde=True)
    axes[0].set_title("Distribution of Concentration (log10)")

    sns.histplot(df.duration, bins=100, ax=axes[1], kde=True)
    axes[1].set_title("Distribution of Duration (log10)")

    plt.tight_layout()
    plt.savefig("data_description.png", dpi=300)
    plt.show()


def print_final_data_description(df: pd.DataFrame):
    logger.info("Final data description:")
    logger.success("\nBASIC INFO:")
    logger.info(f"  Number of rows: {len(df)}")
    logger.info(f"  Number of unique CAS: {df.cas.nunique()}")
    logger.info(f"  Number of unique SMILES: {df.SMILES.nunique()}")
    logger.info(f"  Number of unique endpoints: {df.endpoint.nunique()}")
    logger.info(f"  Number of unique effects: {df.effect.nunique()}")
    logger.info(f"  Number of unique concentration units: {df.conc_unit.nunique()}")
    logger.info(f"  Number of unique duration units: {df.duration_unit.nunique()}")
    logger.info(
        f"  Number of unique species groups: {df.species_group_corrected.nunique()}"
    )
    logger.info(
        f"  Number of unique administration routes: {df.administration_route_categorized.nunique()}"
    )
    logger.info(
        f"  Number of unique organism life stages: {df.organism_lifestage_categorized.nunique()}"
    )

    logger.success("\nDETAILED INFO:")
    logger.info(f"  Endpoint counts: {df.endpoint.value_counts(dropna=False)}")
    logger.info(f"  Effect counts: {df.effect.value_counts(dropna=False)}")
    logger.info(
        f"  Concentration unit counts: {df.conc_unit.value_counts(dropna=False)}"
    )
    logger.info(
        f"  Duration unit counts: {df.duration_unit.value_counts(dropna=False)}"
    )
    logger.info(
        f"  Administration route counts: {df.administration_route_categorized.value_counts(dropna=False)}"
    )
    logger.info(
        f"  Organism life stage counts: {df.organism_lifestage_categorized.value_counts(dropna=False)}"
    )
    logger.info(
        f"  Species group counts: {df.species_group_corrected.value_counts(dropna=False)}"
    )

    logger.success("\nNCBI TAXONOMY INFO:")
    for rank in [
        "superkingdom",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]:
        logger.info(f"  Number of unique {rank}: {df[f'NCBI_rank_{rank}'].nunique()}")
    logger.info(
        f"  Number of rows without species: {df['NCBI_rank_species'].isna().sum()}"
    )

    logger.success("\nCONCENTRATION & DURATION INFO:")
    logger.info(
        f"  Concentration (log10) - mean: {df.conc.mean():.2f}, std: {df.conc.std():.2f}"
    )
    logger.info(
        f"  Duration (log10) - mean: {df.duration.mean():.2f}, std: {df.duration.std():.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess the data")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the input data",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to the output data",
    )
    parser.add_argument(
        "--database_out",
        type=str,
        required=True,
        help="Path to the output duckdb database",
    )
    parser.add_argument(
        "--concentration_thresh",
        type=float,
        default=np.inf,
        help="Threshold for concentration values (values above this will be removed)",
    )
    parser.add_argument(
        "--duration_thresh",
        type=float,
        default=np.inf,
        help="Threshold for duration values (values above this will be removed)",
    )
    parser.add_argument(
        "--log_data",
        action="store_true",
        default=True,
        help="Whether to log-transform the concentration and duration values",
    )
    parser.add_argument(
        "--turn_duration_outliers_to_nan",
        action="store_true",
        default=False,
        help="Whether to turn duration outliers to NaN (outliers are defined as values outside 1.5*IQR)",
    )
    parser.add_argument(
        "--turn_duration_units_other_than_hours_to_nan",
        action="store_true",
        default=True,
        help="Whether to turn duration values with units other than hours to NaN",
    )
    parser.add_argument(
        "--upper_endpoint_effect",
        action="store_true",
        default=True,
        help="Whether to capitalize endpoint and effect values",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging and limited rows",
    )

    args = parser.parse_args()

    setup_logger(level="INFO" if not args.debug else "DEBUG")
    logger.info(f"Arguments: {args}")
    logger.info(f"Processing file: {args.file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Read the input file
    if args.file.endswith(".zip"):
        df = pd.read_csv(args.file, compression="zip", low_memory=False)
    else:
        df = pd.read_csv(args.file, low_memory=False)

    if args.debug:
        df = df.head(100)
    logger.success(f"Loaded data with {len(df)} rows.")

    logger.info("Preprocessing data...")
    df = preprocess(
        df,
        concentration_thresh=args.concentration_thresh,
        duration_thresh=args.duration_thresh,
        log_data=args.log_data,
        turn_duration_outliers_to_nan=args.turn_duration_outliers_to_nan,
        turn_duration_units_other_than_hours_to_nan=args.turn_duration_units_other_than_hours_to_nan,
        upper_endpoint_effect=args.upper_endpoint_effect,
    )

    # Set dtypes
    dtypes = {
        "SK_unique_id": str,
        "data_source": str,
        "species_group": str,
        "species_common_name": str,
        "species_latin_name": str,
        "cas": str,
        "chemical_name": str,
        "conc_unit": str,
        "conc": float,
        "conc_sign": str,
        "duration_unit": str,
        "duration": float,
        "effect": str,
        "endpoint": str,
        "administration_route": str,
        "organism_lifestage": str,
        "organism_habitat": str,
        "SMILES": str,
        "organism_lifestage_categorized": str,
        "administration_route_categorized": str,
        "NCBI_sci_name": str,
        "NCBI_match": str,
        "NCBI_rank_superkingdom": str,
        "NCBI_rank_kingdom": str,
        "NCBI_rank_phylum": str,
        "NCBI_rank_subphylum": str,
        "NCBI_rank_class": str,
        "NCBI_rank_order": str,
        "NCBI_rank_family": str,
        "NCBI_rank_genus": str,
        "NCBI_rank_species": str,
        "NCBI_last_known_rank": str,
        "species_group_corrected": str,
        "duration_missing": str,
    }
    df = df.astype(dtypes)
    logger.success("Data types set successfully.")

    logger.success("Data preprocessing completed.")
    logger.info("Plotting data description...")
    plot_data_description(df)

    logger.info("Printing final data description...")
    print_final_data_description(df)

    if args.out.endswith(".zip"):
        df.to_csv(args.out, compression="zip", index=False)
        df.to_pickle(args.out.replace(".csv.zip", ".pkl.zip"), compression="zip")
    else:
        df.to_csv(args.out, index=False)
        df.to_pickle(args.out.replace(".csv", ".pkl"))

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    logger.success(f"Data with NCBI info saved to {args.out} ({size_mb:.2f} MB)")

    # Save to duckdb database as "Step 7"
    con = duckdb.connect(args.database_out)
    con.execute("CREATE OR REPLACE TABLE step7 AS SELECT * FROM df")
    con.close()

    logger.success(
        f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step7'"
    )


if __name__ == "__main__":
    main()
