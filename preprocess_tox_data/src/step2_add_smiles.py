import os
import duckdb
from loguru import logger
import pandas as pd
import argparse
from setup_logger import setup_logger


def main():

    setup_logger()
    parser = argparse.ArgumentParser(description="Process and pickle a file.")
    parser.add_argument("--file", required=True, help="Input file to filter")
    parser.add_argument("--out", required=True, help="Output concatenated data to file")
    parser.add_argument(
        "--cas-to-smiles-mapping",
        required=True,
        help="Input file to map CAS numbers to SMILES (CSV with columns 'cas' and 'SMILES')",
    )
    parser.add_argument(
        "--database_out",
        required=False,
        help="Output duckdb database file path to save the preprocessed data into table 'step2'",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    logger.info(f"Processing file: {args.file}")
    logger.info(f"Output will be saved to: {args.out}")

    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(args.out)):
        logger.info(f"Created output directory: {os.path.dirname(args.out)}")
        os.makedirs(os.path.dirname(args.out))

    # Read the input file
    if args.file.endswith(".zip"):
        df = pd.read_csv(args.file, compression="zip", low_memory=False)
    else:
        df = pd.read_csv(args.file, low_memory=False)
    logger.success(
        f"File read successfully: {args.file} with {len(df)} rows and {len(df.columns)} columns"
    )

    # Use mapping file to add SMILES to the dataframe
    smiles_mapping_df = pd.read_csv(args.cas_to_smiles_mapping)[["cas", "SMILES"]]
    df = df.merge(smiles_mapping_df, how="left", left_on="cas", right_on="cas")

    # Save to out file as csv
    df.to_csv(args.out, index=False)
    logger.success(
        f"Preprocessed data saved to {args.out} with {len(df)} rows and {len(df.columns)} columns."
    )

    if args.database_out:
        # Save to duckdb database as "Step 2" (overwrite if exists)
        con = duckdb.connect(args.database_out)
        con.execute("CREATE OR REPLACE TABLE step2 AS SELECT * FROM df")
        con.close()

        logger.success(
            f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step2'"
        )


if __name__ == "__main__":
    main()
