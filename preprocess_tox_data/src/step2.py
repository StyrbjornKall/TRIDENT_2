import datetime
import os
import duckdb
from loguru import logger
import pandas as pd
import numpy as np
from utils import print_nunique_cas
import argparse
from setup_logger import setup_logger


def main():

    setup_logger()
    parser = argparse.ArgumentParser(description="Process and pickle a file.")
    parser.add_argument("--file", required=True, help="Input file to filter")
    parser.add_argument("--out", required=True, help="Output concatenated data to file")
    parser.add_argument(
        "--out-cas-list", required=False, help="Output file to save unique CAS numbers"
    )
    parser.add_argument(
        "--database_out",
        required=True,
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

    filters = {
        "conc_sign": ["="],
        "conc_unit": [
            "mg/l",
            "mg/kg",
            "kg/m2",
            "%",
            "ppm",
            "mg/kg bw",
            "m",
            "mg/kg soil",
            "mg/org",
            "mg/l air",
            "mg/kg diet",
            "mg/kg org",
            "ml/kg bw",
            "mg/org/d",
            "mg/kg bw/d",
        ],
        "endpoint": ["noec", "loec", "ec10", "ec50"],
        "effect": [
            "mor",
            "dvp",
            "beh",
            "rep",
            "car",
            "gro",
            "itx",
            "phy",
            "mph",
            "pop",
        ],
    }

    for column, values in filters.items():
        if column in df.columns:
            df = df[df[column].str.lower().str.strip().isin(values)]
            print_nunique_cas(df, f"{column} cleaned")

    logger.info(f"CAS per unit: {df.groupby('conc_unit')['cas'].nunique()}")
    logger.info(f"CAS per endpoint: {df.groupby('endpoint')['cas'].nunique()}")
    logger.info(f"CAS per effect: {df.groupby('effect')['cas'].nunique()}")

    # Save the output file
    if args.out.endswith(".zip"):
        df.to_csv(args.out, compression="zip", index=False)
    else:
        df.to_csv(args.out, index=False)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    logger.success(f"File processed and saved to {args.out} ({size_mb:.2f} MB)")

    # Save unique CAS numbers to a text file
    if args.out_cas_list:
        unique_cas = list(df["cas"].dropna().unique())
        with open(args.out_cas_list, "w") as f:
            for cas in unique_cas:
                f.write(f"{cas}\n")
        logger.success(
            f"Unique CAS numbers saved to {args.out_cas_list} ({len(unique_cas)} unique CAS numbers)"
        )

    # Save to duckdb database as "Step 2"
    con = duckdb.connect(args.database_out)
    con.execute("CREATE OR REPLACE TABLE step2 AS SELECT * FROM df")
    con.execute("CREATE OR REPLACE TABLE step2_cas AS SELECT DISTINCT cas FROM df")
    con.close()

    logger.success(
        f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step2' & 'step2_cas'"
    )


if __name__ == "__main__":
    main()
