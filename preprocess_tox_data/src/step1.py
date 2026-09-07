from datetime import datetime
import os
import pandas as pd
import numpy as np
from loguru import logger
import duckdb
from preprocess_datasets import (
    preprocess_acute_toxicity_data,
    preprocess_carc_toxicity_data,
    preprocess_rtecs_toxicity_data,
    preprocess_repro_toxicity_data,
    preprocess_aqter_toxicity_data,
)
from setup_logger import setup_logger
import argparse


def preprocess_rtecs(rtecs_tox, debug=False):
    logger.info(f"Reading RTECS data from {rtecs_tox}")
    df = pd.read_csv(rtecs_tox, sep=",", encoding="ISO-8859-1", low_memory=False)
    if debug:
        df = df.head(100)

    df["SK_unique_id"] = [
        "RTECS" + i for i in np.arange(1, len(df) + 1, dtype=int).astype(str)
    ]
    df["data_source"] = rtecs_tox.split("/")[-1]
    rtecs_tox = rtecs_tox.replace(".csv", "_SK.txt").replace(
        "toxicity_datasets", "preprocessed"
    )
    df.to_csv(rtecs_tox, sep="\t", index=False)
    logger.info("Preprocessing RTECS")

    # RTECS
    rename_dict = {
        "CAS_Registry_Number": "cas",
        "Endpoint": "endpoint",
        "Route_of_Exposure_or_Administration": "admin_route2",
        "Species_Group_Specific": "species_common_name",
        "Chemical_Name": "chemical_name",
        "Duration_Final": "duration",
        "Duration_Unit": "duration_unit",
        "Value": "conc",
        "Sign": "conc_sign",
        "Unit": "conc_unit",
        "Measurements": "effect1",
        "Type_of_Data": "effect2",
    }

    df_filtered = preprocess_rtecs_toxicity_data(df, rename_dict=rename_dict)

    df_filtered.to_csv(
        rtecs_tox.replace(".txt", "_preprocessed_step0.txt"), sep="\t", index=False
    )
    logger.success(
        f"Finished preprocessing RTECS. Filtered data saved to {rtecs_tox.replace('.txt', '_preprocessed_step0.txt')}"
    )
    return df_filtered


def preprocess_repro(repro_tox, debug=False):
    logger.info(f"Reading Reproductive Toxicity data from {repro_tox}")
    df = pd.read_csv(repro_tox, sep="\t", encoding="ISO-8859-1", low_memory=False)
    if debug:
        df = df.head(100)

    df["SK_unique_id"] = [
        "REPRO" + i for i in np.arange(1, len(df) + 1, dtype=int).astype(str)
    ]
    df["data_source"] = repro_tox.split("/")[-1]
    repro_tox = repro_tox.replace(".txt", "_SK.txt").replace(
        "toxicity_datasets", "preprocessed"
    )
    df.to_csv(repro_tox, sep="\t", index=False)
    logger.info("Preprocessing REPRO")

    # Reproduction
    rename_dict = {
        "CAS": "cas",
        "Endpoint": "effect",
        "Route_of_administration": "admin_route2",
        "Species": "species_common_name",
        "Name": "chemical_name",
        "Duration_Value": "duration",
        "Duration_Unit": "duration_unit",
        "Dose_descriptor": "endpoint",
        "Conc_Value": "conc",
        "Conc_Sign": "conc_sign",
        "Conc_Unit": "conc_unit",
    }

    df_filtered = preprocess_repro_toxicity_data(df, rename_dict=rename_dict)

    df_filtered.to_csv(
        repro_tox.replace(".txt", "_preprocessed_step0.txt"), sep="\t", index=False
    )
    logger.success(
        f"Finished preprocessing REPRO. Filtered data saved to {repro_tox.replace('.txt', '_preprocessed_step0.txt')}"
    )
    return df_filtered


def preprocess_carc(carc_tox, debug=False):
    logger.info(f"Reading Carcinogenicity data from {carc_tox}")
    df = pd.read_csv(carc_tox, sep="\t", encoding="ISO-8859-1", low_memory=False)
    if debug:
        df = df.head(100)

    df["SK_unique_id"] = [
        "CAR" + i for i in np.arange(1, len(df) + 1, dtype=int).astype(str)
    ]
    df["data_source"] = carc_tox.split("/")[-1]
    carc_tox = carc_tox.replace(".txt", "_SK.txt").replace(
        "toxicity_datasets", "preprocessed"
    )
    df.to_csv(carc_tox, sep="\t", index=False)
    logger.info("Preprocessing CAR")

    # Carcinogenicity
    rename_dict = {
        "CAS": "cas",
        "Endpoint": "admin_route1",
        "Route_of_administration": "admin_route2",
        "Species": "species_common_name",
        "Name": "chemical_name",
        "Duration_Value": "duration",
        "Duration_Unit": "duration_unit",
        "Dose_descriptor": "endpoint",
        "Conc_Value": "conc",
        "Conc_Sign": "conc_sign",
        "Conc_Unit": "conc_unit",
    }

    df_filtered = preprocess_carc_toxicity_data(df, rename_dict=rename_dict)

    df_filtered.to_csv(
        carc_tox.replace(".txt", "_preprocessed_step0.txt"), sep="\t", index=False
    )
    logger.success(
        f"Finished preprocessing CAR. Filtered data saved to {carc_tox.replace('.txt', '_preprocessed_step0.txt')}"
    )
    return df_filtered


def preprocess_acute(acute_tox, debug=False):
    logger.info(f"Reading ACUTE data from {acute_tox}")
    df = pd.read_csv(acute_tox, sep="\t", encoding="ISO-8859-1", low_memory=False)
    if debug:
        df = df.head(100)

    df["SK_unique_id"] = [
        "ACUTE" + i for i in np.arange(1, len(df) + 1, dtype=int).astype(str)
    ]
    df["data_source"] = acute_tox.split("/")[-1]
    acute_tox = acute_tox.replace(".txt", "_SK.txt").replace(
        "toxicity_datasets", "preprocessed"
    )
    df.to_csv(acute_tox, sep="\t", index=False)
    logger.info("Preprocessing ACUTE")

    # Acute
    rename_dict = {
        "CAS": "cas",
        "Endpoint": "admin_route1",
        "Route_of_administration": "admin_route2",
        "Species": "species_common_name",
        "Name.1": "chemical_name",
        "Duration_Value": "duration",
        "Duration_Unit": "duration_unit",
        "Measurement": "effect",
        "Dose": "endpoint",
        "Conc_Value": "conc",
        "Conc_Sign": "conc_sign",
        "Conc_Unit": "conc_unit",
    }

    global_filters = {
        "effect": ["mor"],  # Other value, GRO, only has three entries
        "endpoint": ["noec", "loec", "ec10", "ec50"],  # most other are underrepresented
    }

    df_filtered = preprocess_acute_toxicity_data(df, rename_dict=rename_dict)

    df_filtered.to_csv(
        acute_tox.replace(".txt", "_preprocessed_step0.txt"), sep="\t", index=False
    )
    logger.success(
        f"Finished preprocessing ACUTE. Filtered data saved to {acute_tox.replace('.txt', '_preprocessed_step0.txt')}"
    )

    return df_filtered


def preprocess_aqter(aqter_tox, debug=False):
    logger.info(f"Reading AQTER data from {aqter_tox}")
    df = pd.read_csv(aqter_tox, sep="\t", encoding="ISO-8859-1", low_memory=False)
    if debug:
        df = df.head(100)

    df["SK_unique_id"] = [
        "AQTER" + i for i in np.arange(1, len(df) + 1, dtype=int).astype(str)
    ]
    df["data_source"] = aqter_tox.split("/")[-1]
    aqter_tox = aqter_tox.replace(".txt", "_SK.txt").replace(
        "toxicity_datasets", "preprocessed"
    )
    df.to_csv(aqter_tox, sep="\t", index=False)

    logger.info("Preprocessing AQTER")
    # Aquatic

    rename_dict = {
        "CAS": "cas",
        "EC_Number": "ec_number",
        "endpoint": "endpoint",
        "chemical_name": "chemical_name",
        "duration_value": "duration",
        "duration_unit": "duration_unit",
        "conc": "conc",
        "conc_sign": "conc_sign",
        "conc_unit": "conc_unit",
        "effect": "effect",
        "common_name": "species_common_name",
        "latin_name": "species_latin_name",
        "species_group": "species_group",
        "administration_route": "administration_route",
        "organism_lifestage": "organism_lifestage",
    }

    df_filtered = preprocess_aqter_toxicity_data(df, rename_dict=rename_dict)
    df_filtered.to_csv(
        aqter_tox.replace(".txt", "_preprocessed_step0.txt"), sep="\t", index=False
    )
    logger.success(
        f"Finished preprocessing AQTER. Filtered data saved to {aqter_tox.replace('.txt', '_preprocessed_step0.txt')}"
    )
    return df_filtered


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Process and pickle a file.")
    parser.add_argument("--out", required=True, help="Output concatenated data to file")
    parser.add_argument(
        "--database_out", required=True, help="Output file path for duckdb database"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print diagnostics and limit to 10 rows",
    )
    parser.add_argument(
        "--acute_tox", required=False, help="Path to Acute Toxicity file"
    )
    parser.add_argument(
        "--carc_tox", required=False, help="Path to Carcinogenicity file"
    )
    parser.add_argument(
        "--repro_tox", required=False, help="Path to Reproductive Toxicity file"
    )
    parser.add_argument("--rtecs_tox", required=False, help="Path to RTECS file")
    parser.add_argument(
        "--aqter_tox", required=False, help="Path to Aquatic Terrestrial Toxicity file"
    )
    parser.add_argument("--enviro_tox", required=False, help="Path to Envirotox file")

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")
    logger.info("Processing files:")
    if args.acute_tox:
        logger.info(f" - Acute Toxicity: {args.acute_tox}")
    if args.carc_tox:
        logger.info(f" - Carcinogenicity: {args.carc_tox}")
    if args.repro_tox:
        logger.info(f" - Reproductive Toxicity: {args.repro_tox}")
    if args.rtecs_tox:
        logger.info(f" - RTECS: {args.rtecs_tox}")
    if args.aqter_tox:
        logger.info(f" - Aquatic Terrestrial Toxicity: {args.aqter_tox}")
    if args.enviro_tox:
        logger.info(f" - Envirotox: {args.enviro_tox}")
    logger.info(f"Output will be saved to: {args.out}")
    logger.info(f"DuckDB database will be saved to: {args.database_out}")

    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(args.out)):
        logger.info(f"Created output directory: {os.path.dirname(args.out)}")
        os.makedirs(os.path.dirname(args.out))

    # If no toxicity files are provided, exit
    if not (
        args.acute_tox
        or args.carc_tox
        or args.repro_tox
        or args.rtecs_tox
        or args.aqter_tox
        or args.enviro_tox
    ):
        logger.info("No toxicity files provided. Exiting.")
        return
    acute_tox = args.acute_tox
    carc_tox = args.carc_tox
    repro_tox = args.repro_tox
    rtecs_tox = args.rtecs_tox
    aqter_tox = args.aqter_tox
    enviro_tox = args.enviro_tox

    logger.info("Starting preprocessing of toxicity datasets")

    preprocessed_data = []
    if rtecs_tox:
        df_filtered = preprocess_rtecs(rtecs_tox, args.debug)
        preprocessed_data.append(df_filtered)

    if repro_tox:
        df_filtered = preprocess_repro(repro_tox, args.debug)
        preprocessed_data.append(df_filtered)

    if carc_tox:
        df_filtered = preprocess_carc(carc_tox, args.debug)
        preprocessed_data.append(df_filtered)

    if acute_tox:
        df_filtered = preprocess_acute(acute_tox, args.debug)
        preprocessed_data.append(df_filtered)

    if aqter_tox:
        df_filtered = preprocess_aqter(aqter_tox, args.debug)
        preprocessed_data.append(df_filtered)

    if enviro_tox:
        # TODO: add enviro tox preprocessing function
        pass

    # Save concatenated data
    preprocessed_data = pd.concat(
        preprocessed_data, axis=0, ignore_index=True, join="outer"
    )
    preprocessed_data.to_csv(args.out, compression="zip", index=False)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)

    logger.success(
        f"Preprocessing complete. Concatenated data saved to {args.out} ({size_mb:.2f} MB)"
    )

    # Save to duckdb database as "Step 1"
    con = duckdb.connect(args.database_out)

    con.execute("CREATE OR REPLACE TABLE step1 AS SELECT * FROM preprocessed_data")

    logger.success(
        f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step1'"
    )

    # Save original data to duckdb as well
    for original_path in [
        rtecs_tox,
        repro_tox,
        carc_tox,
        acute_tox,
        aqter_tox,
        enviro_tox,
    ]:
        if not original_path:
            continue

        if original_path.endswith(".csv"):
            table_name = original_path.replace(".csv", "_SK.txt").replace(
                "toxicity_datasets", "preprocessed"
            )
        else:
            table_name = original_path.replace(".txt", "_SK.txt").replace(
                "toxicity_datasets", "preprocessed"
            )
        # if the file exists, save it to duckdb
        if table_name and os.path.exists(table_name):
            df = pd.read_csv(table_name, sep="\t", low_memory=False)
            logger.info(f"Saving original data from {table_name} to duckdb database")

            duck_table_name = os.path.splitext(os.path.basename(table_name))[0]
            con.register("source_df", df)
            con.execute(
                f'CREATE OR REPLACE TABLE "{duck_table_name}" AS SELECT * FROM source_df'
            )
            con.unregister("source_df")
    con.close()


if __name__ == "__main__":
    main()
