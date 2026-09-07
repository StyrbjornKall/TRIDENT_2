import os
import json
import duckdb
import pandas as pd
import argparse
import numpy as np
from loguru import logger
from setup_logger import setup_logger
from chem_utils import canonicalize_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# Convert concentrations from molar to mg/L using RDKit
def convert_molars_to_mg_per_l(smiles, conc, unit):
    if unit.lower() != "m":
        return smiles, conc, unit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, conc, unit
        mol_weight = AllChem.CalcExactMolWt(mol)
        if mol_weight == 0:
            return smiles, conc, unit
        return smiles, conc * mol_weight * 1000, "mg/l"  # Convert from M to mg/l
    except Exception as e:
        return smiles, conc, unit


# Function to split species and expand rows
def split_species(df, species_col):
    # Ensure column is a string
    df[species_col] = df[species_col].astype(str)
    # Remove count prefix
    df[species_col] = df[species_col].str.split("; ", n=1).str[-1]
    # Expand species into multiple rows
    df = (
        df.assign(**{species_col: df[species_col].str.split(", ")})
        .explode(species_col)
        .reset_index(drop=True)
    )
    return df


def correct_duplicate_species_groups(df):
    # Get unique species_latin_name and species_group combinations
    dups = df.drop_duplicates(subset=["species_latin_name", "species_group"])[
        ["species_latin_name", "species_group"]
    ]

    # Get a mapping dictionary that does not contain NaNs, this will be used to map the missing species_group values
    mapping = dict(
        zip(
            dups[
                dups.species_latin_name.isin(
                    dups.species_latin_name[
                        dups.species_latin_name.duplicated()
                    ].tolist()
                )
            ]
            .dropna()
            .species_latin_name,
            dups[
                dups.species_latin_name.isin(
                    dups.species_latin_name[
                        dups.species_latin_name.duplicated()
                    ].tolist()
                )
            ]
            .dropna()
            .species_group,
        )
    )

    # Map NaNs to correct species_group in the data
    for i in range(len(df)):
        if pd.isna(df["species_group"][i]):
            df.loc[i, "species_group"] = mapping.get(
                df.loc[i, "species_latin_name"], None
            )

    return df


def main():
    """
    Main function to preprocess toxicity data and save the processed output to a file.
    This script reads an input file containing toxicity data, applies various preprocessing steps
    such as filtering, mapping, and canonicalizing SMILES strings, and saves the processed data
    to an output file. Optionally, it can categorize organism lifestages using a mapping file.
    Command-line arguments:
        --file (str, required): Path to the input file (CSV or ZIP format).
        --lifestage_mapping (str, optional): Path to the lifestage mapping CSV file.
        --administration_route_mapping (str, optional): Path to the administration route mapping CSV file.
        --out (str, required): Path to the output file (CSV or ZIP format).
        --debug Optional: If set, processes only a subset of the data for debugging purposes.
    Preprocessing steps:
        - Drops rows with missing SMILES strings.
        - Filters rows based on concentration and duration thresholds.
        - Maps and renames values in specific columns using predefined configurations.
        - Splits experiments with multiple species into separate rows.
        - Corrects missing species group values based on species Latin names.
        - Removes rows with no species information.
        - Canonicalizes SMILES strings and removes rows with invalid SMILES.
        - Converts concentrations from molar to mg/L using RDKit and SMILES.
        - Categorizes organism lifestages into broader categories using a mapping file.
        - Removes duplicate rows based on specific columns.
    Output:
        Saves the processed data to the specified output file in CSV or ZIP format.
    """
    setup_logger()

    parser = argparse.ArgumentParser(description="Process and pickle a file.")
    parser.add_argument("--file", required=True, help="Input file path")
    parser.add_argument(
        "--lifestage_mapping",
        required=False,
        help="Path to the lifestage mapping csv file",
    )
    parser.add_argument(
        "--administration_route_mapping",
        required=False,
        help="Path to the administration route mapping csv file",
    )
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument(
        "--out-taxa", required=True, help="Output file path for taxa information"
    )
    parser.add_argument(
        "--save-metadata-files",
        action="store_true",
        default=True,
        help="Whether to save metadata files for later use in modeling, including taxid2sciname.json, taxid2spgroup.json, taxid2rank.json, taxid2parent.json",
    )
    parser.add_argument(
        "--database-out",
        required=True,
        help="Output duckdb database file path to save the preprocessed data into table 'step3'",
    )
    parser.add_argument(
        "--max-duration",
        required=False,
        default=np.inf,
        type=float,
        help="Maximum duration to keep",
    )
    parser.add_argument(
        "--max-concentration",
        required=False,
        default=np.inf,
        type=float,
        help="Maximum concentration to keep",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print diagnostics and limit to 10 rows",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Processing file: {args.file} using lifestage mapping: {args.lifestage_mapping} and administration route mapping: {args.administration_route_mapping}"
    )
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

    if args.debug:
        df = df.head(100)

    # Drop rows with missing SMILES
    logger.warning(f"Dropping {df.SMILES.isna().sum()} rows with missing SMILES")
    df = df.dropna(subset=["SMILES"])

    # Drop probably faulty data
    logger.warning(
        f"Dropping {(df.conc > args.max_concentration).sum()} rows with conc>{args.max_concentration}"
    )
    df = df[(df.conc <= args.max_concentration)]
    logger.warning(f"Dropping {(df.conc <= 0).sum()} rows with conc<={0}")
    df = df[(df.conc > 0)]

    logger.warning(
        f"Dropping {((df.duration > args.max_duration) & (df.species_group != 'rodents')).sum()} rows where duration exceeds {args.max_duration} for non-rodents."
    )
    df = df[~((df.duration > args.max_duration) & (df.species_group != "rodents"))]
    df.loc[df.duration <= 0, "duration"] = np.nan

    logger.info("Splitting experiments with several species...")
    before = len(df)
    df = split_species(df, "species_latin_name")
    logger.info(f"Generated {len(df) - before} new experiments.")

    logger.info(
        "Correcting inconsistencies in species_group values using species_latin_name..."
    )
    before = df.species_group.isna().sum()
    # Correct species_group based on species_latin_name
    df = correct_duplicate_species_groups(df)
    logger.info(
        f"Corrected {before - df.species_group.isna().sum()} missing species_group values."
    )

    logger.info("Removing rows with no species information...")
    # Remove rows with no species information
    before = len(df)
    df = df[
        ~(
            (df.species_group.isna())
            & (df.species_latin_name.isna())
            & (df.species_common_name.isna())
        )
    ]
    logger.warning(f"Dropped {before - len(df)} rows with no species information.")

    # Extract unique SMILES strings
    df = df.dropna(subset=["SMILES"])
    unique_smiles = df["SMILES"].unique()

    # Create a dictionary mapping original SMILES to canonical SMILES
    canonical = True
    isomericSmiles = False
    drop_errorenous_smiles = True
    logger.info(
        f"Canonicalizing {len(unique_smiles)} SMILES with canonical={canonical} and isomericSmiles={isomericSmiles} and drop_errorenous_smiles={drop_errorenous_smiles}"
    )
    smiles_to_canonical = {
        smiles: canonicalize_smiles(
            smiles,
            canonical=canonical,
            isomericSmiles=isomericSmiles,
            drop_errorenous_smiles=drop_errorenous_smiles,
        )
        for smiles in unique_smiles
    }

    # Map canonical SMILES back to the DataFrame
    df["SMILES"] = df["SMILES"].map(smiles_to_canonical)

    # Remove rows with None SMILES
    logger.warning(
        f"Dropping {len(unique_smiles) - df.SMILES.dropna().nunique()} non parsable SMILES --> {df.SMILES.isna().sum()} rows"
    )
    # if drop_errorenous_smiles and len(unique_smiles)-df.SMILES[df.SMILES.notna()].nunique() > 1000:
    #    user_input = input(f"Whoa! You are dropping {len(unique_smiles)-df.SMILES[df.SMILES.notna()].nunique()} incorrect SMILES. Are you sure you want to do this? (yes/no): ").strip().lower()
    #    if user_input != 'yes':
    #        logger.info("Operation aborted by the user.")
    #        return
    df = df.dropna(subset=["SMILES"])

    logger.info("Translating concentrations given in molars to mg/l using RDKit...")

    df[["SMILES", "conc", "conc_unit"]] = df.apply(
        lambda row: convert_molars_to_mg_per_l(
            smiles=row["SMILES"], conc=row["conc"], unit=row["conc_unit"]
        ),
        axis=1,
        result_type="expand",
    )

    # Categorize lifestages into broader categories
    if args.lifestage_mapping:
        logger.info("Categorizing organism_lifestage...")
        mapping = pd.read_csv(args.lifestage_mapping, sep=";")
        mapping = dict(
            zip(
                mapping.TERM.str.lower().str.strip(),
                mapping.CATEGORY.str.lower().str.strip(),
            )
        )
        df["organism_lifestage"] = df["organism_lifestage"].str.lower().str.strip()
        df["organism_lifestage_categorized"] = df["organism_lifestage"].map(mapping)
        logger.success(
            f"Created {df.organism_lifestage_categorized.nunique()} broader lifestage categories"
        )

    # Categorize admin route
    if args.administration_route_mapping:
        logger.info("Categorizing administration_route...")
        mapping = pd.read_csv(args.administration_route_mapping, sep=";")
        mapping = dict(
            zip(
                mapping.TERM.str.lower().str.strip(),
                mapping.CATEGORY.str.lower().str.strip(),
            )
        )
        df["administration_route"] = df["administration_route"].str.lower().str.strip()
        df["administration_route_categorized"] = df["administration_route"].map(mapping)
        logger.success(
            f"Created {df.administration_route_categorized.nunique()} broader administration route categories"
        )

    before = len(df)
    df = df.drop_duplicates(
        subset=[
            "SMILES",
            "species_latin_name",
            "species_common_name",
            "organism_lifestage",
            "administration_route",
            "endpoint",
            "effect",
            "conc_unit",
            "conc",
            "duration",
        ]
    )
    logger.info(f"Dropped {before - len(df)} duplicate experiments")

    # Save the output file
    if args.out.endswith(".zip"):
        df.to_csv(args.out, compression="zip", index=False)
    else:
        df.to_csv(args.out, index=False)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    logger.success(f"File processed and saved to {args.out} ({size_mb:.2f} MB)")

    # Save taxa information if specified
    if args.out_taxa:
        df_taxa = df[["species_latin_name", "species_common_name"]].drop_duplicates()
        if args.out_taxa.endswith(".zip"):
            df_taxa.to_csv(args.out_taxa, compression="zip", index=False)
        else:
            df_taxa.to_csv(args.out_taxa, index=False)
        logger.success(f"Taxa information saved to {args.out_taxa}")

    # Save to duckdb database as "Step 3"
    con = duckdb.connect(args.database_out)
    con.execute("CREATE OR REPLACE TABLE step3 AS SELECT * FROM df")
    con.execute(
        "CREATE OR REPLACE TABLE step3_species AS SELECT DISTINCT species_latin_name, species_common_name FROM df"
    )
    con.close()

    logger.success(
        f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step3'"
    )

    if args.save_metadata_files:
        logger.info("Saving metadata files...")
        smiles2chemname = dict(zip(df["SMILES"], df["chemical_name"]))
        with open(f"{os.path.dirname(args.out)}/smiles2chemname.json", "w") as f:
            json.dump(smiles2chemname, f, indent=4)


if __name__ == "__main__":
    main()
