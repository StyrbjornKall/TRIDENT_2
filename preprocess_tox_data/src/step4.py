import pandas as pd
import numpy as np
import os
import json
import argparse
import duckdb
from typing import List, Dict, Tuple
from loguru import logger
from preprocess_taxonomy import (
    TaxonomyPreProcessor,
    construct_ranked_lineage,
    fill_missing_taxonomy,
    map_lineage_to_species_group,
    write_filled_newick_tree,
)
from setup_logger import setup_logger

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def main():
    parser = argparse.ArgumentParser(description="Find NCBI taxa for species names")
    parser.add_argument("--file", required=True, help="Input file path")
    parser.add_argument(
        "--out-taxonomic-info", required=True, help="Output taxonomic info file path"
    )
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument(
        "--database-out", required=True, help="Output duckdb database file path"
    )
    parser.add_argument(
        "--save-filled-tree",
        action="store_true",
        default=True,
        help="Whether to save the filled newick tree with missing taxids filled in",
    )
    parser.add_argument(
        "--save-metadata-files",
        action="store_true",
        default=True,
        help="Whether to save metadata files for later use in modeling, including taxid2sciname.json, taxid2spgroup.json, taxid2rank.json, taxid2parent.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print diagnostics and limit to 10 rows",
    )

    args = parser.parse_args()
    setup_logger(level="DEBUG" if args.debug else "INFO")
    logger.info(f"Arguments: {args}")
    logger.info(f"Processing file: {args.file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out_taxonomic_info), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not (
        args.out_taxonomic_info.endswith(".csv")
        or args.out_taxonomic_info.endswith(".csv.zip")
    ):
        raise ValueError(
            "Output file must have .csv or .csv.zip in the name to save as .csv.zip"
        )

    # Read the input file
    if args.file.endswith(".zip"):
        df = pd.read_csv(args.file, compression="zip", low_memory=False)
    else:
        df = pd.read_csv(args.file, low_memory=False)

    if args.debug:
        df = df.head(100)
    logger.success(
        f"Loaded data with {len(df)} rows and {len(df.species_latin_name.unique())} unique species latin names."
    )

    queries = df.drop_duplicates(subset=["species_latin_name", "species_group"])[
        ["species_latin_name", "species_group"]
    ]

    # Specify the ranks to keep we are interested in for reflecting evolutionary similarity
    ranks = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    # Print run metrics
    logger.info("Running with the following settings:")
    logger.info(f"  ranks: {ranks}")

    # Using the species latin names and optional species groups, get the NCBI taxonomy information
    # This will return a dictionary with the following keys:
    # 'TaxID': a list of the NCBI Taxonomy IDs for each species
    # 'NCBI_match': a list of the NCBI Taxonomy Scientific names for each species
    # 'species_mapping': a mapping of the species names to their NCBI Taxonomy IDs
    # 'lineage': a list of the lineage for each species
    # 'named_lineage': a list of the named lineage for each species
    preprocessor = TaxonomyPreProcessor()
    out = preprocessor.get_NCBI_match(
        species_names=queries["species_latin_name"],
        species_groups=queries["species_group"],
        return_lineage=True,
        return_species_mapping=True,
        return_TaxID=True,
        verify_taxid_using_species_group=False,
        preprocess_using_gbif=False,
        preprocess_using_local_mapping=True,
    )

    # Combine the output into a DataFrame
    taxonomy = pd.concat(
        [
            queries.reset_index(drop=True),
            pd.DataFrame({key: out[key] for key in out if key != "species_mapping"}),
        ],
        axis=1,
    )
    if args.debug:
        logger.debug(f"Output from get_NCBI_match:\n{out}")
        logger.debug(f"Taxonomy DataFrame after concatenation:\n{taxonomy.head()}")

    logger.success(
        f"Out of {len(queries)} supposed species latin names, {taxonomy['NCBI_match'].nunique()} were matched to NCBI taxonomy."
    )

    # The lineage will contain the full lineage for each species, including all ranks, we filter this and add respective columns to only include the ranks we are interested in
    ranked_lineage = preprocessor.filter_lineages_by_ranks(
        taxonomy.lineage.tolist(),
        desired_ranks=ranks,
        return_names=False,  # We return taxIDs, not names since that will be ambiguous, some species have the same name but different taxIDs
    )
    taxonomy = pd.concat(
        [taxonomy.reset_index(drop=True), ranked_lineage.reset_index(drop=True)], axis=1
    )

    logger.debug(
        f"After ranking, we have:\n   -{taxonomy['NCBI_rank_species'].nunique()} species\n   -{taxonomy['NCBI_rank_genus'].nunique()} genus\n   -{taxonomy['NCBI_rank_family'].nunique()} families\n   -{taxonomy['NCBI_rank_order'].nunique()} orders\n   -{taxonomy['NCBI_rank_class'].nunique()} classes\n   -{taxonomy['NCBI_rank_phylum'].nunique()} phyla\n   -{taxonomy['NCBI_rank_kingdom'].nunique()} kingdoms\n   -{taxonomy['NCBI_rank_superkingdom'].nunique()} superkingdoms"
    )

    logger.debug(f"After removing redundant lineage columns:\n{taxonomy.head()}")

    # Flag any species that have no superkingdom, since those are likely to be errors
    taxonomy.loc[taxonomy["NCBI_rank_superkingdom"].isna(), "flag"] = "no superkingdom"
    logger.warning(
        f"Adding flags {taxonomy['flag'].value_counts().get('no superkingdom', 0)} species that had no superkingdom, since those are likely to be errors"
    )

    logger.debug(
        f"After superkingdom filtering, we have:\n   -{taxonomy['NCBI_rank_species'].nunique()} species\n   -{taxonomy['NCBI_rank_genus'].nunique()} genus\n   -{taxonomy['NCBI_rank_family'].nunique()} families\n   -{taxonomy['NCBI_rank_order'].nunique()} orders\n   -{taxonomy['NCBI_rank_class'].nunique()} classes\n   -{taxonomy['NCBI_rank_phylum'].nunique()} phyla\n   -{taxonomy['NCBI_rank_kingdom'].nunique()} kingdoms\n   -{taxonomy['NCBI_rank_superkingdom'].nunique()} superkingdoms"
    )

    # Flag any entries where NCBI_rank_superkingdom is not bacteria-2 or eukaryota-2759
    taxonomy.loc[~taxonomy["NCBI_rank_superkingdom"].isin(["2759", "2"]), "flag"] = (
        "superkingdom not 2 or 2759"
    )
    logger.warning(
        f"Adding flags {taxonomy['flag'].value_counts().get('superkingdom not 2 or 2759', 0)} species that were not bacteria or eukaryota, since those are likely to be errors"
    )

    # Remove all bacteria that are not cyanobacteria
    taxonomy.loc[
        (taxonomy["NCBI_rank_superkingdom"] == "2")
        & (taxonomy["NCBI_rank_phylum"] != "1117"),
        "flag",
    ] = "incorrect bacteria"
    logger.warning(
        f"Adding flags {taxonomy['flag'].value_counts().get('incorrect bacteria', 0)} bacterial species that were not cyanobacteria, since we are not interested in those for this analysis"
    )

    # Manually set any column with no rank between superkingdom and species to None, since those are likely to errors
    taxonomy.loc[
        taxonomy[
            [
                "NCBI_rank_phylum",
                "NCBI_rank_class",
                "NCBI_rank_order",
                "NCBI_rank_family",
                "NCBI_rank_genus",
            ]
        ]
        .isna()
        .all(axis=1),
        "flag",
    ] = "insufficient lineage"
    logger.warning(
        f"Adding flags {taxonomy['flag'].value_counts().get('insufficient lineage', 0)} species that had no rank between superkingdom and species, since those are likely to be errors"
    )

    logger.success(
        f"After filtering, we have:\n   -{taxonomy['NCBI_rank_species'].nunique()} species\n   -{taxonomy['NCBI_rank_genus'].nunique()} genus\n   -{taxonomy['NCBI_rank_family'].nunique()} families\n   -{taxonomy['NCBI_rank_order'].nunique()} orders\n   -{taxonomy['NCBI_rank_class'].nunique()} classes\n   -{taxonomy['NCBI_rank_phylum'].nunique()} phyla\n   -{taxonomy['NCBI_rank_kingdom'].nunique()} kingdoms\n   -{taxonomy['NCBI_rank_superkingdom'].nunique()} superkingdoms"
    )

    # Add taxid and correct scientific names to the original data
    df = df.merge(
        taxonomy[
            [
                "species_latin_name",
                "species_group",
                "NCBI_sci_name",
                "NCBI_match",
                "NCBI_rank_superkingdom",
                "NCBI_rank_kingdom",
                "NCBI_rank_phylum",
                "NCBI_rank_subphylum",
                "NCBI_rank_class",
                "NCBI_rank_order",
                "NCBI_rank_family",
                "NCBI_rank_genus",
                "NCBI_rank_species",
                "NCBI_last_known_rank",
                "flag",
            ]
        ],
        left_on=["species_latin_name", "species_group"],
        right_on=["species_latin_name", "species_group"],
        how="left",
    )

    # Correct ambiguous taxids and map missing species_groups using the NCBI_last_known_rank
    logger.info(
        "Correcting ambiguous species groups and mapping missing species_groups using the NCBI_last_known_rank..."
    )
    df, _ = preprocessor.correct_ambiguous_species_group_using_NCBI(df.copy())
    df, _ = preprocessor.map_missing_species_group_using_NCBI(df.copy())
    taxonomy, _ = preprocessor.correct_ambiguous_species_group_using_NCBI(
        taxonomy.copy()
    )
    taxonomy, _ = preprocessor.map_missing_species_group_using_NCBI(taxonomy.copy())

    # Generating new species groups based on our definitions using lineage information
    logger.info(
        "Generating species groups based on lineage information using our definitions..."
    )
    df["species_group_corrected"] = df.apply(
        lambda row: map_lineage_to_species_group(construct_ranked_lineage(row)), axis=1
    )
    taxonomy["species_group_corrected"] = taxonomy.apply(
        lambda row: map_lineage_to_species_group(construct_ranked_lineage(row)), axis=1
    )
    logger.success(
        f"Corrected {(df['species_group'] != df['species_group_corrected']).sum()} species groups based on lineage information using our definitions."
    )
    logger.info(
        "Filling missing taxids backwards and forwards in the lineage to fill in missing taxonomic information where possible..."
    )
    taxonomy = fill_missing_taxonomy(taxonomy, backwards=True)
    taxonomy = fill_missing_taxonomy(taxonomy, backwards=False)

    # Save the output file
    if args.out_taxonomic_info.endswith(".zip"):
        # If file has .csv in the name, save as .csv.zip
        taxonomy.to_csv(args.out_taxonomic_info, index=False, compression="zip")
        taxonomy.to_pickle(
            args.out_taxonomic_info.replace(".csv.zip", ".pkl.zip"), compression="zip"
        )
    else:
        taxonomy.to_csv(args.out_taxonomic_info, index=False)
        taxonomy.to_pickle(args.out_taxonomic_info.replace(".csv", ".pkl"))

    size_mb = os.path.getsize(args.out_taxonomic_info) / (1024 * 1024)
    logger.success(
        f"Phylogenetic info saved to {args.out_taxonomic_info} ({size_mb:.2f} MB)"
    )

    # Drop rows with flags since we are not interested in those for the downstream analysis, but log how many were dropped for each flag
    logger.warning(
        f"Dropping {df.flag.notna().sum()} rows with flags from the original data, since those are likely to be errors"
    )
    logger.warning(f"   Flag counts:\n{df['flag'].value_counts()}")
    df = df[df["flag"].isna()].drop(columns=["flag"])

    if args.out.endswith(".zip"):
        df.to_csv(args.out, compression="zip", index=False)
        df.to_pickle(args.out.replace(".csv.zip", ".pkl.zip"), compression="zip")
    else:
        df.to_csv(args.out, index=False)
        df.to_pickle(args.out.replace(".csv", ".pkl"))

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    logger.success(f"Data with NCBI info saved to {args.out} ({size_mb:.2f} MB)")

    # Save to duckdb database as "Step 4"
    con = duckdb.connect(args.database_out)
    con.execute("CREATE OR REPLACE TABLE step4 AS SELECT * FROM df")
    con.execute("CREATE OR REPLACE TABLE step4_taxonomy AS SELECT * FROM taxonomy")
    con.close()

    logger.success(
        f"Preprocessed data saved to duckdb database at {args.database_out} into table 'step4' & 'step4_taxonomy'"
    )

    # Optionally save metadata files for later use in modeling
    if args.save_filled_tree:
        logger.info("Saving filled newick tree...")
        tree = write_filled_newick_tree(taxonomy)
        tree.write(
            format=3,
            features=[],
            outfile=f"{os.path.dirname(args.out)}/taxonomy_tree_filled_missing_taxids.nwk",
        )
        logger.success(
            f"Filled newick tree saved to {os.path.dirname(args.out)}/taxonomy_tree_filled_missing_taxids.nwk"
        )
    if args.save_metadata_files:
        logger.info("Saving metadata files...")
        taxid2sciname = (
            df[["NCBI_match", "NCBI_sci_name"]]
            .drop_duplicates()
            .set_index("NCBI_match")["NCBI_sci_name"]
            .to_dict()
        )
        with open(f"{os.path.dirname(args.out)}/taxid2sciname.json", "w") as f:
            json.dump(taxid2sciname, f, indent=4)
        taxid2commonname = (
            df[["NCBI_match", "species_common_name"]]
            .drop_duplicates()
            .set_index("NCBI_match")["species_common_name"]
            .to_dict()
        )
        with open(f"{os.path.dirname(args.out)}/taxid2commonname.json", "w") as f:
            json.dump(taxid2commonname, f, indent=4)
        taxid2spgroup = (
            df[["NCBI_match", "species_group_corrected"]]
            .drop_duplicates()
            .set_index("NCBI_match")["species_group_corrected"]
            .to_dict()
        )
        with open(f"{os.path.dirname(args.out)}/taxid2spgroup.json", "w") as f:
            json.dump(taxid2spgroup, f, indent=4)
        taxid2rank = {}
        for rank in ranks:
            for taxid in taxonomy[f"NCBI_rank_{rank}"].dropna().unique():
                taxid2rank[taxid] = rank
        with open(f"{os.path.dirname(args.out)}/taxid2rank.json", "w") as f:
            json.dump(taxid2rank, f, indent=4)
        taxid2parent = {}
        for rank in ranks:
            parent_rank = (
                ranks[ranks.index(rank) - 1] if ranks.index(rank) > 0 else None
            )
            if parent_rank is None:
                continue
            for taxid in df[f"NCBI_rank_{rank}"].dropna().unique():
                parent_taxid = (
                    taxonomy.loc[
                        taxonomy[f"NCBI_rank_{rank}"] == taxid,
                        f"NCBI_rank_{parent_rank}",
                    ]
                    .dropna()
                    .unique()
                )
                if len(parent_taxid) > 0:
                    taxid2parent[taxid] = parent_taxid[0]
        with open(f"{os.path.dirname(args.out)}/taxid2parent.json", "w") as f:
            json.dump(taxid2parent, f, indent=4)
        logger.success(
            f"Metadata files saved to {os.path.dirname(args.out)}/taxid2sciname.json, taxid2spgroup.json, taxid2rank.json, taxid2parent.json"
        )


if __name__ == "__main__":
    main()
