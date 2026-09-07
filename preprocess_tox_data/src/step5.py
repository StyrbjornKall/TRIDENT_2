import pandas as pd
import numpy as np
import os
import json
import argparse
from typing import List, Dict, Tuple
from loguru import logger
import time
from preprocess_taxonomy import TaxonomyTreeProcessor
from setup_logger import setup_logger

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def main():
    parser = argparse.ArgumentParser(description="Find NCBI taxa for species names")
    parser.add_argument("--file", required=True, help="Input file path")
    parser.add_argument(
        "--out-tree", required=True, help="Output newick tree file path"
    )
    parser.add_argument(
        "--out-dist-matrix", required=True, help="Output distance matrix file path"
    )
    parser.add_argument("--dist2LCA", action="store_true", help="Use distance to LCA")
    parser.add_argument(
        "--ambig",
        required=True,
        default="max",
        help="If comparing different levels, this specifies which dist to use",
    )
    parser.add_argument(
        "--rank2dist",
        action="store_true",
        help="Whether to use the rank as the distance, phylum=7 etc.",
    )
    parser.add_argument(
        "--remove-flagged-species",
        action="store_true",
        help="Whether to remove species that are flagged for removal in the taxonomy file",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=[
            "superkingdom",
            "kingdom",
            "phylum",
            "subphylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ],
        help="Ranks to keep in the tree",
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

    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(args.out_tree), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_dist_matrix), exist_ok=True)

    # Specify the ranks to keep we are interested in for reflecting evolutionary similarity
    remove_flagged_species = args.remove_flagged_species
    ranks = args.ranks
    ambig = args.ambig

    # Print run metrics
    logger.info("Running with the following settings:")
    logger.info(f"  -remove_flagged_species: {remove_flagged_species}")
    logger.info(f"  -ranks: {ranks}")
    logger.info(f"  -dist2LCA: {args.dist2LCA}")
    logger.info(f"  -ambig: {ambig}")
    logger.info(f"  -rank2dist: {args.rank2dist}")

    # Read the taxonomy from the file
    if ".pkl" not in args.file:
        logger.warning(
            "You should use a pickled file to avoid incorrect formatting.\nAttempting to find corresponding pickle file..."
        )
        pkl_file = args.file.replace(".csv", ".pkl")
        if os.path.exists(pkl_file):
            logger.info(
                f"Found pickle file: {pkl_file}\nUsing this file instead of the csv."
            )
            args.file = pkl_file
        else:
            logger.error(
                f"No corresponding pickle file found for {args.file}. Please provide a pickled file to avoid formatting issues."
            )
            return
    taxonomy = pd.read_pickle(args.file)
    # Remove all filled taxids, they should not be used for distance calculation as the are not real
    for rank in ranks:
        taxonomy.loc[
            taxonomy[f"NCBI_rank_{rank}"].str.contains("_"), f"NCBI_rank_{rank}"
        ] = None

    if remove_flagged_species:
        logger.warning(
            f"Removing {taxonomy.flag.notna().sum()} flagged species from the taxonomy"
        )
        taxonomy = taxonomy[taxonomy.flag.isna()]

    if args.debug:
        taxonomy = taxonomy.head(100)

    start = time.time()

    # Now we will work on the tree in order to get the distance matrix
    treeprocessor = TaxonomyTreeProcessor()
    # First fetch the tree using NCBI taxonomy. We use last known rank to get the most complete tree, this will typically be the species taxid
    # We use intermediate_nodes to get all nodes in the tree, this will be removed later however.
    _ = treeprocessor.get_tree(
        TaxID=taxonomy.NCBI_last_known_rank.dropna().unique().tolist(),
        intermediate_nodes=True,
    )
    # We now prune the tree to remove any nodes that are not in the ranks we are interested in.
    # We do not preserve branch length here, as since we kept the intermediate nodes, the branch lengths are not meaningful.
    _ = treeprocessor.prune_tree(ranks=ranks, preserve_branch_length=False)
    # We now adjust the branch lengths to account for missing ranks. For example, if a species has no genus but has family, the edge length between the species and family will be adjusted to 2
    _ = treeprocessor.adjust_branch_lengths_for_missing_ranks(ranks=ranks)
    treeprocessor.plot_tree()

    # Print number of nodes per rank
    for rank in ranks:
        logger.info(
            f"Rank: {rank}, Number of nodes: {len(treeprocessor.tree.search_nodes(rank=rank))}"
        )

    # Save tree to file
    treeprocessor.tree.write(
        format=3,
        features=[],
        outfile=args.out_tree,
        format_root_node=True,
    )

    # Read the tree from the file
    dm = treeprocessor.get_pairwise_distance_matrix(
        ranks=None,  # Ranks are implicit in the tree
        return_pandas=True,
        return_node_sci_names=False,
        use_distance_to_common_ancestor=args.dist2LCA,  # We calculate distances to LCA as opposed to steps between nodes
        treat_ambiguity=ambig,  # If comparing different ranks, we take the max distance
        use_rank_as_distance=args.rank2dist,  # We do not use rank as distance, as we are using the tree to calculate distances
        rank_to_dist=None,  # We do not use rank to distance, as we are using the tree to calculate distances
    )

    end = time.time()
    elapsed = end - start
    logger.info(
        f"Time taken to process tree and calculate distance matrix: {elapsed / 60:.2f} minutes"
    )

    dm.to_csv(
        args.out_dist_matrix,
        index=False,
        header=True,
        compression="zip",
    )
    dm.to_pickle(
        args.out_dist_matrix.replace(".csv.zip", ".pkl.zip"),
        compression="zip",
    )

    size_mb = os.path.getsize(args.out_dist_matrix) / (1024 * 1024)

    logger.success(f"Tree saved to {args.out_tree}")
    logger.success(
        f"Distance matrix saved to {args.out_dist_matrix} ({size_mb:.2f} MB)"
    )

    # Save run config as json in the same directory as the distance matrix
    run_config = {
        "remove_flagged_species": remove_flagged_species,
        "ranks": ranks,
        "dist2LCA": args.dist2LCA,
        "ambig": ambig,
        "rank2dist": args.rank2dist,
    }
    with open(args.out_dist_matrix.replace(".csv.zip", "_run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)
    logger.success(
        f"Run config saved to {args.out_dist_matrix.replace('.csv.zip', '_run_config.json')}"
    )


if __name__ == "__main__":
    main()
