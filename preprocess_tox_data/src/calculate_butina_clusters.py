import h5py
import duckdb
import os
import numpy as np
from rdkit.ML.Cluster import Butina
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import argparse
from loguru import logger
from setup_logger import setup_logger


# Function to perform Butina clustering
def butina_clustering(distance_matrix, n_molecules, cutoff, flattened=False):

    # If the input is a square distance matrix, convert it to a flat vector
    if not flattened:
        # Convert matrix to distance matrix
        distance_matrix = distance_matrix[np.tril_indices(len(distance_matrix), k=-1)]

    # Perform Butina clustering
    clusters = Butina.ClusterData(distance_matrix, n_molecules, cutoff, isDistData=True)
    clusters = [list(cluster) for cluster in clusters]

    # plot stats on the clusters, number of clusters, min median avg max size of clusters
    cluster_sizes = [len(cluster) for cluster in clusters]
    logger.info(f"Number of chemicals: {n_molecules}")
    logger.info(f"Number of clusters: {len(clusters)}")
    logger.info(f"Min cluster size: {min(cluster_sizes)}")
    logger.info(f"Median cluster size: {np.median(cluster_sizes)}")
    logger.info(f"Average cluster size: {np.mean(cluster_sizes)}")
    logger.info(f"Max cluster size: {max(cluster_sizes)}")

    return clusters


def run_butina_clustering_parallel(similarity_matrix, cutoffs, num_processes=None):
    """
    Runs Butina clustering in parallel over multiple cutoff values.

    Args:
        similarity_matrix (ndarray): Square similarity matrix (e.g., Tanimoto).
        cutoffs (list of float): List of cutoff (distance) values to try.
        num_processes (int, optional): Number of parallel processes to use.
                                       If None, uses number of available CPU cores.

    Returns:
        dict: {cutoff: clusters}, where clusters is a list of lists of molecule indices.
    """
    # Convert similarity matrix to condensed distance array (lower triangle as expected by Butina)
    distance_array = (
        1 - similarity_matrix[np.tril_indices(len(similarity_matrix), k=-1)]
    )
    n_molecules = len(similarity_matrix)

    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                butina_clustering, distance_array, n_molecules, cutoff, True
            ): cutoff
            for cutoff in cutoffs
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Clustering"
        ):
            cutoff = futures[future]
            try:
                clusters = future.result()
                results[cutoff] = clusters
            except Exception as e:
                logger.error(f"Error with cutoff {cutoff}: {e}")
                results[cutoff] = None

    return results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Butina clustering on a similarity matrix."
    )
    parser.add_argument(
        "--similarity_matrix",
        type=str,
        required=True,
        help="Path to the HDF5 file containing the similarity matrix.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the output CSV file.",
    )
    parser.add_argument(
        "--database_out",
        type=str,
        required=True,
        help="Path to save the output duckdb database.",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        required=True,
        help="List of cutoff values for clustering.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes to use (default: 4).",
    )
    args = parser.parse_args()
    setup_logger(level="INFO")
    logger.info(f"Arguments: {args}")

    # Read the HDF5 file
    with h5py.File(args.similarity_matrix, "r") as f:
        smiles = [smi.decode("utf-8") for smi in f["smiles"][:]]
        similarity_matrix = f["similarity_matrix"][:]

    results = run_butina_clustering_parallel(
        similarity_matrix, cutoffs=args.cutoffs, num_processes=args.num_processes
    )

    df = pd.DataFrame({"SMILES": smiles})
    for cutoff, clusters in results.items():
        # Construct map between SMILES and cluster indices
        smiles_to_cluster = {}
        for cluster_index, cluster in enumerate(clusters):
            for molecule_index in cluster:
                smiles_to_cluster[smiles[molecule_index]] = cluster_index
        # Add cluster index to DataFrame
        df[f"Cluster_at_cutoff_{cutoff}"] = df["SMILES"].map(smiles_to_cluster)

    # Save the DataFrame to a CSV file
    if args.out.endswith(".csv"):
        df.to_csv(args.out, index=False)
    elif args.out.endswith(".zip"):
        df.to_csv(args.out, index=False, compression="zip")
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    logger.success(f"Saved clusters to {args.out} (size: {size_mb:.2f} MB)")

    # Save to database as well
    con = duckdb.connect(args.database_out)
    con.execute("CREATE TABLE butina_clusters AS SELECT * FROM df")
    logger.success(
        f"Saved clusters to database at {args.database_out} to table 'butina_clusters'"
    )

    # Example usage:
    # python calculate_butina_clusters.py \
    # --similarity_matrix path/to/similarity_matrix.h5 \
    # --cutoffs 0.3 0.4 0.5 \
    # --num_processes 4 \
    # --output_fpath path/to/output_clusters.csv
    # This will read the similarity matrix from the specified HDF5 file, run Butina clustering for the specified cutoff values in parallel
