import os
import duckdb
import json
import pandas as pd
import numpy as np
from mds.mdscuda import MDS as MDS_cuda
import argparse
from loguru import logger
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from preprocess_taxonomy import evaluate_mds_embeddings
from setup_logger import setup_logger


def run_MDS(
    dm=None,
    dim=768,
    metric=True,
    max_iter=5000,
    n_init=1,
    eps="NaN",
):
    if dm is None:
        logger.error(
            "Distance matrix (dm) is not provided. Please provide a distance matrix to calculate MDS embeddings."
        )
        return

    logger.info("Running MDS with the following settings:")
    logger.info(f"Dimensions: {dim}")
    logger.info(f"Max iterations: {max_iter}")
    logger.info(f"Metric MDS: {metric}")
    logger.info(f"Number of initializations: {n_init}")
    start = time.time()
    mds = MDS_cuda(
        n_dims=dim, max_iter=max_iter, n_init=n_init, x_init=None, verbosity=2
    )
    mds_embeddings, sig, sigmas = mds.fit(squareform(dm.astype(np.float32).values))
    end = time.time()
    logger.success(f"MDS completed in {end - start:.2f} seconds")

    logger.info("Creating mapping of taxonomic embeddings to taxa")
    distances_and_embeds = pd.DataFrame(
        zip([c.lower() for c in dm.columns], mds_embeddings.astype(np.float32)),
        columns=["NCBI_taxa", "taxonomic_embedding"],
    )

    distances_and_embeds.set_index("NCBI_taxa", inplace=True)

    return mds_embeddings, distances_and_embeds, sig, sigmas


def evaluate_MDS(
    dm=None,
    mds_embeddings=None,
    sigmas: list = None,
    save_fig_fpath: str = "./shepard_plot.png",
):
    logger.info("Calculating evaluation metrics for MDS embeddings")
    evaluate_mds_embeddings(dm, mds_embeddings)

    # Create Shepard diagram
    logger.info("Creating Shepard diagram for MDS embeddings")
    mds_distance_matrix = pd.DataFrame(
        squareform(pdist(mds_embeddings, "euclidean")),
        columns=dm.columns,
        index=dm.index,
    )

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        x="Original Distances",
        y="MDS Distances",
        data=pd.DataFrame(
            {
                "Original Distances": dm.values.flatten(),
                "MDS Distances": mds_distance_matrix.values.flatten(),
            }
        ),
    )
    plt.xlabel("Original Distances")
    plt.ylabel("MDS Distances")
    plt.title("Violin Plot of MDS Distances by Original Distances")
    plt.grid()
    plt.tight_layout()
    logger.info(f"Saving Shepard diagram to {save_fig_fpath}...")
    plt.savefig(
        save_fig_fpath,
        dpi=300,
        bbox_inches="tight",
    )

    # Plot sigma over iterations
    if sigmas is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(sigmas[1:])
        plt.xlabel("Iteration")
        plt.ylabel("Sigma")
        plt.title("Sigma over MDS Iterations")
        plt.grid()
        plt.tight_layout()
        plt.yscale("log")
        sigmas_fig_fpath = save_fig_fpath.replace(".png", "_sigmas.svg")
        logger.info(f"Saving sigma plot to {sigmas_fig_fpath}...")
        plt.savefig(
            sigmas_fig_fpath,
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and evaluate MDS embeddings"
    )
    parser.add_argument(
        "--dist-matrix",
        required=True,
        help="Input distance matrix file path (pickle format)",
    )
    parser.add_argument(
        "--out-embeddings",
        required=True,
        help="Output MDS embeddings file path (pickle format, compressed with zip, will also be saved as csv)",
    )
    parser.add_argument(
        "--database-out",
        required=False,
        help="Output embeddings to database.",
    )
    parser.add_argument(
        "--rank2dist", action="store_true", help="Use rank to distance conversion"
    )
    parser.add_argument("--dist2LCA", action="store_true", help="Use distance to LCA")
    parser.add_argument(
        "--ambig",
        type=str,
        default="max",
        help="Ambiguity handling method (default: max)",
    )
    parser.add_argument(
        "--dim", type=int, default=768, help="Number of dimensions (default: 768)"
    )
    parser.add_argument(
        "--metric", action="store_true", default=True, help="Use metric MDS"
    )
    parser.add_argument(
        "--max-iter", type=int, default=5000, help="Maximum iterations (default: 5000)"
    )
    parser.add_argument(
        "--n-init", type=int, default=1, help="Number of initializations (default: 1)"
    )
    parser.add_argument(
        "--eps", type=str, default="NaN", help="Epsilon value (default: NaN)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print diagnostics and limit to 100 rows",
    )

    args = parser.parse_args()
    setup_logger(level="DEBUG" if args.debug else "INFO")
    logger.info(f"Arguments: {args}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out_embeddings), exist_ok=True)

    if args.dist_matrix.endswith(".csv") or args.dist_matrix.endswith(".csv.zip"):
        logger.warning(
            f"Loading distance matrix from CSV file is deprecated. Please use pickle format for faster loading. Attempting to load from {args.dist_matrix.replace('.csv', '.pkl')}..."
        )
        args.dist_matrix = args.dist_matrix.replace(".csv", ".pkl")
        if not os.path.exists(args.dist_matrix):
            logger.error(
                f"Pickle file not found at {args.dist_matrix}. Please provide a valid distance matrix file in pickle format."
            )
            exit(1)
    logger.info(f"Loading distance matrix from {args.dist_matrix}...")
    dm = pd.read_pickle(args.dist_matrix)
    if args.debug:
        args.max_iter = 2
        dm = dm.iloc[:100, :100]
    logger.success(f"Distance matrix loaded with shape {dm.shape}")

    mds_embeddings, distances_and_embeds, stress, sigmas = run_MDS(
        dm=dm,
        dim=args.dim,
        metric=args.metric,
        max_iter=args.max_iter,
        n_init=args.n_init,
        eps=args.eps,
    )

    evaluate_MDS(
        dm=dm,
        mds_embeddings=mds_embeddings,
        sigmas=sigmas,
        save_fig_fpath=args.out_embeddings.replace(".zip", "").replace(
            ".pkl", "_shepard_plot.png"
        ),
    )
    logger.success("MDS optimization and evaluation completed.")

    distances_and_embeds["taxonomic_embedding"].to_pickle(
        args.out_embeddings,
        compression="zip",
    )
    size_mb = os.path.getsize(args.out_embeddings) / (1024 * 1024)
    logger.success(
        f"Saved MDS embeddings to {args.out_embeddings} (size: {size_mb:.2f} MB)"
    )

    distances_and_embeds["taxonomic_embedding"].to_csv(
        args.out_embeddings.replace(".pkl", ".csv"),
        compression="zip",
    )
    size_mb = os.path.getsize(args.out_embeddings.replace(".pkl", ".csv")) / (
        1024 * 1024
    )
    logger.success(
        f"Saved MDS embeddings to {args.out_embeddings.replace('.pkl', '.csv')} (size: {size_mb:.2f} MB)"
    )

    # Also save to json
    data_to_save = {}
    for taxid, emb in zip(distances_and_embeds.index, distances_and_embeds.values):
        data_to_save[str(taxid)] = emb.tolist()
    with open(
        args.out_embeddings.replace(".zip", "").replace(".pkl", ".json"), "w"
    ) as f:
        json.dump(data_to_save, f, indent=2)  # Pretty-print with indent

    if args.database_out is not None:
        logger.info("Saving to local database")
        con = duckdb.connect(args.database_out)
        con.execute(
            "CREATE OR REPLACE TABLE step6_mds_embeddings AS SELECT * FROM distances_and_embeds"
        )
        con.close()
        logger.success(
            f"Saved MDS embeddings to database at {args.database_out} to table 'step6_mds_embeddings'"
        )

    # Save run config as json in the same directory as the distance matrix
    run_config = {
        "rank2dist": args.rank2dist,
        "dist2LCA": args.dist2LCA,
        "ambig": args.ambig,
        "dim": args.dim,
        "metric": args.metric,
        "max_iter": args.max_iter,
        "n_init": args.n_init,
        "eps": args.eps,
    }
    with open(args.out_embeddings.replace(".pkl", "_run_config.json"), "w") as f:
        import json

        json.dump(run_config, f, indent=4)
    logger.success(
        f"Run config saved to {args.out_embeddings.replace('.zip', '').replace('.pkl', '_run_config.json')}"
    )
