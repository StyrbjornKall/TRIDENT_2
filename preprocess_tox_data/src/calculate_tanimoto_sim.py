import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import argparse
import h5py
from loguru import logger
from setup_logger import setup_logger


def morgan_fp(smiles_list, radius=2, fpSize=1024):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    fps, valid_indices, unprocessable_indices = [], [], []

    for i, smi in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        try:
            if mol is not None:
                fp = mfpgen.GetFingerprint(mol)
                if radius > 2:
                    arr = np.zeros(
                        (fpSize,), dtype=np.uint16
                    )  # Need to cast to high precision for large radius
                else:
                    arr = np.zeros((fpSize,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                valid_indices.append(i)
            else:
                raise ValueError("Invalid Mol")
        except Exception:
            unprocessable_indices.append(i)

    return fps, valid_indices, unprocessable_indices


def compute_block(i_range, X, bit_counts):
    """Compute a block of the Tanimoto similarity matrix."""
    i_start, i_end = i_range
    A = X[i_start:i_end]
    A_counts = bit_counts[i_start:i_end]

    intersection = A @ X.T
    union = A_counts[:, None] + bit_counts[None, :] - intersection

    block = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union != 0,
    )
    return i_start, i_end, block


# Main
def parallel_tanimoto(X, n_jobs=4, block_size=1000):
    logger.info("Computing Tanimoto similarity in parallel...")
    logger.info(f"Using {n_jobs} parallel jobs with block size {block_size}.")
    N = X.shape[0]
    bit_counts = np.sum(X, axis=1)

    # Prepare block ranges
    block_ranges = [(i, min(i + block_size, N)) for i in range(0, N, block_size)]

    results = []
    with Pool(n_jobs) as pool:
        func = partial(compute_block, X=X, bit_counts=bit_counts)
        for result in tqdm(
            pool.imap(func, block_ranges),
            total=len(block_ranges),
            desc="Parallel Tanimoto",
        ):
            results.append(result)

    # Collect results
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i_start, i_end, block in results:
        sim_matrix[i_start:i_end, :] = block

    return sim_matrix


def main(
    smiles: list[str],
    n_jobs: int,
    block_size: int,
    output_file_name: str,
    radius: int = 2,
    fpSize: int = 1024,
):
    fps, valid_indices, _ = morgan_fp(smiles, radius=radius, fpSize=fpSize)

    sim_matrix = parallel_tanimoto(
        X=np.array(fps), n_jobs=n_jobs, block_size=block_size
    )

    with h5py.File(output_file_name, "w") as h5f:
        h5f.create_dataset("similarity_matrix", data=sim_matrix)
        h5f.create_dataset("smiles", data=[smiles[i] for i in valid_indices])

    size_mb = os.path.getsize(output_file_name) / (1024 * 1024)
    logger.success(
        f"Saved Tanimoto similarity matrix to {output_file_name} (size: {size_mb:.2f} MB)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Tanimoto similarity from SMILES."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the file containing column SMILES.",
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        required=True,
        help="Path to save the output similarity matrix in HDF5 file.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Radius for Morgan fingerprint generation.",
    )
    parser.add_argument(
        "--fpSize", type=int, default=1024, help="Size of the Morgan fingerprint."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="Number of parallel jobs to use."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1000,
        help="Size of each block for parallel computation.",
    )

    # Add usage instructions
    parser.epilog = (
        "Example usage:\n"
        "python tanimoto.py \
            --file input_smiles.csv \
            --radius 2 \
            --fpSize 1024 \
            --n_jobs 4 \
            --block_size 1000 \
            --output_hdf5 output.h5 \n"
        "This will compute the Tanimoto similarity matrix for the SMILES strings in 'input_smiles.txt' "
        "using 4 parallel jobs and a block size of 1000, and save the results to 'output.h5'."
        "The generated similarity matrix and smiles will be stored in the HDF5 file under the key 'similarity_matrix', and 'smiles'."
    )

    args = parser.parse_args()
    setup_logger(level="INFO")
    logger.info(f"Arguments: {args}")
    logger.info(f"Reading SMILES from {args.file}...")

    if args.file.endswith(".txt"):
        with open(args.file, "r") as f:
            smiles = [line.strip() for line in f if line.strip()]
    elif args.file.endswith(".csv"):
        df = pd.read_csv(args.file)
        if "SMILES" not in df.columns:
            raise ValueError("CSV file must contain a 'SMILES' column.")
        smiles = df["SMILES"].dropna().unique().tolist()
    elif args.file.endswith(".csv.zip"):
        df = pd.read_csv(args.file, compression="zip", low_memory=False)
        if "SMILES" not in df.columns:
            raise ValueError("CSV file must contain a 'SMILES' column.")
        smiles = df["SMILES"].dropna().unique().tolist()
    else:
        raise ValueError(
            "Input file must be a .txt or .csv file containing SMILES strings."
        )
    logger.success(f"Read {len(smiles)} unique SMILES from {args.file}.")

    main(
        smiles=smiles,
        n_jobs=args.n_jobs,
        block_size=args.block_size,
        output_file_name=args.output_hdf5,
        radius=args.radius,
        fpSize=args.fpSize,
    )
