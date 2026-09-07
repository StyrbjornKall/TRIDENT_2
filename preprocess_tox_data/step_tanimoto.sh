#!/bin/bash
echo "Running Tanimoto similarity calculation step..."
source .env

uv run $HOME/src/calculate_tanimoto_sim.py \
  --file=$OUTPUT_FILE_STEP7 \
  --output_hdf5=$TANIMOTO_SIMILARITY_HDF5 \
  --fpSize=$FPSIZE \
  --radius=$RADIUS \
  --n_jobs=$N_WORKERS \
  --block_size=$BLOCK_SIZE

echo "Tanimoto similarity calculation step completed successfully."