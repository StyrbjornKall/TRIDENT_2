#!/bin/bash
echo "Running Butina clustering step..."
source .env

uv run $HOME/src/calculate_butina_clusters.py \
  --similarity_matrix=$TANIMOTO_SIMILARITY_HDF5 \
  --out=$BUTINA_CLUSTER_OUTPUT \
  --database_out=$DATABASE \
  --cutoffs 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99 \
  --num_processes=4

echo "Butina clustering step completed successfully."