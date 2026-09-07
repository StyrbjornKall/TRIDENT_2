#!/bin/bash
echo "Running step 5: Generating distance matrix and phylogenetic tree..."

source .env
uv run $HOME/src/step5.py \
  --file=$TAXONOMIC_INFORMATION_STEP4 \
  --out-dist-matrix=$OUTPUT_DIST_MATRIX_STEP5 \
  --out-tree=$OUTPUT_NEWICK_STEP5 \
  --ambig=$TREAT_AMBIGUOUS_DISTANCES \
  --remove-flagged-species \
  #--debug

echo "Step 5 completed successfully."
