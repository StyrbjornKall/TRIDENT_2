#!/bin/bash
echo "Running step 2: Adding SMILES data to the dataset..."

source .env
# Now we filter for a series of criteria to generate the final dataset
uv run $HOME/src/step2_add_smiles.py \
  --file=$OUTPUT_FILE_STEP2 \
  --out=$OUTPUT_FILE_STEP2 \
  --cas-to-smiles-mapping=$OUTPUT_CAS_SMILES_MAPPING_STEP2 \
  --database_out=$DATABASE

# Save file as pkl in python
uv run $HOME/src/pickle_files.py --file=$OUTPUT_FILE_STEP2

echo "Step 2 completed successfully."