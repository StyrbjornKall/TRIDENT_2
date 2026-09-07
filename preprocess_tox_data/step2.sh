#!/bin/bash
echo "Running step 2: Preprocessing toxicity data..."

source .env
# Now we filter for a series of criteria to generate the final dataset
uv run $HOME/src/step2.py \
  --file=$OUTPUT_FILE_STEP1 \
  --out=$OUTPUT_FILE_STEP2 \
  --out-cas-list=$OUTPUT_CAS_LIST_STEP2 \
  --database_out=$DATABASE

# Save file as pkl in python
uv run $HOME/src/pickle_files.py --file=$OUTPUT_FILE_STEP2

echo "Step 2 completed successfully."