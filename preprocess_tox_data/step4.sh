#!/bin/bash
echo "Running step 4: Adding taxonomic information to the dataset..."

source .env
# Now we filter for a series of criteria to generate the final dataset
uv run $HOME/src/step4.py \
  --file=$OUTPUT_FILE_STEP3 \
  --out=$OUTPUT_FILE_STEP4 \
  --database-out=$DATABASE \
  --out-taxonomic-info=$TAXONOMIC_INFORMATION_STEP4 \

# Save file as pkl in python
uv run $HOME/src/pickle_files.py --file=$OUTPUT_FILE_STEP4

echo "Step 4 completed successfully."