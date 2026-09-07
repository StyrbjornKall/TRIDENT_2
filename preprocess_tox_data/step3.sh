#!/bin/bash
echo "Running step 3..."

source .env
# Now we filter for a series of criteria to generate the final dataset
uv run $HOME/src/step3.py \
  --file=$OUTPUT_FILE_STEP2 \
  --lifestage_mapping=$DATA_LS_MAPPING \
  --administration_route_mapping=$DATA_ADMIN_ROUTE_MAPPING \
  --out=$OUTPUT_FILE_STEP3 \
  --out-taxa=$OUTPUT_LATIN_NAMES_STEP3 \
  --database-out=$DATABASE 

# Save file as pkl in python
uv run $HOME/src/pickle_files.py --file=$OUTPUT_FILE_STEP3

echo "Step 3 completed successfully."