#!/bin/bash
echo "Running step 7: Filtering the dataset based on concentration thresholds and other criteria..."

source .env
uv run $HOME/src/step7.py \
  --file=$OUTPUT_FILE_STEP4 \
  --out=$OUTPUT_FILE_STEP7 \
  --database_out=$DATABASE \
  --concentration_thresh=$CONCENTRATION_THRESH \
  --log_data \
  --turn_duration_units_other_than_hours_to_nan \

echo "Step 7 completed successfully."