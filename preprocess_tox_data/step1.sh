#!/bin/bash

echo "Running step 1: Preprocessing toxicity data..."

source .env
uv run $HOME/src/step1.py \
  --out=$OUTPUT_FILE_STEP1 \
  --database_out=$DATABASE \
  --acute_tox $RAW_DATA_ACUTE_TOX \
  --carc_tox  $RAW_DATA_CARC_TOX  \
  --repro_tox $RAW_DATA_REPRO_TOX \
  --rtecs_tox $RAW_DATA_RTECS_TOX \
  --aqter_tox $RAW_DATA_AQTER_TOX \

uv run $HOME/src/pickle_files.py --file=$OUTPUT_FILE_STEP1

echo "Step 1 completed successfully."