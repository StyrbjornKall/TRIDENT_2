#!/usr/bin/env bash
source .env

# Script for batch_inference.py using GPU
SMILES_FILE="data/example_data/smiles.txt"
TAXID_FILE="data/example_data/taxids.txt"
MODEL="models/cn510alz_final_model_epoch_70_hf_trident2"
OUTPUT_DIR="tmp/"

mkdir -p "$OUTPUT_DIR"

echo "=== TRIDENT-2 inference running on ==="
echo "SMILES file : $SMILES_FILE"
echo "Taxid file  : $TAXID_FILE"
echo "Output dir  : $OUTPUT_DIR"
echo "Model       : $MODEL"
echo

uv run scripts/quick_inference.py \
    --smiles_file "$SMILES_FILE" \
    --taxid_file "$TAXID_FILE" \
    --output_pth "$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).csv" \
    --model "$MODEL" \
    --mixed_precision

echo
echo "=== Inference DONE ==="