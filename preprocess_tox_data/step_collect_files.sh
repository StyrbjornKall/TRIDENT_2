#!/bin/bash

source .env

# Create the output directory if it doesn't exist
mkdir -p "${COLLECTED_FILES_DIR}"

# Collect the relevant files from the previous steps
cp ${STORAGE}/preprocessed/step7/preprocessed_tox_data_SK_2026-04-17.pkl.zip ${COLLECTED_FILES_DIR}/
cp ${STORAGE}/preprocessed/step7/preprocessed_tox_data_SK_2026-04-17.csv.zip ${COLLECTED_FILES_DIR}/
cp ${BUTINA_CLUSTER_OUTPUT} ${COLLECTED_FILES_DIR}/
cp ${STORAGE}/preprocessed/step6/taxid2*.json ${COLLECTED_FILES_DIR}/
cp ${TAXONOMIC_INFORMATION_STEP4} ${COLLECTED_FILES_DIR}/
cp ${OUTPUT_NEWICK_STEP5} ${COLLECTED_FILES_DIR}/
cp ${DATABASE} ${COLLECTED_FILES_DIR}/

# Get metadata as well
cp "${STORAGE}/preprocessed/step4/taxid2rank.json" "${COLLECTED_FILES_DIR}/"
cp "${STORAGE}/preprocessed/step4/taxid2sciname.json" "${COLLECTED_FILES_DIR}/"
cp "${STORAGE}/preprocessed/step4/taxid2spgroup.json" "${COLLECTED_FILES_DIR}/"
cp "${STORAGE}/preprocessed/step4/taxid2parent.json" "${COLLECTED_FILES_DIR}/"