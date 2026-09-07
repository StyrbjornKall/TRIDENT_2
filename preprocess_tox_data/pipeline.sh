# !/bin/bash
echo "Running the full pipeline..."

#./step1.sh && 
#./step2.sh && 
./src/cas_to_smiles/get_smiles_from_cas.sh && 
./step2_add_smiles.sh && 
./step3.sh && 
./step4.sh && 
./step5.sh && 
./step7.sh && 
./step_collect_files.sh

echo "Pipeline completed successfully."