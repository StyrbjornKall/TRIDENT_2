#!/bin/env bash
#SBATCH -A NAISS2025-5-534          # find your project with the "projinfo" command
#SBATCH -p alvis                    # what partition to use (usually not needed)
#SBATCH -t 0-6:00:00                # how long time it will take to run
#SBATCH --gpus-per-node=A100:1        # choosing no. GPUs and their type
#SBATCH -J initial_test             # the jobname (not needed)
# #SBATCH -o SOME_FILENAME.out        # name of the output file

source /cephyr/users/skall/Alvis/preprocess_tox_data/.env

# Load modules
ml purge
ml load virtualenv/20.32.0-GCCcore-14.3.0
ml load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

source /mimer/NOBACKUP/groups/snic2022-22-552/skall/venvs/mds_gpu/bin/activate

# Echo the env variables to verify they are loaded correctly
echo "OUTPUT_DIST_MATRIX_STEP5: $OUTPUT_DIST_MATRIX_STEP5"
echo "OUTPUT_MDS_EMBEDDINGS_STEP6: $OUTPUT_MDS_EMBEDDINGS_STEP6"
echo "DATABASE: $DATABASE"
echo "TREAT_AMBIGUOUS_DISTANCES: $TREAT_AMBIGUOUS_DISTANCES"
echo "EMBEDDING_DIM: $EMBEDDING_DIM"
echo "MAX_ITER: $MAX_ITER"
echo "N_INIT: $N_INIT"

python $HOME/src/step6.py \
  --dist-matrix=$OUTPUT_DIST_MATRIX_STEP5 \
  --out-embeddings=$OUTPUT_MDS_EMBEDDINGS_STEP6 \
  --database-out=$DATABASE \
  --ambig=$TREAT_AMBIGUOUS_DISTANCES \
  --dim=$EMBEDDING_DIM \
  --max-iter=$MAX_ITER \
  --n-init=$N_INIT \
  --metric \
