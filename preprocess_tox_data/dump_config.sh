#!/bin/env bash

source /cephyr/users/skall/Alvis/preprocess_tox_data/.env

# Load modules
ml purge
ml load virtualenv/20.32.0-GCCcore-14.3.0
ml load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

source /mimer/NOBACKUP/groups/snic2022-22-552/skall/venvs/mds_gpu/bin/activate

python $HOME/src/dump_config.py \
  --out=$CONFIG_DUMP_PATH \
  --database-out=$DATABASE \
