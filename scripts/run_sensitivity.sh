#!/bin/bash
# Sensitivity of the viewpoint selection + continuous score, on both datasets (resumable). Run from my_code/.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/pca_servo/bin/python
$PY scripts/sensitivity.py --dataset mipnerf360 && $PY scripts/sensitivity.py --dataset lerf_ovs && echo "all done"
