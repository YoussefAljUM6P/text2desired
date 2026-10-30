#!/bin/bash
# Mip-NeRF 360 servo test again with the current view supersampled 4x (anti-aliased, like the desired image which is
# rendered at full resolution then downscaled). Output: out/servo_eval/mipnerf360_same_ss4.csv
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/pca_servo/bin/python
D=/media/youssef/LaCie/data/mip-nerf360/360_v2
for s in room counter kitchen bonsai garden bicycle; do
  $PY scripts/servo_eval.py --runs "out/multiscene/$s/*" --seeds 3 --iterations 600 --supersample 4 \
      --model $D/$s/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/mipnerf360_same_ss4.csv
done
echo "all done"
