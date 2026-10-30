#!/bin/bash
# Photometric servo toward every generated desired image (Mip-NeRF 360 27 queries + LERF-OVS 67 queries), current
# views rendered from the same model. 2 perturbation levels x 3 seeds. Resumable (rows already in the csv are skipped).
# Outputs: out/servo_eval/{mipnerf360,lerf_ovs}_same.csv. Run from my_code/:  bash scripts/run_servo_eval.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/pca_servo/bin/python
D=/media/youssef/LaCie/data/mip-nerf360/360_v2
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
for s in room counter kitchen bonsai garden bicycle; do
  $PY scripts/servo_eval.py --runs "out/multiscene/$s/*" --seeds 3 --iterations 600 --save_traj \
      --model $D/$s/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/mipnerf360_same.csv
done
for s in figurines teatime ramen waldo_kitchen; do
  $PY scripts/servo_eval.py --runs "out/lerf_ovs/$s/*" --seeds 3 --iterations 600 --save_traj \
      --model $L/$s/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/lerf_ovs_same.csv
done
echo "all done"
