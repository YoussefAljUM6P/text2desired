#!/bin/bash
# Photometric servo toward our desired images (rendered from our gsplat model) with the current views rendered from
# another, independently trained 3DGS of the same scene (LangSplat-V2's RGB Gaussians, raw COLMAP frame): the camera
# does not see our render. LERF-OVS, 2 perturbation levels x 3 seeds. Output: out/servo_eval/lerf_ovs_cross.csv.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/pca_servo/bin/python
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
V2=/media/youssef/DATA/langsplatv2_pretrained/LERF
for s in figurines teatime ramen waldo_kitchen; do
  M=$L/$s/output_gsplat_mcmc_df1
  $PY scripts/servo_eval.py --runs "out/lerf_ovs/$s/*" --seeds 3 --iterations 600 --save_traj \
      --model $M/ckpts/ckpt_29999_rank0.pt --cfg $M/cfg.yml \
      --render_ply $V2/$s/${s}_0_1/point_cloud/iteration_10000/point_cloud.ply --csv out/servo_eval/lerf_ovs_cross.csv
done
echo "all done"
