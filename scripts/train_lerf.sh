#!/bin/bash
# gsplat MCMC, data factor 1, 30k iterations, cap 1M (same as the Mip-NeRF 360 models), one LERF-OVS scene after the other
cd /home/youssef/Documents/SW/visual_navigation/mini_PVS_v2_github/gsplat/examples
for s in figurines teatime ramen waldo_kitchen; do
  R=/media/youssef/DATA/lerf_ovs/lerf_ovs/$s/output_gsplat_mcmc_df1
  [ -f $R/ckpts/ckpt_29999_rank0.pt ] && continue
  mkdir -p $R
  ~/anaconda3/envs/gsrl/bin/python simple_trainer.py mcmc --data_dir /media/youssef/DATA/lerf_ovs/lerf_ovs/$s --data_factor 1 --result_dir $R      --disable_viewer --init_opa 0.5 --init_scale 0.1 --strategy.cap-max 1000000 > $R/train.log 2>&1 && echo "done $s" || echo "FAILED $s"
done
echo "all trained"
