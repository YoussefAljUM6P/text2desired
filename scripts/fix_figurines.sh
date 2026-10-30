#!/bin/bash
# figurines: the trainer's scene scale was 45 (outlying cameras) instead of ~2 -> broken model (PSNR 13).
# Retrain with global_scale 0.045 (scene scale ~2, like the other LERF scenes), then re-run its queries and the eval.
cd /home/youssef/Documents/SW/visual_navigation/mini_PVS_v2_github/gsplat/examples
R=/media/youssef/DATA/lerf_ovs/lerf_ovs/figurines/output_gsplat_mcmc_df1; mkdir -p $R
~/anaconda3/envs/gsrl/bin/python simple_trainer.py mcmc --data_dir /media/youssef/DATA/lerf_ovs/lerf_ovs/figurines --data_factor 1 --result_dir $R    --disable_viewer --init_opa 0.5 --init_scale 0.1 --strategy.cap-max 1000000 --global_scale 0.045 > $R/train.log 2>&1
cd /home/youssef/Documents/SW/visual_navigation/gs_imagenav/project_A_text_to_desired_image/my_code && bash scripts/run_lerf_ovs.sh && ~/anaconda3/envs/pca_servo/bin/python scripts/eval_lerf_ovs.py > out/lerf_ovs/eval.log 2>&1
echo "figurines fixed"
