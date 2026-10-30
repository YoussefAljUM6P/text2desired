#!/bin/bash
# Photometric servo toward the baselines' desired images (B1 training photo, B2 CLIP render-and-score), LERF-OVS,
# same protocol as ours (scripts/run_servo_eval.sh). Waits for scripts/goal_benchmark.py to write the baseline runs.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/pca_servo/bin/python
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
while [ ! -f out/benchmark/summary.md ]; do sleep 60; done
for m in refview clip; do
  for s in figurines teatime ramen waldo_kitchen; do
    $PY scripts/servo_eval.py --runs "out/baselines/$m/$s/*" --seeds 3 --iterations 600 \
        --model $L/$s/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/baseline_${m}_same.csv
  done
done
echo "all done"
