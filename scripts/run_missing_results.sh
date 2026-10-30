#!/bin/bash
# Missing results of the journal paper (2026-10-29 night), sequentially on the GPU (each step ~3-5 GB):
#   1. benchmark rows / grounding hits of the 4 LERF-OVS runs redone after the tie-break fix
#   2. 3-D precision / recall of the lifted object per mask source (mask quality)
#   3. score ablation + good-view rate of every sensitivity variant (LERF-OVS)
#   4. B1 with a rendered desired image, and its photometric servo test (same settings as the other methods)
#   nohup bash scripts/run_missing_results.sh > out/missing_results.log 2>&1 &        (from my_code/)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=scripts:.
PY=~/anaconda3/envs/pca_servo/bin/python
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
echo "$(date) 1 tie-break benchmark update"; $PY scripts/tiefix_update.py
echo "$(date) 2 grounding 3-D precision/recall"; $PY scripts/grounding_3d.py
echo "$(date) 3 ablation"; $PY scripts/ablation.py > out/ablation_run.log 2>&1; tail -24 out/ablation_run.log
echo "$(date) 4 B1 rendered desired image + servo"; $PY scripts/refview_render.py
for s in figurines teatime ramen waldo_kitchen; do
  $PY scripts/servo_eval.py --runs "out/baselines/refview_render/$s/*" --seeds 3 --iterations 600 \
      --model $L/$s/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/baseline_refview_render_same.csv
done
echo "$(date) all done"
