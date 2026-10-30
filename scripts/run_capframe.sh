#!/bin/bash
# Baseline B3 CapFrame on the 67 LERF-OVS queries (scripts/capframe_batch.py), then the desired-image benchmark
# (scripts/goal_benchmark.py, adds the method capframe) and the photometric servo toward CapFrame's views.
# Everything heavy lives on the DATA disk (/media/youssef/DATA/capframe: env, HF cache, outputs; out/capframe links there).
# Run from my_code/:  bash scripts/run_capframe.sh
M=$(pwd)
C=/media/youssef/DATA/capframe
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
V2=/media/youssef/DATA/langsplatv2_pretrained/LERF
export HF_HOME=$C/hf_home U2NET_HOME=$C/u2net TMPDIR=$C/tmp PIP_CACHE_DIR=$C/pip_cache \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q INSTALL_OK $C/install.log 2>/dev/null; do sleep 60; done
until grep -q DONE $C/download.log 2>/dev/null; do sleep 60; done
FG=$(ls -d $HF_HOME/hub/models--qihoo360--fg-clip-base/snapshots/*/ | head -1)
for s in teatime figurines ramen waldo_kitchen; do
  # two processes per scene: MLLM stages, then GroundedSAM + Orient Anything + pose optimisation (GPU memory)
  for phase in mllm optim; do
    (cd ~/Documents/SW/capframe && $C/env/bin/python $M/scripts/capframe_batch.py \
        -m $V2/$s/${s}_0_1 -s $C/images/$s --model_root $FG --batch_size 2 \
        --queries $M/out/lerf_ovs_queries_$s.txt --out $M/out/capframe/$s --phase $phase)
  done
done
PY=~/anaconda3/envs/pca_servo/bin/python
OMP_NUM_THREADS=8 $PY scripts/goal_benchmark.py > out/goal_benchmark_capframe.log 2>&1
for s in figurines teatime ramen waldo_kitchen; do
  OMP_NUM_THREADS=8 $PY scripts/servo_eval.py --runs "out/baselines/capframe/$s/*" --seeds 3 --iterations 600 \
      --model $L/$s/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt --csv out/servo_eval/baseline_capframe_same.csv
done
echo "all done"
