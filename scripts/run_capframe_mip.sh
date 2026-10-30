#!/bin/bash
# Night chain (2026-10-29): B3 CapFrame on the 27 Mip-NeRF 360 queries, then the photometric servo test (4x supersampled)
# toward B1, B2 and B3 on Mip-NeRF 360, and toward our 2 runs rerun after the tie-break fix.
# CapFrame needs ~22 GB: it starts only when the tie-break reruns are done and the GPU has >= 21 GB free.
#   nohup bash scripts/run_capframe_mip.sh > out/capframe_mip_run.log 2>&1 &        (from my_code/)
M=$(pwd)
C=/media/youssef/DATA/capframe
D=/media/youssef/LaCie/data/mip-nerf360/360_v2
PY=~/anaconda3/envs/pca_servo/bin/python
export HF_HOME=$C/hf_home U2NET_HOME=$C/u2net TMPDIR=$C/tmp PIP_CACHE_DIR=$C/pip_cache \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "all done" out/rerun_tiefix.log; do sleep 60; done
echo "$(date) tie-break reruns done; waiting for a free GPU"
until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -ge 21000 ]; do sleep 120; done
echo "$(date) GPU free, CapFrame starts"
FG=$(ls -d $HF_HOME/hub/models--qihoo360--fg-clip-base/snapshots/*/ | head -1)
for s in room counter kitchen bonsai garden bicycle; do
  for phase in mllm optim; do
    (cd ~/Documents/SW/capframe && $C/env/bin/python $M/scripts/capframe_batch.py \
        -m $C/mip/$s -s $C/images/mip_$s --model_root $FG --batch_size 2 \
        --queries $M/out/mip_queries_$s.txt --out $M/out/capframe/mip_$s --phase $phase)
  done
  echo "$(date) CapFrame $s done"
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PYTHONPATH=scripts:. $PY scripts/capframe_mip_runs.py
for s in room counter kitchen bonsai garden bicycle; do
  CK=$D/$s/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt
  for m in capframe refview clip; do
    $PY scripts/servo_eval.py --runs "out/baselines/$m/$s/*" --seeds 3 --iterations 600 --supersample 4 \
        --model $CK --csv out/servo_eval/mip_baseline_${m}_ss4.csv
  done
done
for r in bicycle/bicycle room/piano_bench; do
  s=${r%%/*}
  $PY scripts/servo_eval.py --runs "out/multiscene/$r" --seeds 3 --iterations 600 --supersample 4 \
      --model $D/$s/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt \
      --csv out/servo_eval/mipnerf360_same_ss4_tiefix.csv
done
# ablation + good-view rate of the sensitivity variants on LERF-OVS (scripts/ablation.py)
PYTHONPATH=scripts:. $PY scripts/ablation.py > out/ablation_run.log 2>&1
echo "$(date) all done"
