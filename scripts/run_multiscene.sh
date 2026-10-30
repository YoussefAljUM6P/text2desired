#!/bin/bash
# Project A on the Mip-NeRF 360 scenes: every query of the original report, with the gsplat (MCMC, data factor 1,
# 30k iterations, no depth prior) model of each scene. Outputs: out/multiscene/<scene>/<query>/ (+ run.log).
# Run from project_A_text_to_desired_image/my_code/:  bash scripts/run_multiscene.sh
D=/media/youssef/LaCie/data/mip-nerf360/360_v2
PY=~/anaconda3/envs/pca_servo/bin/python
declare -A Q
Q[room]="bookshelf;coffee table;door;lamp;piano;piano bench;plant;radiator;screen;side table;sofa;speaker;television;window"
Q[counter]="egg carton;metal bowl;oven mitt;potted plant"
Q[kitchen]="chair;toy bulldozer"
Q[bonsai]="bicycle;bonsai;cardboard box"
Q[garden]="round table;vase"
Q[bicycle]="bench;bicycle"
for s in room counter kitchen bonsai garden bicycle; do
  M=$D/$s/output_gsplat_1_5_3_data_factor_1
  # the trainer wrote its cfg on another machine: point data_dir to this disk
  sed "s#/media/youssefalj/#/media/youssef/#" $M/cfg.yml > out/multiscene/cfg/$s.yml
  IFS=';' read -ra QS <<< "${Q[$s]}"
  for q in "${QS[@]}"; do
    o=out/multiscene/$s/${q// /_}
    [ -f $o/summary.json ] && continue
    mkdir -p $o
    echo "=== $s / $q"
    $PY scripts/text_to_desired_image.py --query "$q" --model $M/ckpts/ckpt_29999_rank0.pt \
        --cfg out/multiscene/cfg/$s.yml --images $D/$s/images --servo --out $o > $o/run.log 2>&1 \
      || echo "FAILED $s / $q (see $o/run.log)"
  done
done
echo "all done"
