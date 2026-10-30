#!/bin/bash
# Oracle-grounding ablation on LERF-OVS: stage 1 = the dataset's labelled polygon (scripts/gt_ground.py), our
# stages 2-3 unchanged (frozen parameters). Outputs out/lerf_ovs_gt/<scene>/<query>/.
M=$(cd "$(dirname "$0")/.." && pwd)
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
PY=~/anaconda3/envs/pca_servo/bin/python
cd $M && $PY scripts/gt_ground.py
for s in figurines teatime ramen waldo_kitchen; do
  O=$L/$s/output_gsplat_mcmc_df1
  for g in $M/out/gt_ground/$s/*/ground.json; do
    q=$(basename $(dirname $g)); o=$M/out/lerf_ovs_gt/$s/$q
    [ -f $o/summary.json ] && continue
    mkdir -p $o
    $PY $M/scripts/text_to_desired_image.py --query "${q//_/ }" --mask_json $g --model $O/ckpts/ckpt_29999_rank0.pt \
        --cfg $O/cfg.yml --images $L/$s/images --out $o > $o/run.log 2>&1 || echo "FAILED $s / $q"
  done
done
echo "all done"
