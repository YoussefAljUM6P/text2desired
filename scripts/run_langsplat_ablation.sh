#!/bin/bash
# Grounding-module ablation on LERF-OVS: stage 1 by LangSplat-V2 (authors' pretrained models) instead of Grounding
# DINO + SAM, then our stages 2-3 unchanged (frozen parameters), then the same grounding evaluation.
# Outputs: out/langsplat_ground/<scene>/<query>/ (masks), out/lerf_ovs_langsplat/<scene>/<query>/ (runs + eval).
M=$(cd "$(dirname "$0")/.." && pwd)
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
V2=/media/youssef/DATA/langsplatv2_pretrained/LERF
PY=~/anaconda3/envs/pca_servo/bin/python
for s in figurines teatime ramen waldo_kitchen; do
  if [ ! -f $M/out/langsplat_ground/$s/done ]; then
    (cd $M/../litterature/code/LangSplatV2 && ~/anaconda3/envs/langsplat_v2/bin/python $M/scripts/langsplat_ground.py \
        -s $L/$s -m $V2/$s/${s}_0_1 --dataset_name $s --include_feature -r 2 --data_device cpu \
        --queries $M/out/lerf_ovs_queries_$s.txt --out $M/out/langsplat_ground/$s) && touch $M/out/langsplat_ground/$s/done
  fi
  O=$L/$s/output_gsplat_mcmc_df1
  for g in $M/out/langsplat_ground/$s/*/ground.json; do
    q=$(basename $(dirname $g)); o=$M/out/lerf_ovs_langsplat/$s/$q
    [ -f $o/summary.json ] && continue
    mkdir -p $o
    $PY $M/scripts/text_to_desired_image.py --query "${q//_/ }" --mask_json $g --model $O/ckpts/ckpt_29999_rank0.pt \
        --cfg $O/cfg.yml --images $L/$s/images --out $o > $o/run.log 2>&1 || echo "FAILED $s / $q"
  done
done
cd $M && $PY scripts/eval_lerf_ovs.py out/lerf_ovs_langsplat > out/lerf_ovs_langsplat/eval.log 2>&1
echo "all done"
