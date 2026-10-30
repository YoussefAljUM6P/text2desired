#!/bin/bash
# Rerun the 6 runs whose chosen view scored 0 (every candidate tied at 0, picked by candidate order before the
# tie-break fix in text2goal/viewpoint.py, 2026-10-29). Old run directories are moved to out/_before_tiefix/.
#   bash scripts/rerun_tiefix.sh        (from my_code/)
PY=~/anaconda3/envs/pca_servo/bin/python
D=/media/youssef/LaCie/data/mip-nerf360/360_v2
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
B=out/_before_tiefix
backup() { mkdir -p $B/$(dirname $1); [ -d $B/$1 ] || mv $1 $B/$1; mkdir -p $1; }
for sq in "bicycle;bicycle" "room;piano bench"; do
  IFS=';' read -r s q <<< "$sq"; o=multiscene/$s/${q// /_}; backup out/$o
  M=$D/$s/output_gsplat_1_5_3_data_factor_1
  $PY scripts/text_to_desired_image.py --query "$q" --model $M/ckpts/ckpt_29999_rank0.pt \
      --cfg out/multiscene/cfg/$s.yml --images $D/$s/images --servo --out out/$o > out/$o/run.log 2>&1 || echo "FAILED $o"
  echo "done $o"
done
for rsq in "lerf_ovs_langsplat;langsplat_ground;ramen;spoon" "lerf_ovs_langsplat;langsplat_ground;figurines;tesla door handle" \
           "lerf_ovs_gt;gt_ground;ramen;sake cup" "lerf_ovs_gt;gt_ground;ramen;corn"; do
  IFS=';' read -r r g s q <<< "$rsq"; o=$r/$s/${q// /_}; backup out/$o
  O=$L/$s/output_gsplat_mcmc_df1
  $PY scripts/text_to_desired_image.py --query "$q" --mask_json out/$g/$s/${q// /_}/ground.json \
      --model $O/ckpts/ckpt_29999_rank0.pt --cfg $O/cfg.yml --images $L/$s/images --out out/$o > out/$o/run.log 2>&1 \
      || echo "FAILED $o"
  echo "done $o"
done
echo "all done"
