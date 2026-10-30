#!/bin/bash
# Project A on LERF-OVS (held-out test, parameters frozen as tuned on the room scene): every annotated query of the
# 4 scenes, with our gsplat MCMC model of each scene (scripts/train_lerf.sh). Waits for each model to be trained.
# Outputs: out/lerf_ovs/<scene>/<query>/ (+ run.log). Run from my_code/:  bash scripts/run_lerf_ovs.sh
L=/media/youssef/DATA/lerf_ovs/lerf_ovs
PY=~/anaconda3/envs/pca_servo/bin/python
for s in figurines teatime ramen waldo_kitchen; do
  M=$L/$s/output_gsplat_mcmc_df1
  while [ ! -f $M/ckpts/ckpt_29999_rank0.pt ]; do sleep 60; done
  sleep 30                                             # let the trainer finish writing
  $PY - "$L/label/$s" > out/lerf_ovs_queries_$s.txt <<'PYEOF'
import json, glob, sys
qs = sorted({o['category'] for f in glob.glob(sys.argv[1] + '/*.json') for o in json.load(open(f))['objects']})
print('\n'.join(qs))
PYEOF
  while IFS= read -r q; do
    o="out/lerf_ovs/$s/${q// /_}"
    [ -f "$o/summary.json" ] && continue
    mkdir -p "$o"
    echo "=== $s / $q"
    $PY scripts/text_to_desired_image.py --query "$q" --model $M/ckpts/ckpt_29999_rank0.pt \
        --cfg $M/cfg.yml --images $L/$s/images --out "$o" > "$o/run.log" 2>&1 || echo "FAILED $s / $q"
  done < out/lerf_ovs_queries_$s.txt
done
echo "all done"
