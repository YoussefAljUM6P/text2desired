"""Update the benchmark rows (out/benchmark/lerf_ovs_goals.csv) and grounding hits (grounding_hits.csv) of the LERF-OVS
runs redone after the tie-break fix (scripts/rerun_tiefix.sh), with the same metrics as scripts/goal_benchmark.py and
scripts/grounding_hits.py. The other rows are unchanged (their selection did not change). Old files are kept as *.before_tiefix.

  ~/anaconda3/envs/pca_servo/bin/python scripts/tiefix_update.py        (from my_code/)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from goal_benchmark import GS_VS, L, OUT, gt_instances, support_of, view_metrics
from grounding_hits import hit
from text2goal.cameras import load_colmap_views
from text2goal.geometry import up_from_cameras
from text2goal.render import GaussianScene

RERUN = [("ours_langsplat", "lerf_ovs_langsplat", "langsplat_v2", "ramen", "spoon"),
         ("ours_langsplat", "lerf_ovs_langsplat", "langsplat_v2", "figurines", "tesla door handle"),
         ("ours_gt", "lerf_ovs_gt", "gt_labels", "ramen", "sake cup"),
         ("ours_gt", "lerf_ovs_gt", "gt_labels", "ramen", "corn")]


def main():
    fb, fh = OUT / "benchmark/lerf_ovs_goals.csv", OUT / "benchmark/grounding_hits.csv"
    for f in (fb, fh):
        if not f.with_suffix(".csv.before_tiefix").exists():
            shutil.copy(f, f.with_suffix(".csv.before_tiefix"))
    b, h = pd.read_csv(fb), pd.read_csv(fh)
    for s in sorted({r[3] for r in RERUN}):
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        vlist = list(views.values())
        train_c2w = np.stack([v.T_c2w for v in vlist])
        up = up_from_cameras(train_c2w)
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        K, W, H = vlist[0].K, vlist[0].W, vlist[0].H
        for method, root, hcol, s2, q in RERUN:
            if s2 != s:
                continue
            run = OUT / root / s / q.replace(" ", "_")
            g = np.load(run / "goal.npz")
            T = g["T_goal"]
            _, depth = scene.render(T, K, W, H)
            m = view_metrics(gt[q], T, K, W, H, depth, up)
            m["support"] = support_of(T, float(g["d_star"]), train_c2w)
            sel = (b.scene == s) & (b["query"] == q) & (b.method == method)
            old = b.loc[sel].iloc[0].to_dict()
            for k, v in m.items():
                b.loc[sel, k] = v
            sm = json.loads((run / "summary.json").read_text())
            hs = (h.scene == s) & (h["query"] == q)
            h.loc[hs, hcol] = hit(np.array(sm["p_obj"]), gt[q])
            print(f"{s}/{q} [{method}] good_view {old['good_view']} -> {m['good_view']}, visible {old['visible']:.2f} -> "
                  f"{m['visible']:.2f}, hit {int(h.loc[hs, hcol].iloc[0])}", flush=True)
        scene.release()
    b.to_csv(fb, index=False)
    h.to_csv(fh, index=False)


if __name__ == "__main__":
    main()
