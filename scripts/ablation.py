"""Ablation of the viewpoint selection and good-view rate of every sensitivity variant, on LERF-OVS (ground truth).

Stage 1 is not re-run: the object (cloud, centroid, normal, rho, reference view) is rebuilt from each run of ours
(out/lerf_ovs, Grounding DINO + SAM) as in scripts/sensitivity.py. For every variant the viewpoint is selected again and
the new goal view is scored against the ground-truth instances with the criteria of scripts/goal_benchmark.py
(visible, in_frame, center, fill, roll, good_view). Variants:
  the sensitivity variants of scripts/sensitivity.py (one parameter at -50 % / +50 %, continuous score, support floor);
  ablations (ViewParams.ablate): each score term removed (vis, cov, prior, fit: set to 1 in the score and in the
  decision rules), free space ignored (camera at the required distance), and no search: always the PCA-normal view,
  or always the reference-view direction (the training photo's direction, with our distance and level horizon).
"frozen" must reproduce the benchmark's "ours" rows (sanity check).

  ~/anaconda3/envs/pca_servo/bin/python scripts/ablation.py        (from my_code/)
Writes out/ablation/lerf_ovs.csv (resumable) and prints the good-view rate per variant.
"""
import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from goal_benchmark import GS_VS, L, OUT, SCENES, gt_instances, view_metrics
from sensitivity import P0, VARIANTS
from text2goal.cameras import load_colmap_views
from text2goal.geometry import up_from_cameras
from text2goal.render import GaussianScene
from text2goal.viewpoint import select_view

ABLATIONS = {"- vis": ("vis",), "- cov": ("cov",), "- prior": ("prior",), "- fit": ("fit",),
             "- free space": ("free_space",), "PCA normal only": ("normal_only",),
             "reference direction only": ("reference_only",)}
ALL = {**VARIANTS, **{k: {"ablate": v} for k, v in ABLATIONS.items()}}
FIELDS = ["scene", "query", "variant", "method", "chosen", "visible", "in_frame", "center", "fill", "roll_deg",
          "good_view"]


def main():
    path = OUT / "ablation/lerf_ovs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["scene"], r["query"], r["variant"]) for r in csv.DictReader(open(path))} if path.exists() else set()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        w.writeheader()
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        vlist = list(views.values())
        train_c2w = np.stack([v.T_c2w for v in vlist])
        up = up_from_cameras(train_c2w)
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        K, W, H = vlist[0].K, vlist[0].W, vlist[0].H
        for run in sorted((OUT / "lerf_ovs" / s).iterdir()):
            if not (run / "goal.npz").exists():
                continue
            g, sm = np.load(run / "goal.npz"), json.loads((run / "summary.json").read_text())
            q = sm["query"]
            if q not in gt:
                continue
            obj = SimpleNamespace(centroid=g["p_obj"], normal=g["n_obj"], points=g["points"], rho=sm["rho"],
                                  view=views[Path(sm["reference_view"]).stem])
            for name, change in ALL.items():
                if (s, q, name) in done:
                    continue
                try:
                    goal = select_view(obj, scene, up, train_c2w, replace(P0, **change))
                except RuntimeError as e:          # no candidate: a miss, scored as no view
                    print(f"{s}/{q} [{name}]: {e}", flush=True)
                    w.writerow({"scene": s, "query": q, "variant": name, "method": "failed", "chosen": "",
                                "visible": 0.0, "in_frame": 0.0, "center": 1.0, "fill": float("inf"),
                                "roll_deg": 0.0, "good_view": 0})
                    continue
                T = goal["T_goal"]
                _, depth = scene.render(T, K, W, H)
                m = view_metrics(gt[q], T, K, W, H, depth, up)
                w.writerow({"scene": s, "query": q, "variant": name, "method": goal["method"],
                            "chosen": goal["chosen"]["name"], **m})
                f.flush()
                print(f"{s}/{q} [{name}] {goal['chosen']['name']} good_view={m['good_view']}", flush=True)
        scene.release()
    f.close()
    rows = list(csv.DictReader(open(path)))
    for name in ALL:
        R = [int(r["good_view"]) for r in rows if r["variant"] == name]
        if R:
            print(f"{name:40s} good view {100 * np.mean(R):5.1f} % ({sum(R)}/{len(R)})")


if __name__ == "__main__":
    main()
