"""Runtime of each stage (LERF-OVS, 2 queries per scene, one RTX GPU shared with nothing else when possible):
  grounding   Grounding DINO on every training view + SAM on the best view + lifting and PCA (text2goal.localize)
  selection   rendering and scoring the 74 candidates, desired image (text2goal.viewpoint.select_view)
  servo       one photometric servo iteration at 320 px (render + interaction matrix + LM step)
Model loading (detector, SAM, 3DGS) is reported apart, once per scene.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/timing.py        (from my_code/)
Writes out/timing.csv and prints medians.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from goal_benchmark import GS_VS, L, OUT, SCENES
from text2goal.cameras import load_colmap_views
from text2goal.geometry import up_from_cameras
from text2goal.localize import localize
from text2goal.perception import Detector, Segmenter
from text2goal.photometric import gray, photometric_servo
from text2goal.render import GaussianScene
from text2goal.servo import exp_se3
from text2goal.viewpoint import select_view

SAM = "/home/youssef/Documents/SW/checkpoints/sam_vit_h_4b8939.pth"
N_QUERIES, SERVO_IT, WIDTH = 2, 50, 320


def tic():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    rows = []
    t = tic()
    segmenter = Segmenter(SAM)
    load_sam = tic() - t
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)
        train_c2w = np.stack([v.T_c2w for v in views])
        up = up_from_cameras(train_c2w)
        t = tic()
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        load_gs = tic() - t
        queries = [json.loads((r / "summary.json").read_text())["query"]
                   for r in sorted((OUT / "lerf_ovs" / s).iterdir()) if (r / "goal.npz").exists()][:N_QUERIES]
        for q in queries:
            t = tic()
            det = Detector(q, "IDEA-Research/grounding-dino-tiny", 0.2)
            load_det = tic() - t
            t = tic()
            try:
                obj = localize(q, views, scene, det, segmenter)
            except RuntimeError:
                continue
            t_ground = tic() - t
            t = tic()
            goal = select_view(obj, scene, up, train_c2w)
            t_select = tic() - t
            W, H = views[0].W, views[0].H
            sc = WIDTH / W
            Ks = views[0].K.copy()
            Ks[:2] *= sc
            w_, h_ = WIDTH, round(H * sc)
            T_goal = goal["T_goal"]
            I_star = gray(scene.render(T_goal, Ks, w_, h_)[0])
            T0 = T_goal @ exp_se3(np.r_[0.1 * goal["d_star"], 0, 0, 0, np.radians(5), 0])
            t = tic()
            poses, errs = photometric_servo(lambda T: scene.render(T, Ks, w_, h_), T0, I_star, Ks,
                                            iterations=SERVO_IT)
            t_servo = (tic() - t) / len(errs)
            rows.append({"scene": s, "query": q, "views": len(views), "load_sam": load_sam, "load_gs": load_gs,
                         "load_det": load_det, "grounding": t_ground, "selection": t_select, "servo_iter": t_servo})
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            del det
        scene.release()
    out = OUT / "timing.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k in ("load_gs", "load_det", "grounding", "selection", "servo_iter"):
        print(f"{k:12s} median {np.median([r[k] for r in rows]):.3f} s")
    print(f"views per scene: {sorted({(r['scene'], r['views']) for r in rows})}")


if __name__ == "__main__":
    main()
