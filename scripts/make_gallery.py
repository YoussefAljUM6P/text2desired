"""Thumbnails of every method's desired image for the LERF-OVS benchmark, with the ground-truth object outlined.

For each (scene, query, method) of out/benchmark/lerf_ovs_goals.csv: the saved desired image, the GT instance the
benchmark scored it against (most visible one) drawn as its projected convex hull (green) with its centroid (dot),
and the image centre (white cross). Writes out/gallery/<scene>/<query>/<method>.jpg (360 px wide).

  ~/anaconda3/envs/pca_servo/bin/python scripts/make_gallery.py        (from my_code/)
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from goal_benchmark import GS_VS, L, OUT, SCENES, gt_instances, view_metrics
from text2goal.cameras import load_colmap_views
from text2goal.geometry import project, up_from_cameras
from text2goal.render import GaussianScene

WIDTH = 360


def goal_dir(method, s, run):
    return {"ours": OUT / "lerf_ovs", "ours_langsplat": OUT / "lerf_ovs_langsplat", "ours_gt": OUT / "lerf_ovs_gt"}.get(
        method, OUT / "baselines" / method) / s / run


def best_instance(inst, T, K, W, H, depth, up):
    return max(inst, key=lambda P: view_metrics([P], T, K, W, H, depth, up)["visible"])


def main():
    rows = defaultdict(list)
    for r in csv.DictReader(open(OUT / "benchmark/lerf_ovs_goals.csv")):
        rows[r["scene"]].append(r)
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        vlist = list(views.values())
        up = up_from_cameras(np.stack([v.T_c2w for v in vlist]))
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        for r in rows[s]:
            run = r["query"].replace(" ", "_")
            d = goal_dir(r["method"], s, run)
            out = OUT / "gallery" / s / run / f"{r['method']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not (d / "goal.npz").exists():                  # failed method: no image
                continue
            g = np.load(d / "goal.npz")
            T, K, W, H = g["T_goal"], g["K"], int(g["W"]), int(g["H"])
            img = np.asarray(Image.open(d / "desired_image.png").convert("RGB").resize((W, H)))
            _, depth = scene.render(T, K, W, H)
            P = best_instance(gt[r["query"]], T, K, W, H, depth, up)
            u, v, z = project(P, T, K)
            ok = z > 1e-3
            k = WIDTH / W
            img = cv2.resize(np.ascontiguousarray(img), (WIDTH, round(H * k)), interpolation=cv2.INTER_AREA)
            if ok.sum() >= 3:
                pts = np.stack([u[ok], v[ok]], 1) * k
                hull = cv2.convexHull(np.round(pts).astype(np.int32))
                cv2.polylines(img, [hull], True, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.polylines(img, [hull], True, (60, 255, 90), 2, cv2.LINE_AA)
                cu, cv_, cz = project(P.mean(0)[None], T, K)
                if cz[0] > 0:
                    cv2.circle(img, (int(cu[0] * k), int(cv_[0] * k)), 5, (60, 255, 90), -1, cv2.LINE_AA)
            cx, cy = int(K[0, 2] * k), int(K[1, 2] * k)
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.line(img, (cx - 8, cy), (cx + 8, cy), col, th)
                cv2.line(img, (cx, cy - 8), (cx, cy + 8), col, th)
            Image.fromarray(img).save(out, quality=82)
        print(f"{s}: {len(rows[s])} thumbnails", flush=True)
        scene.release()


if __name__ == "__main__":
    main()
