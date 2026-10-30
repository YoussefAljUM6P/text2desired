"""InteriorGS (Habitat-GS validation houses) benchmark, step 1 (habitat-gs env, navmesh only, no rendering).

The scenes are synthetic 3DGS houses without training photographs. A ground robot surveys the house: N_SURVEY level
camera poses at the robot's camera height (H = 1.25 m, as gs_vs_dichotomic_planner), on navigable floor points spread
by farthest-point sampling, with random headings. They play the role of the training views (grounding images, coverage
term). Also saved: the navigable floor on a 10 cm grid (where a candidate camera can stand) and the ObjectNav goal
objects (category -> object centres), used as ground truth.

Frames: Habitat world (y up), which is the frame of the .gs.ply; cameras are OpenCV camera-to-world.
  /media/youssef/DATA/gs_imagenav/envs/habitat-gs/bin/python scripts/interiorgs_survey.py        (from my_code/)
Writes out/interiorgs/<scene>/survey.npz and gt.json.
"""
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np
import habitat_sim

ROOT = Path(__file__).resolve().parents[1]
VAL = "/media/youssef/DATA/gs_imagenav/data/gs_scenes/val"
EPI = "/media/youssef/DATA/gs_imagenav/data/gs_scenes/episodes/objectnav/val/content"
H_CAM, N_SURVEY, GRID = 1.25, 200, 0.10
W, H, HFOV = 640, 480, 90.0


def c2w(pos, yaw):
    """OpenCV camera at floor point pos + H_CAM, looking along yaw (Habitat: y up, -z forward at yaw 0)."""
    f = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(f, up), -up, f], 1)
    T[:3, 3] = pos + np.array([0, H_CAM, 0])
    return T


def main():
    rng = np.random.default_rng(0)
    fx = W / 2 / np.tan(np.radians(HFOV) / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    for f in sorted(glob.glob(f"{EPI}/interior_*.json.gz")):
        sc = os.path.basename(f).split(".")[0]
        out = ROOT / "out/interiorgs" / sc
        out.mkdir(parents=True, exist_ok=True)
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(f"{VAL}/{sc}/{sc}.navmesh")
        lo, hi = (np.asarray(b) for b in pf.get_bounds())
        # navigable floor on a grid: snap each grid point to the navmesh, keep it if it stays within half a cell
        nav = []
        for x in np.arange(lo[0], hi[0], GRID):
            for z in np.arange(lo[2], hi[2], GRID):
                p = np.array([x, lo[1], z])
                q = np.asarray(pf.snap_point(p))
                if np.all(np.isfinite(q)) and np.hypot(q[0] - x, q[2] - z) < GRID / 2:
                    nav.append(q)
        nav = np.unique(np.round(np.array(nav), 3), axis=0)
        # survey: farthest-point sampling over the navigable floor (keeps the views spread), random headings
        idx = [int(rng.integers(len(nav)))]
        d = np.linalg.norm(nav - nav[idx[0]], axis=1)
        for _ in range(min(N_SURVEY, len(nav)) - 1):
            i = int(np.argmax(d))
            idx.append(i)
            d = np.minimum(d, np.linalg.norm(nav - nav[i], axis=1))
        cams = np.stack([c2w(nav[i], rng.uniform(-np.pi, np.pi)) for i in idx])
        np.savez(out / "survey.npz", c2w=cams, K=K, W=W, H=H, navigable=nav, camera_height=H_CAM, grid=GRID)
        d = json.load(gzip.open(f))
        gt = {}
        for key, goals in d["goals_by_category"].items():
            cat = key.split(".gs.ply_")[-1]
            gt[cat] = [{"id": g["object_id"], "position": g["position"],
                        "view_points": [v["agent_state"]["position"] for v in g.get("view_points", [])]} for g in goals]
        (out / "gt.json").write_text(json.dumps(gt))
        print(f"{sc}: {len(nav)} navigable cells, {len(cams)} survey views, {len(gt)} categories, "
              f"{sum(len(v) for v in gt.values())} objects", flush=True)


if __name__ == "__main__":
    main()
