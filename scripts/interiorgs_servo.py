"""InteriorGS benchmark, step 3: photometric servo of a ground robot (3 degrees of freedom: lateral, forward, yaw)
toward the desired images of every method (run directories of scripts/interiorgs_run.py).

Start poses are reachable by the robot: the goal pose shifted on the floor by alpha * d* in a random horizontal
direction and turned by beta about the vertical axis (small: 0.1 d*, 5 deg; large: 0.2 d*, 10 deg), 3 seeds each.
The servo controls only (v_x, v_z, w_y) of the level OpenCV camera (text2goal/photometric.py, dof=(0, 2, 4)), so the
camera stays at its height, level. Success: final position within 0.02 d* and heading within 1 deg of the goal.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/interiorgs_servo.py
Writes out/interiorgs/servo.csv (resumable).
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from servo_eval import pose_error
from text2goal.photometric import gray, photometric_servo
from text2goal.render import GaussianScene
from text2goal.servo import exp_se3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/interiorgs"
VAL = "/media/youssef/DATA/gs_imagenav/data/gs_scenes/val"
LEVELS, SEEDS, WIDTH = [(0.1, 5.0), (0.2, 10.0)], 3, 320


def floor_perturbation(rng, alpha, beta_deg, d_star):
    """Camera-frame motion of a level camera: horizontal translation (x, z) of alpha * d*, then yaw (about y)."""
    a = rng.uniform(0, 2 * np.pi)
    t = alpha * d_star * np.array([np.cos(a), 0.0, np.sin(a)])
    yaw = np.radians(beta_deg) * rng.choice([-1.0, 1.0])
    return exp_se3(np.r_[t, 0, 0, 0]) @ exp_se3(np.r_[0, 0, 0, 0, yaw, 0])


def main():
    path = OUT / "servo.csv"
    fields = ["scene", "query", "method", "alpha", "beta", "seed", "d_star", "t_err_rel", "r_err", "mse", "iterations",
              "success"]
    done = {(r["scene"], r["query"], r["method"], r["alpha"], r["seed"]) for r in csv.DictReader(open(path))} \
        if path.exists() else set()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()
    for sd in sorted(p for p in OUT.iterdir() if (p / "survey.npz").exists()):
        runs = sorted(r for m in ("ours", "refview", "clip") for r in (sd / m).glob("*/goal.npz"))
        if not runs:
            continue
        scene = GaussianScene(f"{VAL}/{sd.name}/{sd.name}.gs.ply")
        for g_path in runs:
            run, method = g_path.parent, g_path.parent.parent.name
            g = np.load(g_path)
            T_goal, d_star, W, H = g["T_goal"], float(g["d_star"]), int(g["W"]), int(g["H"])
            s = WIDTH / W
            w_, h_ = WIDTH, round(H * s)
            K = g["K"].copy()
            K[:2] *= s
            I_star = gray(cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                                     interpolation=cv2.INTER_AREA))
            for alpha, beta in LEVELS:
                for seed in range(SEEDS):
                    key = (sd.name, run.name.replace("_", " "), method, str(alpha), str(seed))
                    if key in done:
                        continue
                    rng = np.random.default_rng(1000 * seed + int(100 * alpha) + int(beta))
                    T0 = T_goal @ floor_perturbation(rng, alpha, beta, d_star)
                    poses, errs = photometric_servo(lambda T: scene.render(T, K, w_, h_), T0, I_star, K,
                                                    iterations=600, dof=(0, 2, 4))
                    t1, r1 = pose_error(poses[-1], T_goal)
                    row = {"scene": sd.name, "query": key[1], "method": method, "alpha": alpha, "beta": beta,
                           "seed": seed, "d_star": d_star, "t_err_rel": t1 / d_star, "r_err": r1, "mse": errs[-1],
                           "iterations": len(errs), "success": int(t1 / d_star < 0.02 and r1 < 1.0)}
                    w.writerow(row)
                    f.flush()
            print(f"{sd.name}/{run.name} [{method}] done", flush=True)
        scene.release()
    f.close()


if __name__ == "__main__":
    main()
