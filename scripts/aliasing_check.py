"""Grey-level MSE between the desired image and the current view rendered AT THE GOAL POSE (no perturbation), as the
servo test builds them (scripts/servo_eval.py): desired image = full-resolution render area-downsampled to 320 px;
current view = direct 320 px render, or 4x supersampled and averaged. A large error at the goal pose itself is
aliasing, not a servo error.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/aliasing_check.py        (from my_code/)
Writes out/servo_eval/aliasing_check.csv and prints per-scene medians.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from goal_benchmark import L
from text2goal.photometric import gray
from text2goal.render import GaussianScene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
SETS = [("mipnerf360", s, OUT / "multiscene" / s, f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt")
        for s in ["room", "counter", "kitchen", "bonsai", "garden", "bicycle"]] + \
       [("lerf_ovs", s, OUT / "lerf_ovs" / s, f"{L}/{s}/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt")
        for s in ["figurines", "teatime", "ramen", "waldo_kitchen"]]
WIDTH = 320


def main():
    rows = []
    for ds, sc, runs_dir, ckpt in SETS:
        scene = GaussianScene(ckpt)
        for run in sorted(p for p in runs_dir.iterdir() if (p / "goal.npz").exists()):
            g = np.load(run / "goal.npz")
            T, W, H = g["T_goal"], int(g["W"]), int(g["H"])
            s = WIDTH / W
            w_, h_ = WIDTH, round(H * s)
            K = g["K"].copy()
            K[:2] *= s
            I_star = gray(cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                                     interpolation=cv2.INTER_AREA))
            row = {"dataset": ds, "scene": sc, "run": run.name, "width": W}
            for k in (1, 4):
                Kk = K.copy()
                Kk[:2, :2] *= k
                Kk[:2, 2] = (K[:2, 2] + 0.5) * k - 0.5
                rgb = scene.render(T, Kk, w_ * k, h_ * k)[0]
                if k > 1:
                    rgb = cv2.resize(rgb, (w_, h_), interpolation=cv2.INTER_AREA)
                row[f"mse_ss{k}"] = float(np.mean((gray(rgb) - I_star) ** 2))
            rows.append(row)
            print(row, flush=True)
        scene.release()
    with open(OUT / "servo_eval/aliasing_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby(["dataset", "scene"])[["mse_ss1", "mse_ss4"]].median().round(0))


if __name__ == "__main__":
    main()
