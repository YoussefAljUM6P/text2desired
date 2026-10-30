"""B1 with a rendered desired image: same goal pose as B1 (the training camera our stage 1 picked) but the desired image is
the 3DGS render at that pose instead of the photograph, so that the servo compares renders with renders as for every
other method. Writes out/baselines/refview_render/<scene>/<query>/ (goal.npz copied, desired_image.png rendered).

  ~/anaconda3/envs/pca_servo/bin/python scripts/refview_render.py        (from my_code/)
"""
import shutil

import numpy as np
from PIL import Image

from goal_benchmark import L, OUT, SCENES
from text2goal.render import GaussianScene


def main():
    for s in SCENES:
        scene = GaussianScene(f"{L}/{s}/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt")
        for run in sorted((OUT / "baselines/refview" / s).iterdir()):
            if not (run / "goal.npz").exists():
                continue
            g = np.load(run / "goal.npz")
            d = OUT / "baselines/refview_render" / s / run.name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(run / "goal.npz", d / "goal.npz")
            shutil.copy(run / "summary.json", d / "summary.json")
            rgb = scene.render(g["T_goal"], g["K"], int(g["W"]), int(g["H"]))[0]
            Image.fromarray(rgb).save(d / "desired_image.png")
        print(f"{s}: done", flush=True)
        scene.release()


if __name__ == "__main__":
    main()
