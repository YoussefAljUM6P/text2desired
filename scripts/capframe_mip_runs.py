"""B3 CapFrame on Mip-NeRF 360: its final poses (out/capframe/mip_<scene>/<query>/capframe.npz, from
scripts/capframe_batch.py on the models exported by scripts/capframe_export_mip.py) as run directories
out/baselines/capframe/<scene>/<query>/ (goal.npz + desired_image.png, at most MAXW px wide) for scripts/servo_eval.py.
The exported models are in our frame, so W2C = [R^T | T] converts directly: c2w = [R | -R T]. Failed queries
(capframe.json status != ok) get no run directory and count as failures.

  ~/anaconda3/envs/pca_servo/bin/python scripts/capframe_mip_runs.py        (from my_code/)
"""
import json

import numpy as np

from goal_benchmark import GS_VS, OUT, central_depth, write_run
from mip_baselines import D, MAXW, SCENES, scaled
from text2goal.cameras import load_colmap_views
from text2goal.render import GaussianScene


def main():
    for s in SCENES:
        root = OUT / "capframe" / f"mip_{s}"
        if not root.exists():
            continue
        v = load_colmap_views(str(OUT / "multiscene/cfg" / f"{s}.yml"), f"{D}/{s}/images", GS_VS)[0]
        Kd, Wd, Hd = scaled(v.K, v.W, v.H, min(v.W, MAXW))
        scene = GaussianScene(f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt")
        for q in sorted(root.iterdir()):
            st = json.loads((q / "capframe.json").read_text())["status"] if (q / "capframe.json").exists() else "none"
            if not (q / "capframe.npz").exists():
                print(f"{s}/{q.name}: {st}", flush=True)
                continue
            c = np.load(q / "capframe.npz")
            T = np.eye(4)
            T[:3, :3] = c["R"]
            T[:3, 3] = -c["R"] @ c["T"]
            rgb, depth = scene.render(T, Kd, Wd, Hd)
            write_run(OUT / "baselines/capframe" / s / q.name, T, Kd, Wd, Hd, rgb, central_depth(depth),
                      q.name.replace("_", " "))
            print(f"{s}/{q.name}: ok", flush=True)
        scene.release()


if __name__ == "__main__":
    main()
