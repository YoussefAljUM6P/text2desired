"""Export the Mip-NeRF 360 gsplat models in the format CapFrame reads, for baseline B3 on Mip-NeRF 360.

CapFrame (scripts/capframe_batch.py) only reads <model>/cameras.json (Inria format), <model>/point_cloud/iteration_*/
point_cloud.ply and the training images (<images>/<img_name>). Unlike LERF-OVS (LangSplat-V2's Gaussians, raw COLMAP
frame), the Gaussians and cameras are written in the frame of OUR model, so CapFrame renders the same reconstruction
as our method and its pose needs no conversion (W2C = [R^T | T] in our frame). Cameras are scaled to the images_4
resolution (about 1000 px wide, as LERF-OVS) and the images are symlinked as <stem> (img_name without extension).

  ~/anaconda3/envs/pca_servo/bin/python scripts/capframe_export_mip.py        (from my_code/)
Writes /media/youssef/DATA/capframe/mip/<scene>/ and /media/youssef/DATA/capframe/images/mip_<scene>/.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from goal_benchmark import GS_VS, OUT
from text2goal.cameras import load_colmap_views

D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
C = Path("/media/youssef/DATA/capframe")
SCENES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room"]


def write_ply(path, s):
    """gsplat splats -> Inria 3DGS ply (same parameterisation: log scales, logit opacities, wxyz quats, SH)."""
    means, scales, quats = s["means"], s["scales"], s["quats"]
    sh0 = s["sh0"].reshape(len(means), 3)                                   # (N, 1, 3)
    rest = s["shN"].transpose(0, 2, 1).reshape(len(means), -1)                 # (N, K, 3) -> (N, 3, K): channel-major
    cols = [means, np.zeros_like(means), sh0, rest, s["opacities"].reshape(-1, 1), scales, quats]
    data = np.concatenate(cols, 1).astype(np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    assert data.shape[1] == len(names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(data)}\n"
                 + "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
        f.write(data.tobytes())


def main():
    for s in SCENES:
        out, img = C / "mip" / s, C / "images" / f"mip_{s}"
        img.mkdir(parents=True, exist_ok=True)
        views = load_colmap_views(str(OUT / "multiscene/cfg" / f"{s}.yml"), f"{D}/{s}/images", GS_VS)
        small = {Path(n).stem: Path(D, s, "images_4", n) for n in os.listdir(f"{D}/{s}/images_4")}
        cams = []
        for i, v in enumerate(views):
            stem = Path(v.name).stem
            w, h = Image.open(small[stem]).size
            k = w / v.W
            R = v.T_c2w[:3, :3]
            assert abs(np.linalg.det(R) - 1) < 1e-4, "camera rotation is not orthonormal"
            cams.append({"id": i, "img_name": stem, "width": w, "height": h, "position": v.T_c2w[:3, 3].tolist(),
                         "rotation": R.tolist(), "fy": float(v.K[1, 1] * k), "fx": float(v.K[0, 0] * k)})
            link = img / stem
            if not link.exists():
                link.symlink_to(small[stem])
        out.mkdir(parents=True, exist_ok=True)
        (out / "cameras.json").write_text(json.dumps(cams))
        (out / "cfg_args").write_text(f"Namespace(sh_degree=3, source_path='{img}', model_path='{out}', "
                                      "images='images', resolution=-1, white_background=False, data_device='cuda', "
                                      "eval=False)")
        ck = torch.load(f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt", map_location="cpu",
                        weights_only=True)["splats"]
        write_ply(out / "point_cloud/iteration_30000/point_cloud.ply", {k: v.numpy() for k, v in ck.items()})
        print(f"{s}: {len(cams)} cameras ({cams[0]['width']}x{cams[0]['height']}), {len(ck['means']):,} Gaussians",
              flush=True)


if __name__ == "__main__":
    main()
