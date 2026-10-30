"""B1 (best training photo) and B2 (CLIP render-and-score) on the Mip-NeRF 360 queries, as in scripts/goal_benchmark.py.

For every run of ours in out/multiscene/<scene>/<query>/ (frozen parameters): B1 is the training photo our stage 1
picked (summary.json reference_view), B2 the level render, among every training camera x 5 yaws x 3 pitches, with
the highest CLIP ViT-L/14 similarity to "a photo of a <query>". No labels on Mip-NeRF 360: the goals are only written
as run directories (goal.npz + desired_image.png, at most MAXW px wide) in out/baselines/<method>/<scene>/<query>/.

  ~/anaconda3/envs/pca_servo/bin/python scripts/mip_baselines.py        (from my_code/)
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from goal_benchmark import GS_VS, OUT, PITCHES, YAWS, ClipScorer, central_depth, write_run
from text2goal.cameras import load_colmap_views
from text2goal.geometry import look_at, up_from_cameras
from text2goal.render import GaussianScene

D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
SCENES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room"]
CLIP_W, MAXW = 256, 1600


def scaled(K, W, H, w):
    k = w / W
    Ks = K.copy()
    Ks[:2] *= k
    return Ks, w, round(H * k)


def main():
    clip = ClipScorer()
    for s in SCENES:
        views = {Path(v.name).stem: v for v in load_colmap_views(str(OUT / "multiscene/cfg" / f"{s}.yml"),
                                                                 f"{D}/{s}/images", GS_VS)}
        vlist = list(views.values())
        up = up_from_cameras(np.stack([v.T_c2w for v in vlist]))
        scene = GaussianScene(f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt")
        K, W, H = vlist[0].K, vlist[0].W, vlist[0].H
        Kq, Wq, Hq = scaled(K, W, H, CLIP_W)
        Kd, Wd, Hd = scaled(K, W, H, min(W, MAXW))

        cands, imgs = [], []
        for v in vlist:
            f0 = v.T_c2w[:3, 2] - (v.T_c2w[:3, 2] @ up) * up
            f0 /= np.linalg.norm(f0)
            side = np.cross(up, f0)
            for yaw in YAWS:
                for pitch in PITCHES:
                    a, b = np.radians(yaw), np.radians(pitch)
                    fwd = np.cos(b) * (np.cos(a) * f0 + np.sin(a) * side) + np.sin(b) * up
                    T = look_at(v.T_c2w[:3, 3], v.T_c2w[:3, 3] + fwd, up)
                    if T is not None:
                        cands.append(T)
                        imgs.append(Image.fromarray(scene.render(T, Kq, Wq, Hq)[0]))
        emb = clip.images(imgs)
        del imgs
        print(f"{s}: {len(cands)} CLIP candidates", flush=True)

        for run in sorted((OUT / "multiscene" / s).iterdir()):
            if not (run / "goal.npz").exists():
                continue
            sm = json.loads((run / "summary.json").read_text())
            q = sm["query"]
            ref = views[Path(sm["reference_view"]).stem]
            T_clip = cands[int(torch.argmax(emb @ clip.text(q)))]
            for method, T in (("refview", ref.T_c2w), ("clip", T_clip)):
                rgb, depth = scene.render(T, Kd, Wd, Hd)
                if method == "refview":
                    rgb = np.asarray(Image.open(ref.image_path).convert("RGB").resize((Wd, Hd), Image.LANCZOS))
                write_run(OUT / "baselines" / method / s / run.name, T, Kd, Wd, Hd, rgb, central_depth(depth), q)
            print(f"{s}/{q}: done", flush=True)
        scene.release()


if __name__ == "__main__":
    main()
