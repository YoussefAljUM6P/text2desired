"""InteriorGS (Habitat-GS validation houses) benchmark, step 2 (pca_servo env): desired images for a ground robot.

Per house (survey and ground truth from scripts/interiorgs_survey.py), every ObjectNav category is a text query:
  ours     grounding as in the paper (Grounding DINO + SAM on the survey views, lifted with the rendered depth) and
           the ground-robot viewpoint selection (text2goal/robot_view.py: navigable floor, camera height, level)
  refview  B1: the survey view in which the object was detected
  clip     B2: survey positions x 8 headings (level), highest CLIP ViT-L/14 similarity to "a photo of a <query>"
Scored against the ObjectNav objects of the category (centres only, no boxes), for the instance the view shows best:
  right_object  our lifted centroid within 1 m of an object centre of the category (grounding)
  visible       the object centre projects inside the image and the surface seen at that pixel (rendered depth,
                back-projected) lies within 1 m of the centre (the centre of a bed or sofa is inside the object)
  centred       horizontal offset of the centre from the image centre <= 15 % of the width (a ground robot cannot
                tilt: the vertical position is not controlled)
  good_view     visible and centred
Run directories (goal.npz, desired_image.png, summary.json) in out/interiorgs/<scene>/<method>/<query>/ for the servo.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/interiorgs_run.py [--scenes ...] [--queries ...]
Writes out/interiorgs/results.csv (resumable).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from goal_benchmark import ClipScorer
from text2goal.cameras import View
from text2goal.geometry import project
from text2goal.localize import localize
from text2goal.perception import Detector, Segmenter
from text2goal.render import GaussianScene
from text2goal.robot_view import robot_pose, select_robot_view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/interiorgs"
VAL = "/media/youssef/DATA/gs_imagenav/data/gs_scenes/val"
SAM = "/home/youssef/Documents/SW/checkpoints/sam_vit_h_4b8939.pth"
UP = np.array([0.0, 1.0, 0.0])                       # Habitat world: y up
FIELDS = ["scene", "query", "method", "right_object", "visible", "centred", "good_view", "h_offset", "distance"]


def score(T, centres, scene, K, W, H):
    """Best instance (visible, then smallest horizontal offset): visible, centred, offset, distance."""
    _, depth = scene.render(T, K, W, H)
    best = {"visible": 0, "centred": 0, "good_view": 0, "h_offset": 1.0, "distance": float("nan")}
    for c in centres:
        u, v, z = project(np.asarray(c)[None], T, K)
        if z[0] <= 0.05 or not (0 <= u[0] < W and 0 <= v[0] < H):
            continue
        dz = depth[int(v[0]), int(u[0])]
        ray = np.array([(u[0] - K[0, 2]) / K[0, 0], (v[0] - K[1, 2]) / K[1, 1], 1.0])
        hit3d = T[:3, :3] @ (ray * dz) + T[:3, 3]
        vis = int(np.isfinite(dz) and np.linalg.norm(hit3d - np.asarray(c)) <= 1.0)
        off = abs(u[0] - K[0, 2]) / W
        cand = {"visible": vis, "centred": int(vis and off <= 0.15), "h_offset": float(off), "distance": float(z[0])}
        cand["good_view"] = cand["centred"]
        if (cand["visible"], -cand["h_offset"]) > (best["visible"], -best["h_offset"]):
            best = cand
    return best


def write_run(d, T, K, W, H, rgb, d_star, query, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "desired_image.png")
    np.savez(d / "goal.npz", T_goal=T, K=K, W=W, H=H, d_star=d_star)
    (d / "summary.json").write_text(json.dumps({"query": query, "d_star": d_star, **(extra or {})}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--queries", nargs="*", default=None)
    args = ap.parse_args()
    path = OUT / "results.csv"
    done = {(r["scene"], r["query"]) for r in csv.DictReader(open(path))} if path.exists() else set()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        w.writeheader()
    clip, segmenter = ClipScorer(), Segmenter(SAM)
    for sd in sorted(p for p in OUT.iterdir() if (p / "survey.npz").exists()):
        sc = sd.name
        if args.scenes and sc not in args.scenes:
            continue
        sv = np.load(sd / "survey.npz")
        gt = json.loads((sd / "gt.json").read_text())
        K, W, H = sv["K"], int(sv["W"]), int(sv["H"])
        height, grid, nav = float(sv["camera_height"]), float(sv["grid"]), sv["navigable"]
        scene = GaussianScene(f"{VAL}/{sc}/{sc}.gs.ply")
        (sd / "images").mkdir(exist_ok=True)
        views = []
        for i, T in enumerate(sv["c2w"]):
            p = sd / "images" / f"survey_{i:03d}.jpg"
            if not p.exists():
                Image.fromarray(scene.render(T, K, W, H)[0]).save(p, quality=92)
            views.append(View(p.name, str(p), T, K, W, H))
        train_c2w = np.stack([v.T_c2w for v in views])
        # B2 candidates: survey positions x 8 headings, CLIP embeddings once per house
        Kq, Wq, Hq = K.copy(), W // 2, H // 2
        Kq[:2] *= 0.5
        cands, imgs = [], []
        for T in train_c2w:
            floor = T[:3, 3] - height * UP
            for a in np.radians(np.arange(0, 360, 45)):
                Tc = robot_pose(floor, T[:3, 3] + np.array([np.sin(a), 0, -np.cos(a)]), height, UP)
                cands.append(Tc)
                imgs.append(Image.fromarray(scene.render(Tc, Kq, Wq, Hq)[0]))
        emb = clip.images(imgs)
        del imgs
        for q, objs in sorted(gt.items()):
            if (sc, q) in done or (args.queries and q not in args.queries):
                continue
            centres = [o["position"] for o in objs]
            res = {}
            try:
                det = Detector(q, "IDEA-Research/grounding-dino-tiny", 0.2)
                obj = localize(q, views, scene, det, segmenter)
                del det
            except RuntimeError as e:                  # no detection: a miss for ours and for B1 (same grounding)
                print(f"{sc}/{q}: {e}", flush=True)
                obj = None
                res["ours"] = res["refview"] = (0, None)
            if obj is not None:
                right = int(min(np.linalg.norm(np.asarray(centres) - obj.centroid, axis=1)) <= 1.0)
                Tr = obj.view.T_c2w                    # B1: the survey view of the detection
                write_run(sd / "refview" / q.replace(" ", "_"), Tr, K, W, H, scene.render(Tr, K, W, H)[0],
                          float(np.linalg.norm(Tr[:3, 3] - obj.centroid)), q)
                res["refview"] = (right, Tr)
                try:
                    goal = select_robot_view(obj, scene, UP, train_c2w, nav, height, grid)
                    T = goal["T_goal"]
                    write_run(sd / "ours" / q.replace(" ", "_"), T, K, W, H, goal["desired_rgb"], goal["d_star"], q,
                              {"chosen": goal["chosen"]["name"], "reference_view": obj.view.name,
                               "p_obj": obj.centroid.tolist(), "rho": obj.rho})
                    res["ours"] = (right, T)
                except RuntimeError as e:              # no reachable candidate: a viewpoint failure of ours only
                    print(f"{sc}/{q}: {e}", flush=True)
                    res["ours"] = (right, None)
            k = int(torch.argmax(emb @ clip.text(q)))
            Tc = cands[k]
            write_run(sd / "clip" / q.replace(" ", "_"), Tc, K, W, H, scene.render(Tc, K, W, H)[0],
                      float(np.median(scene.render(Tc, K, W, H)[1][H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10])), q)
            res["clip"] = ("", Tc)
            for m, (right, T) in res.items():
                m_ = score(T, centres, scene, K, W, H) if T is not None else \
                    {"visible": 0, "centred": 0, "good_view": 0, "h_offset": 1.0, "distance": float("nan")}
                w.writerow({"scene": sc, "query": q, "method": m, "right_object": right, **m_})
                print(f"{sc}/{q} [{m}] right={right} " + " ".join(f"{a}={b}" for a, b in m_.items()), flush=True)
            f.flush()
        scene.release()
    f.close()


if __name__ == "__main__":
    main()
