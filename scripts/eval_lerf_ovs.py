"""Grounding score of Project A on LERF-OVS, with the protocol of LangSplat / Dr. Splat (per annotated frame and query).

For every query and every annotated frame of its scene that contains it, the object's 3D point cloud
(out/lerf_ovs/<scene>/<query>/goal.npz, built by stage 1) is projected into that frame with our 3DGS model's camera;
points hidden in the rendered depth are dropped. Then
  localization : the median projected point lies inside the ground-truth box (LangSplat's localization accuracy)
  IoU          : the projected points, dilated by their own spacing and closed, against the ground-truth polygons
A query with no detection, or no visible point in the frame, counts as a miss (IoU 0).

  ~/anaconda3/envs/pca_servo/bin/python scripts/eval_lerf_ovs.py [runs dir, default out/lerf_ovs]   (from my_code/)
Writes <runs dir>/eval_pairs.csv and <runs dir>/eval_summary.md.
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text2goal.cameras import load_colmap_views  # noqa: E402
from text2goal.render import GaussianScene  # noqa: E402

L = "/media/youssef/DATA/lerf_ovs/lerf_ovs"
GS_VS = "/home/youssef/Documents/SW/visual_navigation/vanilla_gs_vs"
SCENES = ["figurines", "teatime", "ramen", "waldo_kitchen"]
OUT = ROOT / "out/lerf_ovs"


def gt_masks(scene):
    """{frame stem: {category: (mask HxW bool, box x0 y0 x1 y1)}} from the polygon labels."""
    out = {}
    for f in sorted(glob.glob(f"{L}/label/{scene}/*.json")):
        d = json.load(open(f)); W, H = d["info"]["width"], d["info"]["height"]
        per = defaultdict(lambda: np.zeros((H, W), np.uint8))
        for o in d["objects"]:
            cv2.fillPoly(per[o["category"]], [np.round(np.array(o["segmentation"])).astype(np.int32)], 1)
        out[Path(d["info"]["name"]).stem] = {c: (m > 0, cv2.boundingRect(m)) for c, m in per.items()}
    return out


def predicted_mask(P, view, depth):
    """Visible projected points of the object, dilated by their spacing and closed into a region."""
    T = np.linalg.inv(view.T_c2w); Xc = (T[:3, :3] @ P.T + T[:3, 3:4]).T; z = Xc[:, 2]
    ok = z > 1e-3
    uv = (view.K @ (Xc[ok] / z[ok, None]).T).T[:, :2]; z = z[ok]
    H, W = depth.shape
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, z = uv[inb], z[inb]
    ui = uv.astype(int)
    vis = z <= depth[ui[:, 1], ui[:, 0]] * 1.03 + 1e-3
    uv = uv[vis]
    if len(uv) < 5:
        return None, None
    from scipy.spatial import cKDTree
    sp = np.median(cKDTree(uv).query(uv, k=2)[0][:, 1])
    r = int(np.clip(np.ceil(1.5 * sp), 2, 25))
    m = np.zeros((H, W), np.uint8); m[uv[:, 1].astype(int), uv[:, 0].astype(int)] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(cv2.dilate(m, k), cv2.MORPH_CLOSE, k)
    return m > 0, np.median(uv, axis=0)


def main():
    global OUT
    if len(sys.argv) > 1:                       # other runs, e.g. out/lerf_ovs_langsplat (LangSplat-V2 grounding)
        OUT = ROOT / sys.argv[1]
    rows = []
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        if not Path(f"{M}/ckpts/ckpt_29999_rank0.pt").exists():
            print(f"{s}: model not trained yet, skipped"); continue
        gts = gt_masks(s)
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        scene, depths = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt"), {}
        for fr, objs in gts.items():
            v = views[fr]
            if fr not in depths:
                depths[fr] = np.asarray(scene.render(v.T_c2w, v.K, v.W, v.H)[1])
            for cat, (gm, (bx, by, bw, bh)) in objs.items():
                g = OUT / s / cat.replace(" ", "_") / "goal.npz"
                row = {"scene": s, "frame": fr, "query": cat, "detected": g.exists(), "loc": 0, "iou": 0.0}
                if g.exists():
                    pm, c = predicted_mask(np.load(g)["points"], v, depths[fr])
                    if pm is not None:
                        pm = cv2.resize(pm.astype(np.uint8), gm.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                        row["iou"] = float((pm & gm).sum() / max((pm | gm).sum(), 1))
                        row["loc"] = int(bx <= c[0] <= bx + bw and by <= c[1] <= by + bh)
                rows.append(row)
        print(f"{s}: {sum(r['scene'] == s for r in rows)} (frame, query) pairs")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eval_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["| scene | pairs | localization acc. (%) | mIoU (%) | queries detected |", "|---|---|---|---|---|"]
    for s in SCENES + ["mean"]:
        R = [r for r in rows if r["scene"] == s] if s != "mean" else None
        if s == "mean":
            per = [l for l in lines[2:]]
            vals = [[float(x) for x in l.split("|")[3:5]] for l in per]
            if vals:
                lines.append(f"| **mean (of scenes)** | | {np.mean([v[0] for v in vals]):.1f} | {np.mean([v[1] for v in vals]):.1f} | |")
            continue
        if not R:
            continue
        q = {r["query"]: r["detected"] for r in R}
        lines.append(f"| {s} | {len(R)} | {100 * np.mean([r['loc'] for r in R]):.1f} | {100 * np.mean([r['iou'] for r in R]):.1f} "
                     f"| {sum(q.values())}/{len(q)} |")
    (OUT / "eval_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
