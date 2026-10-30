"""Task-level benchmark on LERF-OVS: text -> goal view, ours vs baselines, scored against ground-truth 3D objects.

Ground truth: every labelled polygon of a category (LERF-OVS label frames) is back-projected with our 3DGS depth
(2-px erosion, 5-95 % depth percentiles), merged over frames, cleaned, and split into instances (DBSCAN). A goal
view is scored against the instance it shows best:
  visible      fraction of the instance's points inside the image and not occluded (rendered depth test)
  in_frame     fraction inside the image (occlusion ignored): 1 = whole object in view
  center       distance of the instance centroid's projection to the image centre / image diagonal
  fill         larger projected extent / image size (99th percentile around the centroid)
  roll_deg     tilt of the horizon (image x axis vs the plane orthogonal to the world up)
  support      training-camera support of the viewpoint (same kernel as stage 2): render reliability
  good_view    visible >= 0.5 and in_frame >= 0.9 and center <= 0.15 and 0.15 <= fill <= 0.9
Methods (same queries):
  ours         goal pose of out/lerf_ovs/<scene>/<query>/goal.npz
  refview      B1 retrieval: the training photo our stage 1 picked (best whole Grounding DINO detection);
               same grounding as ours, no viewpoint synthesis
  ours_langsplat  grounding ablation: LangSplat-V2 masks (out/lerf_ovs_langsplat) + our stages 2-3
  ours_gt      oracle grounding: the dataset's labelled polygon (out/lerf_ovs_gt, scripts/gt_ground.py) + stages 2-3
  capframe     B3 CapFrame (scripts/capframe_batch.py, out/capframe/): its final pose, COLMAP frame -> ours
  clip         B2 render-and-score: every training camera x 5 yaws x 3 pitches (level horizon) rendered, the view
               with the highest CLIP ViT-L/14 similarity to "a photo of a <query>" is kept
B1 and B2 goals are also written as run directories (goal.npz + desired_image.png) in out/baselines/<method>/,
so that scripts/servo_eval.py can servo toward them.

  ~/anaconda3/envs/pca_servo/bin/python scripts/goal_benchmark.py              (from my_code/)
Writes out/benchmark/lerf_ovs_goals.csv and out/benchmark/summary.md.
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text2goal.cameras import colmap_to_model_transform, load_colmap_views  # noqa: E402
from text2goal.geometry import backproject_mask, clean_point_cloud, look_at, project, up_from_cameras  # noqa: E402
from text2goal.render import GaussianScene  # noqa: E402
from text2goal.viewpoint import ViewParams  # noqa: E402

L = "/media/youssef/DATA/lerf_ovs/lerf_ovs"
GS_VS = "/home/youssef/Documents/SW/visual_navigation/vanilla_gs_vs"
SCENES = ["figurines", "teatime", "ramen", "waldo_kitchen"]
OUT = ROOT / "out"
YAWS, PITCHES = [-30, -15, 0, 15, 30], [-15, 0, 15]


def gt_instances(scene_name, views, scene):
    """{category: [instance point clouds]} from the polygon labels."""
    pts = defaultdict(list)
    for f in sorted(glob.glob(f"{L}/label/{scene_name}/*.json")):
        d = json.load(open(f))
        v = views[Path(d["info"]["name"]).stem]
        rgb, depth = scene.render(v.T_c2w, v.K, v.W, v.H)
        sx, sy = v.W / d["info"]["width"], v.H / d["info"]["height"]
        for o in d["objects"]:
            m = np.zeros((v.H, v.W), np.uint8)
            poly = np.array(o["segmentation"], np.float64) * [sx, sy]
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            if m.sum() < 30:
                continue
            try:
                pts[o["category"]].append(backproject_mask(m > 0, depth, rgb, v.K, v.T_c2w)[0])
            except ValueError:
                pass
    import open3d as o3d

    out = {}
    for cat, P in pts.items():
        P = np.concatenate(P)
        lo, hi = np.percentile(P, [2, 98], axis=0)
        vox = float(np.linalg.norm(hi - lo)) / 200.0
        P, _ = clean_point_cloud(P, np.zeros_like(P), vox)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        lab = np.asarray(pcd.cluster_dbscan(eps=4 * vox, min_points=10))
        inst = [P[lab == k] for k in range(lab.max() + 1) if (lab == k).sum() >= max(30, 0.05 * len(P))]
        out[cat] = inst or [P]
    return out


def view_metrics(inst, T, K, W, H, depth, up):
    """Metrics of one goal view against the GT instance it shows best (depth: render at T, size W x H)."""
    roll = float(np.degrees(np.arcsin(np.clip(abs(T[:3, 0] @ up), 0, 1))))
    best = None
    for P in inst:
        u, v, z = project(P, T, K)
        inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vis = np.zeros(len(P), bool)
        if inside.any():
            vis[inside] = depth[v[inside].astype(int), u[inside].astype(int)] >= z[inside] * 0.97 - 1e-3
        c = P.mean(0)
        cu, cv, cz = project(c[None], T, K)
        center = float(np.hypot(cu[0] - K[0, 2], cv[0] - K[1, 2]) / np.hypot(W, H)) if cz[0] > 0 else 1.0
        if (z > 1e-3).all():
            fill = float(max(2 * np.percentile(np.abs(u - cu[0]), 99) / W, 2 * np.percentile(np.abs(v - cv[0]), 99) / H))
        else:
            fill = float("inf")
        m = {"visible": float(vis.mean()), "in_frame": float(inside.mean()), "center": min(center, 1.0),
             "fill": fill, "roll_deg": roll}
        if best is None or m["visible"] > best["visible"]:
            best = m
    best["good_view"] = int(best["visible"] >= 0.5 and best["in_frame"] >= 0.9 and best["center"] <= 0.15
                            and 0.15 <= best["fill"] <= 0.9)
    return best


def central_depth(depth):
    h, w = depth.shape
    p = depth[h // 2 - h // 20: h // 2 + h // 20 + 1, w // 2 - w // 20: w // 2 + w // 20 + 1]
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(p)) if len(p) else 1.0


def support_of(T, d, train_c2w, prm=ViewParams()):
    """Training-camera support (stage-2 kernel) of the view T looking at the point at distance d on its axis."""
    p = T[:3, 3] + d * T[:3, 2]
    v = (T[:3, 3] - p) / d
    off = train_c2w[:, :3, 3] - p
    dist = np.linalg.norm(off, axis=1)
    z = np.einsum("nij,ni->nj", train_c2w[:, :3, :3], -off)[:, 2]          # p in each camera: depth > 0 if in front
    ang = np.degrees(np.arccos(np.clip((off / dist[:, None]) @ v, -1, 1)))
    k = np.exp(-0.5 * (ang / prm.support_sigma_deg) ** 2) * np.exp(-0.5 * (np.log(dist / d) / prm.support_sigma_log_distance) ** 2)
    return float((k * (z > 0)).max())


class ClipScorer:
    def __init__(self, name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(name).cuda().eval().half()
        self.proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def images(self, imgs, batch=64):
        out = []
        for i in range(0, len(imgs), batch):
            x = self.proc(images=imgs[i:i + batch], return_tensors="pt")["pixel_values"].cuda().half()
            out.append(torch.nn.functional.normalize(self.model.get_image_features(pixel_values=x), dim=-1).float())
        return torch.cat(out)

    @torch.no_grad()
    def text(self, query):
        t = self.proc(text=[f"a photo of a {query}"], return_tensors="pt", padding=True)
        return torch.nn.functional.normalize(self.model.get_text_features(
            input_ids=t["input_ids"].cuda(), attention_mask=t["attention_mask"].cuda()), dim=-1).float()[0]


def write_run(d, T, K, W, H, rgb, d_star, query):
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "desired_image.png")
    np.savez(d / "goal.npz", T_goal=T, K=K, W=W, H=H, d_star=d_star)
    (d / "summary.json").write_text(json.dumps({"query": query, "d_star": d_star}))


def main():
    rows, clip = [], ClipScorer()
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        vlist = list(views.values())
        train_c2w = np.stack([v.T_c2w for v in vlist])
        up = up_from_cameras(train_c2w)
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        A = colmap_to_model_transform(f"{M}/cfg.yml", GS_VS)
        sA = float(np.cbrt(np.linalg.det(A[:3, :3])))
        gt = gt_instances(s, views, scene)
        print(f"{s}: GT for {len(gt)} categories, {sum(len(v) for v in gt.values())} instances", flush=True)
        K, W, H = vlist[0].K, vlist[0].W, vlist[0].H
        Kq, Wq, Hq = K.copy(), W // 4, H // 4
        Kq[:2] *= 0.25

        # B2 candidates: every training camera x yaw x pitch, level horizon, CLIP image embeddings once per scene
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

        for run in sorted((OUT / "lerf_ovs" / s).iterdir()):
            if not (run / "goal.npz").exists():
                continue
            sm = json.loads((run / "summary.json").read_text())
            q = sm["query"]
            if q not in gt:
                continue
            ref = views[Path(sm["reference_view"]).stem]
            k = int(torch.argmax(emb @ clip.text(q)))
            goals = {"ours": np.load(run / "goal.npz")["T_goal"], "refview": ref.T_c2w, "clip": cands[k]}
            ablation = {m: OUT / d / s / run.name / "goal.npz"   # grounding ablations: other masks + our stages 2-3
                        for m, d in (("ours_langsplat", "lerf_ovs_langsplat"), ("ours_gt", "lerf_ovs_gt"))}
            for m, f in ablation.items():
                if f.exists():
                    goals[m] = np.load(f)["T_goal"]
            cf = OUT / "capframe" / s / run.name
            if (cf / "capframe.npz").exists():         # CapFrame: W2C = [R^T | T] in the COLMAP frame
                c = np.load(cf / "capframe.npz")
                Tc = np.eye(4)
                Tc[:3, :3] = A[:3, :3] / sA @ c["R"]
                Tc[:3, 3] = A[:3, :3] @ (-c["R"] @ c["T"]) + A[:3, 3]
                goals["capframe"] = Tc
            elif (cf / "capframe.json").exists():      # CapFrame ran and failed: a miss, scored at no view
                goals["capframe"] = None
            for method, T in goals.items():
                if T is None:
                    rows.append({"scene": s, "query": q, "method": method, "visible": 0.0, "in_frame": 0.0,
                                 "center": 1.0, "fill": float("inf"), "roll_deg": float("nan"), "good_view": 0,
                                 "support": float("nan")})          # no view: fails every criterion, incl. level
                    continue
                rgb, depth = scene.render(T, K, W, H)
                d = central_depth(depth)
                if method == "refview":
                    rgb = np.asarray(Image.open(ref.image_path).convert("RGB").resize((W, H), Image.BILINEAR))
                if method in ("refview", "clip", "capframe"):
                    write_run(OUT / "baselines" / method / s / run.name, T, K, W, H, rgb, d, q)
                d_axis = float(np.load(ablation.get(method, run / "goal.npz"))["d_star"]) \
                    if method.startswith("ours") else d
                row = {"scene": s, "query": q, "method": method, **view_metrics(gt[q], T, K, W, H, depth, up),
                       "support": support_of(T, d_axis, train_c2w)}
                rows.append(row)
                print(f"{s}/{q} [{method}] " + " ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}"
                                                        for a, b in row.items() if a not in ("scene", "query", "method")),
                      flush=True)
        scene.release()

    (OUT / "benchmark").mkdir(parents=True, exist_ok=True)
    with open(OUT / "benchmark/lerf_ovs_goals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| method | queries | good view (%) | visible | whole in frame (%) | centre offset | fill | roll (deg) | support |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in ["ours", "ours_langsplat", "ours_gt", "refview", "clip", "capframe"]:
        R = [r for r in rows if r["method"] == m]
        if not R:
            continue
        fin = [min(r["fill"], 2.0) for r in R]
        lines.append(f"| {m} | {len(R)} | {100 * np.mean([r['good_view'] for r in R]):.1f} | "
                     f"{np.mean([r['visible'] for r in R]):.2f} | {100 * np.mean([r['in_frame'] >= 0.9 for r in R]):.1f} | "
                     f"{np.median([r['center'] for r in R]):.3f} | {np.median(fin):.2f} | "
                     f"{np.median([r['roll_deg'] for r in R]):.1f} | {np.median([r['support'] for r in R]):.2f} |")
    (OUT / "benchmark/summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
