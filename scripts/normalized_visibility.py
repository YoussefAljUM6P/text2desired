"""Normalised visibility on LERF-OVS: the visible fraction of a ground-truth instance divided by the best visible
fraction any training photo achieves for that instance.

The ground truth is lifted from labels drawn on training photos (scripts/goal_benchmark.py), so its points are the
surfaces those photos see: a training photo (B1) sees most of them by construction, a new viewpoint sees the object
from another side. Normalising by the best training view removes most of this bias and the strictness of the
"visible >= 0.5" test for closed objects (no single view sees more than about half of their surface).

For every method's goal view (same poses as the benchmark: run directories of ours / ablations / baselines, B1 = the
training camera) the instance shown best is chosen as in goal_benchmark.view_metrics (largest raw visible fraction);
visible_norm = visible / best training visible of that instance (capped at 1), good_view_norm = visible_norm >= 0.5 and
the other three tests unchanged.

  ~/anaconda3/envs/pca_servo/bin/python scripts/normalized_visibility.py        (from my_code/)
Writes out/benchmark/lerf_ovs_goals_norm.csv (benchmark rows + best_train_visible, visible_norm, good_view_norm).
"""
import csv
import json
from pathlib import Path

import numpy as np

from goal_benchmark import GS_VS, L, OUT, SCENES, gt_instances
from text2goal.cameras import load_colmap_views
from text2goal.geometry import project
from text2goal.render import GaussianScene

RUN_DIRS = {"ours": OUT / "lerf_ovs", "ours_langsplat": OUT / "lerf_ovs_langsplat", "ours_gt": OUT / "lerf_ovs_gt",
            "clip": OUT / "baselines/clip", "capframe": OUT / "baselines/capframe"}


def visible(P, T, K, W, H, depth):
    """Fraction of the points inside the image and not occluded (same test as goal_benchmark.view_metrics)."""
    u, v, z = project(P, T, K)
    inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis = np.zeros(len(P), bool)
    if inside.any():
        vis[inside] = depth[v[inside].astype(int), u[inside].astype(int)] >= z[inside] * 0.97 - 1e-3
    return float(vis.mean())


def main():
    bench = {(r["scene"], r["query"], r["method"]): r for r in csv.DictReader(open(OUT / "benchmark/lerf_ovs_goals.csv"))}
    out = []
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        vlist = list(views.values())
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        K, W, H = vlist[0].K, vlist[0].W, vlist[0].H
        # best training-photo visibility of every instance
        best = {q: np.zeros(len(inst)) for q, inst in gt.items()}
        for v in vlist:
            depth = scene.render(v.T_c2w, v.K, v.W, v.H)[1]
            for q, inst in gt.items():
                for k, P in enumerate(inst):
                    best[q][k] = max(best[q][k], visible(P, v.T_c2w, v.K, v.W, v.H, depth))
        print(f"{s}: best training visibility of {sum(len(b) for b in best.values())} instances, "
              f"median {np.median(np.concatenate(list(best.values()))):.2f}", flush=True)
        for run in sorted((OUT / "lerf_ovs" / s).iterdir()):
            if not (run / "summary.json").exists():
                continue
            sm = json.loads((run / "summary.json").read_text())
            q = sm["query"]
            if q not in gt:
                continue
            poses = {m: d / s / run.name / "goal.npz" for m, d in RUN_DIRS.items()}
            poses = {m: np.load(f)["T_goal"] for m, f in poses.items() if f.exists()}
            if "reference_view" in sm:
                poses["refview"] = views[Path(sm["reference_view"]).stem].T_c2w
            for m, row in ((m, bench.get((s, q, m))) for m in ["ours", "ours_langsplat", "ours_gt", "refview", "clip",
                                                                 "capframe"]):
                if row is None:
                    continue
                r = dict(row)
                if m in poses:
                    depth = scene.render(poses[m], K, W, H)[1]
                    vis = [visible(P, poses[m], K, W, H, depth) for P in gt[q]]
                    k = int(np.argmax(vis))
                    r["best_train_visible"] = best[q][k]
                    r["visible_norm"] = min(1.0, vis[k] / max(best[q][k], 1e-6))
                else:                                   # failed method: no view
                    r["best_train_visible"], r["visible_norm"] = "", 0.0
                g = (float(r["in_frame"]) >= 0.9 and float(r["center"]) <= 0.15
                     and 0.15 <= float(r["fill"]) <= 0.9)
                r["good_view_norm"] = int(r["visible_norm"] >= 0.5 and g)
                out.append(r)
        scene.release()
    with open(OUT / "benchmark/lerf_ovs_goals_norm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for m in ["ours", "ours_langsplat", "ours_gt", "refview", "clip", "capframe"]:
        R = [r for r in out if r["method"] == m]
        print(f"{m:15s} visible>=0.5 {100 * np.mean([float(r['visible']) >= 0.5 for r in R]):5.1f}  "
              f"norm>=0.5 {100 * np.mean([r['visible_norm'] >= 0.5 for r in R]):5.1f}  "
              f"good {100 * np.mean([int(r['good_view']) for r in R]):5.1f}  "
              f"good_norm {100 * np.mean([r['good_view_norm'] for r in R]):5.1f}")


if __name__ == "__main__":
    main()
