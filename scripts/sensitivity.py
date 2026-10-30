"""Sensitivity of the viewpoint selection (stage 2) to its hand-set parameters, and the continuous-score variant.

Stage 1 is not re-run: the object (cloud, centroid, normal, rho, reference view) is rebuilt from each run's goal.npz
and summary.json. For every variant (one parameter at -50 % / +50 % of its frozen value, or a scoring change), the
viewpoint is selected again and compared with the frozen choice: angle between the viewing directions, d*/d*_frozen,
and the object in the new desired view (in-image fraction, size, centring offset, visibility, support).

  ~/anaconda3/envs/pca_servo/bin/python scripts/sensitivity.py --dataset mipnerf360     (from my_code/)
Writes out/sensitivity/<dataset>.csv and a 1/4-resolution render per (query, variant) in out/sensitivity/renders/.
"""
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text2goal.cameras import load_colmap_views  # noqa: E402
from text2goal.geometry import project, up_from_cameras  # noqa: E402
from text2goal.render import GaussianScene  # noqa: E402
from text2goal.viewpoint import ViewParams, projected_fill, select_view  # noqa: E402

GS_VS = str(ROOT / "third_party")         # gsplat COLMAP parser (gs_vs.datasets.colmap)
D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
L = "/media/youssef/DATA/lerf_ovs/lerf_ovs"
DATASETS = {
    "mipnerf360": [(s, ROOT / f"out/multiscene/{s}", f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt",
                    ROOT / f"out/multiscene/cfg/{s}.yml", f"{D}/{s}/images")
                   for s in ["room", "counter", "kitchen", "bonsai", "garden", "bicycle"]],
    "lerf_ovs": [(s, ROOT / f"out/lerf_ovs/{s}", f"{L}/{s}/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt",
                  f"{L}/{s}/output_gsplat_mcmc_df1/cfg.yml", f"{L}/{s}/images")
                 for s in ["figurines", "teatime", "ramen", "waldo_kitchen"]],
}
P0 = ViewParams()
VARIANTS = {"frozen": {}}
for k in ["fill", "margin", "planarity_threshold", "support_sigma_deg", "support_sigma_log_distance",
          "azimuth_step_deg"]:
    for f, tag in [(0.5, "-50%"), (1.5, "+50%")]:
        VARIANTS[f"{k} {tag}"] = {k: getattr(P0, k) * f}
VARIANTS["fill +50%"] = {"fill": P0.max_fill}          # 0.9 > max_fill 0.85: capped
VARIANTS["margin +50%"] = {"margin": 1.0}              # a margin above 1 would place the camera behind the wall
VARIANTS["continuous"] = {"continuous": True}
VARIANTS["support_floor 0.25"] = {"support_floor": 0.25}
VARIANTS["continuous + support_floor 0.25"] = {"continuous": True, "support_floor": 0.25}


def object_metrics(points, T, K, W, H, p, v, d, up):
    u, vv, z = project(points, T, K)
    inside = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    cu, cv, _ = project(p[None], T, K)
    return {"in_image": float(inside.mean()), "fill": projected_fill(points, p, v, d, up, K, W, H),
            "center_offset": float(np.hypot(cu[0] - K[0, 2], cv[0] - K[1, 2]) / np.hypot(W, H))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names")
    args = ap.parse_args()
    out = ROOT / "out/sensitivity"
    (out / "renders").mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.csv"
    done = {(r["run"], r["variant"]) for r in csv.DictReader(open(path))} if path.exists() else set()
    fields = ["scene", "query", "run", "variant", "method", "chosen", "angle_to_frozen_deg", "d_star",
              "d_ratio", "visibility", "support", "fit", "score", "in_image", "fill", "center_offset"]
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()
    variants = {k: VARIANTS[k] for k in (args.variants or VARIANTS)}
    for scene_name, runs_dir, model, cfg, images in DATASETS[args.dataset]:
        runs = sorted(r for r in runs_dir.iterdir() if (r / "goal.npz").exists())
        if all((str(r), k) in done for r in runs for k in variants):
            continue
        views = {v.name: v for v in load_colmap_views(cfg, images, GS_VS)}
        train_c2w = np.stack([v.T_c2w for v in views.values()])
        up = up_from_cameras(train_c2w)
        scene = GaussianScene(model)
        for run in runs:
            g, sm = np.load(run / "goal.npz"), json.loads((run / "summary.json").read_text())
            obj = SimpleNamespace(centroid=g["p_obj"], normal=g["n_obj"], points=g["points"], rho=sm["rho"],
                                  view=views[sm["reference_view"]])
            v_frozen = np.asarray(sm["v_world"])
            for name, change in variants.items():
                if (str(run), name) in done:
                    continue
                try:
                    goal = select_view(obj, scene, up, train_c2w, replace(P0, **change))
                except RuntimeError as e:
                    print(f"{run.name} / {name}: {e}")
                    continue
                c, v = goal["chosen"], goal["v_world"]
                K, W, H = obj.view.K, obj.view.W, obj.view.H
                row = {"scene": scene_name, "query": run.name, "run": str(run), "variant": name,
                       "method": goal["method"], "chosen": c["name"],
                       "angle_to_frozen_deg": float(np.degrees(np.arccos(np.clip(v @ v_frozen, -1, 1)))),
                       "d_star": goal["d_star"], "d_ratio": goal["d_star"] / sm["d_star"],
                       **{k: c[k] for k in ("visibility", "support", "fit", "score")},
                       **object_metrics(obj.points, goal["T_goal"], K, W, H, obj.centroid, v, goal["d_star"], up)}
                w.writerow(row)
                f.flush()
                Ks = K.copy()
                Ks[:2] *= 0.25
                Image.fromarray(scene.render(goal["T_goal"], Ks, W // 4, H // 4)[0]).save(
                    out / "renders" / f"{scene_name}__{run.name}__{name.replace(' ', '_').replace('%', 'pc')}.jpg")
                print(f"{scene_name}/{run.name} [{name}] {goal['method']} {c['name']} "
                      f"angle {row['angle_to_frozen_deg']:.1f} deg, d ratio {row['d_ratio']:.2f}", flush=True)
        scene.release()
    f.close()


if __name__ == "__main__":
    main()
