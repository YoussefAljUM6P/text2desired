"""Did stage 1 find the right object? Grounding DINO + SAM (out/lerf_ovs) vs LangSplat-V2 (out/lerf_ovs_langsplat).

The 3D object centre p_obj of each run (summary.json) is compared with the GT instances of the query (same GT as
scripts/goal_benchmark.py): hit if p_obj lies within 15 % of an instance's extent (2-98 % percentile box diagonal)
of its nearest point, or inside its box. Writes out/benchmark/grounding_hits.csv and prints a summary.

  ~/anaconda3/envs/pca_servo/bin/python scripts/grounding_hits.py        (from my_code/)
"""
import csv
import json
from pathlib import Path

import numpy as np

from goal_benchmark import GS_VS, L, OUT, SCENES, gt_instances
from text2goal.cameras import load_colmap_views
from text2goal.render import GaussianScene

RUNS = {"gdino_sam": OUT / "lerf_ovs", "langsplat_v2": OUT / "lerf_ovs_langsplat", "gt_labels": OUT / "lerf_ovs_gt"}


def hit(p, inst):
    for P in inst:
        lo, hi = np.percentile(P, [2, 98], axis=0)
        ext = float(np.linalg.norm(hi - lo))
        if np.all(p >= lo) and np.all(p <= hi) or np.min(np.linalg.norm(P - p, axis=1)) <= 0.15 * ext:
            return 1
    return 0


def main():
    rows = []
    for s in SCENES:
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        scene.release()
        for run in sorted((OUT / "lerf_ovs" / s).iterdir()):
            q = json.loads((run / "summary.json").read_text())["query"] if (run / "summary.json").exists() else None
            if q not in gt:
                continue
            row = {"scene": s, "query": q}
            for name, root in RUNS.items():
                f = root / s / run.name / "summary.json"
                sm = json.loads(f.read_text()) if f.exists() else {}
                row[name] = hit(np.array(sm["p_obj"]), gt[q]) if "p_obj" in sm else 0
                row[name + "_score"] = sm.get("detection", {}).get("score", "")
            rows.append(row)
            print(s, q, row["gdino_sam"], row["langsplat_v2"], flush=True)
    with open(OUT / "benchmark/grounding_hits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a, b = np.array([r["gdino_sam"] for r in rows]), np.array([r["langsplat_v2"] for r in rows])
    print(f"{len(rows)} queries | GD+SAM right object {a.mean():.1%} | LangSplat-V2 {b.mean():.1%} | "
          f"both {np.mean(a & b):.1%} | only GD {np.mean(a & ~b.astype(bool)):.1%} | only LS {np.mean(b & ~a.astype(bool)):.1%}")


if __name__ == "__main__":
    main()
