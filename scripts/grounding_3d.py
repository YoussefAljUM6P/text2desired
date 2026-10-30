"""How well does each stage-1 mask source delineate the object? 3-D precision / recall of the lifted object cloud against
the ground-truth instance, on the LERF-OVS queries whose object both groundings find (grounding_hits.csv).

The masks live in different reference views and the labels only in a few frames, so masks are compared in 3-D: the
object cloud each run lifted (goal.npz "points": mask back-projected with the rendered depth and cleaned) against the
ground-truth instance it lies on (scripts/goal_benchmark.py). precision = fraction of lifted points within tau of the
instance (how much background the mask lets in), recall = fraction of instance points within tau of the lifted cloud
(how much of the object it covers), tau = 3 % of the instance extent (2-98 % box diagonal); F1 summarises both.

  ~/anaconda3/envs/pca_servo/bin/python scripts/grounding_3d.py        (from my_code/)
Writes out/benchmark/grounding_3d.csv and paper_mva/tables/grounding_3d.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from goal_benchmark import GS_VS, L, OUT, SCENES, gt_instances
from text2goal.cameras import load_colmap_views
from text2goal.render import GaussianScene

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"gdino_sam": ("Grounding DINO + SAM", OUT / "lerf_ovs"), "langsplat_v2": ("LangSplat-V2", OUT / "lerf_ovs_langsplat"),
           "gt_labels": ("GT label polygon", OUT / "lerf_ovs_gt")}


def pr(P, G, tau):
    dp = cKDTree(G).query(P)[0]
    dg = cKDTree(P).query(G)[0]
    prec, rec = float((dp <= tau).mean()), float((dg <= tau).mean())
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    h = pd.read_csv(OUT / "benchmark/grounding_hits.csv")
    both = h[(h.gdino_sam == 1) & (h.langsplat_v2 == 1)]
    rows = []
    for s in SCENES:
        Q = both[both.scene == s]["query"].tolist()
        if not Q:
            continue
        M = f"{L}/{s}/output_gsplat_mcmc_df1"
        views = {Path(v.name).stem: v for v in load_colmap_views(f"{M}/cfg.yml", f"{L}/{s}/images", GS_VS)}
        scene = GaussianScene(f"{M}/ckpts/ckpt_29999_rank0.pt")
        gt = gt_instances(s, views, scene)
        scene.release()
        for q in Q:
            for src, (_, root) in SOURCES.items():
                f = root / s / q.replace(" ", "_") / "goal.npz"
                if not f.exists():
                    continue
                P = np.load(f)["points"]
                c = P.mean(0)
                G = min(gt[q], key=lambda I: np.linalg.norm(I.mean(0) - c))          # instance the cloud lies on
                lo, hi = np.percentile(G, [2, 98], axis=0)
                prec, rec, f1 = pr(P, G, 0.03 * np.linalg.norm(hi - lo))
                rows.append({"scene": s, "query": q, "source": src, "precision": prec, "recall": rec, "f1": f1})
                print(f"{s}/{q} [{src}] precision {prec:.2f} recall {rec:.2f} F1 {f1:.2f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "benchmark/grounding_3d.csv", index=False)
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"stage-1 masks & precision & recall & F1 \\", r"\midrule"]
    for src, (name, _) in SOURCES.items():
        x = d[d.source == src]
        lines.append(f"{name} & {x.precision.median():.2f} & {x.recall.median():.2f} & {x.f1.median():.2f} \\\\")
        print(f"{name:22s} n={len(x)} median precision {x.precision.median():.2f} recall {x.recall.median():.2f} "
              f"F1 {x.f1.median():.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper_mva/tables/grounding_3d.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
