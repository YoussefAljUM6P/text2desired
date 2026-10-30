"""Rescore the InteriorGS results (out/interiorgs/results.csv) from the saved run directories with the current
interiorgs_run.score (visibility test on the back-projected surface point). Rows without a run directory (failed
grounding) stay misses. Writes out/interiorgs/results_scored.csv and prints the summary per method.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/interiorgs_rescore.py
"""
import json

import numpy as np
import pandas as pd

from interiorgs_run import OUT, VAL, score
from text2goal.render import GaussianScene


def main():
    r = pd.read_csv(OUT / "results.csv")
    rows = []
    for sc, grp in r.groupby("scene"):
        gt = json.loads((OUT / sc / "gt.json").read_text())
        scene = GaussianScene(f"{VAL}/{sc}/{sc}.gs.ply")
        for _, row in grp.iterrows():
            g = OUT / sc / row.method / row["query"].replace(" ", "_") / "goal.npz"
            out = row.to_dict()
            if g.exists():
                z = np.load(g)
                out.update(score(z["T_goal"], [o["position"] for o in gt[row["query"]]], scene, z["K"],
                                 int(z["W"]), int(z["H"])))
            rows.append(out)
        scene.release()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results_scored.csv", index=False)
    for m in ("refview", "clip", "ours"):
        x = d[d.method == m]
        print(f"{m:8s} n={len(x)} visible {100 * x.visible.mean():.0f}% centred {100 * x.centred.mean():.0f}% "
              f"good {100 * x.good_view.mean():.0f}%" + (f" right object {100 * pd.to_numeric(x.right_object, errors='coerce').mean():.0f}%"
                                                       if m == "ours" else ""))


if __name__ == "__main__":
    main()
