"""Which properties of a desired image make the photometric servo succeed? (LERF-OVS, current views from the same model)

For every desired image of ours, B1, B2 and B3, the servo success over its 6 runs (out/servo_eval/*_same.csv) is joined
with the benchmark criteria of the image (out/benchmark/lerf_ovs_goals.csv). Prints the success rate when each criterion
passes / fails, and a binomial logistic regression of the run outcomes on the criteria with one intercept per method
(so that method-level effects, e.g. B1's photograph-vs-render mismatch, do not masquerade as criteria effects).
The 6 runs of an image are not independent: standard errors are quasi-binomial (scaled by the Pearson dispersion of the
image-level fit). A second model adds log d* (the goal distance, which differs by method and sets the perturbation).
Writes paper_mva/tables/servo_criteria.tex.

  ~/anaconda3/envs/pca_servo/bin/python scripts/servo_vs_criteria.py        (from my_code/)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CSVS = {"ours": "lerf_ovs_same", "refview": "baseline_refview_same", "clip": "baseline_clip_same",
        "capframe": "baseline_capframe_same"}


def logistic(X, k, n, iters=50):
    """Binomial logistic regression by IRLS: k successes out of n per row. Returns coefficients and std errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = n * p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        beta = beta + np.linalg.solve(H, X.T @ (k - n * p))
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (n * p * (1 - p))[:, None]))
    phi = max(1.0, float(np.sum((k - n * p) ** 2 / (n * p * (1 - p))) / (len(k) - X.shape[1])))
    return beta, np.sqrt(phi * np.diag(cov)), phi


def main():
    b = pd.read_csv(OUT / "benchmark/lerf_ovs_goals.csv")
    b["qd"] = b["query"].str.replace(" ", "_")
    rows = []
    for m, f in CSVS.items():
        d = pd.read_csv(OUT / "servo_eval" / f"{f}.csv")
        d["scene"], d["qd"] = [Path(r).parent.name for r in d.run], [Path(r).name for r in d.run]
        s = d.groupby(["scene", "qd"]).agg(sum=("success", "sum"), count=("success", "count"),
                                           d_star=("d_star", "first")).reset_index()
        s["method"] = m
        rows.append(s)
    x = pd.concat(rows).merge(b, on=["scene", "qd", "method"])
    fill = x.fill.replace(np.inf, np.nan)
    crit = {"centred": x.center <= 0.15, "size 0.15--0.9": (fill >= 0.15) & (fill <= 0.9),
            "whole in frame": x.in_frame >= 0.9, "visible $\\ge$50\\%": x.visible >= 0.5}
    k, n = x["sum"].to_numpy(float), x["count"].to_numpy(float)
    C = [c.astype(float) for c in crit.values()]
    I = [(x.method == m).astype(float) for m in CSVS]                              # one intercept per method
    beta, se, phi = logistic(np.column_stack(C + I), k, n)
    beta2, se2, phi2 = logistic(np.column_stack(C + [np.log(x.d_star.to_numpy(float))] + I), k, n)
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{success (\%)} & \multicolumn{2}{c}{odds ratio [95\% CI]} \\",
             r"criterion of the desired image & passed & failed & criteria only & with $\log d^\star$ \\", r"\midrule"]

    def cell(b, e):
        pv = 2 * (1 - norm.cdf(abs(b / e)))
        return f"{np.exp(b):.2f} [{np.exp(b - 1.96 * e):.2f}, {np.exp(b + 1.96 * e):.2f}]" + (r"$^\ast$" if pv < 0.05 else ""), pv
    for i, (name, c) in enumerate(crit.items()):
        sp = 100 * x[c]["sum"].sum() / x[c]["count"].sum()
        sf = 100 * x[~c]["sum"].sum() / x[~c]["count"].sum()
        (c1, p1), (c2, p2) = cell(beta[i], se[i]), cell(beta2[i], se2[i])
        lines.append(f"{name} & {sp:.0f} & {sf:.0f} & {c1} & {c2} \\\\")
        print(f"{name:22s} pass {sp:5.1f}%  fail {sf:5.1f}%  {c1} p={p1:.3f} | with log d*: {c2} p={p2:.3f}")
    c3, p3 = cell(beta2[len(C)], se2[len(C)])
    lines.append(f"$\\log d^\\star$ (per unit) & & & & {c3} \\\\")
    print(f"log d*: {c3} p={p3:.4f}; dispersion {phi:.2f} / {phi2:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper_mva/tables/servo_criteria.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(x)} desired images, {int(x['count'].sum())} runs")


if __name__ == "__main__":
    main()
