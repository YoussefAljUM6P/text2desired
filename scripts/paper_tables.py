"""LaTeX tables of the journal paper (paper_mva/tables/*.tex) from the result csv files, so that every number is regenerated.

  benchmark.tex   Table I: desired-image criteria + servo success, ours vs baselines (LERF-OVS, 67 queries)
  grounding.tex   Table II: stage 1 (right object) and good views with GD+SAM / LangSplat-V2 / GT-label masks
  servo.tex       Table III: photometric servo, same / other model, LERF-OVS / Mip-NeRF 360
  sensitivity.tex Table IV: viewpoint selection under +-50 % parameter changes
  servo_failures.tex  how the photometric servo fails (narrow convergence basin), per desired-image method
  ablation.tex    good-view rate (LERF-OVS) of the score ablations and of every sensitivity variant (scripts/ablation.py)
  interiorgs.tex  ground-robot benchmark on InteriorGS (scripts/interiorgs_run.py, _rescore.py, _servo.py)

  ~/anaconda3/envs/pca_servo/bin/python scripts/paper_tables.py        (from my_code/)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT, TAB = ROOT / "out", ROOT / "paper_mva" / "tables"
N = 67
NAMES = {"refview": "B1 detected training photo", "clip": "B2 CLIP render-and-score", "capframe": "B3 CapFrame~\\cite{capframe}",
         "ours": "Ours", "ours_langsplat": "Ours, LangSplat-V2 masks", "ours_gt": "Ours, GT-label masks"}


# paper labels of the sensitivity / ablation variants (scripts/sensitivity.py VARIANTS), with the values used:
# "fill +50%" and "margin +50%" are capped (fill <= max_fill, a margin above 1 would put the camera behind the wall)
VARIANT_LABELS = {
    "fill -50%": r"fill $\phi$ 0.3 ($-$50\%)", "fill +50%": r"fill $\phi$ 0.85 (+42\%, max.)",
    "margin -50%": r"margin $\mu$ 0.45 ($-$50\%)", "margin +50%": r"margin $\mu$ 1.0 (+11\%, max.)",
    "planarity_threshold -50%": r"planarity threshold 0.1 ($-$50\%)",
    "planarity_threshold +50%": r"planarity threshold 0.3 (+50\%)",
    "support_sigma_deg -50%": r"cov.\ $\sigma_\theta$ 7.5$^\circ$ ($-$50\%)",
    "support_sigma_deg +50%": r"cov.\ $\sigma_\theta$ 22.5$^\circ$ (+50\%)",
    "support_sigma_log_distance -50%": r"cov.\ $\sigma_{\log d}$ 0.2 ($-$50\%)",
    "support_sigma_log_distance +50%": r"cov.\ $\sigma_{\log d}$ 0.6 (+50\%)",
    "azimuth_step_deg -50%": r"azimuth step 7.5$^\circ$ ($-$50\%)",
    "azimuth_step_deg +50%": r"azimuth step 22.5$^\circ$ (+50\%)",
    "continuous": "continuous prior weight", "support_floor 0.25": "coverage floor 0.25",
    "continuous + support_floor 0.25": "both"}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def bold_max(vals):
    m = max(v for v in vals if v is not None)
    return ["--" if v is None else (f"\\textbf{{{v:.0f}}}" if abs(v - m) < 0.5 else f"{v:.0f}") for v in vals]


def servo_rate(csv, alpha):
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    return 100 * d[d.alpha == alpha].success.mean()


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(OUT / "benchmark/lerf_ovs_goals.csv")
    fill = b.fill.replace(np.inf, np.nan)
    crit = {"good": b.good_view == 1, "vis": b.visible >= 0.5, "frame": b.in_frame >= 0.9, "ctr": b.center <= 0.15,
            "size": (fill >= 0.15) & (fill <= 0.9), "level": b.roll_deg < 5}
    pct = {c: {m: 100 * s[b.method == m].sum() / N for m in NAMES} for c, s in crit.items()}
    servo_csv = {"ours": "lerf_ovs_same", "refview": "baseline_refview_same", "clip": "baseline_clip_same",
                 "capframe": "baseline_capframe_same"}

    # Table I: all queries, then the queries whose object Grounding DINO + SAM finds (grounding_hits.csv), which
    # separates the viewpoint selection from grounding errors (same columns, servo restricted to those queries)
    methods = ["refview", "clip", "capframe", "ours"]
    hits = pd.read_csv(OUT / "benchmark/grounding_hits.csv")
    right = set(zip(hits.scene[hits.gdino_sam == 1], hits["query"][hits.gdino_sam == 1]))
    lines = [r"\begin{tabular}{@{}lcccccc|cc@{}}", r"\toprule",
             r" & good & visible & whole & centred & size & level & \multicolumn{2}{c}{servo success} \\",
             r"method & view & $\ge$50\% & in frame & & 0.15--0.9 & $<$5$^\circ$ & small & large \\"]
    for title, keep in ((f"all {N} queries", None), (f"right object found ({len(right)} queries)", right)):
        sel = np.ones(len(b), bool) if keep is None else np.array([(sc, q) in keep for sc, q in zip(b.scene, b["query"])])
        n = N if keep is None else len(keep)
        cols = [[100 * (s & sel)[b.method == m].sum() / n for m in methods] for s in crit.values()]
        for a in (0.1, 0.2):
            col = []
            for m in methods:
                f = OUT / "servo_eval" / f"{servo_csv[m]}.csv"
                d = pd.read_csv(f)
                if keep is not None:
                    d = d[[(Path(r).parent.name, Path(r).name.replace("_", " ")) in keep for r in d.run]]
                # over every expected run (3 seeds per query): a method without a desired image fails its runs
                col.append(100 * d[d.alpha == a].success.sum() / (3 * n))
            cols.append(col)
        cols = [bold_max(c) for c in cols]
        ci = {m: wilson(int((crit["good"] & sel)[b.method == m].sum()), n) for m in methods}
        lines += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for i, m in enumerate(methods):
            cells = [cols[0][i] + f"\\,{{\\tiny[{ci[m][0]:.0f},{ci[m][1]:.0f}]}}"] + [c[i] for c in cols[1:]]
            lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "benchmark.tex").write_text("\n".join(lines) + "\n")

    # Table II
    h = pd.read_csv(OUT / "benchmark/grounding_hits.csv")
    g = b.pivot_table(index=["scene", "query"], columns="method", values="good_view").reset_index()
    x = h.merge(g, on=["scene", "query"], how="left")
    rows = [("Grounding DINO + SAM (zero-shot)", "gdino_sam", "ours"), ("LangSplat-V2~\\cite{langsplatv2} (per-scene)", "langsplat_v2", "ours_langsplat")]
    if "gt_labels" in x and "ours_gt" in x:
        rows.append(("GT label polygon (oracle)", "gt_labels", "ours_gt"))
    both = (x.gdino_sam == 1) & (x.langsplat_v2 == 1)
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"stage-1 masks & right & good view & good view & good view \\",
             r" & object & (all 67) & $\mid$ right object & both right (23) \\", r"\midrule"]
    for name, hcol, m in rows:
        if m not in x:
            continue
        hit = x[hcol] == 1
        lines.append(f"{name} & {100 * hit.mean():.0f} & {100 * x[m].fillna(0).mean():.0f} & "
                     f"{100 * x[hit][m].fillna(0).mean():.0f} ({int(hit.sum())}) & {100 * x[both][m].fillna(0).mean():.0f} \\\\")
    lines += [r"\midrule"]
    for m in ["refview", "clip", "capframe"]:
        lines.append(f"{NAMES[m]} & -- & {100 * x[m].fillna(0).mean():.0f} & -- & {100 * x[both][m].fillna(0).mean():.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "grounding.tex").write_text("\n".join(lines) + "\n")

    # Table III
    sv = [("LERF-OVS", "same model", "lerf_ovs_same"), ("LERF-OVS", "other model (LangSplat-V2 3DGS)", "lerf_ovs_cross"),
          ("Mip-NeRF 360", "same model, 320 px render", "mipnerf360_same"),
          ("Mip-NeRF 360", "same model, 4$\\times$ supersampled", "mipnerf360_same_ss4")]
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"dataset & current views & runs & small & large & median final \\",
             r" & & & & & $t/d^\star$, rot. ($^\circ$) \\", r"\midrule"]
    for ds, src, f in sv:
        p = OUT / "servo_eval" / f"{f}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        lines.append(f"{ds} & {src} & {len(d)} & {100 * d[d.alpha == 0.1].success.mean():.0f} & "
                     f"{100 * d[d.alpha == 0.2].success.mean():.0f} & {d.t_err_rel.median():.3f}, {d.r_err.median():.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "servo.tex").write_text("\n".join(lines) + "\n")

    # Table IV
    rows = []
    for ds, f in (("Mip", "mipnerf360"), ("LERF", "lerf_ovs")):
        s = pd.read_csv(OUT / "sensitivity" / f"{f}.csv")
        for v, g in s.groupby("variant", sort=False):
            rows.append((ds, v, 100 * (g.angle_to_frozen_deg < 5).mean()))
    t = pd.DataFrame(rows, columns=["ds", "variant", "same"]).pivot(index="variant", columns="ds", values="same")
    order = [v for v in dict.fromkeys(r[1] for r in rows) if v != "frozen"]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"parameter change & Mip-NeRF 360 & LERF-OVS \\", r"\midrule"]
    for v in order:
        lines.append(f"{VARIANT_LABELS[v]} & {t.loc[v, 'Mip']:.0f} & {t.loc[v, 'LERF']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "sensitivity.tex").write_text("\n".join(lines) + "\n")
    servo_failures()
    ablation()
    interiorgs()
    print("written:", ", ".join(sorted(p.name for p in TAB.glob("*.tex"))))


def servo_failures():
    """Per desired-image method (LERF-OVS, same model): images never reached in 6 runs, success by offset, and the
    final photometric error of failed vs successful runs (failures stall in a local minimum)."""
    rows = [("Ours", "lerf_ovs_same"), ("Ours, other model", "lerf_ovs_cross"),
            ("B1 detected training photo", "baseline_refview_same"),
            ("B1, rendered at the photo's pose", "baseline_refview_render_same"), ("B2 CLIP render-and-score", "baseline_clip_same"),
            ("B3 CapFrame~\\cite{capframe}", "baseline_capframe_same")]
    lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             " & images never & \\multicolumn{2}{c}{success (\\%)} & \\multicolumn{2}{c}{final MSE (median)} \\\\",
             "desired images & reached & small & large & failed & succeeded \\\\", "\\midrule"]
    for name, f in rows:
        d = pd.read_csv(OUT / "servo_eval" / f"{f}.csv")
        per = d.groupby("run").success.sum()
        ok, ko = d[d.success == 1], d[d.success == 0]
        lines.append(f"{name} & {int((per == 0).sum())}/{len(per)} & {100 * d[d.alpha == 0.1].success.mean():.0f} & "
                     f"{100 * d[d.alpha == 0.2].success.mean():.0f} & {ko.mse.median():.0f} & {ok.mse.median():.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "servo_failures.tex").write_text("\n".join(lines) + "\n")



def ablation():
    """Per variant (LERF-OVS, 67 queries): views changed w.r.t. the frozen method, good view (%) over all queries and
    over the queries our grounding finds; *: exact McNemar test against the frozen method, p < 0.05.
    Written only once every variant covers all queries."""
    f = OUT / "ablation/lerf_ovs.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rate = d.groupby("variant").good_view.agg(["mean", "count"])
    if (rate["count"] < N).any():
        print(f"ablation incomplete ({int(rate['count'].sum())} rows): table not written")
        return
    g = pd.read_csv(OUT / "benchmark/grounding_hits.csv")
    ok = g[g.gdino_sam == 1].set_index(["scene", "query"]).index
    ref = d[d.variant == "frozen"].set_index(["scene", "query"])

    def stats(v, keep=None):
        x = d[d.variant == v].set_index(["scene", "query"]).reindex(ref.index)
        r = ref
        if keep is not None:
            x, r = x[x.index.isin(keep)], r[r.index.isin(keep)]
        lost = int(((r.good_view == 1) & (x.good_view == 0)).sum())
        gained = int(((r.good_view == 0) & (x.good_view == 1)).sum())
        p = binomtest(min(lost, gained), lost + gained).pvalue if lost + gained else 1.0
        return 100 * x.good_view.mean(), p, int((x.chosen != r.chosen).sum())

    groups = [("full method", ["frozen"]),
              ("score term removed", ["- vis", "- cov", "- prior", "- fit"]),
              ("free space", ["- free space"]),
              ("search removed", ["PCA normal only", "reference direction only"]),
              ("one parameter $\\pm$50\\%", [v for v in rate.index if v.endswith("%")]),
              ("scoring variants", ["continuous", "support_floor 0.25", "continuous + support_floor 0.25"])]
    names = {"frozen": "frozen parameters", "- vis": "without vis", "- cov": "without cov", "- prior": "without $\\pi$",
             "- fit": "without fit", "- free space": "free space ignored", "PCA normal only": "always the PCA normal",
             "reference direction only": "always the reference direction", "continuous": "continuous prior weight",
             "support_floor 0.25": "coverage floor 0.25", "continuous + support_floor 0.25": "both"}
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r" & & views & \multicolumn{2}{c}{good view (\%)} \\",
             r" & variant & changed & all & right object \\", r"\midrule"]
    for grp, vs in groups:
        for k, v in enumerate(vs):
            label = names.get(v, VARIANT_LABELS.get(v, v))
            a, pa, ch = stats(v)
            b, pb, _ = stats(v, ok)
            cell = lambda r, p: f"{r:.0f}" + ("$^\\ast$" if p < 0.05 else "")
            lines.append(f"{grp if k == 0 else ''} & {label} & {ch} & {cell(a, pa)} & {cell(b, pb)} " + r"\\")
        lines.append(r"\midrule" if grp != groups[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "ablation.tex").write_text("\n".join(lines) + "\n")


def interiorgs():
    """Ground-robot benchmark: grounding, visible, centred, good view (% of queries) and 3-DOF servo success, for all
    queries and for the queries whose object was grounded (B1 shares our grounding: same detection view)."""
    f, fs = OUT / "interiorgs/results_scored.csv", OUT / "interiorgs/servo.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    sv = pd.read_csv(fs) if fs.exists() else None
    ok = set(map(tuple, d[(d.method == "ours") & (d.right_object == 1)][["scene", "query"]].values))
    n, k = d[d.method == "ours"].shape[0], len(ok)
    lines = [r"\begin{tabular}{@{}lcccc|cc@{}}", r"\toprule",
             r" & right & visible & centred & good & \multicolumn{2}{c}{servo success (3 DOF)} \\",
             r"method & object & & (horiz.) & view & small & large \\", r"\midrule"]
    for title, keep in ((f"all queries ($n={n}$)", None), (f"right object found ($n={k}$)", ok)):
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for m, name in (("refview", "B1 survey view"), ("clip", "B2 CLIP render-and-score"), ("ours", "Ours (ground robot)")):
            x = d[d.method == m]
            y = sv[sv.method == m] if sv is not None else None
            if keep is not None:
                x = x.loc[x.set_index(["scene", "query"]).index.isin(list(keep))]
                y = y.loc[y.set_index(["scene", "query"]).index.isin(list(keep))] if y is not None else None
            right = f"{100 * pd.to_numeric(x.right_object, errors='coerce').mean():.0f}" if m != "clip" else "--"
            cells = [right] + [f"{100 * x[c].mean():.0f}" for c in ("visible", "centred", "good_view")]
            for a in (0.1, 0.2):
                z = y[y.alpha == a] if y is not None else None
                nq = len(x)                            # every query: 3 runs per offset, a missing desired image fails
                cells.append(f"{100 * z.success.sum() / (3 * nq):.0f}" if z is not None and len(z) else "--")
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TAB / "interiorgs.tex").write_text("\n".join(lines) + "\n")
    print(f"interiorgs: {n} queries, {k} grounded")

if __name__ == "__main__":
    main()
