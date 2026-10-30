"""Every significance test quoted in the paper, at the level of the desired image (the 3 runs of an image per offset
are not independent): paired tests between methods on the same queries use the per-image success rate (Wilcoxon
signed-rank, two-sided, zero differences dropped) or, for binary outcomes per query, the exact McNemar test.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/paper_stats.py        (from my_code/)
Writes out/paper_stats.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LINES = []


def say(s):
    print(s)
    LINES.append(s)


def mcnemar(a, b):
    """a, b: aligned 0/1 Series. Returns (a only, b only, exact p)."""
    x, y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return x, y, (binomtest(min(x, y), x + y).pvalue if x + y else 1.0)


def per_image(f, key=lambda r: "/".join(Path(r).parts[-2:])):
    d = pd.read_csv(OUT / "servo_eval" / f"{f}.csv")
    d["key"] = d.run.map(key)
    return d.groupby(["key", "alpha"]).success.mean()


def wil(a, b, keys=None):
    j = pd.concat([a, b], axis=1, keys=["a", "b"])
    if keys is not None:
        j = j[j.index.get_level_values(0).isin(keys)]
    j = j.fillna(0.0)                                            # a method without a desired image fails its runs
    diff = j.a - j.b
    p = wilcoxon(diff[diff != 0]).pvalue if (diff != 0).sum() > 0 else 1.0
    return 100 * j.a.mean(), 100 * j.b.mean(), int((diff > 0).sum()), int((diff < 0).sum()), p


def main():
    b = pd.read_csv(OUT / "benchmark/lerf_ovs_goals.csv")
    b["key"] = b.scene + "/" + b["query"].str.replace(" ", "_")
    g = pd.read_csv(OUT / "benchmark/grounding_hits.csv")
    right = set(g[g.gdino_sam == 1].scene + "/" + g[g.gdino_sam == 1]["query"].str.replace(" ", "_"))
    gv = {m: b[b.method == m].set_index("key").good_view for m in b.method.unique()}
    say("== LERF-OVS good view, ours vs baselines (exact McNemar)")
    for subset, keep in (("all 67", None), ("right object 29", right)):
        for m in ("refview", "clip", "capframe"):
            o, x = gv["ours"], gv[m].reindex(gv["ours"].index)
            if keep:
                o, x = o[o.index.isin(keep)], x[x.index.isin(keep)]
            say(f"{subset:16s} ours {int(o.sum())} vs {m} {int(x.sum())}: discordant {mcnemar(o, x)}")
    say("== centring threshold sweep (good view with centre <= t x diagonal), ours vs B1, all 67")
    fill = b.fill.replace(np.inf, np.nan)
    for t in (0.10, 0.15, 0.20, 0.25):
        gvt = (b.visible >= 0.5) & (b.in_frame >= 0.9) & (b.center <= t) & (fill >= 0.15) & (fill <= 0.9)
        say(f"t={t:.2f}: ours {int(gvt[b.method == 'ours'].sum())}, B1 {int(gvt[b.method == 'refview'].sum())}")
    say("== median centre offset (fraction of diagonal) on the right-object subset")
    for m in ("refview", "ours"):
        x = b[(b.method == m) & b.key.isin(right)]
        say(f"{m}: {x.center.median():.3f}")

    say("== LERF-OVS servo, per-image success, ours vs baselines (Wilcoxon); failed images count as 0")
    keys = sorted(set(b.key))
    o = per_image("lerf_ovs_same")
    for m, f in (("refview", "baseline_refview_same"), ("refview_render", "baseline_refview_render_same"),
                 ("clip", "baseline_clip_same"), ("capframe", "baseline_capframe_same")):
        x = per_image(f)
        for a in (0.1, 0.2):
            oa = o.xs(a, level=1).reindex(keys)
            xa = x.xs(a, level=1).reindex(keys)
            say(f"alpha {a}: ours vs {m}: {wil(oa, xa)}")
            if m == "capframe":
                shared = list(x.xs(a, level=1).index)
                say(f"alpha {a}: ours vs capframe on its 61 views: {wil(o.xs(a, level=1).reindex(shared), x.xs(a, level=1).reindex(shared))}")
    say("== same vs other model (ours)")
    c = per_image("lerf_ovs_cross")
    for a in (0.1, 0.2):
        say(f"alpha {a}: same vs cross: {wil(o.xs(a, level=1), c.xs(a, level=1).reindex(o.xs(a, level=1).index))}")
    say("== Mip-NeRF 360: 4x supersampled vs 320 px (same runs)")
    m1, m4 = per_image("mipnerf360_same"), per_image("mipnerf360_same_ss4")
    say(f"both offsets: {wil(m4, m1.reindex(m4.index))}")
    say("== median d* per method (LERF-OVS same-model servo)")
    for f in ("lerf_ovs_same", "baseline_refview_same", "baseline_clip_same", "baseline_capframe_same"):
        say(f"{f}: {pd.read_csv(OUT / 'servo_eval' / f'{f}.csv').d_star.median():.2f}")

    say("== InteriorGS (per query; missing desired images fail)")
    r = pd.read_csv(OUT / "interiorgs/results_scored.csv")
    r["key"] = r.scene + "/" + r["query"]
    s = pd.read_csv(OUT / "interiorgs/servo.csv")
    s["key"] = s.scene + "/" + s["query"]
    grounded = set(r[(r.method == "ours") & (r.right_object == 1)].key)
    rv = {m: r[r.method == m].set_index("key").good_view for m in ("ours", "refview", "clip")}
    for subset, keep in (("all 93", None), (f"grounded {len(grounded)}", grounded)):
        for m in ("refview", "clip"):
            a, x = rv["ours"], rv[m]
            if keep:
                a, x = a[a.index.isin(keep)], x[x.index.isin(keep)]
            say(f"good view {subset}: ours {int(a.sum())} vs {m} {int(x.sum())}: {mcnemar(a, x)}")
        allk = sorted(keep) if keep else sorted(set(r.key))
        sv = {m: s[s.method == m].groupby(["key", "alpha"]).success.mean() for m in ("ours", "refview", "clip")}
        for m in ("refview", "clip"):
            for a in (0.1, 0.2):
                say(f"servo {subset} alpha {a}: ours vs {m}: "
                    f"{wil(sv['ours'].xs(a, level=1).reindex(allk), sv[m].xs(a, level=1).reindex(allk))}")
    say("== InteriorGS median d* per method")
    say(str(s.groupby("method").d_star.median().round(2).to_dict()))
    (OUT / "paper_stats.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
