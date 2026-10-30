"""Figures of the journal paper (paper_mva/figures/); LERF-OVS ones from the outlined thumbnails of scripts/make_gallery.py.

  gallery_lerf.pdf   rows = selected LERF-OVS queries (ROWS), columns = methods (COLS), good-view mark on each image
  grounding_failures.pdf  rows = queries where the detector picks a wrong object (GROUNDING_ROWS), columns = stage-1
                     mask sources followed by the same viewpoint selection (GROUNDING_COLS)
  gallery_mip.pdf    Mip-NeRF 360: rows = selected queries (MIP), columns = B1, B2, ours
                     (baselines from scripts/mip_baselines.py); on ours the photometric servo success over its 6 runs
                     (4x supersampled, out/servo_eval/mipnerf360_same_ss4.csv). No labels, so no good-view mark.

  ~/anaconda3/envs/pca_servo/bin/python scripts/paper_figures.py        (from my_code/)
"""
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
COLS = [("refview", "B1 training photo"), ("clip", "B2 CLIP"), ("capframe", "B3 CapFrame"), ("ours", "Ours")]
# selected queries where our view is a good view (so better than or similar to the training photo B1), all four scenes;
# 20 of the 67 queries qualify, the caption says they are selected (full results: the benchmark table)
ROWS = [("ramen", "kamaboko"), ("figurines", "green toy chair"), ("teatime", "bear nose"), ("waldo_kitchen", "knife"),
        ("waldo_kitchen", "pot"), ("teatime", "apple")]
GROUNDING_COLS = [("ours", "Grounding DINO + SAM"), ("ours_langsplat", "LangSplat-V2"), ("ours_gt", "GT masks")]
GROUNDING_ROWS = [("figurines", "jake"), ("teatime", "three cookies"), ("teatime", "sheep")]

# selected queries where our view is better than or similar to the training photo (B1), at least one per scene that has
# one (none on bicycle); the caption says they are selected: the full results are the servo table
MIP = [("bonsai", "cardboard box"), ("counter", "egg carton"), ("counter", "potted plant"), ("garden", "round table"),
       ("garden", "vase"), ("kitchen", "toy bulldozer"), ("room", "door")]
MIP_COLS = [("refview", "B1 training photo"), ("clip", "B2 CLIP"), ("ours", "Ours")]


def gallery_mip():
    sv = pd.read_csv(OUT / "servo_eval/mipnerf360_same_ss4.csv")
    sv["key"] = sv.run.str.split("/").str[-2:].str.join("/")
    ok, n = sv.groupby("key").success.sum(), sv.groupby("key").size()
    fig, ax = plt.subplots(len(MIP), len(MIP_COLS), figsize=(4.3, 0.96 * len(MIP) + 0.15))
    for i, (s, q) in enumerate(MIP):
        run = q.replace(" ", "_")
        for j, (m, name) in enumerate(MIP_COLS):
            a = ax[i, j]
            a.set_xticks([])
            a.set_yticks([])
            d = OUT / "multiscene" / s / run if m == "ours" else OUT / "baselines" / m / s / run
            im = Image.open(d / "desired_image.png").convert("RGB")
            a.imshow(im.resize((480, round(im.height * 480 / im.width)), Image.LANCZOS))
            if m == "ours":                     # no labels on Mip-NeRF 360: servo success toward our image
                k = f"{s}/{run}"
                a.text(0.97, 0.05, f"servo {ok[k]}/{n[k]}", transform=a.transAxes, ha="right", va="bottom",
                       fontsize=5.5, color="white", bbox=dict(boxstyle="round,pad=0.15", ec="none",
                                                              fc="#1f7a3a" if ok[k] >= n[k] / 2 else "#c03a35"))
            if i == 0:
                a.set_title(name, fontsize=7)
            if j == 0:
                a.set_ylabel(f"{s}\n\u201c{q}\u201d", fontsize=6.5)
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
    plt.subplots_adjust(left=0.075, right=0.995, top=0.965, bottom=0.005, wspace=0.03, hspace=0.04)
    out = ROOT / "paper_mva/figures/gallery_mip.pdf"
    fig.savefig(out, dpi=300)
    print("written", out)


IGS = [("0637", "dining table"), ("0695", "sofa"), ("0651", "tv"), ("0518", "multi person sofa"),
       ("0637", "armchair"), ("0637", "bed"), ("0516", "table lamp")]   # last row: a failure (all candidates occluded)
IGS_COLS = [("refview", "B1 survey view"), ("clip", "B2 CLIP render-and-score"), ("ours", "Ours (ground robot)")]


def gallery_interiorgs():
    """Ground robot in InteriorGS houses: desired images per method, good-view mark (ObjectNav centres)."""
    b = pd.read_csv(OUT / "interiorgs/results_scored.csv")
    fig, ax = plt.subplots(len(IGS), len(IGS_COLS), figsize=(4.3, 1.03 * len(IGS) + 0.15))
    for i, (h, q) in enumerate(IGS):
        sc = next(p.name for p in (OUT / "interiorgs").glob(f"interior_{h}_*"))
        for j, (m, name) in enumerate(IGS_COLS):
            a = ax[i, j]
            a.set_xticks([])
            a.set_yticks([])
            im = Image.open(OUT / "interiorgs" / sc / m / q.replace(" ", "_") / "desired_image.png").convert("RGB")
            a.imshow(im.resize((480, round(im.height * 480 / im.width)), Image.LANCZOS))
            good = int(b[(b.scene == sc) & (b["query"] == q) & (b.method == m)].good_view.iloc[0]) == 1
            a.text(0.03, 0.95, "\u2713" if good else "\u2717", transform=a.transAxes, va="top", fontsize=8,
                   color="white", bbox=dict(boxstyle="round,pad=0.15", fc="#1f7a3a" if good else "#c03a35", ec="none"))
            if i == 0:
                a.set_title(name, fontsize=7)
            if j == 0:
                a.set_ylabel(f"house {h}\n\u201c{q}\u201d", fontsize=6.5)
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
    plt.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.04)
    out = ROOT / "paper_mva/figures/gallery_interiorgs.pdf"
    fig.savefig(out, dpi=300)
    print("written", out)


def lerf_grid(rows, cols, out, width):
    """Outlined thumbnails (scripts/make_gallery.py), rows = (scene, query), columns = (method, title), good-view mark."""
    b = pd.read_csv(OUT / "benchmark/lerf_ovs_goals.csv").set_index(["scene", "query", "method"])
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(width, width / len(cols) * 0.74 * len(rows) + 0.2),
                           squeeze=False)
    for i, (s, q) in enumerate(rows):
        for j, (m, name) in enumerate(cols):
            a = ax[i, j]
            a.set_xticks([])
            a.set_yticks([])
            f = OUT / "gallery" / s / q.replace(" ", "_") / f"{m}.jpg"
            if f.exists():
                a.imshow(Image.open(f))
                good = (s, q, m) in b.index and b.loc[(s, q, m), "good_view"] == 1
                a.text(0.03, 0.95, "✓" if good else "✗", transform=a.transAxes, va="top", fontsize=9,
                       color="white", bbox=dict(boxstyle="round,pad=0.15", fc="#1f7a3a" if good else "#c03a35", ec="none"))
            else:
                a.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=7, transform=a.transAxes)
            if i == 0:
                a.set_title(name, fontsize=7.5)
            if j == 0:
                a.set_ylabel(f"“{q}”", fontsize=7.5)
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
    plt.subplots_adjust(left=0.06, right=0.995, top=1 - 0.25 / fig.get_figheight(), bottom=0.005, wspace=0.03,
                        hspace=0.04)
    fig.savefig(out, dpi=300)
    print("written", out)


def main():
    lerf_grid(ROWS, COLS, ROOT / "paper_mva/figures/gallery_lerf.pdf", 5.16)
    lerf_grid(GROUNDING_ROWS, GROUNDING_COLS, ROOT / "paper_mva/figures/grounding_failures.pdf", 4.3)


if __name__ == "__main__":
    main()
    gallery_mip()
    gallery_interiorgs()
