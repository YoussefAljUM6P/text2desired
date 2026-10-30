"""Stage 2: object geometry -> goal pose and desired image.

The PCA normal is kept when the object is planar and its view is renderable (visible, covered by training
cameras, the whole object fits in free space). Otherwise level candidate views around the object are
rendered and scored, without any learned judge:
  S(v) = visibility * support * shape * fit^2
  visibility : fraction of object points not occluded in the rendered depth
  support    : closeness to a training camera that sees the object (angle and log-distance kernels)
  shape      : planar -> (v . n)_+^2 ; non-planar -> 1/2 + 1/2 broadside alignment
  fit        : min(1, usable free space / distance at which the object spans max_fill of the image)
Each direction gets its own goal distance: min(size-rule distance at `fill`, margin * free distance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import look_at, normalize, project


@dataclass
class ViewParams:
    fill: float = 0.6                 # preferred object size / image size at the goal
    max_fill: float = 0.85            # largest acceptable size (whole object in view)
    margin: float = 0.9               # usable fraction of the free space behind the camera
    min_fit: float = 0.98             # fit required to keep the PCA-normal view / preferred in the search
    planarity_threshold: float = 0.2  # rho below which the PCA normal is trusted
    min_visibility: float = 0.6
    min_support: float = 0.25
    support_sigma_deg: float = 15.0
    support_sigma_log_distance: float = 0.4
    azimuth_step_deg: float = 15.0
    render_scale: float = 0.25        # candidate renders are scored at this resolution
    max_points: int = 5000
    continuous: bool = False          # one score for all objects: shape = w (v.n)^2 + (1 - w)(1/2 + 1/2 broadside),
                                      # w = 1 / (1 + (rho / planarity_threshold)^4); no planar branch
    support_floor: float = 0.0        # candidates below this support are dropped when others remain
    ablate: tuple = ()                # ablations (scripts/ablation.py): score terms replaced by 1 ("vis", "cov",
                                      # "prior", "fit"); "free_space": camera at d_req, free space ignored;
                                      # "normal_only" / "reference_only": that candidate, no search


def visibility(points, T, K, W, H, depth):
    """(visible fraction, in-image fraction) of the object points in a render."""
    u, v, z = project(points, T, K)
    inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return 0.0, 0.0
    rendered = depth[v[inside].astype(int), u[inside].astype(int)]
    visible = np.isfinite(rendered) & (rendered >= z[inside] - (0.05 + 0.03 * z[inside]))
    return float(visible.sum()) / len(points), float(inside.mean())


def free_distance(scene, p, v, up, K, W, H, extent, scale=0.25):
    """Distance from the object to the first surface along +v: median depth of the central image patch of a
    camera placed just outside the object, looking along v."""
    start = p + 0.05 * extent * v
    T = look_at(start, start + v, up)
    if T is None:
        return float("inf")
    Ks = K.copy()
    Ks[:2] *= scale
    w, h = max(8, int(W * scale)), max(8, int(H * scale))
    _, depth = scene.render(T, Ks, w, h)
    patch = depth[h // 2 - h // 20: h // 2 + h // 20 + 1, w // 2 - w // 20: w // 2 + w // 20 + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) + 0.05 * extent if len(patch) else float("inf")


def projected_fill(points, p, v, d, up, K, W, H):
    """Object size in the image (larger of horizontal / vertical, 99th percentile around the center, as a
    fraction of the image) for a camera at p + d v looking at p. >1: the object overflows."""
    T = look_at(p + d * v, p, up)
    if T is None:
        return float("inf")
    cam = (points - T[:3, 3]) @ T[:3, :3]
    if np.any(cam[:, 2] <= 1e-6):
        return float("inf")
    fx = 2 * np.percentile(np.abs(K[0, 0] * cam[:, 0] / cam[:, 2]), 99) / W
    fy = 2 * np.percentile(np.abs(K[1, 1] * cam[:, 1] / cam[:, 2]), 99) / H
    return float(max(fx, fy))


def distance_for_fill(points, p, v, up, K, W, H, fill, extent):
    """Smallest distance at which the projected size is <= fill (bisection)."""
    lo, hi = 0.05 * extent, 200.0 * extent
    if projected_fill(points, p, v, hi, up, K, W, H) > fill:
        return float("nan")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if projected_fill(points, p, v, mid, up, K, W, H) > fill else (lo, mid)
    return float(hi)


def broadside_axis(points, c, up):
    """Horizontal direction of least spread of the object's footprint."""
    flat = (points - c) - np.outer((points - c) @ up, up)
    axis = np.linalg.eigh(np.cov(flat.T))[1][:, 1]
    return normalize(axis - float(axis @ up) * up)


def select_view(obj, scene, up, train_c2w, prm: ViewParams = ViewParams()):
    """Goal pose and desired image for an ObjectGeometry. `train_c2w`: (N, 4, 4) training camera poses."""
    p, n, ref = obj.centroid, normalize(obj.normal), obj.view
    K, W, H = ref.K, ref.W, ref.H
    planar = obj.rho < prm.planarity_threshold
    points = obj.points
    if len(points) > prm.max_points:
        points = points[np.random.default_rng(0).choice(len(points), prm.max_points, replace=False)]

    # training cameras that see the object centroid (5% image margin)
    seeing = []
    for i, T in enumerate(train_c2w):
        u, v, z = project(p[None], T, K)
        if z[0] > 0.05 and 0.05 * W <= u[0] <= 0.95 * W and 0.05 * H <= v[0] <= 0.95 * H:
            seeing.append(i)
    if not seeing:
        raise RuntimeError("No training camera sees the object centroid.")
    off = train_c2w[seeing, :3, 3] - p
    cam_dist = np.linalg.norm(off, axis=1)
    cam_dir = off / cam_dist[:, None]

    def support(v, d):
        ang = np.degrees(np.arccos(np.clip(cam_dir @ v, -1, 1)))
        return float((np.exp(-0.5 * (ang / prm.support_sigma_deg) ** 2)
                      * np.exp(-0.5 * (np.log(cam_dist / d) / prm.support_sigma_log_distance) ** 2)).max())

    # candidates: PCA normal, reference-view direction, level rings at the training-camera elevations
    east = normalize(np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0]))
    north = np.cross(up, east)
    cands = [("pca_normal", n), ("reference_view", normalize(ref.T_c2w[:3, 3] - p))]
    for el in np.percentile(np.degrees(np.arcsin(np.clip(cam_dir @ up, -1, 1))), [25, 50, 75]):
        for az in np.arange(0.0, 360.0, prm.azimuth_step_deg):
            a, e = np.radians(az), np.radians(el)
            cands.append((f"ring_el{el:+.0f}_az{az:03.0f}",
                          normalize(np.cos(e) * (np.cos(a) * east + np.sin(a) * north) + np.sin(e) * up)))

    broad = broadside_axis(points, p, up)
    s = prm.render_scale
    Ks = K.copy()
    Ks[:2] *= s
    ws, hs = max(1, round(W * s)), max(1, round(H * s))
    centered = points - points.mean(0)
    main_axis = np.linalg.svd(np.cov(centered.T))[0][:, 0]
    extent = float(np.ptp(np.percentile(points @ main_axis, [2, 98])))

    rows = []
    for name, v in cands:
        d_req = distance_for_fill(points, p, v, up, K, W, H, prm.fill, extent)
        d_min = distance_for_fill(points, p, v, up, K, W, H, prm.max_fill, extent)
        if not (np.isfinite(d_req) and np.isfinite(d_min)):
            continue
        free = np.inf if "free_space" in prm.ablate else free_distance(scene, p, v, up, K, W, H, extent)
        d = float(min(d_req, prm.margin * free))
        T = look_at(p + d * v, p, up)
        if T is None:
            continue
        rgb, depth = scene.render(T, Ks, ws, hs)
        vis, in_image = visibility(points, T, Ks, ws, hs, depth)
        horiz = v - float(v @ up) * up
        r = {"name": name, "v": v, "T_goal": T, "d": d, "d_required": d_req, "free_distance": free,
             "fit": float(min(1.0, prm.margin * free / d_min)), "visibility": vis, "in_image": in_image,
             "support": support(v, d), "frontality": max(0.0, float(v @ n)),
             "broadside": abs(float(normalize(horiz) @ broad)) if np.linalg.norm(horiz) > 1e-6 else 0.0,
             "render": rgb}
        if prm.continuous:
            wp = 1.0 / (1.0 + (obj.rho / prm.planarity_threshold) ** 4)
            shape = wp * r["frontality"] ** 2 + (1 - wp) * (0.5 + 0.5 * r["broadside"])
        else:
            shape = r["frontality"] ** 2 if planar else 0.5 + 0.5 * r["broadside"]
        # ablation: a removed term is 1 everywhere (score, planar-normal rule, fit preference, tie-break); the
        # measured value is kept as <term>_measured
        for key, term in (("visibility", "vis"), ("support", "cov"), ("fit", "fit")):
            if term in prm.ablate:
                r[key + "_measured"], r[key] = r[key], 1.0
        if "prior" in prm.ablate:
            shape = 1.0
        r["score"] = r["visibility"] * r["support"] * shape * r["fit"] ** 2
        rows.append(r)

    supported = [r for r in rows if r["support"] >= prm.support_floor] or rows
    normal_row = next((r for r in supported if r["name"] == "pca_normal"), None)
    fixed = {"normal_only": "pca_normal", "reference_only": "reference_view"}
    only = next((fixed[k] for k in prm.ablate if k in fixed), None)
    if only is not None:
        chosen = next((r for r in rows if r["name"] == only), None)
        if chosen is None:
            raise RuntimeError(f"ablation {only}: candidate not renderable")
        method = only + "_ablation"
    elif prm.continuous:
        fitting = [r for r in supported if r["fit"] >= prm.min_fit]
        chosen = max(fitting or supported, key=lambda r: (r["score"], r["visibility"] * r["support"] * r["fit"] ** 2))
        method = "continuous_search"
    elif (planar and normal_row is not None and normal_row["visibility"] >= prm.min_visibility
            and normal_row["support"] >= prm.min_support and normal_row["fit"] >= prm.min_fit):
        chosen, method = normal_row, "pca_normal_planar"
    else:
        pool = [r for r in supported if r["name"] != "pca_normal" or planar]
        fitting = [r for r in pool if r["fit"] >= prm.min_fit]  # whole object in view is a hard preference
        # ties (in practice: every score 0, e.g. a planar prior that no level candidate can face) are broken by the
        # score without the shape prior, not by candidate order
        chosen = max(fitting or pool, key=lambda r: (r["score"], r["visibility"] * r["support"] * r["fit"] ** 2))
        method = "planar_fallback_search" if planar else "nonplanar_render_search"

    # start pose for the servo: the reference view if the normal was kept, else the training camera closest
    # to the goal direction that sees the object from the chosen side
    v = chosen["v"]
    T_start, start = ref.T_c2w, "reference_view"
    if method != "pca_normal_planar":
        ok = [i for i in seeing if (train_c2w[i][:3, :3].T @ v)[2] < -0.3]
        if ok:
            i = max(ok, key=lambda i: float(normalize(train_c2w[i][:3, 3] - p) @ v))
            T_start, start = train_c2w[i], f"training_camera_{i}"

    desired_rgb, desired_depth = scene.render(chosen["T_goal"], K, W, H)
    print(f"[View] rho={obj.rho:.3f} planar={planar} {method}: {chosen['name']} vis={chosen['visibility']:.2f} "
          f"sup={chosen['support']:.2f} fit={chosen['fit']:.2f} d={chosen['d']:.2f} "
          f"(required {chosen['d_required']:.2f}, free {chosen['free_distance']:.2f})")
    return {"method": method, "chosen": chosen, "candidates": rows, "v_world": v, "d_star": chosen["d"],
            "T_goal": chosen["T_goal"], "T_start": T_start, "start_source": start,
            "desired_rgb": desired_rgb, "desired_depth": desired_depth,
            "angle_to_pca_normal_deg": float(np.degrees(np.arccos(np.clip(v @ n, -1, 1))))}
