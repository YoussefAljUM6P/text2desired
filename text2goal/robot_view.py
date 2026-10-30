"""Viewpoint selection for a ground robot (3 degrees of freedom: x, z on the floor and yaw; fixed camera height and
zero pitch), with the score of viewpoint.py: S = vis . cov . prior . fit^2.

Differences with the free-flying camera (viewpoint.select_view):
  candidates  24 horizontal directions u around the object (no elevation); the camera stands on the navigable floor
              at the robot's camera height, level, yawed toward the object centroid (pitch 0: the object is centred
              horizontally only)
  distance    d_req: the object's extents fill at most `fill` of the image (as select_view, with the camera level);
              the camera stands at the navigable floor point along u closest to d_req (searched between 0.4 and 1.5
              d_req); fit = min(1, d / d_min) with d_min the distance at which the object still fits (max_fill)
  free space  replaced by reachability: the navigable floor (navmesh cells) instead of the rendered free distance
  cov         training-camera kernel on the horizontal direction and log distance (survey views)
"""
import numpy as np

from .geometry import normalize
from .viewpoint import ViewParams, broadside_axis, distance_for_fill, visibility


def robot_pose(floor, target, height, up):
    """OpenCV camera-to-world at floor point + height * up, level, yawed toward target (no pitch)."""
    c = floor + height * up
    f = target - c
    f = normalize(f - (f @ up) * up)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(f, up), -up, f], 1)
    T[:3, 3] = c
    return T


def select_robot_view(obj, scene, up, train_c2w, navigable, height, grid, prm: ViewParams = ViewParams()):
    """Goal pose for a ground robot. navigable: (M, 3) navigable floor points (grid cells of size `grid`)."""
    p, n, ref = obj.centroid, normalize(obj.normal), obj.view
    K, W, H = ref.K, ref.W, ref.H
    planar = obj.rho < prm.planarity_threshold
    points = obj.points
    if len(points) > prm.max_points:
        points = points[np.random.default_rng(0).choice(len(points), prm.max_points, replace=False)]
    e1 = normalize(np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0]))
    e2 = np.cross(up, e1)
    floor_y = navigable @ up
    horiz = navigable - np.outer(floor_y, up)
    p_h = p - (p @ up) * up
    # training (survey) cameras seen from the object: horizontal direction and distance
    off = train_c2w[:, :3, 3] - p
    off_h = off - np.outer(off @ up, up)
    cam_dist = np.linalg.norm(off_h, axis=1)
    cam_dir = off_h / np.maximum(cam_dist, 1e-9)[:, None]
    broad = broadside_axis(points, p, up)
    centered = points - points.mean(0)
    main_axis = np.linalg.svd(np.cov(centered.T))[0][:, 0]
    extent = float(np.ptp(np.percentile(points @ main_axis, [2, 98])))
    s = prm.render_scale
    Ks = K.copy()
    Ks[:2] *= s
    ws, hs = max(1, round(W * s)), max(1, round(H * s))

    rows = []
    for az in np.arange(0.0, 360.0, prm.azimuth_step_deg):
        a = np.radians(az)
        u = np.cos(a) * e1 + np.sin(a) * e2
        d_req = distance_for_fill(points, p, u, up, K, W, H, prm.fill, extent)
        d_min = distance_for_fill(points, p, u, up, K, W, H, prm.max_fill, extent)
        if not (np.isfinite(d_req) and np.isfinite(d_min)):
            continue
        # navigable floor points close to the ray p_h + t u (within one grid cell), t in [0.4, 1.5] d_req
        rel = horiz - p_h
        t = rel @ u
        lateral = np.linalg.norm(rel - np.outer(t, u), axis=1)
        ok = (lateral <= grid) & (t >= 0.4 * d_req) & (t <= 1.5 * d_req)
        if not ok.any():
            continue
        i = np.flatnonzero(ok)[np.argmin(np.abs(t[ok] - d_req))]
        d = float(t[i])
        T = robot_pose(navigable[i], p, height, up)
        rgb, depth = scene.render(T, Ks, ws, hs)
        vis, in_image = visibility(points, T, Ks, ws, hs, depth)
        ang = np.degrees(np.arccos(np.clip(cam_dir @ u, -1, 1)))
        cov = float((np.exp(-0.5 * (ang / prm.support_sigma_deg) ** 2)
                     * np.exp(-0.5 * (np.log(np.maximum(cam_dist, 1e-6) / d) / prm.support_sigma_log_distance) ** 2)).max())
        fit = float(min(1.0, d / d_min))
        front = max(0.0, float(u @ n))
        bs = abs(float(u @ broad))
        shape = front ** 2 if planar else 0.5 + 0.5 * bs
        rows.append({"name": f"robot_az{az:03.0f}", "v": u, "T_goal": T, "d": d, "d_required": d_req, "fit": fit,
                     "visibility": vis, "in_image": in_image, "support": cov, "frontality": front, "broadside": bs,
                     "score": vis * cov * shape * fit ** 2, "render": rgb})
    if not rows:
        raise RuntimeError("no reachable candidate")
    fitting = [r for r in rows if r["fit"] >= prm.min_fit]
    chosen = max(fitting or rows, key=lambda r: (r["score"], r["visibility"] * r["support"] * r["fit"] ** 2))
    desired_rgb, desired_depth = scene.render(chosen["T_goal"], K, W, H)
    print(f"[Robot] rho={obj.rho:.3f} planar={planar}: {chosen['name']} vis={chosen['visibility']:.2f} "
          f"cov={chosen['support']:.2f} fit={chosen['fit']:.2f} d={chosen['d']:.2f} (required {chosen['d_required']:.2f})")
    return {"chosen": chosen, "candidates": rows, "T_goal": chosen["T_goal"], "d_star": chosen["d"],
            "desired_rgb": desired_rgb, "desired_depth": desired_depth, "v_world": chosen["v"]}
