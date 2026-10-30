"""Stage 1: text query -> 3D object geometry.

Grounding DINO scores every training view; the reference view is the best detection whose box does not
touch the image border, provided it shows the same object as the overall best detection. SAM segments the
box in the 3DGS render of that view; the eroded mask is back-projected with the rendered depth, cleaned,
and analysed with PCA (centroid, dominant-plane normal oriented toward the reference camera, planarity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .geometry import backproject_mask, clean_point_cloud, normalize, pca


@dataclass
class ObjectGeometry:
    query: str
    view: object                 # reference View
    box: list                    # detection box in the reference render (pixels at view.W x view.H)
    mask: np.ndarray             # SAM mask in the reference render
    rgb: np.ndarray              # reference render
    points: np.ndarray           # cleaned object cloud (world)
    colors: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray           # PCA normal, oriented toward the reference camera
    eigenvalues: np.ndarray      # descending
    rho: float                   # lambda3 / lambda2
    detection: dict = field(default_factory=dict)


def scan(views, detector):
    """Detections of the query in every view, best first."""
    hits = []
    for k, v in enumerate(views):
        det = detector(Image.open(v.image_path))
        if det is not None:
            hits.append(det | {"view": v, "image_size": Image.open(v.image_path).size})
        if (k + 1) % 50 == 0 or k + 1 == len(views):
            print(f"[GD] {k + 1}/{len(views)} views, best={max((h['score'] for h in hits), default=0):.3f}")
    if not hits:
        raise RuntimeError(f"No detection of '{detector.prompt}'.")
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def _box_in_view(hit):
    """Detection box rescaled from the image file size to the camera size."""
    v, (w, h) = hit["view"], hit["image_size"]
    sx, sy = v.W / w, v.H / h
    x0, y0, x1, y1 = hit["box"]
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def touches_border(hit, margin=0.01):
    w, h = hit["image_size"]
    x0, y0, x1, y1 = hit["box"]
    return x0 <= margin * w or y0 <= margin * h or x1 >= (1 - margin) * w or y1 >= (1 - margin) * h


def reference_detection(hits, scene, scale=0.25):
    """Best non-truncated detection showing the same object as the top one (a truncated box biases the
    centroid, but a whole box of another object is worse). Same object: the 3D point under the top box
    center (median rendered depth over the central 40% of the box) projects inside the candidate's box."""
    top = hits[0]
    if not touches_border(top):
        return top
    v = top["view"]
    K = v.K.copy()
    K[:2] *= scale
    w, h = int(v.W * scale), int(v.H * scale)
    _, depth = scene.render(v.T_c2w, K, w, h)
    x0, y0, x1, y1 = (c * scale for c in _box_in_view(top))
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    patch = depth[int(cy - 0.2 * bh): int(cy + 0.2 * bh) + 1, int(cx - 0.2 * bw): int(cx + 0.2 * bw) + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if not len(patch):
        return top
    Z = float(np.median(patch))
    p = v.T_c2w[:3, :3] @ np.array([(cx - K[0, 2]) / K[0, 0] * Z, (cy - K[1, 2]) / K[1, 1] * Z, Z]) + v.T_c2w[:3, 3]
    for hit in hits[1:]:
        if touches_border(hit):
            continue
        c = hit["view"]
        q = c.T_c2w[:3, :3].T @ (p - c.T_c2w[:3, 3])
        if q[2] <= 0:
            continue
        u, vv = c.K[0, 0] * q[0] / q[2] + c.K[0, 2], c.K[1, 1] * q[1] / q[2] + c.K[1, 2]
        bx0, by0, bx1, by1 = _box_in_view(hit)
        if bx0 <= u <= bx1 and by0 <= vv <= by1:
            print(f"[Ground] top view {v.name} is truncated; using {c.name} (score {hit['score']:.3f})")
            return hit
    print(f"[Ground] top view {v.name} is truncated and no whole view of the same object exists; keeping it")
    return top


def localize(query, views, scene, detector, segmenter, voxel_size=None, nb_neighbors=20, std_ratio=2.0):
    hit = reference_detection(scan(views, detector), scene)
    v = hit["view"]
    rgb, depth = scene.render(v.T_c2w, v.K, v.W, v.H)
    box = _box_in_view(hit)
    mask = segmenter(rgb, box)
    return object_from_mask(query, v, rgb, depth, mask, box, {k: hit[k] for k in ("score", "confidence", "box", "area")},
                            voxel_size, nb_neighbors, std_ratio)


def object_from_mask(query, v, rgb, depth, mask, box, detection, voxel_size=None, nb_neighbors=20, std_ratio=2.0):
    """Back-project a mask of the reference render, clean, PCA. Also used with masks from another grounding
    module (scripts/langsplat_ground.py)."""
    raw, raw_colors = backproject_mask(mask, depth, rgb, v.K, v.T_c2w)
    if voxel_size is None:
        lo, hi = np.percentile(raw, [2, 98], axis=0)
        voxel_size = float(np.linalg.norm(hi - lo)) / 200.0
    points, colors = clean_point_cloud(raw, raw_colors, voxel_size, nb_neighbors, std_ratio)
    g = pca(points)
    n = g["normal"] if g["normal"] @ normalize(v.T_c2w[:3, 3] - g["centroid"]) >= 0 else -g["normal"]
    print(f"[Ground] '{query}': view {v.name} score {detection.get('score', float('nan')):.3f}, {len(points)} points, "
          f"rho {g['rho']:.3f}")
    return ObjectGeometry(query, v, box, mask, rgb, points, colors, g["centroid"], n, g["eigenvalues"], g["rho"],
                          detection)
