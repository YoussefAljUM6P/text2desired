"""Camera and point-cloud geometry.

Conventions (OpenCV): T_c2w = [R | C], R[:, 0] image right, R[:, 1] image down, R[:, 2] optical axis;
p_cam = R^T (p_world - C).
"""
from __future__ import annotations

import numpy as np


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def look_at(center, target, up) -> np.ndarray | None:
    """Camera-to-world pose at `center` looking at `target` with a level horizon (None if looking along up)."""
    z = normalize(np.asarray(target) - center)
    y = -(up - float(up @ z) * z)
    if np.linalg.norm(y) < 1e-3:
        return None
    y = normalize(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = np.cross(y, z), y, z, center
    return T


def project(points, T_c2w, K):
    """Pixel coordinates (u, v) and depth z of world points."""
    cam = (np.asarray(points) - T_c2w[:3, 3]) @ T_c2w[:3, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2], z


def backproject_mask(mask, depth, rgb, K, T_c2w, erosion=2, percentiles=(5.0, 95.0)):
    """World points (and colors in [0, 1]) of the masked pixels. The mask is eroded to drop unstable edge
    pixels and depths outside the given percentiles are discarded (floaters, background behind the object)."""
    from scipy.ndimage import binary_erosion

    ys, xs = np.where(binary_erosion(mask, iterations=erosion))
    z = depth[ys, xs].astype(np.float64)
    colors = rgb[ys, xs].astype(np.float64) / 255.0
    finite = np.isfinite(z) & (z > 0)
    if not finite.any():
        raise ValueError("No valid depth inside the mask.")
    lo, hi = np.percentile(z[finite], percentiles)
    keep = finite & (z >= lo) & (z <= hi)
    ys, xs, z, colors = ys[keep], xs[keep], z[keep], colors[keep]
    cam = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)
    return cam @ np.asarray(T_c2w, np.float64)[:3, :3].T + T_c2w[:3, 3], colors


def clean_point_cloud(points, colors, voxel_size=None, nb_neighbors=20, std_ratio=2.0):
    """Voxel downsampling (uniform 3D density, so the near side of an oblique object does not dominate the
    centroid), then statistical outlier removal."""
    import open3d as o3d

    ok = np.isfinite(points).all(1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[ok]))
    pcd.colors = o3d.utility.Vector3dVector(colors[ok])
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(pcd.points) < 4:
        raise ValueError("Too few points after cleaning.")
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def pca(points) -> dict:
    """Centroid, axes e1 >= e2 >= normal (smallest-variance axis, arbitrary sign), eigenvalues (descending)
    and planarity ratio rho = lambda3 / lambda2 (small for a planar object)."""
    c = points.mean(0)
    lam, vec = np.linalg.eigh(np.cov((points - c).T))
    lam, vec = lam[::-1], vec[:, ::-1]
    return {"centroid": c, "e1": vec[:, 0], "e2": vec[:, 1], "normal": normalize(vec[:, 2]),
            "eigenvalues": lam, "rho": float(lam[2] / (lam[1] + 1e-12))}


def up_from_cameras(c2w) -> np.ndarray:
    """World up direction from camera poses, for scenes whose frame is not gravity-aligned. Hand-held
    captures have little roll, so every image-right axis is close to horizontal: up is the direction most
    orthogonal to all of them, signed like the mean image-up axis (-y)."""
    x = c2w[:, :3, 0]
    up = normalize(np.linalg.eigh(x.T @ x)[1][:, 0])
    return up if float(up @ -c2w[:, :3, 1].mean(0)) >= 0 else -up
