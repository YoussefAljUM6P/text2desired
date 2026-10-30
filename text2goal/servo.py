"""Stage 3 (consumer): analytic 6-feature servo that reaches the goal pose geometrically.

s = [x, y, n_x, n_y, (d - d*)/d*, g_x], s* = 0, with (x, y) the normalized image coordinates of the object
centroid, n the viewing direction (object -> camera) in the camera frame (n_z < 0: visible side), d the
camera-object distance and g the world up direction in the camera frame (roll). Damped least squares
v = -lambda (L^T L + mu^2 I)^-1 L^T s, integrated on SE(3): T <- T exp(v).
No rendering, detector or numerical Jacobian. Online detector correction (YA_code run_online_servo.py) is
not included here.
"""
from __future__ import annotations

import numpy as np

from .geometry import normalize


def skew(w):
    x, y, z = w
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])


def exp_se3(xi):
    xi = np.asarray(xi, np.float64)
    phi, t = xi[3:], np.linalg.norm(xi[3:])
    S = skew(phi)
    if t < 1e-10:
        R, J = np.eye(3) + S + 0.5 * S @ S, np.eye(3) + 0.5 * S + S @ S / 6
    else:
        R = np.eye(3) + np.sin(t) / t * S + (1 - np.cos(t)) / t**2 * S @ S
        J = np.eye(3) + (1 - np.cos(t)) / t**2 * S + (t - np.sin(t)) / t**3 * S @ S
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, J @ xi[:3]
    return T


def features(T, p, n_world, d_star, up=None, K=None, min_Z=0.05):
    """Feature vector s, interaction matrix L and readable errors at pose T."""
    R, C = T[:3, :3], T[:3, 3]
    X, Y, Z = R.T @ (p - C)
    if Z <= min_Z:
        raise ValueError(f"object behind or too close to the camera (Z={Z:.3f})")
    x, y = X / Z, Y / Z
    nx, ny, nz = normalize(R.T @ n_world)
    d = float(np.linalg.norm(C - p))
    L = np.array([
        [-1 / Z, 0, x / Z, x * y, -(1 + x * x), y],
        [0, -1 / Z, y / Z, 1 + y * y, -x * y, -x],
        [0, 0, 0, 0, -nz, ny],
        [0, 0, 0, nz, 0, -nx],
        [-X / (d * d_star), -Y / (d * d_star), -Z / (d * d_star), 0, 0, 0]])
    s = [x, y, nx, ny, (d - d_star) / d_star]
    err = {"orientation_deg": float(np.degrees(np.arccos(np.clip(-nz, -1, 1)))), "distance": d, "nz": float(nz)}
    if up is not None:
        gx, gy, gz = normalize(R.T @ up)
        L = np.vstack([L, [0, 0, 0, 0, -gz, gy]])
        s.append(gx)
        err["roll_deg"] = float(np.degrees(np.arctan2(gx, -gy)))
    if K is not None:
        err["center_px"] = float(np.hypot(K[0, 0] * x, K[1, 1] * y))
    return np.array(s), L, err


def servo(T_start, p, v_world, d_star, K, up=None, gain=0.3, damping=0.01, iterations=100, min_abs_nz=0.1):
    """Run the fixed-length servo loop. An update that would put the target behind the camera or make the
    normal rows singular (|n_z| < min_abs_nz) is rejected and the pose kept. Returns poses and error history."""
    T = np.asarray(T_start, np.float64).copy()
    poses, history = [T.copy()], []
    for _ in range(iterations):
        s, L, err = features(T, p, v_world, d_star, up, K)
        history.append(err)
        if err["nz"] >= 0 or abs(err["nz"]) < min_abs_nz:
            poses.append(T.copy())
            continue
        cmd = np.linalg.solve(L.T @ L + damping**2 * np.eye(6), -gain * L.T @ s)
        cand = T @ exp_se3(cmd)
        try:
            nz = features(cand, p, v_world, d_star, up, K)[2]["nz"]
            if nz < 0 and abs(nz) >= min_abs_nz:
                T = cand
        except ValueError:
            pass
        poses.append(T.copy())
    history.append(features(T, p, v_world, d_star, up, K)[2])
    return np.stack(poses), history
