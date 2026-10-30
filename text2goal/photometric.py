"""Image-based check of a desired image: direct (photometric) visual servoing toward it.

Unlike servo.py (geometric features at a known goal pose, converges by construction), this law only sees images:
the current view is rendered from a 3DGS model, the error is e = I - I* on grey levels, and the interaction matrix
L_I = -grad(I)^T L_x uses the current rendered depth (Collewet & Marchand 2011). Levenberg-Marquardt update
v = -gain (H + mu diag H)^-1 L^T e, H = L^T L, integrated on SE(3). Same law as vanilla_gs_vs (photometric).
"""
from __future__ import annotations

import cv2
import numpy as np

from .servo import exp_se3

# 7-tap Gaussian-derivative filter (ViSP / vanilla_gs_vs)
_C = np.array([112.0, 913.0, 2047.0]) / 8418.0
_DERIV = np.array([-_C[2], -_C[1], -_C[0], 0.0, _C[0], _C[1], _C[2]], np.float32)


def gray(rgb):
    return (rgb.astype(np.float32) @ np.array([0.2989, 0.5870, 0.1140], np.float32))


def error_and_interaction(I, Z, I_star, K, border=10):
    """Photometric error (N,) and interaction matrix (N, 6) on the image minus a border."""
    Ix = cv2.filter2D(I, -1, _DERIV[None], borderType=cv2.BORDER_REFLECT)
    Iy = cv2.filter2D(I, -1, _DERIV[:, None], borderType=cv2.BORDER_REFLECT)
    H, W = I.shape
    sl = (slice(border, H - border), slice(border, W - border))
    v, u = np.mgrid[sl]
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    Z = Z[sl]
    Zi = np.where(np.isfinite(Z) & (Z > 1e-6), 1.0 / np.where(Z > 1e-6, Z, 1.0), 1.0)
    gx, gy = Ix[sl] * K[0, 0], Iy[sl] * K[1, 1]
    L = -np.stack([gx * -Zi, gy * -Zi, (gx * x + gy * y) * Zi, gx * x * y + gy * (1 + y * y),
                   -gx * (1 + x * x) - gy * x * y, gx * y - gy * x], -1)
    return (I[sl] - I_star[sl]).reshape(-1), L.reshape(-1, 6)


def photometric_servo(render, T_start, I_star, K, iterations=400, gain=0.5, mu=0.01, tol=1.0, dof=None):
    """render(T) -> (rgb uint8, depth). I_star: desired grey image at the render size. Stops when the mean
    squared grey-level error falls below `tol` or stalls. dof: indices of the controlled components of the camera
    twist (vx, vy, vz, wx, wy, wz; OpenCV camera frame), e.g. (0, 2, 4) for a ground robot (lateral, forward, yaw);
    None = all six. Returns poses (n, 4, 4) and the error history."""
    T = np.asarray(T_start, np.float64).copy()
    poses, errs = [T.copy()], []
    for _ in range(iterations):
        rgb, Z = render(T)
        e, L = error_and_interaction(gray(rgb), Z, I_star, K)
        errs.append(float(np.mean(e * e)))
        if errs[-1] < tol or (len(errs) > 60 and errs[-1] > 0.999 * errs[-50]):
            break
        if dof is not None:
            L = L[:, list(dof)]
        Hs = L.T @ L
        v = -gain * np.linalg.solve(Hs + mu * np.diag(np.diag(Hs)), L.T @ e)
        if dof is not None:                       # uncontrolled components stay at zero
            full = np.zeros(6)
            full[list(dof)] = v
            v = full
        T = T @ exp_se3(v)
        poses.append(T.copy())
    return np.stack(poses), errs
