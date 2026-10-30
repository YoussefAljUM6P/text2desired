"""3DGS scene that renders RGB + expected depth at an OpenCV camera pose (gsplat rasterizer).

Loads either a gsplat training checkpoint (`splats` dict) or a standard 3DGS .ply (Inria format,
Habitat-GS `.gs.ply`).
"""
from __future__ import annotations

import gc

import numpy as np
import torch


def _load_ckpt(path, device):
    s = torch.load(path, map_location=device, weights_only=True)["splats"]
    return {"means": s["means"], "quats": s["quats"], "scales": torch.exp(s["scales"]),
            "opacities": torch.sigmoid(s["opacities"]), "colors": torch.cat([s["sh0"], s["shN"]], 1)}


def _load_ply(path, device):
    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                assert line.split()[1] == "float", line
                names.append(line.split()[-1])
        a = np.fromfile(f, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    g = lambda *ks: torch.from_numpy(a[:, [col[k] for k in ks]].copy()).to(device)  # noqa: E731
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    sh0 = g("f_dc_0", "f_dc_1", "f_dc_2")[:, None]
    shN = g(*rest).reshape(n, 3, -1).transpose(1, 2) if rest else sh0[:, :0]
    return {"means": g("x", "y", "z"), "quats": g("rot_0", "rot_1", "rot_2", "rot_3"),
            "scales": torch.exp(g("scale_0", "scale_1", "scale_2")),
            "opacities": torch.sigmoid(g("opacity")[:, 0]), "colors": torch.cat([sh0, shN], 1)}


class GaussianScene:
    def __init__(self, path, device="cuda"):
        self.device = device
        self.splats = (_load_ply if str(path).endswith(".ply") else _load_ckpt)(str(path), device)
        self.sh_degree = int(np.sqrt(self.splats["colors"].shape[1])) - 1
        print(f"[3DGS] {len(self.splats['means']):,} Gaussians, SH degree {self.sh_degree}")

    @torch.no_grad()
    def render(self, T_c2w, K, W, H):
        """uint8 RGB (H, W, 3) and float32 expected depth (H, W)."""
        from gsplat.rendering import rasterization

        view = torch.from_numpy(np.linalg.inv(np.asarray(T_c2w, np.float64))).float().to(self.device)
        Ks = torch.from_numpy(np.asarray(K, np.float64)).float().to(self.device)
        s = self.splats
        out, _, _ = rasterization(s["means"], s["quats"], s["scales"], s["opacities"], s["colors"],
                                  view[None], Ks[None], int(W), int(H), sh_degree=self.sh_degree,
                                  packed=True, render_mode="RGB+ED", camera_model="pinhole")
        rgb = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return rgb, out[0, ..., 3].cpu().numpy().astype(np.float32)

    def release(self):
        self.splats.clear()
        gc.collect()
        torch.cuda.empty_cache()


class TransformedScene:
    """A 3DGS model trained in another world frame (e.g. raw COLMAP, Inria 3DGS / LangSplat), rendered at poses of
    our frame. M: 4x4 similarity mapping its frame to ours (p_ours = s R p_its + t). Depth is returned in our units."""

    def __init__(self, path, M, device="cuda"):
        self.inner = GaussianScene(path, device)
        A = np.asarray(M, np.float64)
        self.s = float(np.cbrt(np.linalg.det(A[:3, :3])))
        self.R, self.t = A[:3, :3] / self.s, A[:3, 3]

    def render(self, T_c2w, K, W, H):
        T = np.eye(4)
        T[:3, :3] = self.R.T @ T_c2w[:3, :3]
        T[:3, 3] = self.R.T @ (T_c2w[:3, 3] - self.t) / self.s
        rgb, depth = self.inner.render(T, K, W, H)
        return rgb, depth * self.s

    def release(self):
        self.inner.release()
