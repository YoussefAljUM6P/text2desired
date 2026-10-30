"""Training views: image file + camera (T_c2w OpenCV, K, W, H) in the frame of the 3DGS model."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class View:
    name: str
    image_path: str
    T_c2w: np.ndarray
    K: np.ndarray
    W: int
    H: int


def load_colmap_views(cfg_yml, images_dir, gs_vs_dir):
    """Views of a gsplat-trained COLMAP scene, using the same parser (and world normalization) as training.

    cfg_yml: the gsplat run's cfg.yml (data_dir, data_factor, normalize_world_space are read from it);
    gs_vs_dir: repository providing gs_vs.datasets.colmap.Parser. Images are matched to cameras by stem.
    """
    raw = {}
    for line in open(cfg_yml, encoding="utf-8"):
        if ":" in line:
            k, v = line.split(":", 1)
            raw[k.strip()] = v.strip()
    sys.path.insert(0, str(gs_vs_dir))
    from gs_vs.datasets.colmap import Parser

    p = Parser(data_dir=raw.get("data_dir", ""), factor=int(raw.get("data_factor", 1)),
               normalize=raw.get("normalize_world_space", "true").lower() == "true", test_every=8)
    by_stem = {Path(n).stem: i for i, n in enumerate(p.image_names)}
    views = []
    for path in sorted(Path(images_dir).iterdir()):
        i = by_stem.get(path.stem)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or i is None:
            continue
        cid = p.camera_ids[i]
        W, H = p.imsize_dict[cid]
        views.append(View(path.name, str(path), np.asarray(p.camtoworlds[i], np.float64),
                          np.asarray(p.Ks_dict[cid], np.float64), int(W), int(H)))
    return views


def colmap_to_model_transform(cfg_yml, gs_vs_dir):
    """4x4 similarity mapping raw COLMAP coordinates to the world of the gsplat model (identity if not normalized)."""
    raw = {}
    for line in open(cfg_yml, encoding="utf-8"):
        if ":" in line:
            k, v = line.split(":", 1)
            raw[k.strip()] = v.strip()
    sys.path.insert(0, str(gs_vs_dir))
    from gs_vs.datasets.colmap import Parser

    p = Parser(data_dir=raw.get("data_dir", ""), factor=int(raw.get("data_factor", 1)),
               normalize=raw.get("normalize_world_space", "true").lower() == "true", test_every=8)
    return np.asarray(p.transform, np.float64)
