"""Oracle grounding for LERF-OVS: stage 1 replaced by the dataset's own labels (upper bound for stages 2-3).

For every query, the reference view is the label frame where the query's largest labelled polygon is, and the mask
is that polygon. Writes out/gt_ground/<scene>/<query>/{mask.png, ground.json} for text_to_desired_image.py --mask_json.

  ~/anaconda3/envs/pca_servo/bin/python scripts/gt_ground.py        (from my_code/)
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np

L = "/media/youssef/DATA/lerf_ovs/lerf_ovs"
OUT = Path(__file__).resolve().parents[1] / "out" / "gt_ground"

for s in ["figurines", "teatime", "ramen", "waldo_kitchen"]:
    best = {}
    for f in sorted(glob.glob(f"{L}/label/{s}/*.json")):
        d = json.load(open(f))
        for o in d["objects"]:
            poly = np.array(o["segmentation"], np.float64)
            area = abs(cv2.contourArea(poly.astype(np.float32)))
            if area > best.get(o["category"], (0,))[0]:
                best[o["category"]] = (area, d["info"]["name"], poly, d["info"]["width"], d["info"]["height"])
    for q, (area, name, poly, w, h) in best.items():
        o = OUT / s / q.replace(" ", "_")
        o.mkdir(parents=True, exist_ok=True)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
        cv2.imwrite(str(o / "mask.png"), m)
        (o / "ground.json").write_text(json.dumps({"query": q, "image": Path(name).stem, "score": 1.0,
                                                   "mask": str(o / "mask.png"), "source": "gt_labels"}, indent=2))
    print(s, len(best), "queries")
