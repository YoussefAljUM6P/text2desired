"""Text query -> object geometry -> goal pose + desired image -> (optional) analytic servo check.

Example (Mip-NeRF 360 room, 3DGS trained with the MoGe-2 depth prior, env pca_servo):
  ~/anaconda3/envs/pca_servo/bin/python scripts/text_to_desired_image.py --query piano \
    --model /media/youssef/LaCie/data/mip-nerf360/360_v2/room/output_3DGS_inria/moge2_depth/gsplat_ckpt_30000.pt \
    --cfg <YA_code>/config/scenes/room_colmap_frame_cfg.yml \
    --images /media/youssef/LaCie/data/mip-nerf360/360_v2/room/images --servo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text2goal.cameras import load_colmap_views  # noqa: E402
from text2goal.geometry import up_from_cameras  # noqa: E402
from text2goal.localize import localize, object_from_mask  # noqa: E402
from text2goal.perception import Detector, Segmenter  # noqa: E402
from text2goal.render import GaussianScene  # noqa: E402
from text2goal.servo import servo  # noqa: E402
from text2goal.viewpoint import ViewParams, select_view  # noqa: E402


def contact_sheet(rows, chosen, path, n=8, cols=4):
    from PIL import ImageDraw

    first = [r for r in rows if r["name"] in ("pca_normal", "reference_view")]
    rest = sorted((r for r in rows if r not in first), key=lambda r: r["score"], reverse=True)
    tiles = []
    for r in (first + rest)[:n]:
        tile = Image.fromarray(r["render"])
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width, 30), fill=(0, 0, 0))
        draw.text((4, 2), f"{r['name']}{' <== CHOSEN' if r is chosen else ''} d={r['d']:.2f} fit={r['fit']:.2f}\n"
                  f"vis={r['visibility']:.2f} sup={r['support']:.2f} score={r['score']:.3f}",
                  fill=(255, 255, 0) if r is chosen else (255, 255, 255))
        tiles.append(tile)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * w, (k // cols) * h))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True)
    ap.add_argument("--model", required=True, help="gsplat checkpoint (.pt) or 3DGS .ply")
    ap.add_argument("--cfg", required=True, help="gsplat cfg.yml of the COLMAP scene (cameras)")
    ap.add_argument("--images", required=True, help="training images scanned by the detector")
    ap.add_argument("--gs_vs_dir", default=str(Path(__file__).resolve().parents[1] / "third_party"))
    ap.add_argument("--sam_ckpt", default="/home/youssef/Documents/SW/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--gd_model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--gd_threshold", type=float, default=0.2)
    ap.add_argument("--servo", action="store_true", help="also run the analytic servo from the start pose")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask_json", default=None, help="external grounding (scripts/langsplat_ground.py): json with "
                    "the reference image name, a mask png and a score; replaces Grounding DINO + SAM")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "out" / args.query.replace(" ", "_"))
    out.mkdir(parents=True, exist_ok=True)

    views = load_colmap_views(args.cfg, args.images, args.gs_vs_dir)
    train_c2w = np.stack([v.T_c2w for v in views])
    up = up_from_cameras(train_c2w)
    scene = GaussianScene(args.model)
    if args.mask_json:
        info = json.loads(Path(args.mask_json).read_text())
        v = next(v for v in views if Path(v.name).stem == Path(info["image"]).stem)
        rgb, depth = scene.render(v.T_c2w, v.K, v.W, v.H)
        mask = np.asarray(Image.open(info["mask"]).convert("L").resize((v.W, v.H), Image.NEAREST)) > 127
        ys, xs = np.where(mask)
        obj = object_from_mask(args.query, v, rgb, depth, mask, [xs.min(), ys.min(), xs.max(), ys.max()],
                               {"score": info["score"], "source": info.get("source", "external")})
    else:
        detector = Detector(args.query, args.gd_model, args.gd_threshold)
        segmenter = Segmenter(args.sam_ckpt)
        obj = localize(args.query, views, scene, detector, segmenter)
        del detector, segmenter
    goal = select_view(obj, scene, up, train_c2w, ViewParams())

    Image.fromarray(obj.rgb).save(out / "reference_render.png")
    Image.fromarray(obj.mask.astype(np.uint8) * 255).save(out / "reference_mask.png")
    Image.fromarray(goal["desired_rgb"]).save(out / "desired_image.png")
    np.save(out / "desired_depth.npy", goal["desired_depth"])
    contact_sheet(goal["candidates"], goal["chosen"], out / "candidates.png")
    np.savez(out / "goal.npz", T_goal=goal["T_goal"], T_start=goal["T_start"], v_world=goal["v_world"],
             d_star=goal["d_star"], p_obj=obj.centroid, n_obj=obj.normal, K=obj.view.K, W=obj.view.W,
             H=obj.view.H, up_world=up, points=obj.points)
    c = goal["chosen"]
    summary = {"query": args.query, "reference_view": obj.view.name, "detection": obj.detection,
               "rho": obj.rho, "eigenvalues": obj.eigenvalues.tolist(), "p_obj": obj.centroid.tolist(),
               "n_obj": obj.normal.tolist(), "method": goal["method"], "chosen": c["name"],
               "v_world": goal["v_world"].tolist(), "d_star": goal["d_star"],
               "angle_to_pca_normal_deg": goal["angle_to_pca_normal_deg"], "start_source": goal["start_source"],
               **{k: c[k] for k in ("visibility", "support", "fit", "frontality", "broadside", "score",
                                    "free_distance", "d_required")}}

    if args.servo:
        poses, hist = servo(goal["T_start"], obj.centroid, goal["v_world"], goal["d_star"], obj.view.K, up)
        rgb, _ = scene.render(poses[-1], obj.view.K, obj.view.W, obj.view.H)
        Image.fromarray(np.hstack([goal["desired_rgb"], rgb])).save(out / "desired_vs_servo_final.png")
        np.save(out / "servo_poses.npy", poses)
        f = hist[-1]
        summary["servo_final"] = {**f, "pose_error_m": float(np.linalg.norm(poses[-1][:3, 3] - goal["T_goal"][:3, 3])),
                                  "pose_error_deg": float(np.degrees(np.arccos(np.clip(
                                      (np.trace(poses[-1][:3, :3].T @ goal["T_goal"][:3, :3]) - 1) / 2, -1, 1))))}
    scene.release()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
