"""Image-based servo test of the generated desired images (the real test; servo.py converges by construction).

For every run directory (goal.npz + desired_image.png written by text_to_desired_image.py), start from the goal pose
perturbed in the camera frame (translation alpha * d* in a random direction, rotation beta about a random axis) and
servo photometrically (text2goal/photometric.py) toward desired_image.png. The current views are rendered from
--model (same model that produced the desired image) or, with --render_ply, from another 3DGS model of the scene
trained in the raw COLMAP frame (e.g. LangSplat-V2's), to mimic a camera that does not see our render.
Success: final position error < 2 % of d* and rotation error < 1 deg.

  ~/anaconda3/envs/pca_servo/bin/python scripts/servo_eval.py --runs 'out/multiscene/room/*' \
      --model <ckpt.pt> --csv out/servo_eval/room.csv
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text2goal.cameras import colmap_to_model_transform  # noqa: E402
from text2goal.photometric import gray, photometric_servo  # noqa: E402
from text2goal.render import GaussianScene, TransformedScene  # noqa: E402
from text2goal.servo import exp_se3  # noqa: E402


def perturbation(rng, alpha, beta_deg, d_star):
    t = rng.normal(size=3)
    w = rng.normal(size=3)
    return exp_se3(np.r_[0, 0, 0, np.radians(beta_deg) * w / np.linalg.norm(w)]) @ \
        exp_se3(np.r_[alpha * d_star * t / np.linalg.norm(t), 0, 0, 0])


def pose_error(T, T_goal):
    D = np.linalg.inv(T_goal) @ T
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, nargs="+", help="run directories or globs")
    ap.add_argument("--model", required=True, help="3DGS model of the desired images (our frame)")
    ap.add_argument("--render_ply", default=None, help="other model, raw COLMAP frame (needs --cfg)")
    ap.add_argument("--render_model", default=None, help="other model already in our frame")
    ap.add_argument("--cfg", default=None, help="gsplat cfg.yml of --model (COLMAP -> our frame)")
    ap.add_argument("--gs_vs_dir", default="/home/youssef/Documents/SW/visual_navigation/vanilla_gs_vs")
    ap.add_argument("--width", type=int, default=320, help="servo image width")
    ap.add_argument("--levels", default="0.1,5;0.2,10", help="perturbations 'alpha,beta_deg;...'")
    ap.add_argument("--supersample", type=int, default=1, help="render the current view k times larger and area-"
                    "average it down (anti-aliasing, as a real camera; 3DGS trained on large photos aliases at 320 px)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--save_traj", action="store_true", help="save the final image next to the desired one")
    args = ap.parse_args()

    runs = sorted({Path(r) for pat in args.runs for r in glob.glob(pat) if (Path(r) / "goal.npz").exists()})
    if args.render_ply:
        scene = TransformedScene(args.render_ply, colmap_to_model_transform(args.cfg, args.gs_vs_dir))
        source = "cross:" + Path(args.render_ply).parent.parent.parent.name
    elif args.render_model:
        scene, source = GaussianScene(args.render_model), "cross:" + Path(args.render_model).name
    else:
        scene, source = GaussianScene(args.model), "same"
    levels = [tuple(map(float, l.split(","))) for l in args.levels.split(";")]
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["run"], r["alpha"], r["beta"], r["seed"]) for r in csv.DictReader(open(out))}
    f = open(out, "a", newline="")
    fields = ["run", "source", "alpha", "beta", "seed", "d_star", "t_err0", "r_err0", "t_err", "r_err",
              "t_err_rel", "mse0", "mse", "iterations", "success"]
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()

    for run in runs:
        g = np.load(run / "goal.npz")
        T_goal, d_star, W, H = g["T_goal"], float(g["d_star"]), int(g["W"]), int(g["H"])
        s = args.width / W
        w_, h_ = args.width, round(H * s)
        K = g["K"].copy()
        K[:2] *= s
        I_star = gray(cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                                 interpolation=cv2.INTER_AREA))
        k = args.supersample
        Kk = K.copy()
        Kk[:2, :2] *= k
        Kk[:2, 2] = (K[:2, 2] + 0.5) * k - 0.5                 # same pixel grid after k x k averaging

        def render(T):
            if k == 1:
                return scene.render(T, K, w_, h_)
            rgb, Z = scene.render(T, Kk, w_ * k, h_ * k)
            return (cv2.resize(rgb, (w_, h_), interpolation=cv2.INTER_AREA),
                    np.ascontiguousarray(Z[k // 2::k, k // 2::k][:h_, :w_]))
        for alpha, beta in levels:
            for seed in range(args.seeds):
                if (str(run), str(alpha), str(beta), str(seed)) in done:
                    continue
                rng = np.random.default_rng(1000 * seed + int(100 * alpha) + int(beta))
                T0 = T_goal @ perturbation(rng, alpha, beta, d_star)
                poses, errs = photometric_servo(render, T0, I_star, K, iterations=args.iterations)
                t0, r0 = pose_error(T0, T_goal)
                t1, r1 = pose_error(poses[-1], T_goal)
                row = {"run": str(run), "source": source, "alpha": alpha, "beta": beta, "seed": seed,
                       "d_star": d_star, "t_err0": t0, "r_err0": r0, "t_err": t1, "r_err": r1,
                       "t_err_rel": t1 / d_star, "mse0": errs[0], "mse": errs[-1], "iterations": len(errs),
                       "success": int(t1 / d_star < 0.02 and r1 < 1.0)}
                w.writerow(row)
                f.flush()
                print(f"{run.name:28s} a={alpha} b={beta} s={seed}: t {t0 / d_star:.3f}->{t1 / d_star:.4f} d*, "
                      f"r {r0:.2f}->{r1:.3f} deg, mse {errs[0]:.0f}->{errs[-1]:.1f}, {len(errs)} it "
                      f"{'OK' if row['success'] else 'FAIL'}", flush=True)
                if args.save_traj and seed == 0:
                    rgb, _ = render(poses[-1])
                    d = cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                                   interpolation=cv2.INTER_AREA)
                    Image.fromarray(np.hstack([d, render(T0)[0], rgb])).save(
                        out.parent / f"{run.parent.name}_{run.name}_{source.replace(':', '_')}_a{alpha}.png")
    f.close()


if __name__ == "__main__":
    main()
