"""Videos for the project page (../webpage/videos/), from runs of ours.

  servo_<scene>_<query>.mp4       dashboard (2 x 3, as Mustapha's photometric VS dashboard) of the photometric servo
                                  from a large perturbation (0.2 d*, 10 deg, seed 0, current view supersampled 4x):
                                  current | desired | current - desired, then photometric error | camera velocity
                                  (per-iteration twist, linear solid / angular dashed) | 3-D camera trajectory
  candidates_<scene>_<query>.mp4  the viewpoint selection: the candidates covered by training photos (cov >= 0.25;
                                  the others render unreliable space and are only counted), lowest score first,
                                  with vis, cov, fit and S, then the selected view (stage 1 is not re-run: object from the run's goal.npz)

  ~/anaconda3/envs/pca_servo/bin/python scripts/make_videos.py        (from my_code/)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goal_benchmark import GS_VS, L
from servo_eval import perturbation
from text2goal.cameras import load_colmap_views
from text2goal.geometry import up_from_cameras
from text2goal.photometric import gray, photometric_servo
from text2goal.render import GaussianScene
from text2goal.viewpoint import ViewParams, select_view

ROOT = Path(__file__).resolve().parents[1]
OUT, VID = ROOT / "out", ROOT.parent / "webpage" / "videos"
D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
PW = 320                                       # panel width
MIP = lambda s: (f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt",  # noqa: E731
                 OUT / "multiscene/cfg" / f"{s}.yml", f"{D}/{s}/images", OUT / "multiscene" / s)
LERF = lambda s: (f"{L}/{s}/output_gsplat_mcmc_df1/ckpts/ckpt_29999_rank0.pt",  # noqa: E731
                  f"{L}/{s}/output_gsplat_mcmc_df1/cfg.yml", f"{L}/{s}/images", OUT / "lerf_ovs" / s)
SERVO = [("room", "piano", MIP), ("counter", "egg_carton", MIP), ("kitchen", "toy_bulldozer", MIP),
         ("garden", "vase", MIP), ("ramen", "kamaboko", LERF), ("figurines", "green_toy_chair", LERF)]
CANDIDATES = [("room", "piano", MIP), ("ramen", "kamaboko", LERF)]


def label(img, lines, color=(255, 255, 255)):
    img = img.copy()
    for i, t in enumerate(lines):
        y = 18 + 18 * i
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def even(img):
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def log_se3(T):
    """Twist (v, w) with exp_se3((v, w)) = T."""
    R, t = T[:3, :3], T[:3, 3]
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-8:
        return np.r_[t, 0, 0, 0]
    Wx = th / (2 * np.sin(th)) * (R - R.T)
    w = np.array([Wx[2, 1], Wx[0, 2], Wx[1, 0]])
    Vinv = np.eye(3) - 0.5 * Wx + (1 - th * np.sin(th) / (2 * (1 - np.cos(th)))) / th ** 2 * (Wx @ Wx)
    return np.r_[Vinv @ t, w]


def servo_video(scene, run, name, query, k=4, alpha=0.2, beta=10.0):
    g = np.load(run / "goal.npz")
    T_goal, d_star, W, H = g["T_goal"], float(g["d_star"]), int(g["W"]), int(g["H"])
    s = PW / W
    w_, h_ = PW, round(H * s)
    K = g["K"].copy()
    K[:2] *= s
    Kk = K.copy()
    Kk[:2, :2] *= k
    Kk[:2, 2] = (K[:2, 2] + 0.5) * k - 0.5
    desired = cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                         interpolation=cv2.INTER_AREA)
    I_star = gray(desired)

    def render(T):
        rgb, Z = scene.render(T, Kk, w_ * k, h_ * k)
        return cv2.resize(rgb, (w_, h_), interpolation=cv2.INTER_AREA), np.ascontiguousarray(Z[k // 2::k, k // 2::k])

    T0 = T_goal @ perturbation(np.random.default_rng(1000 * 0 + int(100 * alpha) + int(beta)), alpha, beta, d_star)
    poses, errs = photometric_servo(render, T0, I_star, K, iterations=600)
    n = len(poses)
    vel = np.array([log_se3(np.linalg.inv(poses[i]) @ poses[i + 1]) for i in range(n - 1)] + [np.zeros(6)])
    vel[:, :3] /= d_star                                       # translation in units of d*
    errs = list(errs) + [errs[-1]] * (n - len(errs))
    pos = (poses[:, :3, 3] - T_goal[:3, 3]) / d_star          # camera centre relative to the goal, in d*

    fig = plt.figure(figsize=(12, 6.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28, left=0.06, right=0.97, top=0.88,
                          bottom=0.08)
    title = fig.suptitle("", fontsize=12)
    ims = []
    for c, (t, img) in enumerate([("Current image", desired), ("Desired image", desired),
                                  ("Current \u2212 desired", desired)]):
        a = fig.add_subplot(gs[0, c])
        ims.append(a.imshow(img, cmap="gray", vmin=-255 if c == 2 else 0, vmax=255))
        a.set_title(t, fontsize=10)
        a.axis("off")
    ae = fig.add_subplot(gs[1, 0])
    ae.semilogy(range(n), errs, color="#d03b3b", lw=1.2, alpha=0.25)
    le, = ae.semilogy([], [], color="#d03b3b", lw=1.6)
    ae.set(title="Photometric error", xlabel="iteration", ylabel="mean squared grey-level error", xlim=(0, n))
    av = fig.add_subplot(gs[1, 1])
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    lv = [av.plot([], [], color=cols[i], lw=1.3, label=f"$v_{'xyz'[i]}$")[0] for i in range(3)]
    lw = [av.plot([], [], color=cols[i], lw=1.3, ls="--", label=f"$\\omega_{'xyz'[i]}$")[0] for i in range(3)]
    lim = 1.1 * max(np.abs(vel).max(), 1e-6)
    av.set(title="Camera velocity (per iteration)", xlabel="iteration", ylabel="$d^\\star$ / it, rad / it",
           xlim=(0, n), ylim=(-lim, lim))
    av.legend(ncol=2, fontsize=7, loc="upper right")
    at = fig.add_subplot(gs[1, 2], projection="3d")
    at.plot(*pos.T, color="#2a78d6", lw=1, alpha=0.25)
    lt, = at.plot([], [], [], color="#2a78d6", lw=1.6, label="trajectory")
    at.scatter(*pos[0], color="#1f7a3a", s=25, label="start")
    pc = at.scatter(*pos[0], color="#2a78d6", marker="^", s=30, label="current")
    at.scatter(0, 0, 0, color="#d03b3b", marker="*", s=80, label="goal")
    at.set_title("Camera trajectory (in $d^\\star$, goal at 0)", fontsize=10)
    at.legend(fontsize=7, loc="upper left")
    at.tick_params(labelsize=6)
    for a in (ae, av):
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        a.title.set_fontsize(10)
    frames = []
    for i in range(n):
        cur = render(poses[i])[0]
        ims[0].set_data(cur)
        ims[2].set_data(gray(cur).astype(np.float32) - I_star.astype(np.float32))
        x = np.arange(i + 1)
        le.set_data(x, errs[: i + 1])
        for j in range(3):
            lv[j].set_data(x, vel[: i + 1, j])
            lw[j].set_data(x, vel[: i + 1, 3 + j])
        lt.set_data_3d(*pos[: i + 1].T)
        pc._offsets3d = tuple(pos[i : i + 1].T)
        t_err = np.linalg.norm(pos[i])
        title.set_text(f"\u201c{query}\u201d \u2014 photometric visual servoing toward the generated desired image \u2014 "
                       f"iteration {i}, distance to goal {t_err:.3f} $d^\\star$")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    frames += [frames[-1]] * 50                      # hold the last frame
    imageio.mimsave(VID / f"servo_{name}.mp4", [even(f) for f in frames], fps=25, quality=7, macro_block_size=1)
    print(f"servo_{name}: {n} iterations, final distance {np.linalg.norm(pos[-1]):.4f} d*", flush=True)


def candidates_video(scene, run, views, train_c2w, up, name):
    g, sm = np.load(run / "goal.npz"), json.loads((run / "summary.json").read_text())
    obj = SimpleNamespace(centroid=g["p_obj"], normal=g["n_obj"], points=g["points"], rho=sm["rho"],
                          view=views[Path(sm["reference_view"]).stem])
    goal = select_view(obj, scene, up, train_c2w, ViewParams())
    rows, chosen = goal["candidates"], goal["chosen"]["name"]
    # candidates far from every training photo render empty or blurry space: the coverage term rejects them, the
    # video says how many and shows only the covered ones (cov >= 0.25), lowest score first, then the selected view
    shown = sorted([r for r in rows if r["support"] >= 0.25 and r is not goal["chosen"]], key=lambda r: r["score"])
    best = max(r["score"] for r in rows) or 1.0
    h = round(rows[0]["render"].shape[0] * 2 * PW / rows[0]["render"].shape[1])
    card = np.full((h, 2 * PW, 3), 30, np.uint8)
    card = label(card, [f"query '{name.split('_', 1)[1].replace('_', ' ')}': {len(rows)} candidate views",
                        f"{len(rows) - len(shown) - 1} rejected: no training photo nearby (cov < 0.25),",
                        "the 3DGS render is unreliable there",
                        f"showing the {len(shown)} others, lowest score first, then the selected view"])
    frames = [card] * 75
    for r in shown + [goal["chosen"]]:
        img = cv2.resize(r["render"], (2 * PW, h))
        last = r is goal["chosen"]
        img = label(img, [("SELECTED: " if last else "candidate: ") + r["name"],
                          f"vis {r['visibility']:.2f}  cov {r['support']:.2f}  fit {r['fit']:.2f}",
                          f"S = {r['score']:.3f}"], (90, 255, 120) if last else (255, 255, 255))
        bar = int((img.shape[1] - 16) * r["score"] / best)
        cv2.rectangle(img, (8, img.shape[0] - 14), (8 + bar, img.shape[0] - 6), (90, 255, 120) if last
                      else (240, 200, 60), -1)
        frames += [img] * (75 if last else 10)
    imageio.mimsave(VID / f"candidates_{name}.mp4", [even(f) for f in frames], fps=25, quality=7, macro_block_size=1)
    print(f"candidates_{name}: {len(rows)} candidates, chosen {chosen}", flush=True)


def main():
    VID.mkdir(parents=True, exist_ok=True)
    for s, q, src in SERVO + CANDIDATES:
        model, cfg, images, runs = src(s)
        scene = GaussianScene(model)
        if (s, q, src) in SERVO and not (VID / f"servo_{s}_{q}.mp4").exists():
            servo_video(scene, runs / q, f"{s}_{q}", q.replace("_", " "))
        if (s, q, src) in CANDIDATES and not (VID / f"candidates_{s}_{q}.mp4").exists():
            views = {Path(v.name).stem: v for v in load_colmap_views(str(cfg), images, GS_VS)}
            train_c2w = np.stack([v.T_c2w for v in views.values()])
            candidates_video(scene, runs / q, views, train_c2w, up_from_cameras(train_c2w), f"{s}_{q}")
        scene.release()


if __name__ == "__main__":
    main()
