"""Dashboard videos of how the desired image is found (../webpage/videos/goal_<scene>_<query>.mp4), 2 x 3 panels,
after Mustapha's PCA visual servoing dashboard.

From a run of ours (goal.npz: object cloud, centroid, PCA normal, selected viewing direction v, goal pose, d*; and
servo_poses.npy: the analytic 6-feature servo of text2goal/servo.py from the start training camera to the goal pose):
  row 1: current render (object centroid +, image centre x) | 3-D: object cloud, PCA normal, selected direction v,
         camera trajectory and optical axis, goal camera | desired image (render at the goal pose)
  row 2: centring error (px) | orientation error (deg) and distance error (% of d*) | camera velocity per iteration
         (linear in d*, solid; angular in rad, dashed)

  ~/anaconda3/envs/pca_servo/bin/python scripts/goal_dashboard.py        (from my_code/)
"""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from goal_benchmark import L
from make_videos import even, log_se3
from text2goal.geometry import project
from text2goal.render import GaussianScene
from text2goal.servo import features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, VID = ROOT / "out", ROOT.parent / "webpage" / "videos"
D = "/media/youssef/LaCie/data/mip-nerf360/360_v2"
MIP = lambda s: f"{D}/{s}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"  # noqa: E731
RUNS = [("multiscene", "room", "piano", MIP("room")), ("multiscene", "counter", "egg_carton", MIP("counter")),
        ("multiscene", "kitchen", "toy_bulldozer", MIP("kitchen")), ("multiscene", "garden", "vase", MIP("garden")),
        ("multiscene", "room", "door", MIP("room")), ("multiscene", "bonsai", "cardboard_box", MIP("bonsai"))]
PW = 480


def frustum(T, f, aspect):
    """Wire-frame segments of a camera pyramid (OpenCV axes: x right, y down, z forward), depth f, in world."""
    w, h = 0.5 * f, 0.5 * f / aspect
    c = T[:3, 3]
    corners = [T[:3, :3] @ np.array([sx * w, sy * h, f]) + c for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    segs = [(c, q) for q in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    segs.append((corners[0] + 0.25 * (corners[1] - corners[0]), 0.5 * (corners[0] + corners[1])
                 - 0.3 * (corners[3] - corners[0]) * 0.5))                      # small "up" tick on the top edge
    return segs


def dashboard(scene, run, name, query):
    g, sm = np.load(run / "goal.npz"), json.loads((run / "summary.json").read_text())
    poses = np.load(run / "servo_poses.npy")
    p, n, v, d_star = g["p_obj"], g["n_obj"], g["v_world"], float(g["d_star"])
    up, T_goal = g["up_world"], g["T_goal"]
    W, H = int(g["W"]), int(g["H"])
    s = PW / W
    K = g["K"].copy()
    K[:2] *= s
    w_, h_ = PW, round(H * s)
    err = [features(T, p, v, d_star, up, g["K"])[2] for T in poses]
    cen = np.array([e["center_px"] for e in err])
    ori = np.array([e["orientation_deg"] for e in err])
    dis = np.array([100 * abs(e["distance"] - d_star) / d_star for e in err])
    conv = np.flatnonzero((cen < 0.5) & (ori < 0.05) & (dis < 0.1))
    N = min(len(poses), (int(conv[0]) if len(conv) else len(poses)) + 8)      # stop shortly after convergence
    poses, cen, ori, dis = poses[:N], cen[:N], ori[:N], dis[:N]
    vel = np.array([log_se3(np.linalg.inv(poses[i]) @ poses[i + 1]) for i in range(len(poses) - 1)] + [np.zeros(6)])
    vel[:, :3] /= d_star
    desired = cv2.resize(np.asarray(Image.open(run / "desired_image.png").convert("RGB")), (w_, h_),
                         interpolation=cv2.INTER_AREA)
    pts = g["points"][np.random.default_rng(0).choice(len(g["points"]), min(2500, len(g["points"])), replace=False)]
    # texture: colour of each point in the desired image where it is visible (depth test), grey elsewhere
    full = np.asarray(Image.open(run / "desired_image.png").convert("RGB"))
    depth = np.load(run / "desired_depth.npy")
    uu, vv_, zz = project(pts, T_goal, g["K"])
    ui, vi = np.round(uu).astype(int), np.round(vv_).astype(int)
    ok = (zz > 0) & (ui >= 0) & (ui < full.shape[1]) & (vi >= 0) & (vi < full.shape[0])
    ok[ok] &= depth[vi[ok], ui[ok]] >= zz[ok] * 0.97 - 1e-3
    rgb_pts = np.full((len(pts), 3), 0.6)
    rgb_pts[ok] = full[vi[ok], ui[ok]] / 255.0
    # display frame: world up -> z, selected direction v in the x-z plane, centred on the object
    ez = up / np.linalg.norm(up)
    ex = v - (v @ ez) * ez
    ex /= np.linalg.norm(ex)
    Rd = np.stack([ex, np.cross(ez, ex), ez])
    to_d = lambda X: (np.atleast_2d(X) - p) @ Rd.T  # noqa: E731
    C = to_d(poses[:, :3, 3])

    fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1, 1.25, 1], hspace=0.3, wspace=0.22, left=0.05, right=0.97, top=0.89,
                          bottom=0.07)
    title = fig.suptitle("", fontsize=12)
    a_cur = fig.add_subplot(gs[0, 0])
    im_cur = a_cur.imshow(desired)
    a_cur.set_title("Current render: object centre (+) and image centre (×)", fontsize=9)
    a_cur.axis("off")
    mark, = a_cur.plot([], [], "+", color="#ff3030", ms=14, mew=2.5)
    a_cur.plot(K[0, 2], K[1, 2], "x", color="white", ms=10, mew=2)
    a3 = fig.add_subplot(gs[0, 1], projection="3d")
    a3.scatter(*to_d(pts).T, s=14, c=rgb_pts, depthshade=False, label="object cloud (textured)")
    L = 0.6 * d_star
    a3.plot(*to_d(np.stack([p, p + L * n])).T, "--", color="#d03b3b", lw=1.5, label="PCA normal")
    a3.plot(*to_d(np.stack([p, p + d_star * v])).T, color="#d03b3b", lw=2.2, label="selected direction v")
    a3.plot(*C.T, color="#2a78d6", lw=1, alpha=0.25)
    traj, = a3.plot([], [], [], color="#2a78d6", lw=1.8, label="camera centre trajectory")
    f = 0.3 * d_star
    fd = lambda T: [tuple(map(tuple, to_d(np.array(sg)))) for sg in frustum(T, f, W / H)]  # noqa: E731
    a3.add_collection3d(Line3DCollection(fd(poses[0]), colors="#1f7a3a", linewidths=1.0, label="start camera"))
    a3.add_collection3d(Line3DCollection(fd(T_goal), colors="#d03b3b", linewidths=1.0, linestyles="--",
                                         label="goal camera"))
    ghosts = Line3DCollection(fd(poses[0]), colors="#2a78d6", linewidths=0.6, alpha=0.3)
    a3.add_collection3d(ghosts, autolim=False)
    cam = Line3DCollection(fd(poses[0]), colors="#1a4f9c", linewidths=1.8, label="current camera")
    a3.add_collection3d(cam, autolim=False)
    axis, = a3.plot([], [], [], color="#00c8d0", lw=2.2, label="optical axis")
    centre, = a3.plot([], [], [], "o", color="#ff9f1c", ms=5, label="camera centre C")
    a3.view_init(elev=18, azim=-70)
    allp = np.vstack([to_d(pts), C, to_d(T_goal[:3, 3])] + [np.array(sg[1]) for sg in fd(poses[0]) + fd(T_goal)])
    c0, r = allp.mean(0), 0.5 * np.ptp(allp, 0).max()
    a3.set(xlim=(c0[0] - r, c0[0] + r), ylim=(c0[1] - r, c0[1] + r), zlim=(c0[2] - r, c0[2] + r))
    a3.set_box_aspect((1, 1, 1))
    a3.tick_params(labelsize=6)
    a3.set(xlabel="along v (horizontal)", ylabel="", zlabel="up")
    a3.xaxis.label.set_size(7)
    a3.zaxis.label.set_size(7)
    a3.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(-0.42, 0.5), frameon=False)
    a3.set_title("3-D (up vertical): textured object cloud, PCA normal, selected direction", fontsize=9)
    a_des = fig.add_subplot(gs[0, 2])
    a_des.imshow(desired)
    a_des.set_title(f"Desired image (selected: {sm['chosen']}, d* = {d_star:.2f})", fontsize=9)
    a_des.axis("off")
    plots = []
    for c, (t, ys, cols, labels, yl) in enumerate([
            ("Centring error", [cen], ["#2a78d6"], ["centre (px)"], "pixels"),
            ("Orientation and distance error", [ori, dis], ["#eb6834", "#6250d6"], ["orientation (deg)",
                                                                                     "distance (% of d*)"], ""),
            ("Camera velocity (per iteration)", [vel[:, i] for i in range(6)],
             ["#2a78d6", "#eb6834", "#1baf7a"] * 2,
             ["$v_x$", "$v_y$", "$v_z$", "$\\omega_x$", "$\\omega_y$", "$\\omega_z$"], "d* / it, rad / it")]):
        a = fig.add_subplot(gs[1, c])
        lines = []
        for k, (y, col, lab) in enumerate(zip(ys, cols, labels)):
            a.plot(range(N), y, color=col, lw=1, alpha=0.2)
            lines.append((a.plot([], [], color=col, lw=1.6, ls="--" if c == 2 and k >= 3 else "-", label=lab)[0], y))
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
        a.set(title=t, xlabel="iteration", ylabel=yl, xlim=(0, N - 1), ylim=(lo - 0.05 * (hi - lo + 1e-9),
                                                                            hi + 0.05 * (hi - lo + 1e-9)))
        a.title.set_fontsize(10)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        a.legend(fontsize=7, ncol=2 if c == 2 else 1)
        plots += lines
    frames = []
    for i in range(N):
        T = poses[i]
        rgb = scene.render(T, K, w_, h_)[0]
        im_cur.set_data(rgb)
        u, vv, z = project(p[None], T, K)
        mark.set_data([u[0]] if z[0] > 0 else [], [vv[0]] if z[0] > 0 else [])
        traj.set_data_3d(*C[: i + 1].T)
        cam.set_segments(fd(T))
        ghosts.set_segments([sg for j in range(0, i, 3) for sg in fd(poses[j])])
        ax_end = T[:3, 3] + np.linalg.norm(p - T[:3, 3]) * T[:3, 2]          # optical axis up to the object depth
        axis.set_data_3d(*to_d(np.stack([T[:3, 3], ax_end])).T)
        centre.set_data_3d(*to_d(T[:3, 3]).T)
        for line, y in plots:
            line.set_data(np.arange(i + 1), y[: i + 1])
        title.set_text(f"“{query}” — reaching the selected viewpoint — iteration {i}: centre "
                       f"{cen[i]:.1f} px, orientation {ori[i]:.2f}°, distance error {dis[i]:.2f} %")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    frames += [frames[-1]] * 15
    imageio.mimsave(VID / f"goal_{name}.mp4", [even(f) for f in frames], fps=5, quality=7, macro_block_size=1)
    print(f"goal_{name}: {N} poses, final centre {cen[-1]:.2f} px, orientation {ori[-1]:.3f} deg", flush=True)


def main():
    VID.mkdir(parents=True, exist_ok=True)
    for root, s, q, model in RUNS:
        if (VID / f"goal_{s}_{q}.mp4").exists():
            continue
        scene = GaussianScene(model)
        dashboard(scene, OUT / root / s / q, f"{s}_{q}", q.replace("_", " "))
        scene.release()


if __name__ == "__main__":
    main()
