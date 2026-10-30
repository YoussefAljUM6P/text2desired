"""Navigation dashboards of the ground robot in InteriorGS houses (../webpage/videos/igs_<house>_<query>.mp4), 2 x 3.

For a query, the robot starts at the survey view in which it detected the object (scripts/interiorgs_run.py), the
goal is our selected viewpoint (text2goal/robot_view.py), and the robot (camera at 1.25 m, level; forward speed and
yaw rate) reaches it in two phases:
  navigation  shortest path on the navigable 10 cm floor grid (A*, cost raised near walls), followed by a unicycle
              path follower until the robot is within NAV_STOP of the goal floor point, then a turn in place toward
              the goal heading (within NAV_YAW)
  servo       the 3-DOF photometric visual servo (text2goal/photometric.py, dof = vx, vz, wy) toward the desired image
Panels: current view (object centre +, image centre x) | top view: navigable floor, object cloud, the 24 floor
candidates coloured by score, chosen goal, planned path, robot trajectory and camera | desired image;
distance and heading to the goal | photometric error | robot velocities.

  PYTHONPATH=scripts:. ~/anaconda3/envs/pca_servo/bin/python scripts/interiorgs_dashboard.py        (from my_code/)
"""
import heapq
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from make_videos import even, log_se3
from text2goal.cameras import View
from text2goal.geometry import project
from text2goal.localize import localize
from text2goal.perception import Detector, Segmenter
from text2goal.photometric import gray, photometric_servo
from text2goal.render import GaussianScene
from text2goal.robot_view import robot_pose, select_robot_view

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, VID = ROOT / "out/interiorgs", ROOT.parent / "webpage" / "videos"
VAL = "/media/youssef/DATA/gs_imagenav/data/gs_scenes/val"
SAM = "/home/youssef/Documents/SW/checkpoints/sam_vit_h_4b8939.pth"
UP = np.array([0.0, 1.0, 0.0])
QUERIES = [("0637", "dining table"), ("0695", "sofa"), ("0651", "tv"), ("0637", "bed")]
V_MAX, W_MAX, LOOKAHEAD = 0.06, np.radians(6.0), 0.5     # m / step, rad / step, m
NAV_STOP, NAV_YAW = 0.15, np.radians(8.0)                # hand-over to the servo
K_V, K_W = 0.1, 0.25                                     # proportional gains (1 / step)
A_V, A_W = 0.005, np.radians(1.0)                        # acceleration limits (m / step², rad / step²)
PW, SW = 480, 320                                        # display and servo widths


def astar(nav, grid, start, goal):
    """Shortest 8-connected path on the navigable grid (x, z), cost raised within 0.4 m of non-navigable cells."""
    ij = np.round(nav[:, [0, 2]] / grid).astype(int)
    lo = ij.min(0) - 2
    occ = np.zeros(tuple(ij.max(0) - lo + 3), bool)
    occ[tuple((ij - lo).T)] = True
    clear = distance_transform_edt(occ) * grid
    cell = lambda p: tuple(np.round(p[[0, 2]] / grid).astype(int) - lo)  # noqa: E731
    near = lambda p: tuple(ij[np.argmin(np.linalg.norm(nav[:, [0, 2]] - p[[0, 2]], axis=1))] - lo)  # noqa: E731
    s, g = near(start), near(goal)
    dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    while pq:
        _, u = heapq.heappop(pq)
        if u == g:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                v = (u[0] + di, u[1] + dj)
                if (di or dj) and 0 <= v[0] < occ.shape[0] and 0 <= v[1] < occ.shape[1] and occ[v]:
                    c = dist[u] + grid * np.hypot(di, dj) * (1 + 2.0 * max(0.0, 0.4 - clear[v]) / 0.4)
                    if c < dist.get(v, np.inf):
                        dist[v], prev[v] = c, u
                        heapq.heappush(pq, (c + grid * np.hypot(v[0] - g[0], v[1] - g[1]), v))
    path, u = [g], g
    while u != s:
        u = prev[u]
        path.append(u)
    xz = (np.array(path[::-1]) + lo) * grid
    return np.column_stack([xz[:, 0], np.full(len(xz), goal[1]), xz[:, 1]])


def pose_xz(x, z, yaw, height, floor_y):
    """Level camera at (x, floor_y + height, z) looking along yaw (forward = (cos, 0, sin))."""
    f = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    c = np.array([x, floor_y, z])
    return robot_pose(c, c + height * UP + f, height, UP)


def yaw_of(T):
    return float(np.arctan2(T[2, 2], T[0, 2]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def drive(path, T_start, T_goal, height):
    """Unicycle pure pursuit then a turn in place; returns the list of camera poses.
    Proportional laws with saturation and acceleration limits, and a lookahead point interpolated along the path,
    so that the velocities are continuous."""
    floor_y = T_start[1, 3] - height
    x, z, yaw = T_start[0, 3], T_start[2, 3], yaw_of(T_start)
    gx, gz, gyaw = T_goal[0, 3], T_goal[2, 3], yaw_of(T_goal)
    P = path[:, [0, 2]]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    on_path = lambda t: np.array([np.interp(t, arc, P[:, 0]), np.interp(t, arc, P[:, 1])])  # noqa: E731
    poses, s_k, v, w = [pose_xz(x, z, yaw, height, floor_y)], 0.0, 0.0, 0.0

    def step(v_ref, w_ref):
        nonlocal x, z, yaw, v, w
        v += float(np.clip(v_ref - v, -A_V, A_V))
        w += float(np.clip(w_ref - w, -A_W, A_W))
        yaw += w
        x, z = x + v * np.cos(yaw), z + v * np.sin(yaw)
        poses.append(pose_xz(x, z, yaw, height, floor_y))

    for _ in range(3000):
        dist = np.hypot(gx - x, gz - z)
        if dist < NAV_STOP and v < 1e-3:
            break
        ts = np.linspace(s_k, min(s_k + 2 * LOOKAHEAD, arc[-1]), 50)     # closest path point, never backwards
        s_k = ts[np.argmin(np.linalg.norm([on_path(t) - (x, z) for t in ts], axis=1))]
        tx, tz = on_path(min(s_k + LOOKAHEAD, arc[-1]))
        err = wrap(np.arctan2(tz - z, tx - x) - yaw)
        w_ref = float(np.clip(K_W * err, -W_MAX, W_MAX))
        v_ref = 0.0 if dist < NAV_STOP else min(V_MAX, K_V * (dist - NAV_STOP) + 0.01) * max(0.0, np.cos(err))
        step(v_ref, w_ref)
    for _ in range(300):
        err = wrap(gyaw - yaw)
        if abs(err) < NAV_YAW and abs(w) < A_W:
            break
        step(0.0, float(np.clip(K_W * err, -W_MAX, W_MAX)))
    return poses


def dashboard(sc, query, scene, views, sv, segmenter, name):
    K, W, H = sv["K"], int(sv["W"]), int(sv["H"])
    height, grid, nav = float(sv["camera_height"]), float(sv["grid"]), sv["navigable"]
    train_c2w = np.stack([v.T_c2w for v in views])
    obj = localize(query, views, scene, Detector(query, "IDEA-Research/grounding-dino-tiny", 0.2), segmenter)
    goal = select_robot_view(obj, scene, UP, train_c2w, nav, height, grid)
    T_goal, d_star, p = goal["T_goal"], float(goal["d_star"]), obj.centroid
    T_start = obj.view.T_c2w
    path = astar(nav, grid, T_start[:3, 3] - height * UP, T_goal[:3, 3] - height * UP)
    nav_poses = drive(path, T_start, T_goal, height)
    # servo at SW px toward the desired image (downsampled as in the servo test)
    s = SW / W
    Ks = K.copy()
    Ks[:2] *= s
    ws, hs = SW, round(H * s)
    I_star = gray(cv2.resize(goal["desired_rgb"], (ws, hs), interpolation=cv2.INTER_AREA))
    sv_poses, sv_err = photometric_servo(lambda T: scene.render(T, Ks, ws, hs), nav_poses[-1], I_star, Ks,
                                         iterations=600, dof=(0, 2, 4))
    sv_poses = list(sv_poses)[1:]
    conv = next((i for i, e in enumerate(sv_err) if e < 5.0), len(sv_err) - 1)
    stride = max(1, len(sv_poses) // 150)
    keep = list(range(0, min(len(sv_poses), conv + 8 * stride), stride))
    poses = nav_poses + [sv_poses[i] for i in keep]
    phase = np.array([0] * len(nav_poses) + [1] * len(keep))
    N = len(poses)
    # per-frame errors and velocities (camera frame: v_x lateral, v_z forward, w_y yaw)
    gfloor = T_goal[[0, 2], 3]
    dist = np.array([np.hypot(*(T[[0, 2], 3] - gfloor)) for T in poses])
    head = np.array([abs(np.degrees(wrap(yaw_of(T) - yaw_of(T_goal)))) for T in poses])
    mse = np.array([float(np.mean((gray(scene.render(T, Ks, ws, hs)[0]) - I_star) ** 2)) for T in poses])
    vel = np.array([log_se3(np.linalg.inv(poses[i]) @ poses[i + 1]) for i in range(N - 1)] + [np.zeros(6)])
    t1, r1 = dist[-1], head[-1]
    print(f"{name}: path {len(path)} cells, {len(nav_poses)} nav steps, servo {len(sv_err)} it, "
          f"final {100 * t1 / d_star:.2f}% d*, {r1:.2f} deg", flush=True)

    sd = PW / W
    Kd = K.copy()
    Kd[:2] *= sd
    wd, hd = PW, round(H * sd)
    desired = cv2.resize(goal["desired_rgb"], (wd, hd), interpolation=cv2.INTER_AREA)
    fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1, 1.1, 1], hspace=0.42, wspace=0.18,
                          left=0.04, right=0.98, top=0.89, bottom=0.07)
    title = fig.suptitle("", fontsize=12)
    a_cur = fig.add_subplot(gs[0, 0])
    im_cur = a_cur.imshow(desired)
    a_cur.set_title("Current view: object centre (+) and image centre (×)", fontsize=9)
    a_cur.axis("off")
    mark, = a_cur.plot([], [], "+", color="#ff3030", ms=14, mew=2.5)
    a_cur.plot(Kd[0, 2], Kd[1, 2], "x", color="white", ms=10, mew=2)
    # top view (x right, z down as in a floor plan)
    a_map = fig.add_subplot(gs[0, 1])
    C = np.array([T[[0, 2], 3] for T in poses])
    cands = np.array([r["T_goal"][[0, 2], 3] for r in goal["candidates"]])
    sc_c = np.array([r["score"] for r in goal["candidates"]])
    box = np.vstack([C, cands, p[[0, 2]][None], path[:, [0, 2]]])
    c0, r = box.mean(0), 0.5 * np.ptp(box, 0).max() + 0.8
    inb = np.all(np.abs(nav[:, [0, 2]] - c0) < r, axis=1)
    a_map.scatter(nav[inb, 0], nav[inb, 2], s=2, c="#d9d9d9", marker="s", linewidths=0, label="navigable floor")
    pts = obj.points[np.random.default_rng(0).choice(len(obj.points), min(2500, len(obj.points)), replace=False)]
    cols = obj.colors if obj.colors.max() <= 1 else obj.colors / 255.0
    cols = cols[np.random.default_rng(0).choice(len(obj.points), min(2500, len(obj.points)), replace=False)]
    a_map.scatter(pts[:, 0], pts[:, 2], s=3, c=cols, linewidths=0, label="object cloud")
    a_map.scatter(cands[:, 0], cands[:, 1], s=28, c=sc_c, cmap="viridis", edgecolors="k", linewidths=0.3, zorder=3,
                  label="floor candidates (score)")
    a_map.plot(path[:, 0], path[:, 2], "--", color="#7f7f7f", lw=1, label="planned path")
    traj, = a_map.plot([], [], color="#2a78d6", lw=1.8, label="robot trajectory")
    gfwd = T_goal[[0, 2], 2]
    a_map.plot(*gfloor, "*", color="#d03b3b", ms=14, zorder=4, label="goal (selected view)")
    a_map.annotate("", xy=gfloor + 0.5 * gfwd, xytext=gfloor, arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.5))
    wedge = plt.Polygon(np.zeros((3, 2)), closed=True, fc="#2a78d6", ec="#1a4f9c", alpha=0.45, lw=1.2, zorder=5)
    a_map.add_patch(wedge)
    a_map.plot(*C[0], "o", color="#1f7a3a", ms=6, zorder=4, label="start (detection view)")
    a_map.set(xlim=(c0[0] - r, c0[0] + r), ylim=(c0[1] + r, c0[1] - r), aspect="equal")
    a_map.tick_params(labelsize=6)
    a_map.set_xlabel("x (m)", fontsize=7)
    a_map.set_ylabel("z (m)", fontsize=7)
    a_map.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, markerscale=1.5)
    a_map.set_title("Top view: navigation then visual servoing", fontsize=9)
    a_des = fig.add_subplot(gs[0, 2])
    a_des.imshow(desired)
    a_des.set_title(f"Desired image ({goal['chosen']['name']}, d* = {d_star:.2f} m)", fontsize=9)
    a_des.axis("off")
    hand = len(nav_poses) - 0.5
    series = [("Distance and heading to the goal", [dist, head], ["#2a78d6", "#eb6834"],
               ["floor distance (m)", "heading error (deg)"], "", False),
              ("Photometric error", [mse], ["#6250d6"], ["MSE (grey levels)"], "", True),
              ("Robot velocities (per step)", [100 * vel[:, 0], 100 * vel[:, 2], np.degrees(vel[:, 4])],
               ["#2a78d6", "#1baf7a", "#eb6834"], ["$v_x$ lateral (cm)", "$v_z$ forward (cm)", "$\\omega_y$ (deg)"], "",
               False)]
    plots = []
    for c, (t, ys, cl, labels, yl, logy) in enumerate(series):
        a = fig.add_subplot(gs[1, c])
        a.axvspan(hand, N, color="#efe9fb", zorder=0)
        a.text(hand + 1, 0.97, "servo", transform=a.get_xaxis_transform(), fontsize=7, va="top", color="#6250d6")
        a.text(1, 0.97, "navigation", transform=a.get_xaxis_transform(), fontsize=7, va="top", color="#555555")
        for y, col, lab in zip(ys, cl, labels):
            a.plot(range(N), y, color=col, lw=1, alpha=0.2)
            plots.append((a.plot([], [], color=col, lw=1.6, label=lab)[0], y))
        if logy:
            a.set_yscale("log")
        else:
            lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
            a.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
        a.set(title=t, xlabel="step", ylabel=yl, xlim=(0, N - 1))
        a.title.set_fontsize(10)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        a.legend(fontsize=7, loc="center right")
    frames = []
    for i in range(N):
        T = poses[i]
        im_cur.set_data(scene.render(T, Kd, wd, hd)[0])
        u, vv, z = project(p[None], T, Kd)
        on = z[0] > 0 and 0 <= u[0] < wd and 0 <= vv[0] < hd
        mark.set_data([u[0]] if on else [], [vv[0]] if on else [])
        traj.set_data(C[: i + 1, 0], C[: i + 1, 1])
        f = T[[0, 2], 2] / np.linalg.norm(T[[0, 2], 2])
        lft = np.array([np.cos(np.radians(45)) * f[0] - np.sin(np.radians(45)) * f[1],
                        np.sin(np.radians(45)) * f[0] + np.cos(np.radians(45)) * f[1]])
        rgt = np.array([np.cos(np.radians(-45)) * f[0] - np.sin(np.radians(-45)) * f[1],
                        np.sin(np.radians(-45)) * f[0] + np.cos(np.radians(-45)) * f[1]])
        q = C[i]
        wedge.set_xy(np.array([q + 0.7 * lft, q, q + 0.7 * rgt]))
        for line, y in plots:
            line.set_data(np.arange(i + 1), y[: i + 1])
        title.set_text(f"“{query}” (InteriorGS house {sc[9:13]}, ground robot) — "
                       f"{'navigation' if phase[i] == 0 else 'visual servoing'}, step {i}: "
                       f"{dist[i]:.2f} m, {head[i]:.1f}° from the goal")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    frames += [frames[-1]] * 20
    imageio.mimsave(VID / f"igs_{name}.mp4", [even(fr) for fr in frames], fps=12, quality=7, macro_block_size=1)
    (OUT / "dashboards").mkdir(exist_ok=True)
    (OUT / "dashboards" / f"{name}.json").write_text(json.dumps(
        {"query": query, "scene": sc, "nav_steps": len(nav_poses), "servo_iterations": len(sv_err),
         "final_floor_distance_m": float(t1), "final_heading_deg": float(r1), "d_star": d_star,
         "final_mse": float(sv_err[-1])}))


def main():
    VID.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter(SAM)
    for house, q in QUERIES:
        sd = next(p for p in OUT.glob(f"interior_{house}_*"))
        name = f"{house}_{q.replace(' ', '_')}"
        if (VID / f"igs_{name}.mp4").exists():
            continue
        sv = np.load(sd / "survey.npz")
        K, W, H = sv["K"], int(sv["W"]), int(sv["H"])
        views = [View(p.name, str(p), T, K, W, H) for p, T in
                 zip(sorted((sd / "images").glob("survey_*.jpg")), sv["c2w"])]
        scene = GaussianScene(f"{VAL}/{sd.name}/{sd.name}.gs.ply")
        dashboard(sd.name, q, scene, views, sv, segmenter, name)
        scene.release()


if __name__ == "__main__":
    main()
