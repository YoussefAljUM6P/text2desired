"""Assets of the pipeline figure (Fig. 1) of the journal paper, from the frozen-parameter run of "piano" (room).

  paper_mva/figures/pipeline_v2/
    reference.jpg       reference render with the SAM mask (tinted) and the detection box
    cand_normal.jpg     candidate along the PCA normal, cand_chosen.jpg the selected candidate (from candidates.png)
    cand_normal_outside.jpg  outside view of the room: the PCA-normal camera (red, inside the piano) and the selected
                        camera (green); the render from the PCA-normal pose itself shows only the inside of the piano
    desired.jpg         desired image
    servo_start.jpg     servo start (large perturbation, seed 0) and servo_final.jpg its last view (320 px)
    values.tex          numbers quoted in the figure
Layout: paper_mva/figures/pipeline_v2.tex (TikZ, drawn at the text width, not rescaled).

  ~/anaconda3/envs/pca_servo/bin/python scripts/pipeline_figure.py        (from my_code/)
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "out/multiscene/room/piano"
STRIP = ROOT / "out/servo_eval/room_piano_same_a0.2.png"       # desired | start | final, seed 0, 320 px renders
FIG = ROOT / "paper_mva/figures/pipeline_v2"
W = 640
CKPT = "/media/youssef/LaCie/data/mip-nerf360/360_v2/room/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"
NORMAL_DIST = 0.13                                              # distance of the PCA-normal candidate (its tile)


def look_at(eye, target, up):
    """Camera-to-world pose (OpenCV axes: x right, y down, z forward) at eye looking at target, horizon level."""
    z = (target - eye) / np.linalg.norm(target - eye)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
    T[:3, 3] = eye
    return T


def outside_view(g):
    """Render the room from above and beside the selected camera, with both candidate cameras drawn as frusta."""
    from PIL import ImageDraw
    from text2goal.geometry import project
    from text2goal.render import GaussianScene
    p, n, up, v = g["p_obj"], g["n_obj"], g["up_world"], g["v_world"]
    T_normal = look_at(p + NORMAL_DIST * n, p, up)
    T_chosen = g["T_goal"]
    eye = p + 2.6 * v + 0.3 * up                             # behind the selected camera, near the photos
    T_view = look_at(eye, p, up)
    Wf, Hf = int(g["W"]), int(g["H"])
    K = g["K"].copy()
    s = 960 / Wf
    K[:2] *= s
    K[0, 0] *= 0.7
    K[1, 1] *= 0.7
    w, h = 960, round(Hf * s)
    scene = GaussianScene(CKPT)
    img = Image.fromarray(scene.render(T_view, K, w, h)[0])
    scene.release()
    d = ImageDraw.Draw(img)
    for T, col, sz in [(T_normal, (220, 30, 30), 0.18), (T_chosen, (30, 190, 60), 0.25)]:
        Kc = g["K"]
        corners = [np.array([(u - Kc[0, 2]) / Kc[0, 0], (vv - Kc[1, 2]) / Kc[1, 1], 1.0]) * sz
                   for u, vv in [(0, 0), (Wf, 0), (Wf, Hf), (0, Hf)]]
        P = np.array([T[:3, 3]] + [T[:3, :3] @ c + T[:3, 3] for c in corners])
        u, vv, z = project(P, T_view, K)
        q = list(zip(u, vv))
        for i in range(1, 5):
            d.line([q[0], q[i]], fill=col, width=5)
        d.line(q[1:] + [q[1]], fill=col, width=5)
        d.ellipse([q[0][0] - 7, q[0][1] - 7, q[0][0] + 7, q[0][1] + 7], fill=col)
    return img.crop((round(0.25 * w), round(0.25 * h), round(0.8 * w), round(0.8 * h)))   # around the piano


def save(img, name, w=W):
    img = img.convert("RGB")
    img.resize((w, round(img.height * w / img.width)), Image.LANCZOS).save(FIG / name, quality=90)


def candidate(grid, label_h, k, cols=4, rows=2):
    tw, th = grid.width // cols, grid.height // rows
    x, y = (k % cols) * tw, (k // cols) * th
    return grid.crop((x, y + label_h, x + tw, y + th))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sm = json.loads((RUN / "summary.json").read_text())

    ref = np.asarray(Image.open(RUN / "reference_render.png").convert("RGB")).astype(np.float32)
    mask = np.asarray(Image.open(RUN / "reference_mask.png").convert("L")) > 127
    ref[mask] = 0.55 * ref[mask] + 0.45 * np.array([220, 40, 40])
    img = Image.fromarray(ref.astype(np.uint8))
    from PIL import ImageDraw
    x0, y0, x1, y1 = sm["detection"]["box"]
    ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=(30, 200, 60), width=36)
    save(img, "reference.jpg")

    grid = Image.open(RUN / "candidates.png")
    label_h = round(grid.height / 2 * 0.065)                    # text band on top of each tile
    save(candidate(grid, label_h, 0), "cand_normal.jpg")        # tile 0: pca_normal
    save(candidate(grid, label_h, 1), "cand_chosen.jpg")        # tile 1: reference_view, chosen
    save(outside_view(np.load(RUN / "goal.npz")), "cand_normal_outside.jpg")
    save(Image.open(RUN / "desired_image.png"), "desired.jpg")

    s = Image.open(STRIP)
    w3 = s.width // 3
    save(s.crop((w3, 0, 2 * w3, s.height)), "servo_start.jpg", w3)
    save(s.crop((2 * w3, 0, 3 * w3, s.height)), "servo_final.jpg", w3)

    r = next(r for r in csv.DictReader(open(ROOT / "out/servo_eval/mipnerf360_same.csv"))
             if r["run"].endswith("room/piano") and r["alpha"] == "0.2" and r["seed"] == "0")
    vals = {"PipeQuery": sm["query"], "PipeRho": f"{sm['rho']:.2f}", "PipeDist": f"{sm['d_star']:.2f}",
            "PipeChosenVis": f"{sm['visibility']:.2f}", "PipeChosenCov": f"{sm['support']:.2f}",
            "PipeChosenFit": f"{sm['fit']:.2f}", "PipeAngle": f"{sm['angle_to_pca_normal_deg']:.0f}",
            "PipeServoIt": r["iterations"], "PipeServoT": f"{float(r['t_err_rel']):.4f}",
            "PipeServoR": f"{float(r['r_err']):.2f}",
            # candidate along the PCA normal: printed on its tile in candidates.png (not in summary.json)
            "PipeNormalDist": "0.13", "PipeNormalFit": "0.12", "PipeNormalVis": "0.01", "PipeNormalCov": "0.00"}
    (FIG / "values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()))
    print(vals)


if __name__ == "__main__":
    main()
