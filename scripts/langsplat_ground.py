"""Stage 1 with LangSplat-V2 instead of Grounding DINO + SAM (grounding-module ablation).

For every training view, the 3 language-feature levels of the authors' pretrained LangSplat-V2 model are rendered
and turned into relevancy maps for all the queries of the scene (OpenCLIP ViT-B/16, canonical negatives, as in
their eval_lerf.py). Reference view of a query = the view and level with the highest smoothed relevancy (their
localization rule); its mask = their segmentation rule (min-max normalised relevancy > 0.4, smoothed).
Writes <out>/<query>/{mask.png, ground.json}; then text_to_desired_image.py --mask_json runs stages 2-3.

Run from the LangSplatV2 repository with its env:
  cd litterature/code/LangSplatV2
  ~/anaconda3/envs/langsplat_v2/bin/python <my_code>/scripts/langsplat_ground.py -s <lerf_ovs>/teatime \
      -m <langsplatv2_pretrained>/LERF/teatime/teatime_0_1 --dataset_name teatime --include_feature \
      --queries <my_code>/out/lerf_ovs_queries_teatime.txt --out <my_code>/out/langsplat_ground/teatime
"""
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from eval.openclip_encoder import OpenCLIPNetwork  # noqa: E402
from eval_lerf import render_language_feature_map_quick, smooth_cuda  # noqa: E402
from gaussian_renderer import GaussianModel  # noqa: E402
from scene import Scene  # noqa: E402


def get_weights_and_indices(logits, k):
    """Top-k softmax codes, renormalised (utils/vq_utils.py, without its CPU/GPU indexing bug on recent torch)."""
    w, i = torch.topk(logits.softmax(dim=1), k, dim=1)
    return w / (w.sum(1, keepdim=True) + 1e-10), i


def combined_model(dataset, args):
    """The 3 levels in one model (evaluate_quick of eval_lerf.py)."""
    g = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.ckpt_paths[0]
    scene = Scene(dataset, g, shuffle=False)
    g.restore(torch.load(os.path.join(args.ckpt_paths[0], f"chkpnt{args.checkpoint}.pth"))[0], args, mode="test")
    books, weights, indices = [], [], []
    for lvl, path in enumerate(args.ckpt_paths):
        m = GaussianModel(dataset.sh_degree)
        m.restore(torch.load(os.path.join(path, f"chkpnt{args.checkpoint}.pth"))[0], args, mode="test")
        books.append(m._language_feature_codebooks.view(-1, 512))
        w, i = get_weights_and_indices(m._language_feature_logits, 4)
        weights.append(w)
        indices.append(i + int(lvl * m._language_feature_codebooks.shape[1]))
    g._language_feature_codebooks = torch.stack(books, 0)
    g._language_feature_weights = torch.cat(weights, 1)
    g._language_feature_indices = torch.cat(indices, 1).float().contiguous()       # the CUDA kernel reads floats
    return g, scene.getTrainCameras()


@torch.no_grad()
def relevancy(g, view, pipe, bg, args, clip):
    feat = render_language_feature_map_quick(g, view, pipe, bg, args).permute(0, 2, 3, 1)
    valid = clip.get_max_across_quick(feat)                                  # (3 levels, n queries, h, w)
    n_lvl, n_q, h, w = valid.shape
    pool = torch.nn.AvgPool2d(29, stride=1, padding=14, count_include_pad=False)
    smoothed = pool(valid.reshape(-1, 1, h, w)).reshape(n_lvl, n_q, h, w)
    return valid, smoothed


def main():
    parser = ArgumentParser()
    model, pipeline = ModelParams(parser, sentinel=True), PipelineParams(parser)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    args = get_combined_args(parser)
    args.quick_render = True
    root = Path(args.model_path).parent
    args.ckpt_paths = [str(root / f"{args.dataset_name}_0_{lvl}") for lvl in (1, 2, 3)]
    queries = [q for q in Path(args.queries).read_text().splitlines() if q.strip()]
    dataset, pipe = model.extract(args), pipeline.extract(args)
    g, views = combined_model(dataset, args)
    bg = torch.zeros(3, device="cuda")
    clip = OpenCLIPNetwork("cuda")
    clip.set_positives(queries)

    best = {q: (-1.0, -1, -1) for q in queries}                             # score, view index, level
    for i, view in enumerate(views):
        _, sm = relevancy(g, view, pipe, bg, args, clip)
        s = sm.amax(dim=(2, 3))                                             # (levels, queries)
        for k, q in enumerate(queries):
            lvl = int(torch.argmax(s[:, k]))
            if float(s[lvl, k]) > best[q][0]:
                best[q] = (float(s[lvl, k]), i, lvl)
        if (i + 1) % 50 == 0:
            print(f"[LangSplat-V2] {i + 1}/{len(views)} views", flush=True)

    for k, q in enumerate(queries):
        score, i, lvl = best[q]
        valid, sm = relevancy(g, views[i], pipe, bg, args, clip)
        r = 0.5 * (sm[lvl, k] + valid[lvl, k])
        r = (r - r.min()) / (r.max() - r.min() + 1e-9) * 2 - 1
        mask = smooth_cuda((r.clip(0, 1) > args.mask_thresh).to(torch.uint8)).cpu().numpy().astype(np.uint8) * 255
        o = Path(args.out) / q.replace(" ", "_")
        o.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(o / "mask.png")
        (o / "ground.json").write_text(json.dumps({"query": q, "image": views[i].image_name, "score": score,
                                                   "level": lvl + 1, "mask": str((o / "mask.png").resolve()),
                                                   "source": "langsplat_v2"}, indent=2))
        print(f"[LangSplat-V2] '{q}': view {views[i].image_name} level {lvl + 1} score {score:.3f}, "
              f"mask {100 * (mask > 0).mean():.1f} %", flush=True)


if __name__ == "__main__":
    main()
