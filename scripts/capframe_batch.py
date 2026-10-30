"""Baseline B3: CapFrame (Li et al., ECCV 2026, arXiv 2608.30342), text -> camera pose in a 3DGS scene.

Runs the authors' pipeline (optimize.py __main__: Qwen3-VL questions + FG-CLIP/MLLM view scoring, GroundedSAM,
Orient Anything, pose optimisation with orientation + layout losses) for every LERF-OVS query of a scene, with
three adaptations only:
  - Qwen3-VL-32B-Instruct (their default) loaded in 4-bit NF4 (bitsandbytes, embedding
    table and output head on the CPU): the bf16 model needs ~66 GB, we have 24 GB; MLLM image batch 2 instead of 32 (memory only);
  - the MLLM and FG-CLIP are loaded once per scene, not once per query;
  - the instruction is templated from the query: "A view of the <query>, in the center of the frame."
3DGS: LangSplat-V2's RGB Gaussians (Inria format, raw COLMAP frame), the scene format CapFrame expects. LangSplat's
cameras.json gives image names without extension, so --images points to a folder of extension-less links.
Writes <out>/<query>/capframe.npz (R, T in CapFrame's convention: W2C = [R^T | T], COLMAP frame; fx fy W H),
the final render and the instruction. scripts/goal_benchmark.py converts them to our frame (method capframe).

Run from the CapFrame repository with its env:
  cd ~/Documents/SW/capframe && /media/youssef/DATA/capframe/env/bin/python <my_code>/scripts/capframe_batch.py \
     -m <langsplatv2>/LERF/teatime/teatime_0_1 -s <capframe data>/images/teatime --model_root <fg-clip-base> \
     --queries <my_code>/out/lerf_ovs_queries_teatime.txt --out <my_code>/out/capframe/teatime --phase mllm
  then the same command with --phase optim (a new process: the MLLM's GPU memory is only freed at exit).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
TEMPLATE = "A view of the {}, in the center of the frame."
ORIENT_CACHE = "/media/youssef/DATA/capframe/orient_cache/"     # the home disk is full: weights on DATA


def take(flag):
    """Remove `flag value` from sys.argv (CapFrame's parser does not know it)."""
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


def main():
    queries_file, out_root, phase = take("--queries"), Path(take("--out")), take("--phase")
    stage_file = out_root / "mllm_stages.pkl"
    import transformers
    from transformers import BitsAndBytesConfig

    # 4-bit NF4 language layers and vision encoder (standard 4-bit loading). The token-embedding table (1.6 GB, a lookup) and the
    # output head (1.6 GB, applied to the last token only when generating) are then moved to and run on the CPU
    # (accelerate's "cpu" device_map would stream their weights to the GPU at every call: no memory saved).
    q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                            llm_int8_skip_modules=["lm_head"])
    base = transformers.Qwen3VLForConditionalGeneration.from_pretrained.__func__

    class OnCPU(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m.cpu()

        def forward(self, x):
            dev = x.device
            return self.m(x.cpu()).to(dev)

        @property
        def weight(self):
            return self.m.weight

    def from_pretrained_4bit(cls, *a, **k):
        k.pop("dtype", None)
        k["device_map"] = {"": 0}
        model = base(cls, *a, quantization_config=q4, dtype=torch.bfloat16, **k)
        lm = model.model.language_model
        lm.embed_tokens = OnCPU(lm.embed_tokens)
        model.lm_head = OnCPU(model.lm_head)
        torch.cuda.empty_cache()
        return model

    transformers.Qwen3VLForConditionalGeneration.from_pretrained = classmethod(from_pretrained_4bit)

    import mllm as mllm_module
    import optimize as opt_module
    from CameraGS import GaussianModel, searchForMaxIteration, update_pose
    from configs.arguments import get_args_parser
    from GroundedSAM.inference import GroundedSAM
    from OrientAnything.inference import OrientAnything
    from rendering import Renderer
    from utils import load_topk_cameras

    mllm_module.Qwen3VLForConditionalGeneration = transformers.Qwen3VLForConditionalGeneration
    args = get_args_parser()
    opt_module.args = args                                   # Optimizer.run reads the module-level args
    args.save_image_every = 10 ** 9                          # no render every 20 steps (~5 GB over all queries)
    queries = [q for q in Path(queries_file).read_text().splitlines() if q.strip()]
    todo = [q for q in queries if not (out_root / q.replace(" ", "_") / "capframe.npz").exists()]
    vertical = args.camera_vertical

    # Phase A (--phase mllm): every MLLM call of CapFrame's pipeline, for all queries, saved to mllm_stages.pkl (the
    # 4-bit 32B model leaves no GPU memory for GroundedSAM / Orient Anything / the pose optimisation, and is not
    # freed by del: phase B (--phase optim) runs in a new process). OrientParser only reads the object list, so
    # calling it before GroundedSAM changes nothing.
    import pickle
    if phase == "mllm":
        out_root.mkdir(parents=True, exist_ok=True)
        args.user_input = TEMPLATE.format(queries[0])
        parser = mllm_module.MLLM(args, score_topk=args.score_topk, score_threshold=args.score_threshold)
    mllm_out, seconds = pickle.load(open(stage_file, "rb")) if stage_file.exists() else ({}, {})
    for q in (todo if phase == "mllm" else []):
        if q in mllm_out and not isinstance(mllm_out[q], str):
            continue
        text = TEMPLATE.format(q)
        args.user_input = parser.user_prompt = parser.text_prompt = text
        t0 = time.time()
        try:
            questions = parser.TextQuestioner(text)
            cam_topk, score_topk = parser.ImageEvaluator(questions)
            data = load_topk_cameras(cam_topk, score_topk, args.camera_path, args.model.source_path, args.device)
            objects = parser.ObjectParser(data)
            mllm_out[q] = (cam_topk, score_topk, objects, parser.LayoutParser(objects), parser.OrientParser(objects))
        except Exception as e:
            import traceback
            traceback.print_exc()
            mllm_out[q] = f"failed (MLLM stage): {type(e).__name__}: {e}"
        seconds[q] = time.time() - t0
        print(f"[CapFrame] MLLM stages '{q}': {seconds[q]:.0f} s", flush=True)
        pickle.dump((mllm_out, seconds), open(stage_file, "wb"))
        torch.cuda.empty_cache()
    if phase == "mllm":
        return                                               # the process exit frees the GPU for phase B

    # Phase B (--phase optim): GroundedSAM, Orient Anything and the pose optimisation, as in optimize.py
    it = searchForMaxIteration(os.path.join(args.model.model_path, "point_cloud"))
    gaussians = GaussianModel(args.model.sh_degree)
    gaussians.load_ply(os.path.join(args.model.model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply"))
    for q in todo:
        o = out_root / q.replace(" ", "_")
        o.mkdir(parents=True, exist_ok=True)
        text = TEMPLATE.format(q)
        args.user_input = text
        args.camera_vertical = vertical
        args.render_path = str(o / "renders")
        args.camera_R_path, args.camera_T_path = str(o / "camera_R.npy"), str(o / "camera_T.npy")
        args.processed_image_dir = str(o / "orient") + "/"
        os.makedirs(args.render_path, exist_ok=True)
        t0 = time.time() - seconds.get(q, 0.0)
        try:
            if isinstance(mllm_out.get(q, "failed: no MLLM stage output"), str):
                mllm_out.setdefault(q, "failed: no MLLM stage output")
                raise RuntimeError(mllm_out[q])
            cam_topk, score_topk, objects, layout, orient = mllm_out[q]
            data = load_topk_cameras(cam_topk, score_topk, args.camera_path, args.model.source_path, args.device)
            renderer = Renderer(data, gaussians, args)
            data = renderer.render_depth(data)
            out = {}
            data = GroundedSAM(objects, args.device).infer(data)
            out["layout"] = layout
            mask = renderer.get_mask_index(data, args)
            if orient != -1:
                data = OrientAnything(args, cache_dir=ORIENT_CACHE).infer(data, orient, mask)
                out["orient"] = orient
                args.camera_vertical = False
            optimizer = opt_module.Optimizer(data, mask, gaussians, out, args)
            optimizer.run()
            cam = optimizer.cameras[optimizer.idx]
            with torch.no_grad():
                update_pose(cam)                             # apply the last Adam step
            np.savez(o / "capframe.npz", R=cam.R.detach().cpu().numpy(), T=cam.T.detach().cpu().numpy(),
                     fx=optimizer.fx, fy=optimizer.fy, W=optimizer.width, H=optimizer.height,
                     start_camera_id=cam.camera_id)
            status = "ok"
        except Exception as e:                               # a failed query counts as a miss
            import traceback
            traceback.print_exc()
            status = f"failed: {type(e).__name__}: {e}"
        (o / "capframe.json").write_text(json.dumps({"query": q, "instruction": text, "status": status,
                                                     "seconds": time.time() - t0}, indent=2))
        print(f"[CapFrame] '{q}': {status} ({time.time() - t0:.0f} s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
