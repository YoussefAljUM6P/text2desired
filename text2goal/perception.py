"""Open-vocabulary detection (Grounding DINO, Hugging Face transformers) and box-prompted segmentation (SAM)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

STOP_WORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or", "by", "for", "from", "is", "it"}


class Detector:
    def __init__(self, query, model_id="IDEA-Research/grounding-dino-tiny", threshold=0.2, device="cuda", max_area=1.0):
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device, self.threshold, self.max_area = device, threshold, max_area
        words = str(query).split()
        self.keywords = {w.lower() for w in words if w.lower() not in STOP_WORDS}
        self.prompt = " ".join([w for w in words if w.lower() not in STOP_WORDS] or words) + " ."

    @torch.no_grad()
    def __call__(self, image: Image.Image):
        """Best detection whose label matches the query: dict(score, confidence, box xyxy, area fraction),
        or None. score = confidence * (0.75 + 0.25 * min(area / 0.02, 1)) slightly penalizes tiny boxes."""
        image = image.convert("RGB")
        W, H = image.size
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        r = self.processor.post_process_grounded_object_detection(
            self.model(**inputs), input_ids=inputs.input_ids, threshold=self.threshold,
            text_threshold=self.threshold, target_sizes=[(H, W)])[0]
        scores, boxes, labels = r["scores"].tolist(), r["boxes"].tolist(), r.get("text_labels", [])
        if not scores:
            return None
        matched = [i for i, lab in enumerate(labels) if any(k in str(lab).lower() for k in self.keywords)]
        if not matched:
            if labels:
                return None
            matched = list(range(len(scores)))
        # boxes covering almost the whole image (e.g. a render facing a blank wall) are not object detections
        matched = [j for j in matched if (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) <= self.max_area * W * H]
        if not matched:
            return None
        i = max(matched, key=lambda j: scores[j])
        x0, y0, x1, y1 = boxes[i]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (W * H)
        return {"score": scores[i] * (0.75 + 0.25 * min(area / 0.02, 1.0)), "confidence": scores[i],
                "box": [x0, y0, x1, y1], "area": area}


class Segmenter:
    def __init__(self, sam_ckpt, model_type="vit_h", device="cuda"):
        from segment_anything import SamPredictor, sam_model_registry

        self.predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_ckpt).to(device))

    def __call__(self, rgb: np.ndarray, box) -> np.ndarray:
        """Single SAM mask (bool H x W) for a box prompt."""
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1])
        self.predictor.set_image(rgb)
        masks, _, _ = self.predictor.predict(box=np.array([x0, y0, x1, y1], np.float32), multimask_output=False)
        if not masks[0].any():
            raise ValueError("SAM returned an empty mask.")
        return masks[0].astype(bool)
