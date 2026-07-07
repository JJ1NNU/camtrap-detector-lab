"""SAM 3 concept detector (image model). Returns bbox + mask per prompt.

Lazy-imports sam3 so MD-only experiments don't need it installed.
"""
from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .base import Detector, Detection
from ..registry import DETECTORS

@DETECTORS.register("sam3")
class SAM3Detector(Detector):
    name = "sam3"
    produces_masks = True

    def __init__(self, prompts=("animal", "bird"), conf=0.3, device="cuda",
                 checkpoint_path=None, bpe_path=None, autocast_dtype="float16"):
        self.prompts = list(prompts)
        self.conf = float(conf)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.autocast_dtype = autocast_dtype
        self._proc = None
        self._torch = None
        self._dtype = None

    def _lazy(self):
        if self._proc is not None:
            return
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        kw = {}
        if self.bpe_path:
            kw["bpe_path"] = self.bpe_path
        if self.checkpoint_path:
            kw["checkpoint_path"] = self.checkpoint_path
        model = build_sam3_image_model(**kw)
        self._proc = Sam3Processor(model, confidence_threshold=self.conf)
        self._torch = torch
        self._dtype = torch.float16 if self.autocast_dtype == "float16" else torch.bfloat16

    def detect(self, image_bgr) -> List[Detection]:
        self._lazy()
        import PIL.Image
        torch = self._torch
        pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        dets: List[Detection] = []
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=self._dtype):
                state = self._proc.set_image(pil)
                for p in self.prompts:
                    out = self._proc.set_text_prompt(state=state, prompt=p)
                    masks = out.get("masks")
                    scores = out.get("scores")
                    if masks is None:
                        continue
                    for i in range(len(masks)):
                        s = 1.0
                        try:
                            s = float(scores[i])
                        except Exception:
                            pass
                        if s < self.conf:
                            continue
                        m = masks[i]
                        if isinstance(m, torch.Tensor):
                            m = m.detach().cpu().numpy()
                        m = np.squeeze(np.asarray(m))
                        if m.ndim > 2:
                            m = m[0]
                        m = m > 0.5
                        ys, xs = np.where(m)
                        if xs.size == 0:
                            continue
                        dets.append(Detection(
                            (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                            s, p, -1, mask=m))
        return dets
