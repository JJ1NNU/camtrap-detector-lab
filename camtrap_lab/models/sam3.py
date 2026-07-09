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

    def _hf_login(self):
        """facebook/sam3 는 gated repo → HF 토큰이 있어야 다운로드됨.
        HF_TOKEN(또는 HUGGING_FACE_HUB_TOKEN) 환경변수가 있으면 로그인한다."""
        import os
        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN")
               or os.environ.get("HUGGINGFACE_TOKEN"))
        if not tok:
            return
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
        except Exception:
            pass  # 토큰이 있으면 hf_hub 다운로드가 env 로도 인증됨

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
        else:
            self._hf_login()  # gated facebook/sam3 가중치 인증
        try:
            model = build_sam3_image_model(**kw)
        except Exception as e:
            raise RuntimeError(
                "SAM3 가중치 로드 실패. checkpoint_path 가 null 이면 가중치를 gated HF repo "
                "'facebook/sam3' 에서 받습니다. (1) huggingface.co/facebook/sam3 에서 접근 승인, "
                "(2) HF_TOKEN 환경변수로 토큰 제공(또는 model.checkpoint_path 에 로컬 .pt 지정)이 "
                f"필요합니다. 원본 오류: {e}") from e
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
