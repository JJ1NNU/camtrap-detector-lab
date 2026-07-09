"""Flock-routing inference strategy (detection-only port of the Kaggle pipeline).

Flow per frame:
  1) MegaDetector (the `model:` block) does a first pass.
  2) Flock gate: if #detections >= flock_th -> treat as a bird flock.
  3) Day/night split by HSV saturation mean.
       day   -> RT-DETR + YOLO-World ensemble (day_models) + conservative merge
       night -> SAM3 (night_model) + aggressive merge; MD fallback if unavailable
  4) Not a flock -> return the MegaDetector result (mammals / few individuals).

Intentionally omitted (they are classification, not detection):
  - the EfficientNet species CNN,
  - the maxn==3 + CNN "promote to flock" rule.

NOTE: the original decided flock over a whole video's MAX detection count; this runs
on the lab's streaming per-frame interface, so each frame decides independently.
"""
from __future__ import annotations
import os
from typing import List
from .base import InferenceStrategy
from ..registry import STRATEGIES, DETECTORS
from ..models.base import Detection
from ..utils.boxmerge import merge_param, frame_saturation_bgr


@STRATEGIES.register("flock_router")
class FlockRouterStrategy(InferenceStrategy):
    name = "flock_router"

    def __init__(self, detector, flock_th=4, night_sat_th=10, bird_label="bird",
                 day_merge=None, night_merge=None, day_models=None, night_model=None):
        super().__init__(detector)          # detector = MegaDetector: first pass + mammal/fallback
        self.flock_th = int(flock_th)
        self.night_sat_th = float(night_sat_th)
        self.bird_label = bird_label
        self.day_merge = day_merge or dict(pad=0.0, merge_iou=0.70, merge_ioa=0.95)
        self.night_merge = night_merge or dict(pad=0.2, merge_iou=0.20, merge_ioa=0.50)
        # Sub-detectors built via the factory (constructed now; weights load lazily on first use).
        self._day = [DETECTORS.build(dict(m)) for m in (day_models or [])]
        self._night = self._build_night(night_model)

    def _build_night(self, night_model):
        if not night_model:
            return None
        w = night_model.get("weights")
        has_hf = bool(night_model.get("hf_repo") and night_model.get("hf_filename"))
        # 로컬 파일도 없고 HF 소스도 없을 때만 미리 폴백. (HF 소스가 있으면 첫 사용 시 다운로드 시도)
        if not has_hf and w and not os.path.exists(w):
            print(f"[flock_router] night weights not found ({w}) and no hf_repo/hf_filename "
                  f"-> night uses MegaDetector fallback")
            return None
        return DETECTORS.build(dict(night_model))

    @staticmethod
    def _boxes(dets):
        return [list(d.bbox) for d in dets]

    def run(self, image_bgr) -> List[Detection]:
        md_dets = self.detector.detect(image_bgr)
        if len(md_dets) < self.flock_th:
            return md_dets                     # mammal / few individuals

        H, W = image_bgr.shape[:2]
        if frame_saturation_bgr(image_bgr) < self.night_sat_th:     # night / IR
            if self._night is None:
                return md_dets                 # SAM3 unavailable -> MD fallback
            try:
                night_dets = self._night.detect(image_bgr)   # 첫 호출에서 가중치 로드/다운로드
            except Exception as e:
                print(f"[flock_router] SAM3 야간 추론 실패 ({str(e)[:100]}) "
                      f"-> 이후 야간은 MegaDetector 폴백")
                self._night = None             # 매 프레임 재시도 방지
                return md_dets
            merged = merge_param(self._boxes(night_dets), (H, W), **self.night_merge)
        else:                                                       # day
            boxes = []
            for det in self._day:
                boxes += self._boxes(det.detect(image_bgr))
            if not boxes:
                return md_dets                 # ensemble found nothing -> MD result
            merged = merge_param(boxes, (H, W), **self.day_merge)

        return [Detection((b[0], b[1], b[2], b[3]), 1.0, self.bird_label) for b in merged]
