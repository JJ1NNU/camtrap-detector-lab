"""Ultralytics-based detectors for the flock router: YOLO-World, RT-DETR, SAM3.

All lazy-import `ultralytics`, so MD-only or facebook-sam3 experiments never need it.
NOTE on color order: ultralytics `predict()` expects **BGR** numpy (OpenCV order) and
converts to RGB internally, so we pass the frame straight through with NO cvtColor.
(The Kaggle snippet passed RGB, which silently swaps R/B — corrected here.)
"""
from __future__ import annotations
import os
import tempfile
from typing import List
import cv2
import numpy as np
from .base import Detector, Detection
from ..registry import DETECTORS


@DETECTORS.register("yoloworld")
class YOLOWorldDetector(Detector):
    """Open-vocabulary YOLO-World. `classes` sets the text vocabulary (e.g. duck/bird)."""
    name = "yoloworld"

    def __init__(self, weights="yolov8x-worldv2.pt", classes=("duck", "bird"),
                 conf=0.25, device="cuda", label="bird"):
        self.weights = weights
        self.classes = list(classes)
        self.conf = float(conf)
        self.device = device
        self.label = label
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        from ultralytics import YOLOWorld
        self._model = YOLOWorld(self.weights)
        self._model.set_classes(self.classes)

    def detect(self, image_bgr) -> List[Detection]:
        self._lazy()
        r = self._model.predict(image_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        dets: List[Detection] = []
        if r.boxes is not None:
            xy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            for b, s in zip(xy, sc):
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                dets.append(Detection((x1, y1, x2, y2), float(s), self.label))
        return dets


@DETECTORS.register("rtdetr")
class RTDETRDetector(Detector):
    """RT-DETR (COCO). `keep_class_ids` filters to specific classes (14 = bird)."""
    name = "rtdetr"

    def __init__(self, weights="rtdetr-x.pt", keep_class_ids=(14,), conf=0.25,
                 device="cuda", label="bird"):
        self.weights = weights
        self.keep = set(int(c) for c in keep_class_ids) if keep_class_ids else None
        self.conf = float(conf)
        self.device = device
        self.label = label
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        from ultralytics import RTDETR
        self._model = RTDETR(self.weights)

    def detect(self, image_bgr) -> List[Detection]:
        self._lazy()
        r = self._model.predict(image_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        dets: List[Detection] = []
        if r.boxes is not None:
            xy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy()
            for b, s, c in zip(xy, sc, cl):
                if self.keep is not None and int(c) not in self.keep:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                dets.append(Detection((x1, y1, x2, y2), float(s), self.label, int(c)))
        return dets


@DETECTORS.register("sam3_ultra")
class SAM3UltralyticsDetector(Detector):
    """Ultralytics SAM3 semantic (text-prompt) predictor for night/IR birds.

    Needs a local `sam3.pt` (gated on Hugging Face; ultralytics won't auto-download it).
    Distinct from the facebook `sam3` detector already in this repo.
    """
    name = "sam3_ultra"
    produces_masks = True

    def __init__(self, weights="sam3.pt", prompts=("duck", "bird"), conf=0.25,
                 quantize=16, device="cuda", label="bird"):
        self.weights = weights
        self.prompts = list(prompts)
        self.conf = float(conf)
        self.quantize = quantize
        self.device = device
        self.label = label
        self._predictor = None
        self._tmp = os.path.join(tempfile.gettempdir(), "_camtrap_sam3_frame.jpg")

    def _lazy(self):
        if self._predictor is not None:
            return
        from ultralytics.models.sam import SAM3SemanticPredictor
        self._predictor = SAM3SemanticPredictor(overrides=dict(
            conf=self.conf, task="segment", mode="predict",
            model=self.weights, quantize=self.quantize, verbose=False))

    def detect(self, image_bgr) -> List[Detection]:
        self._lazy()
        # Mirror the Kaggle flow: write the BGR frame to disk, set_image(path), prompt.
        cv2.imwrite(self._tmp, image_bgr)
        self._predictor.set_image(self._tmp)
        results = self._predictor(text=self.prompts)
        r = results[0] if isinstance(results, (list, tuple)) else results
        dets: List[Detection] = []
        if getattr(r, "boxes", None) is not None:
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                dets.append(Detection((x1, y1, x2, y2), 1.0, self.label))
        elif getattr(r, "masks", None) is not None:
            for m in r.masks.data.cpu().numpy():
                ys, xs = np.where(m > 0.5)
                if len(xs):
                    dets.append(Detection((float(xs.min()), float(ys.min()),
                                           float(xs.max()), float(ys.max())), 1.0, self.label))
        return dets
