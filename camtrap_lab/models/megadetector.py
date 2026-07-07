"""MegaDetector v5a / v6 via Pytorch-Wildlife. Lazy-imported to avoid dep clashes.

If your installed Pytorch-Wildlife exposes a slightly different constructor or
`single_image_detection` signature, adjust ONLY this file (clean separation).
"""
from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .base import Detector, Detection
from ..registry import DETECTORS

class _PWBase(Detector):
    def __init__(self, version=None, device="cuda", conf=0.2, pretrained=True, weights=None):
        self.version = version
        self.device = device
        self.conf = float(conf)
        self._pretrained = pretrained
        self._weights = weights
        self._model = None

    def _pw_class(self, pw):
        raise NotImplementedError

    def _lazy(self):
        if self._model is not None:
            return
        from PytorchWildlife.models import detection as pw  # lazy import
        cls = self._pw_class(pw)
        base = dict(device=self.device, pretrained=self._pretrained)
        trials = []
        if self.version:
            trials.append({**base, "version": self.version})
        trials.append(base)
        last_err = None
        for kw in trials:
            try:
                if self._weights:
                    kw = {**kw, "weights": self._weights}
                self._model = cls(**kw)
                return
            except TypeError as e:
                last_err = e
                continue
        raise last_err if last_err else RuntimeError("failed to build MegaDetector")

    def _infer(self, img_rgb):
        # Handle PW API differences robustly.
        m = self._model
        for call in (
            lambda: m.single_image_detection(img_rgb),
            lambda: m.single_image_detection(img=img_rgb),
        ):
            try:
                return call()
            except TypeError:
                continue
        raise RuntimeError("single_image_detection signature mismatch; edit megadetector.py")

    def detect(self, image_bgr) -> List[Detection]:
        self._lazy()
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._infer(img_rgb)
        return self._parse(res)

    def _parse(self, res) -> List[Detection]:
        dets: List[Detection] = []
        d = res.get("detections") if isinstance(res, dict) else res
        labels = res.get("labels") if isinstance(res, dict) else None
        if d is None:
            return dets
        xyxy = getattr(d, "xyxy", None)
        conf = getattr(d, "confidence", None)
        cls = getattr(d, "class_id", None)
        if xyxy is None:
            return dets
        for i in range(len(xyxy)):
            s = float(conf[i]) if conf is not None else 1.0
            if s < self.conf:
                continue
            cid = int(cls[i]) if cls is not None else -1
            if labels is not None and i < len(labels):
                lab = str(labels[i]).split()[0]
            else:
                lab = {0: "animal", 1: "person", 2: "vehicle"}.get(cid, str(cid))
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            dets.append(Detection((x1, y1, x2, y2), s, lab, cid))
        return dets

@DETECTORS.register("mdv5a")
class MegaDetectorV5a(_PWBase):
    name = "mdv5a"
    def __init__(self, device="cuda", conf=0.2, version="a", **kw):
        super().__init__(version=version, device=device, conf=conf, **kw)
    def _pw_class(self, pw):
        return pw.MegaDetectorV5

@DETECTORS.register("mdv6")
class MegaDetectorV6(_PWBase):
    name = "mdv6"
    def __init__(self, device="cuda", conf=0.2, version="MDV6-yolov9-c", **kw):
        super().__init__(version=version, device=device, conf=conf, **kw)
    def _pw_class(self, pw):
        return pw.MegaDetectorV6
