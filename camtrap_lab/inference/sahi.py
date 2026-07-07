"""SAHI-style tiled inference. Works for any Detector (bbox or bbox+mask)."""
from __future__ import annotations
import numpy as np
from .base import InferenceStrategy
from .nms import nms
from ..registry import STRATEGIES
from ..models.base import Detection

@STRATEGIES.register("sahi")
class SahiStrategy(InferenceStrategy):
    name = "sahi"
    def __init__(self, detector, tile_size=640, overlap=0.2, full_frame_pass=True, nms_iou=0.5):
        super().__init__(detector)
        self.tile = int(tile_size)
        self.overlap = float(overlap)
        self.full = bool(full_frame_pass)
        self.nms_iou = float(nms_iou)

    def _tiles(self, img):
        H, W = img.shape[:2]
        t = self.tile
        if t >= max(H, W):
            return [(img, 0, 0)]
        step = max(1, int(t * (1 - self.overlap)))
        xs = list(range(0, max(W - t, 0) + 1, step))
        ys = list(range(0, max(H - t, 0) + 1, step))
        if not xs or xs[-1] != W - t: xs.append(max(W - t, 0))
        if not ys or ys[-1] != H - t: ys.append(max(H - t, 0))
        xs, ys = sorted(set(xs)), sorted(set(ys))
        return [(img[y:y + t, x:x + t].copy(), x, y) for y in ys for x in xs]

    def run(self, image_bgr):
        H, W = image_bgr.shape[:2]
        tiles = self._tiles(image_bgr)
        if self.full:
            tiles.append((image_bgr, 0, 0))
        merged = []
        for tile, x0, y0 in tiles:
            for d in self.detector.detect(tile):
                x1, y1, x2, y2 = d.bbox
                fm = None
                if d.mask is not None:
                    fm = np.zeros((H, W), bool)
                    th, tw = d.mask.shape[:2]
                    fm[y0:y0 + th, x0:x0 + tw] = d.mask[:H - y0, :W - x0]
                merged.append(Detection((x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                                        d.score, d.label, d.class_id, mask=fm))
        return nms(merged, self.nms_iou)
