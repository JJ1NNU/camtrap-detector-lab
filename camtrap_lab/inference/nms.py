from __future__ import annotations
from typing import List
import numpy as np
from ..models.base import Detection

def nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    if not dets:
        return []
    b = np.array([d.bbox for d in dets], dtype=float)
    s = np.array([d.score for d in dets], dtype=float)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return [dets[i] for i in keep]
