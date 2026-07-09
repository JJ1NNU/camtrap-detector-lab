"""Union-find box merge (count flock individuals) + frame saturation. Ported from the
Kaggle camera-trap pipeline. Used by the `flock_router` strategy."""
from __future__ import annotations
import cv2
import numpy as np


def frame_saturation_bgr(image_bgr) -> float:
    """HSV saturation mean. IR/night footage is ~grayscale, so this value is very low."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean())


def merge_param(boxes, hw, pad=0.0, merge_iou=0.70, merge_ioa=0.95):
    """Merge overlapping boxes via union-find (purpose: counting individuals).
      boxes : list of [x1,y1,x2,y2]
      hw    : (H, W) of the frame
      pad       -> enlarge boxes before overlap test (bigger = merges more)
      merge_iou -> merge if IoU >= this (smaller = merges more)
      merge_ioa -> merge if one box is contained in another by >= this ratio
    Returns list of merged [x1,y1,x2,y2] (ints).
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, np.float32)
    H, W = hw[:2]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] -= w * pad / 2
    boxes[:, 2] += w * pad / 2
    boxes[:, 1] -= h * pad / 2
    boxes[:, 3] += h * pad / 2
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H)

    n = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def iou_ioa(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0., 0., 0.
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6), inter / (a1 + 1e-6), inter / (a2 + 1e-6)

    for i in range(n):
        for j in range(i + 1, n):
            iou, ia, ja = iou_ioa(boxes[i], boxes[j])
            if iou >= merge_iou or ia >= merge_ioa or ja >= merge_ioa:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[int(boxes[idxs][:, 0].min()), int(boxes[idxs][:, 1].min()),
             int(boxes[idxs][:, 2].max()), int(boxes[idxs][:, 3].max())]
            for idxs in groups.values()]
