"""Encode segmentation masks compactly for storage (area / polygon / RLE)."""
from __future__ import annotations
import base64, zlib
import cv2
import numpy as np

def mask_area(m):
    return int(m.sum())

def mask_polygon(m, max_pts=80):
    m8 = (m.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if len(c) > max_pts:
        idx = np.linspace(0, len(c) - 1, max_pts).astype(int)
        c = c[idx]
    return c.reshape(-1, 2).tolist()

def mask_rle(m):
    packed = np.packbits(m.astype(np.uint8))
    comp = zlib.compress(packed.tobytes())
    return {"h": int(m.shape[0]), "w": int(m.shape[1]),
            "data": base64.b64encode(comp).decode("ascii")}

def encode_mask(m, mode, polygon_max_pts=80):
    if m is None or mode == "none":
        return None
    if mode == "area":
        return {"area": mask_area(m)}
    if mode == "polygon":
        return {"area": mask_area(m), "polygon": mask_polygon(m, polygon_max_pts)}
    if mode == "rle":
        return {"area": mask_area(m), "rle": mask_rle(m)}
    return None
