"""Optional annotated-video writer (BBox + mask overlay). Size-control options included."""
from __future__ import annotations
import os, subprocess
import cv2
import numpy as np

def _color(k):
    rng = np.random.RandomState((k * 2654435761) % (2 ** 32))
    return tuple(int(c) for c in rng.randint(64, 256, 3))

def draw(frame, dets, mask_alpha=0.5, box_thick=2, draw_masks=True):
    out = frame.copy()
    if draw_masks:
        ov = out.copy()
        for i, d in enumerate(dets):
            if d.mask is not None:
                ov[d.mask] = _color(hash(d.label) % 99991 + i)
        out = cv2.addWeighted(ov, mask_alpha, out, 1 - mask_alpha, 0)
    for i, d in enumerate(dets):
        c = _color(hash(d.label) % 99991 + i)
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, box_thick)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out

class VideoVisualizer:
    """size controls: max_side (downscale), fps, crf (H.264 quality; higher=smaller)."""
    def __init__(self, run_dir, fps=8.0, mask_alpha=0.5, box_thick=2,
                 max_side=1280, crf=28, draw_masks=True):
        self.dir = os.path.join(run_dir, "viz"); os.makedirs(self.dir, exist_ok=True)
        self.fps = float(fps); self.mask_alpha = mask_alpha; self.box_thick = box_thick
        self.max_side = int(max_side or 0); self.crf = int(crf); self.draw_masks = draw_masks

    def _resize(self, f):
        if not self.max_side:
            return f
        h, w = f.shape[:2]; sc = self.max_side / max(h, w)
        return cv2.resize(f, (int(w * sc), int(h * sc))) if sc < 1.0 else f

    def write(self, video_name, annotated_frames):
        if not annotated_frames:
            return ""
        frames = [self._resize(f) for f in annotated_frames]
        H, W = frames[0].shape[:2]
        stem = os.path.splitext(video_name)[0]
        tmp = os.path.join(self.dir, stem + "_viz.tmp.mp4")
        out = os.path.join(self.dir, stem + "_viz.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, self.fps), (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        try:
            # H.264 + CRF 로 대폭 압축 (crf 23=고화질/큼, 28=기본, 32=작음)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                            "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", str(self.crf), "-pix_fmt", "yuv420p", out], check=True)
            os.remove(tmp)
        except Exception:
            os.replace(tmp, out)
        return out
