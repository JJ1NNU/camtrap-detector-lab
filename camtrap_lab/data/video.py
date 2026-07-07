"""Video preprocessing: fps subsampling, resize (max_side), optional first-N-seconds clip."""
from __future__ import annotations
import cv2

class VideoPreprocessor:
    def __init__(self, fps=None, max_side=0, clip_seconds=None):
        self.fps = fps
        self.max_side = int(max_side or 0)
        self.clip_seconds = clip_seconds

    def meta(self, path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        return fps, total

    def iter_frames(self, path):
        """Yields (frame_index, time_sec, frame_bgr) for kept frames."""
        cap = cv2.VideoCapture(path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        n_src = total
        if self.clip_seconds:
            lim = int(round(src_fps * self.clip_seconds))
            n_src = min(total, lim) if total else lim
        tgt = self.fps or src_fps
        step = max(1, int(round(src_fps / tgt)))
        idx = 0
        while True:
            ok, f = cap.read()
            if not ok or (n_src and idx >= n_src):
                break
            if idx % step == 0:
                if self.max_side:
                    h, w = f.shape[:2]
                    sc = self.max_side / max(h, w)
                    if sc < 1.0:
                        f = cv2.resize(f, (int(w * sc), int(h * sc)))
                yield idx, idx / src_fps, f
            idx += 1
        cap.release()
