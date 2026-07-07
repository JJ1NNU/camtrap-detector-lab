"""Crash-safe, append-only result writer. Saves config, per-video timing, and ALL outputs."""
from __future__ import annotations
import csv, json, os
from ..config import dump_config
from ..utils.mask import encode_mask

class ResultWriter:
    def __init__(self, run_dir, save_masks="polygon"):
        self.run_dir = run_dir
        self.save_masks = save_masks
        os.makedirs(run_dir, exist_ok=True)
        self.det_path = os.path.join(run_dir, "detections.csv")
        self.time_path = os.path.join(run_dir, "timings.csv")
        self._det_fields = ["video", "frame_idx", "frame_time_sec", "model", "strategy",
                            "det_id", "label", "class_id", "score",
                            "x1", "y1", "x2", "y2", "cx", "cy", "w", "h",
                            "mask_area", "mask_json"]
        self._time_fields = ["video", "frames", "total_infer_sec", "sec_per_frame", "total_detections"]
        self._init(self.det_path, self._det_fields)
        self._init(self.time_path, self._time_fields)

    def _init(self, path, fields):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def save_config(self, cfg):
        dump_config(cfg, os.path.join(self.run_dir, "config.yaml"))

    def write_frame(self, video, model, strategy, frame_idx, frame_time, dets):
        rows = []
        for j, d in enumerate(dets):
            x1, y1, x2, y2 = d.bbox
            enc = encode_mask(d.mask, self.save_masks)
            rows.append({
                "video": video, "frame_idx": frame_idx, "frame_time_sec": round(frame_time, 3),
                "model": model, "strategy": strategy, "det_id": j,
                "label": d.label, "class_id": d.class_id, "score": round(float(d.score), 4),
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                "cx": round((x1 + x2) / 2, 1), "cy": round((y1 + y2) / 2, 1),
                "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                "mask_area": (enc or {}).get("area"),
                "mask_json": json.dumps(enc, ensure_ascii=False) if enc else "",
            })
        if rows:
            with open(self.det_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._det_fields)
                for r in rows:
                    w.writerow(r)
        return len(rows)

    def write_timing(self, video, frames, total_sec, total_dets):
        with open(self.time_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._time_fields).writerow({
                "video": video, "frames": frames, "total_infer_sec": round(total_sec, 3),
                "sec_per_frame": round(total_sec / max(1, frames), 4),
                "total_detections": total_dets})
