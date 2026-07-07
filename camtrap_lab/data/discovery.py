from __future__ import annotations
import glob, os

def find_videos(input_dir, exts):
    exts = tuple(exts)
    paths = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
    return sorted(p for p in paths if p.endswith(exts))
