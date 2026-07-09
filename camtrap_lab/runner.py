"""Experiment orchestrator: builds objects from YAML via the factory and runs everything."""
from __future__ import annotations
import os, time
from .config import load_config
from .registry import DETECTORS, STRATEGIES
from .data.video import VideoPreprocessor
from .data.discovery import find_videos
from .results.writer import ResultWriter
from .results.visualizer import VideoVisualizer, draw
from .utils.gitsync import GitSync
# Import side-effect registrations (no hardcoded dispatch):
from .models import megadetector, sam3, ultralytics_models   # noqa: F401
from .inference import whole, sahi, flock_router             # noqa: F401

def run_experiment(config_path: str) -> str:
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    run_dir = os.path.join(exp["output_dir"], exp["name"])
    rcfg = cfg.get("results", {}) or {}

    writer = ResultWriter(run_dir, save_masks=rcfg.get("save_masks", "polygon"),
                          polygon_max_pts=rcfg.get("mask_polygon_max_pts", 80))
    writer.save_config(cfg)

    pre = cfg["data"].get("preprocess", {}) or {}
    prep = VideoPreprocessor(fps=pre.get("fps"), max_side=pre.get("max_side", 0),
                             clip_seconds=pre.get("clip_seconds"))
    videos = find_videos(cfg["data"]["input_dir"], cfg["data"]["video_exts"])

    detector = DETECTORS.build(dict(cfg["model"]))
    strat_cfg = dict(cfg["strategy"]); strat_cfg["detector"] = detector
    strategy = STRATEGIES.build(strat_cfg)

    save_viz = bool(rcfg.get("save_viz", False))
    viz = VideoVisualizer(
        run_dir,
        fps=rcfg.get("viz_fps", (pre.get("fps") or 8.0)),
        max_side=rcfg.get("viz_max_side", 1280),
        crf=rcfg.get("viz_crf", 28),
        draw_masks=rcfg.get("viz_draw_masks", True),
    ) if save_viz else None

    gcfg = cfg.get("git", {}) or {}
    git = GitSync(gcfg.get("repo_dir", "."), branch=gcfg.get("branch", "main"),
                  remote=gcfg.get("remote", "origin"),
                  author_name=gcfg.get("author_name", "colab-bot"),
                  author_email=gcfg.get("author_email", "colab@example.com"),
                  enabled=gcfg.get("enabled", False))
    push_every = rcfg.get("push_every_video", True)

    model_name = cfg["model"]["name"]; strat_name = cfg["strategy"]["name"]
    print(f"[run] {exp['name']} | videos={len(videos)} | model={model_name} strategy={strat_name}")

    for vp in videos:
        vid = os.path.basename(vp)
        t0 = time.time(); nf = 0; ndet = 0; annotated = []
        for fidx, ftime, frame in prep.iter_frames(vp):
            dets = strategy.run(frame)
            ndet += writer.write_frame(vid, model_name, strat_name, fidx, ftime, dets)
            if viz is not None:
                annotated.append(draw(frame, dets, draw_masks=rcfg.get("viz_draw_masks", True)))
            nf += 1
        total_sec = time.time() - t0
        writer.write_timing(vid, nf, total_sec, ndet)
        if viz is not None:
            viz.write(vid, annotated)
        print(f"  {vid}: frames={nf} dets={ndet} {total_sec:.1f}s")
        if push_every and git.enabled:
            ok = git.commit_push(f"results: {exp['name']}/{vid}")
            print(f"    git push: {'ok' if ok else 'skip/fail'}")

    if git.enabled:
        git.commit_push(f"results: {exp['name']} complete")
    print(f"[done] -> {run_dir}")
    return run_dir
