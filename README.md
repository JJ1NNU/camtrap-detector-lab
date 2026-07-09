# camtrap-detector-lab

A **factory-pattern, config-driven** lab for benchmarking wildlife **detectors** on
camera-trap video — built to run on **Google Colab**, read data from **Google Drive**,
save every output, and **push results to GitHub in real time**.

Models supported out of the box: **MegaDetector v5a**, **MegaDetector v6**, **SAM 3**
(image concept detector). Inference strategies: **whole-image** and **SAHI tiling**.

---

## 1. Why this structure

Everything is driven by a YAML file. Code is split by responsibility, and objects are
built by a **registry/factory** — so there are **no `if model == ...` chains** anywhere.
Adding a new model or a new inference method is a one-file change + a decorator.

```
camtrap_lab/
  registry.py          # generic name->class registry (the factory)
  config.py            # YAML load / merge / dump
  data/
    video.py           # VideoPreprocessor: fps, resize (max_side), clip_seconds
    discovery.py       # find videos under a Drive folder
  models/
    base.py            # Detector ABC + Detection dataclass
    megadetector.py    # mdv5a, mdv6  (Pytorch-Wildlife, lazy import)
    sam3.py            # sam3         (official sam3 image model, lazy import)
  inference/
    base.py            # InferenceStrategy ABC
    whole.py           # whole-image
    sahi.py            # SAHI tiling (works for bbox and bbox+mask)
    nms.py             # box NMS merge
  results/
    writer.py          # append-only CSV: config + timings + ALL outputs
    visualizer.py      # optional annotated mp4 (bbox + mask)
  utils/
    mask.py            # mask encoding: area / polygon / RLE
    gitsync.py         # commit + push per video
  runner.py            # orchestrates one experiment from a YAML
configs/               # example experiments (edit these)
scripts/run_experiment.py
notebooks/colab_run.ipynb
```

> Note: the referenced Notion design doc requires login, so this layout follows standard
> clean-architecture / factory conventions. If your doc mandates specific names, rename
> the registry keys / modules — nothing else depends on hardcoded strings.

---

## 2. Quick start (Colab)

Open `notebooks/colab_run.ipynb` and run top-to-bottom. It will:

1. Mount Google Drive.
2. Clone **your fork** of this repo into `/content/camtrap-detector-lab` using a GitHub token.
3. Install dependencies (and SAM 3 if needed), then **restart** the runtime.
4. Run an experiment: `!python scripts/run_experiment.py --config configs/mdv6_sahi.yaml`.
5. Results are written under `runs/<experiment_name>/` **and pushed to GitHub after each video**.

### GitHub token
Create a **fine-grained PAT** with `contents: read/write` on your fork. In Colab, store it
via `from google.colab import userdata` (Secrets, key `GITHUB_TOKEN`) — the notebook reads it
and sets the authenticated remote. Never hardcode the token in a config.

---

## 3. Configuring experiments (YAML)

Every knob lives in the YAML — nothing is hardcoded.

### `data.preprocess`
| key | meaning |
|---|---|
| `fps` | target sampling fps (`null` = source fps) |
| `max_side` | resize so the long side = this many px (`0` = original) |
| `clip_seconds` | only the first N seconds (`null` = whole video) |

### `model` (choose one `name`)
- **`mdv5a`** — `device`, `conf`, `version` (`"a"`/`"b"`).
- **`mdv6`** — `device`, `conf`, `version` (`MDV6-yolov9-c/e`, `MDV6-yolov10-c/e`, `MDV6-rtdetr-c`).
- **`sam3`** — `device`, `conf`, `prompts` (list of concepts), `autocast_dtype` (`float16` on T4),
  `checkpoint_path` (`null` = HF `facebook/sam3`, gated), `bpe_path`.

### `strategy` (choose one `name`)
- **`whole`** — no params.
- **`sahi`** — `tile_size`, `overlap`, `full_frame_pass`, `nms_iou`.

### `results`
| key | meaning |
|---|---|
| `save_masks` | `none` / `area` / `polygon` / `rle` (segmentation storage) |
| `save_viz` | write an annotated mp4 per video |
| `push_every_video` | git push after each video (real-time) |

### `git`
`enabled`, `repo_dir`, `remote`, `branch`, `author_name`, `author_email`.

---


## 4. What gets saved  (`runs/<name>/`)

- **`config.yaml`** — the exact resolved config used.
- **`timings.csv`** — one row per video: `frames, total_infer_sec, sec_per_frame, total_detections`.
- **`detections.csv`** — one row per detection with **all model outputs**:
  `video, frame_idx, frame_time_sec, model, strategy, det_id, label, class_id, score,
  x1,y1,x2,y2, cx,cy,w,h, mask_area, mask_json` (mask_json holds polygon/RLE per `save_masks`).
- **`viz/*.mp4`** — optional annotated videos (bbox + mask).

CSVs are **append-only** (crash-safe) and pushed to GitHub after each video.

---

## 5. Add a new model (factory pattern)

Create `camtrap_lab/models/mymodel.py`:

```python
from .base import Detector, Detection
from ..registry import DETECTORS

@DETECTORS.register("mymodel")
class MyModel(Detector):
    name = "mymodel"
    produces_masks = False
    def __init__(self, device="cuda", conf=0.25, **kw):
        ...
    def detect(self, image_bgr):
        return [Detection((x1,y1,x2,y2), score, label)]
```

Import it in `runner.py` (next to the others), then set `model.name: mymodel` in a YAML.
Same pattern for a new `strategy` via `STRATEGIES.register(...)`.

---

## 6. Troubleshooting

- **numpy / opencv conflict**: SAM 3 pins `numpy<2`; Colab's default opencv wants `numpy>=2`.
  The notebook installs `opencv-python-headless==4.10.0.84` + `numpy==1.26.4` and **restarts**.
- **Pytorch-Wildlife API drift**: if `MegaDetectorV5/V6` constructor or `single_image_detection`
  differs in your PW version, edit only `models/megadetector.py`.
- **SAM 3 gated weights**: request access to `facebook/sam3` on Hugging Face and `hf auth login`.
- **SAM 3.1**: intentionally not used — the public multiplex checkpoint currently fails to load
  (`offload_state_to_cpu`, issue #526). Stick to `sam3`.
