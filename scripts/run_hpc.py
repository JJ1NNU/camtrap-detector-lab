#!/usr/bin/env python
"""
================================================================================
camtrap-detector-lab · HPC / 슈퍼컴퓨터 러너  (colab_run.ipynb 의 .py 버전)
================================================================================
Colab 전용 부분(드라이브 마운트 / userdata 토큰 / /content clone / 세션 재시작)을
전부 걷어내고, 환경마다 바뀌는 값만 아래 CONFIG 블록에 "" 로 비워 두었습니다.

[사용법]
    # 저장소 루트에서
    export PYTHONPATH=.
    python scripts/run_hpc.py

    # 또는 SLURM 배치 (같은 폴더의 submit.sbatch 참고)
    sbatch scripts/submit.sbatch

[설치는 미리 (로그인 노드 / conda·venv 안에서) 1회만]
    pip install -U numpy==1.26.4 opencv-python-headless==4.10.0.84 PyYAML PytorchWildlife
    # flock_router(YOLO-World + RT-DETR + SAM3) 를 쓸 때만:
    pip install -U "ultralytics>=8.3.237" supervision
    pip install git+https://github.com/ultralytics/CLIP.git
    # facebook sam3 전략을 쓸 때만:
    pip install "sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git"
================================================================================
"""
from __future__ import annotations
import os
import sys

# ==============================================================================
# ⚙️  CONFIG — 환경(HPC/서버)마다 바뀌는 값만 여기서 채우세요. "" 는 비운 곳.
# ==============================================================================

# ---- 저장소 루트 (절대경로). ""면 이 파일 위치로 자동 추정 ----
REPO_DIR = ""                 # 예: "/home/id/camtrap-detector-lab" 또는 "/scratch/id/camtrap-detector-lab"

# ---- 사용할 실험 config (REPO_DIR 기준 상대경로 또는 절대경로) ----
CONFIG_FILE = ""              # 예: "configs/mdv6-yolo10-c_sahi.yaml" / "configs/sam3.yaml" / "configs/flock_router.yaml"

# ---- 입력 영상 폴더 (하위 폴더 재귀 탐색) ----
INPUT_DIR = ""                # 예: "/scratch/id/data/camtrap_videos"

# ---- 결과 저장 루트 (runs/<name>/ 가 이 아래 생성). ""면 <REPO_DIR>/runs ----
OUTPUT_DIR = ""               # 예: "/scratch/id/camtrap_runs"

# ---- 실행 이름. ""면 config yaml 의 experiment.name 사용 ----
EXPERIMENT_NAME = ""          # 예: "mdv6_night_run1"

# ---- 연산 장치 ----
DEVICE = "cuda"               # GPU 없으면 "cpu"

# ---- (flock_router 전용) 야간 SAM3 가중치 sam3.pt 절대경로. ""면 야간은 MD 폴백 ----
SAM3_WEIGHTS = ""             # 예: "/scratch/id/weights/sam3.pt"

# ---- gated HF 다운로드용 토큰 (configs/sam3.yaml = facebook/sam3 를 쓸 때 필요) ----
#      먼저 huggingface.co/facebook/sam3 에서 접근 승인을 받아야 함.
HF_TOKEN = ""                 # 예: "hf_xxxxxxxx"  (오프라인 노드면 아래 SAM3_CHECKPOINT 사용 권장)

# ---- (configs/sam3.yaml 전용) 로컬 facebook-sam3 이미지 체크포인트 .pt ----
#      지정하면 HF 다운로드 없이 이 파일을 사용 (외부망 차단 노드에서 필수) ----
SAM3_CHECKPOINT = ""          # 예: "/scratch/id/weights/sam3_image.pt"

# ---- GitHub 실시간 push. HPC 컴퓨트 노드는 보통 외부망 차단 → 기본 off ----
ENABLE_GIT = False            # True 로 켤 경우 remote 인증(토큰/SSH)이 노드에서 되는지 먼저 확인

# ==============================================================================
#  이 아래는 보통 수정할 필요 없음
# ==============================================================================


def _resolve_repo_dir() -> str:
    if REPO_DIR:
        return os.path.abspath(REPO_DIR)
    # 이 파일: <repo>/scripts/run_hpc.py  → 상위의 상위가 저장소 루트
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _set_device_recursive(obj, device):
    """cfg 트리(dict/list) 안의 모든 'device' 키를 DEVICE 로 통일 (하위 모델 포함)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "device" and isinstance(v, str):
                obj[k] = device
            else:
                _set_device_recursive(v, device)
    elif isinstance(obj, list):
        for v in obj:
            _set_device_recursive(v, device)


def build_resolved_config(repo_dir: str):
    """base yaml 을 읽어 환경별 override 를 얹은 dict + 저장 경로를 돌려준다."""
    from camtrap_lab.config import load_config, deep_update, dump_config

    if not CONFIG_FILE:
        sys.exit("[run_hpc] CONFIG_FILE 을 지정하세요 (예: configs/mdv6-yolo10-c_sahi.yaml)")
    cfg_path = CONFIG_FILE if os.path.isabs(CONFIG_FILE) else os.path.join(repo_dir, CONFIG_FILE)
    if not os.path.isfile(cfg_path):
        sys.exit(f"[run_hpc] config 파일 없음: {cfg_path}")
    base = load_config(cfg_path)

    # --- 환경별 override (비운 값은 건드리지 않음) ---
    ov = {}
    if INPUT_DIR:
        ov.setdefault("data", {})["input_dir"] = INPUT_DIR
    if OUTPUT_DIR:
        ov.setdefault("experiment", {})["output_dir"] = OUTPUT_DIR
    if EXPERIMENT_NAME:
        ov.setdefault("experiment", {})["name"] = EXPERIMENT_NAME
    ov.setdefault("git", {})["enabled"] = bool(ENABLE_GIT)
    ov["git"]["repo_dir"] = repo_dir
    cfg = deep_update(base, ov)

    # OUTPUT_DIR 를 안 줬으면 기본 <REPO_DIR>/runs 로
    if not OUTPUT_DIR and not base.get("experiment", {}).get("output_dir"):
        cfg["experiment"]["output_dir"] = os.path.join(repo_dir, "runs")

    # device 통일 (하위 day_models/night_model 포함)
    if DEVICE:
        _set_device_recursive(cfg, DEVICE)

    # flock_router 야간 SAM3 가중치 경로 주입 (해당 전략일 때만)
    strat = cfg.get("strategy", {}) or {}
    if strat.get("name") == "flock_router" and strat.get("night_model"):
        if SAM3_WEIGHTS:
            strat["night_model"]["weights"] = SAM3_WEIGHTS

    # facebook-sam3 모델: 로컬 체크포인트가 있으면 주입 (HF 다운로드 불필요)
    if cfg.get("model", {}).get("name") == "sam3" and SAM3_CHECKPOINT:
        cfg["model"]["checkpoint_path"] = SAM3_CHECKPOINT

    # 해석된 config 를 저장(재현용). run_experiment 는 파일 경로를 받으므로 필요.
    out_root = cfg["experiment"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)
    resolved_path = os.path.join(out_root, f"_resolved_{cfg['experiment']['name']}.yaml")
    dump_config(cfg, resolved_path)
    return cfg, resolved_path


def main():
    repo_dir = _resolve_repo_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)   # PYTHONPATH 를 안 걸어도 import 되도록

    if HF_TOKEN:                       # gated facebook/sam3 다운로드 인증
        os.environ["HF_TOKEN"] = HF_TOKEN

    # ---- 환경 점검 (Colab notebook 셀 3 대응) ----
    import numpy, cv2, torch
    print(f"[env] numpy {numpy.__version__} | cv2 {cv2.__version__} | "
          f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    assert hasattr(cv2, "VideoCapture"), "cv2 깨짐 → opencv-python-headless 재설치 필요"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        sys.exit("[run_hpc] DEVICE=cuda 인데 GPU 가 안 보입니다. GPU 노드/모듈 로드를 확인하세요.")

    from camtrap_lab.registry import DETECTORS, STRATEGIES
    from camtrap_lab.data.discovery import find_videos
    from camtrap_lab.runner import run_experiment  # side-effect 로 모든 모델/전략 등록
    print(f"[env] detectors={DETECTORS.available()} | strategies={STRATEGIES.available()}")

    # ---- config 해석 + 입력 영상 검증 (셀 4-2 대응) ----
    cfg, resolved_path = build_resolved_config(repo_dir)
    idir = cfg["data"]["input_dir"]
    if not os.path.isdir(idir):
        sys.exit(f"[run_hpc] input_dir 폴더 없음: {idir}  (INPUT_DIR 확인)")
    vids = find_videos(idir, cfg["data"]["video_exts"])
    print(f"[data] input_dir={idir} | 매칭 영상 {len(vids)}개")
    for v in vids[:5]:
        print("   -", v)
    if not vids:
        sys.exit("[run_hpc] 영상 0개 → INPUT_DIR 경로 또는 data.video_exts(대소문자 포함) 확인")

    # ---- 실행 (셀 5 대응) ----
    print(f"[run] config={resolved_path}")
    run_dir = run_experiment(resolved_path)

    # ---- 결과 요약 (셀 6 대응) ----
    tpath = os.path.join(run_dir, "timings.csv")
    if os.path.isfile(tpath):
        print("\n[timings.csv]")
        with open(tpath, encoding="utf-8") as f:
            sys.stdout.write(f.read())
    print(f"\n[done] -> {run_dir}")


if __name__ == "__main__":
    main()
