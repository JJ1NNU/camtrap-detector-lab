#!/bin/bash
# ==============================================================================
# camtrap-detector-lab · SLURM 배치 예시
#   sbatch scripts/submit.sbatch
# 아래 빈칸(""/<...>)은 사용하는 클러스터에 맞게 채우세요. 클러스터마다 다릅니다.
# ==============================================================================
#SBATCH --job-name=camtrap
#SBATCH --partition=              # 예: gpu / a100 / normal  ← 클러스터 파티션명
#SBATCH --account=               # 예: my_project           ← 없으면 이 줄 삭제
#SBATCH --gres=gpu:1             # GPU 1장 (클러스터 표기법에 맞게, 예: gpu:a100:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out  # 로그 경로 (logs/ 폴더 미리 생성)
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- 환경 준비 (클러스터마다 다름: module 또는 conda 중 택1) --------------------
# module load cuda/12.1                       # 예시
# module load python/3.10                     # 예시
# source ~/miniconda3/etc/profile.d/conda.sh  # conda 쓸 경우
# conda activate camtrap                      # 미리 만든 env 이름

# ---- 저장소 루트로 이동 (채우세요) --------------------------------------------
REPO_DIR=""                      # 예: /home/id/camtrap-detector-lab
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"

# 오프라인 노드면 가중치 자동 다운로드가 막히므로 미리 캐시해두고 아래를 켜세요:
# export HF_HUB_OFFLINE=1
# export TORCH_HOME="$REPO_DIR/.cache/torch"

# ---- 실행 ---------------------------------------------------------------------
# 경로/모델 등 세부 설정은 scripts/run_hpc.py 상단 CONFIG 블록에서 채웁니다.
python scripts/run_hpc.py
