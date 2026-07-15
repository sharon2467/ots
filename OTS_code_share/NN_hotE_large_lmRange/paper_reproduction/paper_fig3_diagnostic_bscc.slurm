#!/bin/bash
#SBATCH --job-name=ots_fig3_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=bscc_paper_fig3_diag_%j.out
#SBATCH --error=bscc_paper_fig3_diag_%j.err

set -euo pipefail

module purge
module load miniforge3/24.1
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x

export PYTHONPATH="/home/bingxing2/apps/anaconda/2021.11/envs/h310/lib/python3.10/site-packages"
export MPLBACKEND=Agg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${OTS_ROOT_DIR:-$PROJECT_ROOT}"
TASK_DIR="$ROOT_DIR/paper_reproduction"
SOURCE_JOB_ID="${OTS_PAPER_SOURCE_JOB_ID:-1347041}"
DATASET_ID="${OTS_PAPER_DATASET_ID:-paper_10gaussian_v1_seed20260715}"
RUN_DIR="$ROOT_DIR/runs/paper_fig3_diagnostic_${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"
cd "$ROOT_DIR"

python "$TASK_DIR/diagnose_paper_fig3_supergaussian.py" \
  --model "$ROOT_DIR/runs/paper_fig3_${SOURCE_JOB_ID}/paper_fig3_model.pt" \
  --test-data "$ROOT_DIR/datasets/${DATASET_ID}/dataset_test.parquet" \
  --output "$RUN_DIR/supergaussian_forward_model_diagnostic.json" \
  --device cuda

sha256sum "$RUN_DIR/supergaussian_forward_model_diagnostic.json" \
  > "$RUN_DIR/SHA256SUMS.txt"

echo "DIAGNOSTIC_RUN_DIR=$RUN_DIR"
