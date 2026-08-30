#!/bin/bash
#SBATCH --job-name=ots_canonical_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00

source /etc/profile >/dev/null 2>&1 || true
module purge
module load miniforge3/24.1
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x

set -euo pipefail

PROJECT_ROOT=/home/bingxing2/home/scx6090/yhh/ots
cd "$PROJECT_ROOT"

for required_name in OTS_RUN_ID OTS_RUN_DIR OTS_DATASET_ID OTS_CHECKPOINT_ID; do
    if [[ -z "${!required_name:-}" ]]; then
        echo "Missing required environment variable: $required_name" >&2
        exit 2
    fi
done

case "$OTS_RUN_DIR" in
    ""|.|..|*/*|*\\*)
        echo "OTS_RUN_DIR must be one safe runs/ child name." >&2
        exit 2
        ;;
esac

export PYTHONPATH=/home/bingxing2/apps/anaconda/2021.11/envs/h310/lib/python3.10/site-packages
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RUN_ID=$OTS_RUN_ID"
echo "RUN_DIR=runs/$OTS_RUN_DIR"
echo "DATASET_ID=$OTS_DATASET_ID"
echo "CHECKPOINT_ID=$OTS_CHECKPOINT_ID"
module list
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)"

command=(
    python main.py all
    --run-id "$OTS_RUN_ID"
    --run-dir "runs/$OTS_RUN_DIR"
    --dataset-id "$OTS_DATASET_ID"
    --checkpoint-id "$OTS_CHECKPOINT_ID"
    --train-samples "${OTS_TRAIN_SAMPLES:-5000}"
    --test-samples "${OTS_TEST_SAMPLES:-100}"
    --seed "${OTS_SEED:-20260710}"
    --validation-fraction "${OTS_VALIDATION_FRACTION:-0.10}"
    --candidates "${OTS_CANDIDATES:-512,256;1024,512}"
    --batch-size "${OTS_BATCH_SIZE:-128}"
    --max-epochs "${OTS_MAX_EPOCHS:-250}"
    --patience "${OTS_PATIENCE:-25}"
    --learning-rate "${OTS_LEARNING_RATE:-1e-3}"
    --weight-decay "${OTS_WEIGHT_DECAY:-1e-5}"
    --raw-loss-weight "${OTS_RAW_LOSS_WEIGHT:-0.10}"
    --gradient-clip-norm "${OTS_GRADIENT_CLIP_NORM:-5.0}"
    --evaluation-batch-size "${OTS_EVALUATION_BATCH_SIZE:-128}"
    --plot-samples "${OTS_PLOT_SAMPLES:-6}"
    --detail-index "${OTS_DETAIL_INDEX:-0}"
    --device "${OTS_DEVICE:-cuda}"
)

if [[ "${OTS_OVERWRITE:-0}" == "1" ]]; then
    command+=(--overwrite)
fi

"${command[@]}"
echo "RUN_END=$(date --iso-8601=seconds)"
