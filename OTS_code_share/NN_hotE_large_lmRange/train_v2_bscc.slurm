#!/bin/bash
#SBATCH --job-name=ots_v2_bscc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=bscc_train_v2_%j.out
#SBATCH --error=bscc_train_v2_%j.err

source /etc/profile >/dev/null 2>&1 || true
module purge
module load miniforge3/24.1
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="/home/bingxing2/apps/anaconda/2021.11/envs/h310/lib/python3.10/site-packages"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "WORKDIR=$PWD"
module list
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys, torch, numpy, pandas, pyarrow, matplotlib; print('Python', sys.version.split()[0]); print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available()); print('NumPy', numpy.__version__); print('Pandas', pandas.__version__); print('PyArrow', pyarrow.__version__); print('Matplotlib', matplotlib.__version__); assert torch.cuda.is_available()"

echo "Running train_v2 numerical self-test"
python train_v2.py --self-test
echo "SELF_TEST_PASSED"

if [[ "${SELF_TEST_ONLY:-0}" == "1" ]]; then
    echo "SELF_TEST_ONLY_COMPLETE"
    exit 0
fi

if [[ ! -s dataset_train.parquet || ! -s dataset_test.parquet ]]; then
    echo "DATA_GENERATION_REQUIRED"
    python genDataSet.py
else
    echo "USING_EXISTING_DATASETS"
fi

python -c "import pandas as pd; train=pd.read_parquet('dataset_train.parquet'); test=pd.read_parquet('dataset_test.parquet'); print('DATASET_SHAPES', train.shape, test.shape); assert train.shape == (5001, 1384); assert test.shape == (101, 1384); assert train.iloc[1:].notna().all().all(); assert test.iloc[1:].notna().all().all()"

output_dir="runs/bscc_v2_${SLURM_JOB_ID}"
mkdir -p "$output_dir"

train_args=(
    --train-data dataset_train.parquet
    --test-data dataset_test.parquet
    --output-dir "$output_dir"
    --device cuda
    --seed 20260710
    --validation-fraction 0.10
    --batch-size 128
    --max-epochs 250
    --patience 25
    --learning-rate 1e-3
    --weight-decay 1e-5
    --candidates '512,256;1024,512'
)

if [[ -s OTS_Te_model.pickle ]]; then
    echo "LEGACY_MODEL_PRESENT"
    train_args+=(--legacy-model OTS_Te_model.pickle)
else
    echo "LEGACY_MODEL_ABSENT; using --skip-legacy"
    train_args+=(--skip-legacy)
fi

python train_v2.py "${train_args[@]}"

(
    cd "$output_dir"
    sha256sum OTS_Te_model_v2.pt metrics_v2.json predictions_v2.npz \
        training_history_v2.json learning_curves_v2.png comparison_v2.png \
        > SHA256SUMS.txt
)

printf '%s\n' "$SLURM_JOB_ID" > LAST_BSCC_V2_JOB.txt
printf '%s\n' "$output_dir" > LAST_BSCC_V2_RESULT_DIR.txt
echo "RESULT_DIR=$output_dir"
find "$output_dir" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo "RUN_END=$(date --iso-8601=seconds)"
