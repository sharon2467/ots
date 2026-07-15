#!/bin/bash
#SBATCH --job-name=ots_v2_capacity
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=train_v2_capacity_%j.out
#SBATCH --error=train_v2_capacity_%j.err

set -euo pipefail

module purge
module load python/3.10.10-intel-2021.4.0
module load pytorch/1.6.0
module list

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/.python-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

OUTPUT_DIR="runs/v2_capacity_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

date
echo "Running v2 numerical self-test before the capacity experiment"
python train_v2.py --self-test
echo "SELF_TEST_PASSED; starting capacity experiment"

python train_v2.py \
  --train-data dataset_train.parquet \
  --test-data dataset_test.parquet \
  --legacy-model OTS_Te_model.pickle \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  --batch-size 128 \
  --max-epochs 350 \
  --patience 40 \
  --candidates '2048,1024;5000,5000'

sha256sum \
  "$OUTPUT_DIR/OTS_Te_model_v2.pt" \
  "$OUTPUT_DIR/metrics_v2.json" \
  "$OUTPUT_DIR/training_history_v2.json" \
  "$OUTPUT_DIR/predictions_v2.npz" \
  "$OUTPUT_DIR/learning_curves_v2.png" \
  "$OUTPUT_DIR/comparison_v2.png"
date
