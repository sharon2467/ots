#!/bin/bash
#SBATCH --job-name=ots_v2_rawloss
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=train_v2_rawloss_%j.out
#SBATCH --error=train_v2_rawloss_%j.err

set -euo pipefail

module purge
module load python/3.10.10-intel-2021.4.0
module load pytorch/1.6.0
module list

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/.python-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

OUTPUT_DIR="runs/v2_rawloss_${SLURM_JOB_ID}"
RAW_LOSS_WEIGHT="${OTS_RAW_LOSS_WEIGHT:-0.50}"
mkdir -p "$OUTPUT_DIR"

date
echo "Running v2 numerical self-test before the raw-loss experiment"
python train_v2.py --self-test
echo "SELF_TEST_PASSED; starting raw-loss experiment with weight ${RAW_LOSS_WEIGHT}"

python train_v2.py \
  --train-data dataset_train.parquet \
  --test-data dataset_test.parquet \
  --legacy-model OTS_Te_model.pickle \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  --batch-size 128 \
  --max-epochs 350 \
  --patience 40 \
  --raw-loss-weight "$RAW_LOSS_WEIGHT" \
  --candidates '2048,1024;5000,5000'

sha256sum \
  "$OUTPUT_DIR/OTS_Te_model_v2.pt" \
  "$OUTPUT_DIR/metrics_v2.json" \
  "$OUTPUT_DIR/training_history_v2.json" \
  "$OUTPUT_DIR/predictions_v2.npz" \
  "$OUTPUT_DIR/learning_curves_v2.png" \
  "$OUTPUT_DIR/comparison_v2.png"
date
