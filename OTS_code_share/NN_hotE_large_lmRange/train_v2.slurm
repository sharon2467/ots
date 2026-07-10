#!/bin/bash
#SBATCH --job-name=ots_v2
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=train_v2_%j.out
#SBATCH --error=train_v2_%j.err

set -euo pipefail

module purge
module load python/3.10.10-intel-2021.4.0
module load pytorch/1.6.0
module list

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/.python-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUTPUT_DIR="runs/v2_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

date
python train_v2.py \
  --train-data dataset_train.parquet \
  --test-data dataset_test.parquet \
  --legacy-model OTS_Te_model.pickle \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  --batch-size 128 \
  --max-epochs 250 \
  --patience 25 \
  --candidates '512,256;1024,512'

cp "$OUTPUT_DIR/OTS_Te_model_v2.pt" ./OTS_Te_model_v2.pt
cp "$OUTPUT_DIR/metrics_v2.json" ./metrics_v2.json
cp "$OUTPUT_DIR/comparison_v2.png" ./comparison_v2.png
date
