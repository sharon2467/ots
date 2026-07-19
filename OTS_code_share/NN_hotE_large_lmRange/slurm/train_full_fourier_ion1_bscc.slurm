#!/bin/bash
#SBATCH --job-name=ots_full_i1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=bscc_full_fourier_ion1_%j.out
#SBATCH --error=bscc_full_fourier_ion1_%j.err

source /etc/profile >/dev/null 2>&1 || true
module purge
module load miniforge3/24.1
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${OTS_ROOT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/train_v2.py" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTHONPATH="/home/bingxing2/apps/anaconda/2021.11/envs/h310/lib/python3.10/site-packages"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

DATASET_ID="${OTS_FULL_I1_DATASET_ID:-full_fourier_ion1_v1_seed20260716}"
DATA_DIR="${OTS_FULL_I1_DATA_DIR:-datasets/${DATASET_ID}}"
RUN_DIR="${OTS_FULL_I1_RUN_DIR:-runs/full_fourier_ion1_${SLURM_JOB_ID}}"
TRAIN_SEED="${OTS_FULL_I1_TRAIN_SEED:-20260716}"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "WORKDIR=$PWD"
echo "DATASET_ID=$DATASET_ID"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_DIR=$RUN_DIR"
module list
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys, torch, numpy, pandas, pyarrow, matplotlib; print('Python', sys.version.split()[0]); print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available()); print('NumPy', numpy.__version__); print('Pandas', pandas.__version__); print('PyArrow', pyarrow.__version__); print('Matplotlib', matplotlib.__version__)"
if [[ "${SELF_TEST_ONLY:-0}" != "1" ]]; then
    python -c "import torch; assert torch.cuda.is_available(), 'Formal training requires a CUDA-compatible node.'"
fi

echo "Running full-Fourier forward-model self-test"
python genDataSet_full_fourier.py --self-test
echo "Running train_v2 numerical self-test"
python train_v2.py --target-kind ion1 --self-test
echo "FULL_FOURIER_SELF_TESTS_PASSED"

if [[ "${SELF_TEST_ONLY:-0}" == "1" ]]; then
    echo "SELF_TEST_ONLY_COMPLETE"
    exit 0
fi

if [[ -s "$DATA_DIR/dataset_train.parquet" \
      && -s "$DATA_DIR/dataset_test.parquet" \
      && -s "$DATA_DIR/dataset_manifest.json" ]]; then
    echo "USING_EXISTING_FULL_FOURIER_DATASET"
elif [[ ! -e "$DATA_DIR/dataset_train.parquet" \
        && ! -e "$DATA_DIR/dataset_test.parquet" \
        && ! -e "$DATA_DIR/dataset_manifest.json" ]]; then
    mkdir -p "$DATA_DIR"
    python genDataSet_full_fourier.py \
        --output-dir "$DATA_DIR" \
        --dataset-id "$DATASET_ID" \
        --train-samples 5000 \
        --test-samples 100 \
        --seed 20260716 \
        --progress-every 100
else
    echo "PARTIAL_FULL_FOURIER_DATASET_FOUND; refusing automatic overwrite" >&2
    find "$DATA_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort >&2
    exit 2
fi

python - "$DATA_DIR" "$DATASET_ID" <<'PY'
import hashlib
import json
from pathlib import Path
import pandas as pd
import sys

data_dir = Path(sys.argv[1])
dataset_id = sys.argv[2]
manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
assert manifest["dataset_id"] == dataset_id
assert manifest["family"] == "full_fourier_variable_ion1"
assert manifest["target"]["species"] == "ion1"
assert manifest["fixed_distributions"]["electron"]["family"] == "Maxwellian"
assert manifest["fixed_distributions"]["ion2"]["family"] == "Maxwellian"

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

train_path = data_dir / "dataset_train.parquet"
test_path = data_dir / "dataset_test.parquet"
assert pd.read_parquet(train_path).shape == (5001, 1384)
assert pd.read_parquet(test_path).shape == (101, 1384)
assert digest(train_path) == manifest["files"][train_path.name]["sha256"]
assert digest(test_path) == manifest["files"][test_path.name]["sha256"]
print("FULL_FOURIER_DATASET_VALIDATED", dataset_id)
PY

mkdir -p "$RUN_DIR"
cp "$DATA_DIR/dataset_manifest.json" "$RUN_DIR/dataset_manifest.json"

python train_v2.py \
    --train-data "$DATA_DIR/dataset_train.parquet" \
    --test-data "$DATA_DIR/dataset_test.parquet" \
    --dataset-manifest "$DATA_DIR/dataset_manifest.json" \
    --output-dir "$RUN_DIR" \
    --target-kind ion1 \
    --device cuda \
    --seed "$TRAIN_SEED" \
    --validation-fraction 0.10 \
    --batch-size 128 \
    --max-epochs 250 \
    --patience 25 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --raw-loss-weight 1.0 \
    --candidates '1024,512;5000,5000' \
    --skip-legacy

python - "$RUN_DIR/predictions_v2.npz" "$RUN_DIR/ion1_test_diagnostics.json" "$RUN_DIR/ion1_worst_cases.png" <<'PY'
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

predictions_path = Path(sys.argv[1])
diagnostics_path = Path(sys.argv[2])
figure_path = Path(sys.argv[3])
payload = np.load(predictions_path)
target = payload["target_coefficients"].astype(np.float64)
predicted = payload["v2_coefficients"].astype(np.float64)
k_grid = payload["k_grid"].astype(np.float64)
mode_count = len(k_grid)
dk = k_grid[2] - k_grid[1]
dx = 2.0 * math.pi / (mode_count * dk)
velocity = np.linspace(-1.0, 1.0, mode_count) * mode_count * dx / 2.0
weights = np.ones(mode_count)
weights[[0, -1]] = 0.5
phase = np.outer(k_grid, velocity)
cosine = weights[:, None] * np.cos(phase)
sine = weights[:, None] * np.sin(phase)

def reconstruct(coefficients):
    real = coefficients[:, :mode_count]
    imag = coefficients[:, mode_count:]
    return (real @ cosine - imag @ sine) * dk / (2.0 * math.pi)

truth = reconstruct(target)
estimate = reconstruct(predicted)
error = estimate - truth
relative_l2 = np.sqrt(
    np.sum(error**2, axis=1) / np.maximum(np.sum(truth**2, axis=1), 1.0e-30)
)
sample_r2 = 1.0 - np.sum(error**2, axis=1) / np.maximum(
    np.sum((truth - truth.mean(axis=1, keepdims=True))**2, axis=1), 1.0e-30
)
negative_area = np.trapz(np.maximum(-estimate, 0.0), velocity, axis=1)
absolute_area = np.maximum(np.trapz(np.abs(estimate), velocity, axis=1), 1.0e-30)
integral = np.trapz(estimate, velocity, axis=1)
worst_indices = np.argsort(relative_l2)[-6:][::-1]
best_indices = np.argsort(relative_l2)[:6]

diagnostics = {
    "sample_count": int(len(target)),
    "relative_l2": {
        "mean": float(relative_l2.mean()),
        "median": float(np.median(relative_l2)),
        "p90": float(np.quantile(relative_l2, 0.90)),
        "p95": float(np.quantile(relative_l2, 0.95)),
        "max": float(relative_l2.max()),
    },
    "sample_r2": {
        "mean": float(sample_r2.mean()),
        "median": float(np.median(sample_r2)),
        "min": float(sample_r2.min()),
    },
    "negative_mass_fraction": {
        "mean": float(np.mean(negative_area / absolute_area)),
        "max": float(np.max(negative_area / absolute_area)),
    },
    "reconstructed_integral": {
        "mean": float(integral.mean()),
        "standard_deviation": float(integral.std()),
        "minimum": float(integral.min()),
        "maximum": float(integral.max()),
    },
    "worst_indices": [int(value) for value in worst_indices],
    "best_indices": [int(value) for value in best_indices],
    "per_sample": [
        {
            "index": int(index),
            "relative_l2": float(relative_l2[index]),
            "r2": float(sample_r2[index]),
            "negative_mass_fraction": float(
                negative_area[index] / absolute_area[index]
            ),
            "integral": float(integral[index]),
        }
        for index in range(len(target))
    ],
}
diagnostics_path.write_text(
    json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

figure, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
for axis, index in zip(axes.flat, worst_indices):
    axis.plot(velocity, truth[index], color="black", label="truth")
    axis.plot(velocity, estimate[index], color="#2563eb", linestyle="--", label="v2")
    axis.set_title(
        "test {}  relative L2={:.3f}  R2={:.3f}".format(
            int(index), relative_l2[index], sample_r2[index]
        )
    )
    axis.set_xlabel("velocity (m/s)")
    axis.set_ylabel("ion-1 VDF")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig(figure_path, dpi=170)
plt.close(figure)
print("ION1_TEST_DIAGNOSTICS_COMPLETE", json.dumps(diagnostics["relative_l2"], sort_keys=True))
PY

python - "$RUN_DIR" "$DATASET_ID" "$DATA_DIR" <<'PY'
import datetime
import json
import os
from pathlib import Path
import platform
import sys

import torch

run_dir = Path(sys.argv[1])
dataset_id = sys.argv[2]
data_dir = sys.argv[3]
metrics = json.loads((run_dir / "metrics_v2.json").read_text(encoding="utf-8"))
manifest = {
    "format_version": 1,
    "workflow": "full_fourier_variable_ion1_v2",
    "job_id": os.environ["SLURM_JOB_ID"],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": platform.node(),
    "dataset_id": dataset_id,
    "dataset_dir": data_dir,
    "run_dir": str(run_dir),
    "target_kind": metrics["target_kind"],
    "selected_candidate": metrics["selected_candidate"]["hidden_sizes"],
    "best_epoch": metrics["selected_candidate"]["best_epoch"],
    "raw_loss_weight": metrics["raw_loss_weight"],
    "test_metrics": metrics["v2"],
    "environment": {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    },
}
(run_dir / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

(
    cd "$RUN_DIR"
    sha256sum \
        "$PROJECT_ROOT/phyEqns/ots_full_fourier.py" \
        "$PROJECT_ROOT/genDataSet_full_fourier.py" \
        "$PROJECT_ROOT/train_v2.py" \
        "$PROJECT_ROOT/slurm/train_full_fourier_ion1_bscc.slurm" \
        > SOURCE_SHA256SUMS.txt
    sha256sum \
        OTS_i1_model_v2.pt \
        metrics_v2.json \
        predictions_v2.npz \
        training_history_v2.json \
        learning_curves_v2.png \
        comparison_v2.png \
        dataset_manifest.json \
        ion1_test_diagnostics.json \
        ion1_worst_cases.png \
        run_manifest.json \
        SOURCE_SHA256SUMS.txt \
        > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt
)

MODEL_DIR="models/${DATASET_ID}/job_${SLURM_JOB_ID}"
mkdir -p "$MODEL_DIR"
ln -sfn ../../../"$RUN_DIR"/OTS_i1_model_v2.pt "$MODEL_DIR/model.pt"
ln -sfn ../../../"$RUN_DIR"/metrics_v2.json "$MODEL_DIR/metrics.json"
ln -sfn ../../../"$RUN_DIR"/run_manifest.json "$MODEL_DIR/run_manifest.json"
printf '%s\n' "job_${SLURM_JOB_ID}" > "models/${DATASET_ID}/LATEST.txt"
printf '%s\n' "$SLURM_JOB_ID" > LAST_BSCC_FULL_FOURIER_ION1_JOB.txt
printf '%s\n' "$DATA_DIR" > LAST_BSCC_FULL_FOURIER_ION1_DATA_DIR.txt
printf '%s\n' "$RUN_DIR" > LAST_BSCC_FULL_FOURIER_ION1_RESULT_DIR.txt

echo "RESULT_DIR=$RUN_DIR"
find "$DATA_DIR" -maxdepth 1 -type f -printf 'DATA %f %s bytes\n' | sort
find "$RUN_DIR" -maxdepth 1 -type f -printf 'RESULT %f %s bytes\n' | sort
echo "RUN_END=$(date --iso-8601=seconds)"
