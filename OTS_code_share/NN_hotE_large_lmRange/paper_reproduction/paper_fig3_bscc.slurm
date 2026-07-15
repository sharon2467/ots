#!/bin/bash
#SBATCH --job-name=ots_paper_fig3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=bscc_paper_fig3_%j.out
#SBATCH --error=bscc_paper_fig3_%j.err

source /etc/profile >/dev/null 2>&1 || true
module purge
module load miniforge3/24.1
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${OTS_ROOT_DIR:-$PROJECT_ROOT}"
TASK_DIR="$ROOT_DIR/paper_reproduction"
cd "$ROOT_DIR"
export PYTHONPATH="/home/bingxing2/apps/anaconda/2021.11/envs/h310/lib/python3.10/site-packages"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

DATASET_ID="${OTS_PAPER_DATASET_ID:-paper_10gaussian_v1_seed20260715}"
DATA_DIR="${OTS_PAPER_DATA_DIR:-datasets/${DATASET_ID}}"
RUN_DIR="${OTS_PAPER_RUN_DIR:-runs/paper_fig3_${SLURM_JOB_ID}}"
SCHEDULE="${OTS_PAPER_SCHEDULE:-3000:1e-4,6000:1e-5,10000:1e-6,100000:1e-7}"
TRAIN_SEED="${OTS_PAPER_TRAIN_SEED:-20260715}"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "WORKDIR=$PWD"
echo "DATASET_ID=$DATASET_ID"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "SCHEDULE=$SCHEDULE"
module list
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys, torch, numpy, pandas, pyarrow, matplotlib; print('Python', sys.version.split()[0]); print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available()); print('NumPy', numpy.__version__); print('Pandas', pandas.__version__); print('PyArrow', pyarrow.__version__); print('Matplotlib', matplotlib.__version__); assert torch.cuda.is_available()"

python "$TASK_DIR/generate_paper_dataset.py" --output-dir "$DATA_DIR" --self-test
python "$TASK_DIR/train_paper_fig3.py" \
  --train-data "$DATA_DIR/dataset_train.parquet" \
  --test-data "$DATA_DIR/dataset_test.parquet" \
  --dataset-manifest "$DATA_DIR/dataset_manifest.json" \
  --output-dir "$RUN_DIR" \
  --self-test
python "$TASK_DIR/reproduce_paper_fig3.py" \
  --model "$RUN_DIR/paper_fig3_model.pt" \
  --test-data "$DATA_DIR/dataset_test.parquet" \
  --output-dir "$RUN_DIR" \
  --self-test
echo "PAPER_SELF_TESTS_PASSED"

# Register the pre-existing hot-electron dataset/model without moving or
# duplicating the physical Parquet/model files.  These symlinks preserve the
# old root-level commands while making the two data families explicit.
HOT_DATASET_ID="hot_e_large_range_job1346661"
HOT_DATA_DIR="datasets/${HOT_DATASET_ID}"
HOT_MODEL_DIR="models/${HOT_DATASET_ID}/job_1346661"
mkdir -p "$HOT_DATA_DIR" "$HOT_MODEL_DIR"
ln -sfn ../../dataset_train.parquet "$HOT_DATA_DIR/dataset_train.parquet"
ln -sfn ../../dataset_test.parquet "$HOT_DATA_DIR/dataset_test.parquet"
ln -sfn ../../../runs/bscc_v2_1346661/OTS_Te_model_v2.pt "$HOT_MODEL_DIR/model.pt"
ln -sfn ../../../runs/bscc_v2_1346661/metrics_v2.json "$HOT_MODEL_DIR/metrics.json"
python - "$HOT_DATA_DIR/dataset_manifest.json" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
root = Path.cwd()

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

files = {}
for name in ("dataset_train.parquet", "dataset_test.parquet"):
    path = root / name
    files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
manifest = {
    "dataset_id": "hot_e_large_range_job1346661",
    "family": "hot_e_large_wavelength_range",
    "registered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "generator": "genDataSet.py",
    "seed": None,
    "physical_layout": "root-level files retained; version directory uses symlinks",
    "model_run": "runs/bscc_v2_1346661",
    "files": files,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ -s "$DATA_DIR/dataset_train.parquet" && -s "$DATA_DIR/dataset_test.parquet" && -s "$DATA_DIR/dataset_manifest.json" ]]; then
    echo "USING_EXISTING_PAPER_DATASET"
elif [[ ! -e "$DATA_DIR/dataset_train.parquet" && ! -e "$DATA_DIR/dataset_test.parquet" && ! -e "$DATA_DIR/dataset_manifest.json" ]]; then
    mkdir -p "$DATA_DIR"
    python "$TASK_DIR/generate_paper_dataset.py" \
      --output-dir "$DATA_DIR" \
      --dataset-id "$DATASET_ID" \
      --train-samples 5000 \
      --test-samples 100 \
      --seed 20260715 \
      --progress-every 100
else
    echo "PARTIAL_PAPER_DATASET_FOUND; refusing automatic overwrite" >&2
    find "$DATA_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort >&2
    exit 2
fi

python - "$DATA_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import pandas as pd
import sys

data_dir = Path(sys.argv[1])
manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
assert manifest["family"] == "paper_10gaussian"
assert manifest["gaussians_per_evdf"] == 10
assert manifest["gaussian_center_c"] == [-0.2, 0.2]
assert manifest["gaussian_sigma_c"] == [0.02, 0.08]
assert manifest["gaussian_amplitude"] == [0.5, 1.5]

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
print("PAPER_DATASET_VALIDATED", manifest["dataset_id"])
PY

mkdir -p "$RUN_DIR"
train_args=(
  --train-data "$DATA_DIR/dataset_train.parquet"
  --test-data "$DATA_DIR/dataset_test.parquet"
  --dataset-manifest "$DATA_DIR/dataset_manifest.json"
  --output-dir "$RUN_DIR"
  --device cuda
  --seed "$TRAIN_SEED"
  --schedule "$SCHEDULE"
  --log-every 200
  --checkpoint-every 1000
)
if [[ -n "${OTS_PAPER_RESUME:-}" ]]; then
    train_args+=(--resume "$OTS_PAPER_RESUME")
fi
if [[ -n "${OTS_PAPER_MAX_GLOBAL_STEPS:-}" ]]; then
    train_args+=(--max-global-steps "$OTS_PAPER_MAX_GLOBAL_STEPS")
fi
python "$TASK_DIR/train_paper_fig3.py" "${train_args[@]}" &
TRAIN_PID=$!
forward_training_signal() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -USR1 "$TRAIN_PID"
  fi
}
trap forward_training_signal USR1 TERM
set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
trap - USR1 TERM
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  echo "PAPER_TRAINING_EXIT_STATUS=$TRAIN_STATUS" >&2
  exit "$TRAIN_STATUS"
fi

python "$TASK_DIR/reproduce_paper_fig3.py" \
  --model "$RUN_DIR/paper_fig3_model.pt" \
  --test-data "$DATA_DIR/dataset_test.parquet" \
  --output-dir "$RUN_DIR" \
  --test-indices 0,1 \
  --device cuda

python - "$RUN_DIR/paper_training_metrics.json" <<'PY'
import json
from pathlib import Path
import sys

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics["requested_steps"] == 119000:
    assert metrics["full_schedule_completed"]
    assert metrics["completed_steps"] == 119000
print("PAPER_TRAINING_VALIDATED", metrics["completed_steps"])
PY

(
  cd "$RUN_DIR"
  sha256sum paper_fig3_model.pt paper_training_metrics.json \
    paper_training_history.json paper_training_curve.png \
    paper_fig3_reproduction.png fig3a_test_evdf.png fig3b_test_evdf.png \
    fig3c_supergaussian_m3p5.png fig3d_supergaussian_m5.png \
    paper_fig3_metrics.json > SHA256SUMS.txt
)

if [[ "${OTS_PAPER_REGISTER:-1}" == "1" ]]; then
  PAPER_MODEL_DIR="models/${DATASET_ID}/job_${SLURM_JOB_ID}"
  mkdir -p "$PAPER_MODEL_DIR"
  ln -sfn ../../../"$RUN_DIR"/paper_fig3_model.pt "$PAPER_MODEL_DIR/model.pt"
  ln -sfn ../../../"$RUN_DIR"/paper_training_metrics.json "$PAPER_MODEL_DIR/training_metrics.json"
  ln -sfn ../../../"$RUN_DIR"/paper_fig3_metrics.json "$PAPER_MODEL_DIR/fig3_metrics.json"
  printf '%s\n' "job_${SLURM_JOB_ID}" > "models/${DATASET_ID}/LATEST.txt"
  printf '%s\n' "$SLURM_JOB_ID" > LAST_BSCC_PAPER_FIG3_JOB.txt
  printf '%s\n' "$DATA_DIR" > LAST_BSCC_PAPER_DATA_DIR.txt
  printf '%s\n' "$RUN_DIR" > LAST_BSCC_PAPER_FIG3_RESULT_DIR.txt
else
  echo "RESULT_REGISTRATION_SKIPPED"
fi
echo "RESULT_DIR=$RUN_DIR"
find "$DATA_DIR" -maxdepth 1 -type f -printf 'DATA %f %s bytes\n' | sort
find "$RUN_DIR" -maxdepth 1 -type f -printf 'RESULT %f %s bytes\n' | sort
echo "RUN_END=$(date --iso-8601=seconds)"
