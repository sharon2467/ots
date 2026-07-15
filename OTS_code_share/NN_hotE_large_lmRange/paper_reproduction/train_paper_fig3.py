#!/usr/bin/env python3
"""Train the paper Figure-3 OTS network on the versioned 10-Gaussian data.

The default model and schedule follow the paper text: 1224 -> 5000 -> 5000 ->
160 with LeakyReLU, full-batch Adam, and 3k/6k/10k/100k optimization steps at
1e-4/1e-5/1e-6/1e-7.  Loss is MSE after a differentiable inverse Fourier
transform.  EVDF values are multiplied by c before MSE so that the loss is
dimensionless; the paper reports an O(1e-4) loss but does not document this
scaling convention.
"""

from __future__ import print_function

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_DIM = 1224
FOURIER_MODES = 80
OUTPUT_DIM = 160
SPEED_OF_LIGHT = 2.99792458e8
DEFAULT_SCHEDULE = "3000:1e-4,6000:1e-5,10000:1e-6,100000:1e-7"
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-global-steps", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class PaperOTSModel(nn.Module):
    def __init__(self, hidden_1=5000, hidden_2=5000):
        super(PaperOTSModel, self).__init__()
        self.hidden_1 = int(hidden_1)
        self.hidden_2 = int(hidden_2)
        self.layer_1 = nn.Linear(INPUT_DIM, self.hidden_1)
        self.layer_2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output = nn.Linear(self.hidden_2, OUTPUT_DIM)
        self.activation = nn.LeakyReLU()

    def forward(self, spectra):
        hidden = self.activation(self.layer_1(spectra))
        hidden = self.activation(self.layer_2(hidden))
        return self.output(hidden)


def parse_schedule(text):
    schedule = []
    for item in text.split(","):
        steps_text, learning_rate_text = item.strip().split(":", 1)
        steps = int(steps_text)
        learning_rate = float(learning_rate_text)
        if steps <= 0 or learning_rate <= 0.0:
            raise ValueError("Schedule entries must be positive: {}".format(item))
        schedule.append((steps, learning_rate))
    if not schedule:
        raise ValueError("Training schedule cannot be empty")
    return schedule


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(path):
    values = pq.read_table(str(path)).to_pandas().values
    if values.shape[1] < INPUT_DIM + OUTPUT_DIM or values.shape[0] < 2:
        raise ValueError("Unexpected dataset shape {} in {}".format(values.shape, path))
    values = values[:, : INPUT_DIM + OUTPUT_DIM].astype(np.float32, copy=False)
    metadata = values[0]
    physical = values[1:]
    spectra = torch.from_numpy(physical[:, :INPUT_DIM].copy())
    coefficients = torch.from_numpy(physical[:, INPUT_DIM:].copy())
    wavelengths_nm = torch.from_numpy(metadata[:INPUT_DIM].copy())
    k_grid = torch.from_numpy(metadata[INPUT_DIM : INPUT_DIM + FOURIER_MODES].copy())
    if not torch.isfinite(spectra).all() or not torch.isfinite(coefficients).all():
        raise ValueError("Non-finite dataset values in {}".format(path))
    return spectra, coefficients, wavelengths_nm, k_grid


def inverse_velocity_grid(k_grid):
    mode_count = len(k_grid)
    dk = k_grid[2] - k_grid[1]
    dx = 2.0 * math.pi / (mode_count * dk)
    return torch.linspace(-1.0, 1.0, mode_count, dtype=k_grid.dtype) * mode_count * dx / 2.0


def inverse_fourier_matrix(k_grid, velocity):
    """Return a real matrix mapping [Re(F), Im(F)] to real f(v)."""
    differences = k_grid[1:] - k_grid[:-1]
    weights = torch.empty_like(k_grid)
    weights[0] = differences[0] / 2.0
    weights[-1] = differences[-1] / 2.0
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2.0
    phase = k_grid[:, None] * velocity[None, :]
    real_matrix = weights[:, None] * torch.cos(phase) / (2.0 * math.pi)
    imaginary_matrix = -weights[:, None] * torch.sin(phase) / (2.0 * math.pi)
    return torch.cat((real_matrix, imaginary_matrix), dim=0)


def regression_metrics(target, prediction):
    difference = prediction - target
    residual = difference.square().sum()
    total = (target - target.mean()).square().sum().clamp(min=1.0e-30)
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "r2": (1.0 - residual / total).item(),
    }


def evdf_metrics(target, prediction, velocity):
    metrics = regression_metrics(target, prediction)
    difference_norm = torch.linalg.vector_norm(prediction - target, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1).clamp(min=1.0e-30)
    relative = difference_norm / target_norm
    negative = torch.trapz(torch.clamp(-prediction, min=0.0), velocity, dim=1)
    absolute = torch.trapz(prediction.abs(), velocity, dim=1).clamp(min=1.0e-30)
    metrics.update(
        {
            "relative_l2_mean": relative.mean().item(),
            "relative_l2_median": relative.median().item(),
            "negative_mass_fraction_mean": (negative / absolute).mean().item(),
        }
    )
    return metrics


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def save_resume(path, model, optimizer, global_step, history, schedule, args):
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "history": history,
            "schedule": schedule,
            "seed": args.seed,
        },
        path,
    )


def handle_stop_signal(signum, frame):
    del frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("STOP_SIGNAL_RECEIVED signum={}".format(signum), flush=True)


def validate_paper_contract(wavelengths_nm, manifest):
    expected = (
        (0, 349.0),
        (578, 353.0),
        (579, 375.0),
        (882, 500.0),
        (883, 200.0),
        (1223, 340.0),
    )
    for index, expected_value in expected:
        if abs(wavelengths_nm[index].item() - expected_value) > 1.0e-3:
            raise ValueError(
                "Paper wavelength contract mismatch at {}: {} vs {}".format(
                    index, wavelengths_nm[index].item(), expected_value
                )
            )
    if manifest.get("family") != "paper_10gaussian":
        raise ValueError("Dataset manifest is not the paper_10gaussian family")
    if manifest.get("gaussians_per_evdf") != 10:
        raise ValueError("Dataset manifest does not specify ten Gaussians")


def self_test():
    k_grid = torch.linspace(-2.0, 2.0, FOURIER_MODES)
    velocity = inverse_velocity_grid(k_grid)
    matrix = inverse_fourier_matrix(k_grid, velocity)
    assert tuple(matrix.shape) == (OUTPUT_DIM, FOURIER_MODES)
    model = PaperOTSModel(64, 32)
    spectra = torch.randn(3, INPUT_DIM)
    target = torch.randn(3, OUTPUT_DIM)
    prediction = model(spectra)
    loss = F.mse_loss(prediction.matmul(matrix), target.matmul(matrix))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("PAPER_TRAIN_SELF_TEST_PASSED loss={:.6e}".format(loss.item()))


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    schedule = parse_schedule(args.schedule)
    total_requested_steps = sum(item[0] for item in schedule)
    if args.max_global_steps is not None and args.max_global_steps <= 0:
        raise ValueError("--max-global-steps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    train_x, train_y, wavelengths_nm, k_grid = load_dataset(args.train_data)
    test_x, test_y, test_wavelengths_nm, test_k_grid = load_dataset(args.test_data)
    validate_paper_contract(wavelengths_nm, manifest)
    if not torch.equal(wavelengths_nm, test_wavelengths_nm) or not torch.equal(k_grid, test_k_grid):
        raise ValueError("Training and test metadata grids differ")
    if len(train_x) != manifest["train_physical_rows"] or len(test_x) != manifest["test_physical_rows"]:
        raise ValueError("Dataset row counts disagree with the manifest")

    set_determinism(args.seed)
    device = choose_device(args.device)
    model = PaperOTSModel().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule[0][1])
    velocity = inverse_velocity_grid(k_grid)
    inverse_matrix = inverse_fourier_matrix(k_grid, velocity)
    scaled_inverse_matrix = (inverse_matrix * SPEED_OF_LIGHT).to(device)
    train_x_device = train_x.to(device)
    train_y_device = train_y.to(device)
    target_evdf_scaled = train_y_device.matmul(scaled_inverse_matrix)
    resume_path = args.output_dir / "paper_training_resume.pt"
    history = []
    global_step = 0

    if args.resume is not None:
        resume = torch.load(str(args.resume), map_location=device)
        if resume["schedule"] != schedule:
            raise ValueError("Resume checkpoint schedule differs from requested schedule")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        global_step = int(resume["global_step"])
        history = list(resume["history"])
        print("RESUMED_FROM_STEP={}".format(global_step), flush=True)

    print("DEVICE={}".format(device), flush=True)
    if device.type == "cuda":
        print("GPU={}".format(torch.cuda.get_device_name(device)), flush=True)
    print("PARAMETER_COUNT={}".format(parameter_count), flush=True)
    print("TRAIN_ROWS={} TEST_ROWS={}".format(len(train_x), len(test_x)), flush=True)
    print("SCHEDULE={} TOTAL_STEPS={}".format(schedule, total_requested_steps), flush=True)
    print("LOSS=EVDF_MSE SCALE=c", flush=True)

    signal.signal(signal.SIGTERM, handle_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, handle_stop_signal)

    started = time.time()
    cumulative_before_stage = 0
    last_loss = None
    model.train()
    for stage_index, (stage_steps, learning_rate) in enumerate(schedule, start=1):
        completed_in_stage = max(0, min(stage_steps, global_step - cumulative_before_stage))
        cumulative_after_stage = cumulative_before_stage + stage_steps
        if global_step >= cumulative_after_stage:
            cumulative_before_stage = cumulative_after_stage
            continue
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        print(
            "STAGE_START index={} lr={} completed={} target={}".format(
                stage_index, learning_rate, completed_in_stage, stage_steps
            ),
            flush=True,
        )
        for local_step in range(completed_in_stage, stage_steps):
            optimizer.zero_grad()
            prediction = model(train_x_device)
            predicted_evdf_scaled = prediction.matmul(scaled_inverse_matrix)
            loss = F.mse_loss(predicted_evdf_scaled, target_evdf_scaled)
            loss.backward()
            optimizer.step()
            global_step += 1
            last_loss = loss.item()

            should_log = (
                global_step == 1
                or global_step % args.log_every == 0
                or local_step + 1 == stage_steps
            )
            if should_log:
                elapsed = time.time() - started
                record = {
                    "global_step": global_step,
                    "stage": stage_index,
                    "stage_step": local_step + 1,
                    "learning_rate": learning_rate,
                    "train_evdf_mse_scaled": last_loss,
                    "elapsed_seconds": elapsed,
                }
                history.append(record)
                print("TRAIN_PROGRESS {}".format(json.dumps(record, sort_keys=True)), flush=True)

            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                save_resume(
                    resume_path, model, optimizer, global_step, history, schedule, args
                )
                print("RESUME_CHECKPOINT_SAVED step={}".format(global_step), flush=True)

            if STOP_REQUESTED:
                save_resume(
                    resume_path, model, optimizer, global_step, history, schedule, args
                )
                print("TRAINING_PAUSED_AT_STEP={}".format(global_step), flush=True)
                raise SystemExit(99)
            if args.max_global_steps is not None and global_step >= args.max_global_steps:
                print("MAX_GLOBAL_STEPS_REACHED={}".format(global_step), flush=True)
                break

        cumulative_before_stage = cumulative_after_stage
        if args.max_global_steps is not None and global_step >= args.max_global_steps:
            break

    elapsed_seconds = time.time() - started
    model.eval()
    with torch.no_grad():
        test_prediction = model(test_x.to(device)).cpu()
    test_velocity = velocity.float()
    physical_inverse_matrix = inverse_matrix.float()
    target_evdf = test_y.matmul(physical_inverse_matrix)
    predicted_evdf = test_prediction.matmul(physical_inverse_matrix)
    coefficient_summary = regression_metrics(test_y, test_prediction)
    evdf_summary = evdf_metrics(target_evdf, predicted_evdf, test_velocity)

    finished_full_schedule = global_step >= total_requested_steps
    checkpoint = {
        "model_state": dict(
            (name, tensor.detach().cpu()) for name, tensor in model.state_dict().items()
        ),
        "architecture": [INPUT_DIM, 5000, 5000, OUTPUT_DIM],
        "parameter_count": parameter_count,
        "seed": args.seed,
        "schedule": schedule,
        "completed_steps": global_step,
        "full_schedule_completed": finished_full_schedule,
        "loss": "MSE(c * inverse_fourier(coefficients))",
        "k_grid": k_grid,
        "wavelengths_nm": wavelengths_nm,
        "dataset_manifest": manifest,
        "test_metrics": {"coefficient": coefficient_summary, "evdf": evdf_summary},
    }
    model_path = args.output_dir / "paper_fig3_model.pt"
    atomic_torch_save(checkpoint, model_path)
    history_path = args.output_dir / "paper_training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "architecture": [INPUT_DIM, 5000, 5000, OUTPUT_DIM],
        "parameter_count": parameter_count,
        "schedule": schedule,
        "requested_steps": total_requested_steps,
        "completed_steps": global_step,
        "full_schedule_completed": finished_full_schedule,
        "elapsed_seconds": elapsed_seconds,
        "final_train_evdf_mse_scaled": last_loss,
        "loss_definition": "MSE(c * inverse_fourier(coefficients))",
        "dataset_id": manifest["dataset_id"],
        "dataset_files": manifest["files"],
        "test": {"coefficient": coefficient_summary, "evdf": evdf_summary},
        "artifacts": {
            "model": model_path.name,
            "history": history_path.name,
            "training_curve": "paper_training_curve.png",
        },
    }
    metrics_path = args.output_dir / "paper_training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(
        [item["global_step"] for item in history],
        [item["train_evdf_mse_scaled"] for item in history],
        color="#2563eb",
    )
    axis.set_xlabel("Full-batch Adam step")
    axis.set_ylabel("EVDF MSE after scaling by c")
    axis.set_title("Paper Figure 3 training schedule")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "paper_training_curve.png", dpi=200)
    plt.close(figure)

    if resume_path.exists() and finished_full_schedule:
        resume_path.unlink()
    print("PAPER_TRAINING_RESULT {}".format(json.dumps(metrics, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
