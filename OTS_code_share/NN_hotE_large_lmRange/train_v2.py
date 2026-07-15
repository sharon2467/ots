# -*- coding: utf-8 -*-
"""Validated OTS training and legacy/new-model comparison.

This entry point is intentionally compatible with Python 3.6 and the
PyTorch 1.6/1.7 environment used by the cluster module.  It fixes the main
weaknesses of ``main.py`` without changing the legacy pipeline in place:

* every physical row is used (the legacy reader silently drops the last row);
* a deterministic validation split controls scheduling and early stopping;
* mini-batches replace 119,000 full-dataset optimization steps;
* inputs and each Fourier coefficient are standardized from training data;
* predicted Fourier coefficients are projected onto Hermitian symmetry;
* two compact architectures are compared on the same validation split;
* the best checkpoint, metrics, histories, predictions, and plots are saved;
* the legacy pickle is evaluated with a real-valued inverse transform that
  preserves both real and imaginary Fourier coefficients without casting
  complex arrays to float.
"""

from __future__ import print_function

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import load_model as load_legacy_model


INPUT_DIM = 1224
FOURIER_MODES = 80
OUTPUT_DIM = FOURIER_MODES * 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="dataset_train.parquet")
    parser.add_argument("--test-data", default="dataset_test.parquet")
    parser.add_argument("--legacy-model", default="OTS_Te_model.pickle")
    parser.add_argument("--output-dir", default="runs/v2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--raw-loss-weight",
        type=float,
        default=0.10,
        help="Weight of raw coefficient MSE; the remainder weights standardized MSE.",
    )
    parser.add_argument(
        "--candidates",
        default="512,256;1024,512",
        help="Semicolon-separated hidden layer sizes.",
    )
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this job.")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_parquet_dataset(path):
    """Read the legacy metadata-first Parquet layout without dropping rows."""
    table = pq.read_table(path)
    values = table.to_pandas().values
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < INPUT_DIM + OUTPUT_DIM:
        raise ValueError("Unexpected dataset shape in {}: {}".format(path, values.shape))

    values = values[:, : INPUT_DIM + OUTPUT_DIM].astype(np.float32, copy=False)
    metadata = values[0]
    samples = values[1:]
    spectra = torch.from_numpy(samples[:, :INPUT_DIM].copy())
    coefficients = torch.from_numpy(samples[:, INPUT_DIM:].copy())
    wavelengths_nm = torch.from_numpy(metadata[:INPUT_DIM].copy())
    k_grid = torch.from_numpy(metadata[INPUT_DIM : INPUT_DIM + FOURIER_MODES].copy())

    if not torch.isfinite(spectra).all() or not torch.isfinite(coefficients).all():
        raise ValueError("Non-finite values were found in {}".format(path))
    if spectra.size(1) != INPUT_DIM or coefficients.size(1) != OUTPUT_DIM:
        raise ValueError("Feature/label dimensions do not match the OTS model contract.")
    return spectra, coefficients, wavelengths_nm, k_grid


def split_train_validation(spectra, coefficients, fraction, seed):
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(spectra), generator=generator)
    validation_count = max(1, int(round(len(spectra) * fraction)))
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    return (
        spectra[training_indices],
        coefficients[training_indices],
        spectra[validation_indices],
        coefficients[validation_indices],
    )


def normalization_statistics(x_train, y_train):
    x_mean = x_train.mean(dim=0)
    x_scale = x_train.std(dim=0, unbiased=False).clamp(min=1.0e-4)
    y_mean = y_train.mean(dim=0)
    y_scale = y_train.std(dim=0, unbiased=False).clamp(min=1.0e-8)
    return x_mean, x_scale, y_mean, y_scale


def parse_candidates(text):
    parsed = []
    for raw_candidate in text.split(";"):
        sizes = tuple(int(item.strip()) for item in raw_candidate.split(",") if item.strip())
        if len(sizes) != 2 or min(sizes) <= 0:
            raise ValueError("Each candidate must contain two positive hidden sizes.")
        parsed.append(sizes)
    if not parsed:
        raise ValueError("At least one model candidate is required.")
    return parsed


class OTSRegressorV2(nn.Module):
    def __init__(self, hidden_sizes, dropout=0.02):
        super(OTSRegressorV2, self).__init__()
        first, second = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, first),
            nn.LayerNorm(first),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(second, OUTPUT_DIM),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.05)
                nn.init.zeros_(module.bias)

    def forward(self, spectra):
        return self.network(spectra)


def hermitian_project(coefficients):
    """Project [Re(F), Im(F)] so F(-k) = conjugate(F(k))."""
    real = coefficients[:, :FOURIER_MODES]
    imag = coefficients[:, FOURIER_MODES:]
    projected_real = 0.5 * (real + torch.flip(real, dims=(1,)))
    projected_imag = 0.5 * (imag - torch.flip(imag, dims=(1,)))
    return torch.cat((projected_real, projected_imag), dim=1)


def coefficient_loss(
    predicted_normalized, target_raw, y_mean, y_scale, raw_loss_weight
):
    predicted_raw = predicted_normalized * y_scale + y_mean
    predicted_raw = hermitian_project(predicted_raw)
    predicted_projected = (predicted_raw - y_mean) / y_scale
    target_normalized = (target_raw - y_mean) / y_scale

    scaled_mse = F.mse_loss(predicted_projected, target_normalized)
    raw_scale = y_scale.square().mean().clamp(min=1.0e-16)
    raw_mse = F.mse_loss(predicted_raw, target_raw) / raw_scale
    return (1.0 - raw_loss_weight) * scaled_mse + raw_loss_weight * raw_mse


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    y_mean,
    y_scale,
    raw_loss_weight,
    training,
):
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for spectra, target in loader:
        spectra = spectra.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            predicted = model(spectra)
            loss = coefficient_loss(
                predicted, target, y_mean, y_scale, raw_loss_weight
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_count = spectra.size(0)
        total_loss += loss.item() * batch_count
        total_count += batch_count
    return total_loss / max(1, total_count)


def cpu_state_dict(model):
    return dict((name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items())


def train_candidate(
    hidden_sizes,
    train_loader,
    validation_loader,
    device,
    y_mean,
    y_scale,
    args,
):
    dropout = 0.0 if max(hidden_sizes) <= 512 else 0.02
    model = OTSRegressorV2(hidden_sizes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1.0e-6
    )

    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    started = time.time()

    y_mean_device = y_mean.to(device)
    y_scale_device = y_scale.to(device)
    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            y_mean_device,
            y_scale_device,
            args.raw_loss_weight,
            training=True,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            optimizer=None,
            device=device,
            y_mean=y_mean_device,
            y_scale=y_scale_device,
            raw_loss_weight=args.raw_loss_weight,
            training=False,
        )
        scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        improved = validation_loss < best_validation - 1.0e-7
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "candidate={} epoch={:03d} train={:.6e} val={:.6e} lr={:.3e}{}".format(
                "x".join(str(value) for value in hidden_sizes),
                epoch,
                train_loss,
                validation_loss,
                learning_rate,
                " best" if improved else "",
            ),
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print("Early stopping after epoch {}.".format(epoch), flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    return {
        "hidden_sizes": list(hidden_sizes),
        "dropout": dropout,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_loss": best_validation,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "state_dict": best_state,
    }


def reconstruct_real_evdf(coefficients, k_grid):
    """Real inverse Fourier transform using Re/Im parts explicitly."""
    coefficients = coefficients.float().cpu()
    k_grid = k_grid.float().cpu()
    if len(k_grid) != FOURIER_MODES:
        raise ValueError("Expected {} Fourier modes.".format(FOURIER_MODES))
    dk = k_grid[2] - k_grid[1]
    mode_count = len(k_grid)
    dx = 2.0 * math.pi / (mode_count * dk)
    velocity = torch.linspace(-1.0, 1.0, mode_count) * mode_count * dx / 2.0

    trapezoid_weights = torch.ones_like(k_grid)
    trapezoid_weights[0] = 0.5
    trapezoid_weights[-1] = 0.5
    phase = k_grid.view(-1, 1) * velocity.view(1, -1)
    cosine_basis = trapezoid_weights.view(-1, 1) * torch.cos(phase)
    sine_basis = trapezoid_weights.view(-1, 1) * torch.sin(phase)

    real = coefficients[:, :FOURIER_MODES]
    imag = coefficients[:, FOURIER_MODES:]
    evdf = (torch.mm(real, cosine_basis) - torch.mm(imag, sine_basis))
    evdf = evdf * dk / (2.0 * math.pi)
    return velocity, evdf


def prediction_metrics(target, predicted, k_grid):
    target = target.float().cpu()
    predicted = predicted.float().cpu()
    error = predicted - target
    target_centered = target - target.mean()
    coefficient_r2 = 1.0 - error.square().sum() / target_centered.square().sum().clamp(min=1.0e-30)

    velocity, true_evdf = reconstruct_real_evdf(target, k_grid)
    _, predicted_evdf = reconstruct_real_evdf(predicted, k_grid)
    evdf_error = predicted_evdf - true_evdf
    evdf_centered = true_evdf - true_evdf.mean()
    evdf_r2 = 1.0 - evdf_error.square().sum() / evdf_centered.square().sum().clamp(min=1.0e-30)
    relative_l2 = torch.sqrt(
        evdf_error.square().sum(dim=1)
        / true_evdf.square().sum(dim=1).clamp(min=1.0e-30)
    )

    negative_area = torch.trapz(torch.relu(-predicted_evdf), velocity, dim=1)
    absolute_area = torch.trapz(predicted_evdf.abs(), velocity, dim=1).clamp(min=1.0e-30)
    real = predicted[:, :FOURIER_MODES]
    imag = predicted[:, FOURIER_MODES:]
    symmetry_error = (
        (real - torch.flip(real, dims=(1,))).abs().mean()
        + (imag + torch.flip(imag, dims=(1,))).abs().mean()
    )

    return {
        "coefficient_mae": error.abs().mean().item(),
        "coefficient_rmse": torch.sqrt(error.square().mean()).item(),
        "coefficient_r2": coefficient_r2.item(),
        "hermitian_symmetry_mae": symmetry_error.item(),
        "evdf_mae": evdf_error.abs().mean().item(),
        "evdf_rmse": torch.sqrt(evdf_error.square().mean()).item(),
        "evdf_r2": evdf_r2.item(),
        "evdf_relative_l2_mean": relative_l2.mean().item(),
        "evdf_relative_l2_median": relative_l2.median().item(),
        "negative_mass_fraction_mean": (negative_area / absolute_area).mean().item(),
    }


def predict_legacy(model_path, spectra, batch_size, device):
    model = load_legacy_model(model_path)
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            batch = spectra[start : start + batch_size].to(device)
            predictions.append(model(batch).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(predictions, dim=0)


def predict_v2(model, spectra, x_mean, x_scale, y_mean, y_scale, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            batch = (spectra[start : start + batch_size] - x_mean) / x_scale
            predicted_normalized = model(batch.to(device)).cpu()
            predicted_raw = predicted_normalized * y_scale + y_mean
            predictions.append(hermitian_project(predicted_raw))
    return torch.cat(predictions, dim=0)


def save_learning_curves(candidate_results, output_path):
    figure, axes = plt.subplots(1, len(candidate_results), figsize=(6 * len(candidate_results), 4))
    if len(candidate_results) == 1:
        axes = [axes]
    for axis, result in zip(axes, candidate_results):
        epochs = [item["epoch"] for item in result["history"]]
        train_loss = [item["train_loss"] for item in result["history"]]
        validation_loss = [item["validation_loss"] for item in result["history"]]
        axis.semilogy(epochs, train_loss, label="train")
        axis.semilogy(epochs, validation_loss, label="validation")
        axis.axvline(result["best_epoch"], color="k", linestyle=":", linewidth=1)
        axis.set_title("{} (best epoch {})".format(
            "x".join(str(value) for value in result["hidden_sizes"]), result["best_epoch"]
        ))
        axis.set_xlabel("epoch")
        axis.set_ylabel("normalized loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_comparison_plot(k_grid, target, legacy, improved, output_path, count=6):
    sample_count = min(count, len(target))
    velocity, true_evdf = reconstruct_real_evdf(target[:sample_count], k_grid)
    _, improved_evdf = reconstruct_real_evdf(improved[:sample_count], k_grid)
    legacy_evdf = None
    if legacy is not None:
        _, legacy_evdf = reconstruct_real_evdf(legacy[:sample_count], k_grid)

    rows = int(math.ceil(sample_count / 2.0))
    figure, axes = plt.subplots(rows, 2, figsize=(11, 3.2 * rows), squeeze=False)
    for index in range(sample_count):
        axis = axes[index // 2][index % 2]
        axis.plot(velocity.numpy(), true_evdf[index].numpy(), "k-", label="truth")
        if legacy_evdf is not None:
            axis.plot(velocity.numpy(), legacy_evdf[index].numpy(), color="#de7b0a", linestyle="--", label="legacy")
        axis.plot(velocity.numpy(), improved_evdf[index].numpy(), color="#2563eb", linestyle="-.", label="v2")
        axis.set_title("test sample {}".format(index + 1))
        axis.set_xlabel("velocity (m/s)")
        axis.set_ylabel("EVDF")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for index in range(sample_count, rows * 2):
        axes[index // 2][index % 2].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def serializable_candidate(result):
    return dict((key, value) for key, value in result.items() if key != "state_dict")


def run_self_test():
    torch.manual_seed(7)
    k_grid = torch.linspace(-1.0, 1.0, FOURIER_MODES)
    target = torch.randn(8, OUTPUT_DIM)
    target = hermitian_project(target)
    predicted = hermitian_project(target + 0.01 * torch.randn_like(target))
    model = OTSRegressorV2((64, 32), dropout=0.0)
    output = model(torch.randn(8, INPUT_DIM))
    metrics = prediction_metrics(target, predicted, k_grid)
    assert output.shape == (8, OUTPUT_DIM)
    assert all(np.isfinite(value) for value in metrics.values())
    print("SELF_TEST_OK", json.dumps(metrics, sort_keys=True))


def main():
    args = parse_args()
    if not 0.0 <= args.raw_loss_weight <= 1.0:
        raise ValueError("--raw-loss-weight must be between 0 and 1.")
    if args.self_test:
        run_self_test()
        return

    set_determinism(args.seed)
    device = choose_device(args.device)
    thread_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    torch.set_num_threads(max(1, thread_count))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("Python:", sys.version.replace("\n", " "), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "Loss weights: standardized={:.2f} raw={:.2f}".format(
            1.0 - args.raw_loss_weight, args.raw_loss_weight
        ),
        flush=True,
    )

    spectra, coefficients, wavelengths_nm, k_grid = load_parquet_dataset(args.train_data)
    test_spectra, test_coefficients, _, test_k_grid = load_parquet_dataset(args.test_data)
    if not torch.allclose(k_grid, test_k_grid, rtol=1.0e-5, atol=1.0e-12):
        raise ValueError("Training and test datasets use different Fourier grids.")

    x_train, y_train, x_validation, y_validation = split_train_validation(
        spectra, coefficients, args.validation_fraction, args.seed
    )
    x_mean, x_scale, y_mean, y_scale = normalization_statistics(x_train, y_train)
    normalized_train = (x_train - x_mean) / x_scale
    normalized_validation = (x_validation - x_mean) / x_scale
    pin_memory = device.type == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(normalized_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        TensorDataset(normalized_validation, y_validation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print(
        "Rows: train={} validation={} test={} (metadata excluded)".format(
            len(x_train), len(x_validation), len(test_spectra)
        ),
        flush=True,
    )

    candidate_results = []
    for hidden_sizes in parse_candidates(args.candidates):
        print("\nTraining candidate {}".format(hidden_sizes), flush=True)
        set_determinism(args.seed)
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(args.seed)
        candidate_results.append(
            train_candidate(
                hidden_sizes,
                train_loader,
                validation_loader,
                device,
                y_mean,
                y_scale,
                args,
            )
        )

    best_result = min(candidate_results, key=lambda item: item["best_validation_loss"])
    best_model = OTSRegressorV2(
        tuple(best_result["hidden_sizes"]), dropout=best_result["dropout"]
    )
    best_model.load_state_dict(best_result["state_dict"])
    best_model.to(device)

    improved_predictions = predict_v2(
        best_model,
        test_spectra,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        args.batch_size,
        device,
    )
    improved_metrics = prediction_metrics(test_coefficients, improved_predictions, k_grid)

    legacy_predictions = None
    legacy_metrics_raw = None
    legacy_metrics_projected = None
    if not args.skip_legacy and os.path.isfile(args.legacy_model):
        print("\nEvaluating legacy model...", flush=True)
        legacy_predictions = predict_legacy(
            args.legacy_model, test_spectra, args.batch_size, device
        )
        legacy_metrics_raw = prediction_metrics(test_coefficients, legacy_predictions, k_grid)
        legacy_predictions = hermitian_project(legacy_predictions)
        legacy_metrics_projected = prediction_metrics(
            test_coefficients, legacy_predictions, k_grid
        )

    checkpoint_path = os.path.join(args.output_dir, "OTS_Te_model_v2.pt")
    checkpoint = {
        "format_version": 2,
        "model_class": "OTSRegressorV2",
        "model_state": best_result["state_dict"],
        "hidden_sizes": best_result["hidden_sizes"],
        "dropout": best_result["dropout"],
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "k_grid": k_grid,
        "wavelengths_nm": wavelengths_nm,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "raw_loss_weight": args.raw_loss_weight,
        "test_metrics": improved_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "raw_loss_weight": args.raw_loss_weight,
        "rows": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test": len(test_spectra),
        },
        "selected_candidate": serializable_candidate(best_result),
        "candidates": [serializable_candidate(item) for item in candidate_results],
        "legacy_raw": legacy_metrics_raw,
        "legacy_physics_projected": legacy_metrics_projected,
        "v2": improved_metrics,
        "artifacts": {
            "checkpoint": checkpoint_path,
            "history": os.path.join(args.output_dir, "training_history_v2.json"),
            "metrics": os.path.join(args.output_dir, "metrics_v2.json"),
            "predictions": os.path.join(args.output_dir, "predictions_v2.npz"),
            "learning_curves": os.path.join(args.output_dir, "learning_curves_v2.png"),
            "comparison": os.path.join(args.output_dir, "comparison_v2.png"),
        },
    }

    history_path = summary["artifacts"]["history"]
    with open(history_path, "w") as handle:
        json.dump(
            [serializable_candidate(item) for item in candidate_results],
            handle,
            indent=2,
            sort_keys=True,
        )
    with open(summary["artifacts"]["metrics"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        summary["artifacts"]["predictions"],
        target_coefficients=test_coefficients.numpy(),
        legacy_coefficients=None if legacy_predictions is None else legacy_predictions.numpy(),
        v2_coefficients=improved_predictions.numpy(),
        k_grid=k_grid.numpy(),
    )
    save_learning_curves(candidate_results, summary["artifacts"]["learning_curves"])
    save_comparison_plot(
        k_grid,
        test_coefficients,
        legacy_predictions,
        improved_predictions,
        summary["artifacts"]["comparison"],
    )

    print("\nSelected candidate:", best_result["hidden_sizes"], flush=True)
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
