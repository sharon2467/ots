# -*- coding: utf-8 -*-
"""Independent frozen-checkpoint evaluation for the canonical OTS pipeline."""

import json
import math
import os
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import FOURIER_MODES, INPUT_DIM, OUTPUT_DIM, load_dataset, OTSDataset, SPECIES_BLOCKS, dataset_layout, _physical_config
from model import hermitian_project, load_model_checkpoint


DEFAULT_EVALUATION_CONFIG = {
    "batch_size": 128,
    "device": "auto",
    "plot_samples": 6,
    "detail_index": 0,
}


def _real_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a tensor.".format(name))
    if torch.is_complex(value):
        raise ValueError("{} must be real-valued.".format(name))
    if not value.dtype.is_floating_point:
        value = value.to(torch.float32)
    dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    value = value.to(dtype=dtype)
    if not torch.isfinite(value).all():
        raise ValueError("{} contains non-finite values.".format(name))
    return value


def _validate_k_grid(k_grid):
    k_grid = _real_tensor(k_grid, "k_grid")
    if k_grid.ndim != 1 or len(k_grid) < 3:
        raise ValueError("k_grid must be a vector with at least 3 points.")
    spacing = k_grid[1:] - k_grid[:-1]
    if not torch.all(spacing > 0):
        raise ValueError("k_grid must be strictly increasing.")
    tolerance = torch.abs(spacing.mean()) * 1.0e-4
    if not torch.all(torch.abs(spacing - spacing.mean()) <= tolerance):
        raise ValueError("k_grid must be approximately uniform.")
    return k_grid


def reconstruct_real_evdf(coefficients, k_grid, *, velocity=None):
    """Reconstruct real EVDFs with an explicit trapezoidal inverse integral."""

    coefficients = _real_tensor(coefficients, "coefficients")
    if coefficients.ndim != 2 or coefficients.shape[1] != 2 * len(k_grid):
        raise ValueError("coefficients must have shape [N, 2M] for the supplied K grid.")
    if coefficients.shape[0] <= 0:
        raise ValueError("coefficients must contain at least one sample.")
    k_grid = _validate_k_grid(k_grid).to(
        device=coefficients.device, dtype=coefficients.dtype
    )

    mode_count = len(k_grid)
    dk = k_grid[2] - k_grid[1]
    dx = 2.0 * math.pi / (mode_count * dk)
    if velocity is None:
        velocity = (
            torch.linspace(
                -1.0,
                1.0,
                mode_count,
                device=coefficients.device,
                dtype=coefficients.dtype,
            )
            * mode_count
            * dx
            / 2.0
        )
    else:
        velocity = torch.as_tensor(velocity, dtype=coefficients.dtype, device=coefficients.device)
        if velocity.ndim != 1 or len(velocity) < 2 or not torch.isfinite(velocity).all() or not torch.all(velocity[1:] > velocity[:-1]):
            raise ValueError("velocity must be finite and strictly increasing.")

    trapezoid_weights = torch.ones_like(k_grid)
    trapezoid_weights[0] = 0.5
    trapezoid_weights[-1] = 0.5
    phase = k_grid.reshape(-1, 1) * velocity.reshape(1, -1)
    cosine_basis = trapezoid_weights.reshape(-1, 1) * torch.cos(phase)
    sine_basis = trapezoid_weights.reshape(-1, 1) * torch.sin(phase)
    real = coefficients[:, :mode_count]
    imaginary = coefficients[:, mode_count:]
    evdf = real @ cosine_basis - imaginary @ sine_basis
    evdf = evdf * dk / (2.0 * math.pi)
    if not torch.isfinite(evdf).all() or not torch.isfinite(velocity).all():
        raise ValueError("EVDF reconstruction produced non-finite values.")
    return velocity, evdf


def predict_coefficients(model, spectra, checkpoint, batch_size, device):
    """Predict raw projected coefficients in batches without gradients."""

    spectra = _real_tensor(spectra, "spectra")
    if spectra.ndim != 2 or spectra.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Spectral width must match checkpoint input_dim.")
    if spectra.shape[0] <= 0:
        raise ValueError("spectra must contain at least one sample.")
    if (
        not isinstance(batch_size, Integral)
        or isinstance(batch_size, bool)
        or int(batch_size) <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    batch_size = int(batch_size)
    device = torch.device(device)
    dtype = torch.float64 if checkpoint["dtype"] == "float64" else torch.float32
    x_mean = checkpoint["x_mean"].to(device=device, dtype=dtype)
    x_scale = checkpoint["x_scale"].to(device=device, dtype=dtype)
    y_mean = checkpoint["y_mean"].to(device=device, dtype=dtype)
    y_scale = checkpoint["y_scale"].to(device=device, dtype=dtype)

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            batch = spectra[start : start + batch_size].to(device=device, dtype=dtype)
            normalized_batch = (batch - x_mean) / x_scale
            predicted_normalized = model(normalized_batch)
            predicted_raw = predicted_normalized * y_scale + y_mean
            predictions.append(hermitian_project(predicted_raw, fourier_modes=checkpoint["fourier_modes"]).cpu())
    predicted = torch.cat(predictions, dim=0)
    if predicted.shape != (len(spectra), checkpoint["output_dim"]):
        raise ValueError("Model predictions have an invalid shape.")
    if not torch.isfinite(predicted).all():
        raise ValueError("Model predictions contain non-finite values.")
    return predicted


def prediction_metrics(target, predicted, k_grid, *, velocity=None):
    """Compute the fixed coefficient, Hermitian, and EVDF metric set."""

    target = _real_tensor(target, "target").cpu()
    predicted = _real_tensor(predicted, "predicted").cpu()
    if (
        target.ndim != 2
        or predicted.ndim != 2
        or target.shape != predicted.shape
        or target.shape[0] <= 0
        or target.shape[1] != 2 * len(k_grid)
    ):
        raise ValueError("target and predicted must have equal shape [N, 2M].")
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    predicted = predicted.to(dtype=dtype)
    k_grid = _validate_k_grid(k_grid).cpu().to(dtype=dtype)
    denominator_floor = 1.0e-30

    coefficient_error = predicted - target
    coefficient_centered = target - target.mean()
    coefficient_r2 = 1.0 - coefficient_error.square().sum() / torch.clamp(
        coefficient_centered.square().sum(), min=denominator_floor
    )

    velocity, true_evdf = reconstruct_real_evdf(target, k_grid, velocity=velocity)
    _, predicted_evdf = reconstruct_real_evdf(predicted, k_grid, velocity=velocity)
    evdf_error = predicted_evdf - true_evdf
    evdf_centered = true_evdf - true_evdf.mean()
    evdf_r2 = 1.0 - evdf_error.square().sum() / torch.clamp(
        evdf_centered.square().sum(), min=denominator_floor
    )
    relative_l2 = torch.sqrt(
        evdf_error.square().sum(dim=1)
        / torch.clamp(true_evdf.square().sum(dim=1), min=denominator_floor)
    )

    negative_area = torch.trapz(torch.relu(-predicted_evdf), velocity, dim=1)
    absolute_area = torch.clamp(
        torch.trapz(predicted_evdf.abs(), velocity, dim=1), min=denominator_floor
    )
    real = predicted[:, :len(k_grid)]
    imaginary = predicted[:, len(k_grid):]
    symmetry_error = (real - torch.flip(real, dims=(1,))).abs().mean() + (
        imaginary + torch.flip(imaginary, dims=(1,))
    ).abs().mean()
    sorted_relative_l2 = torch.sort(relative_l2).values
    middle = len(sorted_relative_l2) // 2
    if len(sorted_relative_l2) % 2:
        relative_l2_median = sorted_relative_l2[middle]
    else:
        relative_l2_median = 0.5 * (
            sorted_relative_l2[middle - 1] + sorted_relative_l2[middle]
        )

    metrics = {
        "coefficient_mae": float(coefficient_error.abs().mean().item()),
        "coefficient_rmse": float(torch.sqrt(coefficient_error.square().mean()).item()),
        "coefficient_r2": float(coefficient_r2.item()),
        "hermitian_symmetry_mae": float(symmetry_error.item()),
        "evdf_mae": float(evdf_error.abs().mean().item()),
        "evdf_rmse": float(torch.sqrt(evdf_error.square().mean()).item()),
        "evdf_r2": float(evdf_r2.item()),
        "evdf_relative_l2_mean": float(relative_l2.mean().item()),
        "evdf_relative_l2_median": float(relative_l2_median.item()),
        "negative_mass_fraction_mean": float(
            (negative_area / absolute_area).mean().item()
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Prediction metrics contain non-finite values.")
    return metrics


def _choose_device(requested):
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be one of: cpu, cuda, auto.")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_config(config):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")
    unexpected = sorted(set(config) - set(DEFAULT_EVALUATION_CONFIG))
    if unexpected:
        raise ValueError("Unsupported evaluation config fields: {}.".format(unexpected))
    resolved = dict(DEFAULT_EVALUATION_CONFIG)
    resolved.update(config)
    for name in ("batch_size", "plot_samples"):
        value = resolved[name]
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or int(value) <= 0
        ):
            raise ValueError("{} must be a positive integer.".format(name))
        resolved[name] = int(value)
    detail_index = resolved["detail_index"]
    if (
        not isinstance(detail_index, Integral)
        or isinstance(detail_index, bool)
        or int(detail_index) < 0
    ):
        raise ValueError("detail_index must be a non-negative integer.")
    resolved["detail_index"] = int(detail_index)
    _choose_device(resolved["device"])
    return resolved


def _validate_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string.".format(name))


def _same_path(left, right):
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _relative_path(path, base):
    return Path(os.path.relpath(str(path.resolve()), str(base.resolve()))).as_posix()


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _write_predictions(path, target, predicted, k_grid, *, velocity=None):
    velocity, true_evdf = reconstruct_real_evdf(target, k_grid, velocity=velocity)
    _, predicted_evdf = reconstruct_real_evdf(predicted, k_grid, velocity=velocity)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    np.savez_compressed(
        str(temporary),
        target_coefficients=target.detach().cpu().numpy(),
        predicted_coefficients=predicted.detach().cpu().numpy(),
        k_grid=k_grid.detach().cpu().numpy(),
        velocity_m_s=velocity.detach().cpu().numpy(),
        target_evdf=true_evdf.detach().cpu().numpy(),
        predicted_evdf=predicted_evdf.detach().cpu().numpy(),
    )
    os.replace(str(temporary), str(path))


def _save_comparison(path, target, predicted, k_grid, sample_count, *, velocity=None, species_name="e"):
    sample_count = min(sample_count, len(target))
    velocity, true_evdf = reconstruct_real_evdf(target[:sample_count], k_grid, velocity=velocity)
    _, predicted_evdf = reconstruct_real_evdf(predicted[:sample_count], k_grid, velocity=velocity)
    columns = 2
    rows = int(math.ceil(sample_count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    velocity_values = velocity.detach().cpu().numpy()
    for index in range(sample_count):
        axis = axes[index // columns][index % columns]
        axis.plot(
            velocity_values,
            true_evdf[index].detach().cpu().numpy(),
            color="black",
            label="truth",
        )
        axis.plot(
            velocity_values,
            predicted_evdf[index].detach().cpu().numpy(),
            color="#2563eb",
            linestyle="--",
            label="prediction",
        )
        axis.set_title("sample {}".format(index + 1))
        axis.set_xlabel("velocity (m/s)")
        axis.set_ylabel(species_name + " VDF")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for index in range(sample_count, rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(str(temporary), dpi=170)
    plt.close(figure)
    os.replace(str(temporary), str(path))


def _save_sample_detail(
    path,
    dataset,
    predicted,
    k_grid,
    sample_index,
    *, velocity=None, species_name="e",
):
    velocity, true_evdf = reconstruct_real_evdf(
        dataset.coefficients[sample_index : sample_index + 1], k_grid, velocity=velocity
    )
    _, predicted_evdf = reconstruct_real_evdf(
        predicted[sample_index : sample_index + 1], k_grid, velocity=velocity
    )
    wavelengths = dataset.wavelengths_nm.detach().cpu().numpy()
    spectrum = dataset.spectra[sample_index].detach().cpu().numpy()
    target = dataset.coefficients[sample_index].detach().cpu()
    candidate = predicted[sample_index].detach().cpu()
    k_values = k_grid.detach().cpu().numpy()

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    spectral_axis = axes[0, 0]
    lengths = ([len(band) for band in dataset.layout["wavelength_bands_nm"]]
               if dataset.layout else [579, 304, 341])
    start = 0
    for index, length in enumerate(lengths):
        stop = start + length
        spectral_axis.plot(wavelengths[start:stop], spectrum[start:stop], label="band {}".format(index + 1))
        start = stop
    spectral_axis.set_xlabel("wavelength (nm)")
    spectral_axis.set_ylabel("normalized intensity")
    spectral_axis.set_title("input spectrum")
    spectral_axis.grid(alpha=0.2)
    spectral_axis.legend(fontsize=8)

    axes[0, 1].plot(k_values, target[:len(k_grid)].numpy(), "k-", label="truth")
    axes[0, 1].plot(
        k_values,
        candidate[:len(k_grid)].numpy(),
        color="#2563eb",
        linestyle="--",
        label="prediction",
    )
    axes[0, 1].set_title("Fourier real part")
    axes[0, 1].set_xlabel("K (s/m)")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(k_values, target[len(k_grid):].numpy(), "k-", label="truth")
    axes[1, 0].plot(
        k_values,
        candidate[len(k_grid):].numpy(),
        color="#2563eb",
        linestyle="--",
        label="prediction",
    )
    axes[1, 0].set_title("Fourier imaginary part")
    axes[1, 0].set_xlabel("K (s/m)")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    velocity_values = velocity.detach().cpu().numpy()
    axes[1, 1].plot(
        velocity_values,
        true_evdf[0].detach().cpu().numpy(),
        "k-",
        label="truth",
    )
    axes[1, 1].plot(
        velocity_values,
        predicted_evdf[0].detach().cpu().numpy(),
        color="#2563eb",
        linestyle="--",
        label="prediction",
    )
    axes[1, 1].set_title("reconstructed " + species_name + " VDF")
    axes[1, 1].set_xlabel("velocity (m/s)")
    axes[1, 1].set_ylabel(species_name + " VDF")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("sample {} detail".format(sample_index + 1))
    figure.tight_layout()
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(str(temporary), dpi=170)
    plt.close(figure)
    os.replace(str(temporary), str(path))


def evaluate_checkpoint(
    run_id,
    dataset_id,
    checkpoint_id,
    checkpoint_path,
    test_data_path,
    manifest_path,
    output_dir,
    config,
    overwrite=False,
):
    """Evaluate one frozen checkpoint against the independent dataset split."""

    for value, name in (
        (run_id, "run_id"),
        (dataset_id, "dataset_id"),
        (checkpoint_id, "checkpoint_id"),
    ):
        _validate_id(value, name)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean.")
    config = _resolve_config(config)
    device = _choose_device(config["device"])

    checkpoint_path = Path(checkpoint_path)
    test_data_path = Path(test_data_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.npz"
    comparison_path = output_dir / "comparison.png"
    detail_path = output_dir / "sample_detail.png"
    outputs = (metrics_path, predictions_path, comparison_path, detail_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation artifacts: {}".format(existing)
        )

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("Manifest dataset_id does not match the evaluation request.")
    dataset_source_run_id = manifest.get("run_id")
    _validate_id(dataset_source_run_id, "manifest run_id")
    try:
        manifest_test_path = manifest_path.parent / manifest["splits"]["test"]["file"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Manifest does not contain a valid test file reference."
        ) from error
    if not _same_path(test_data_path, manifest_test_path):
        raise ValueError("test_data_path does not match manifest splits.test.file.")

    model, checkpoint = load_model_checkpoint(checkpoint_path, device, checkpoint_id)
    dataset = load_dataset(test_data_path, manifest.get("physical_configuration"))
    checkpoint_layout = checkpoint.get("layout")
    if checkpoint_layout is not None:
        if dataset.layout != checkpoint_layout:
            raise ValueError("Checkpoint and test dataset layouts do not match.")
    elif dataset.layout is not None and dataset.layout != dataset_layout(_physical_config()):
        raise ValueError("Old checkpoint requires the original fixed_ion layout.")
    manifest_sample_count = manifest["splits"]["test"].get("physical_samples")
    if manifest_sample_count != len(dataset.spectra):
        raise ValueError("Manifest and dataset physical sample counts do not match.")
    checkpoint_k = checkpoint["k_grid"].detach().cpu().to(dataset.k_grid.dtype)
    checkpoint_wavelengths = (
        checkpoint["wavelengths_nm"].detach().cpu().to(dataset.wavelengths_nm.dtype)
    )
    if dataset.k_grid.shape != checkpoint_k.shape or not torch.allclose(dataset.k_grid, checkpoint_k, rtol=1.0e-5, atol=1.0e-12):
        raise ValueError("Checkpoint and dataset K grids do not match.")
    if dataset.wavelengths_nm.shape != checkpoint_wavelengths.shape or not torch.allclose(
        dataset.wavelengths_nm,
        checkpoint_wavelengths,
        rtol=1.0e-5,
        atol=1.0e-4,
    ):
        raise ValueError("Checkpoint and dataset wavelength grids do not match.")
    if config["detail_index"] >= len(dataset.spectra):
        raise ValueError("detail_index is outside the dataset sample range.")

    predicted = predict_coefficients(
        model,
        dataset.spectra,
        checkpoint,
        config["batch_size"],
        device,
    )
    modes = checkpoint["fourier_modes"]
    selected = [(species, key) for species, key in SPECIES_BLOCKS if checkpoint_layout[key]["used_in_model"]] if checkpoint_layout else [("e", "electron_dis")]
    per_species = selected != [("e", "electron_dis")]
    metrics, prediction_paths, comparison_paths, detail_paths = {}, {}, {}, {}
    for index, (species, key) in enumerate(selected):
        folder = output_dir / species if per_species else output_dir
        species_predictions = folder / "predictions.npz"
        species_comparison = folder / "comparison.png"
        species_detail = folder / "sample_detail.png"
        if not overwrite and any(path.exists() for path in (species_predictions, species_comparison, species_detail)):
            raise FileExistsError("Refusing to overwrite existing species evaluation artifacts.")
        prediction_paths[species], comparison_paths[species], detail_paths[species] = species_predictions, species_comparison, species_detail
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (species, key) in enumerate(selected):
        coefficients = dataset.coefficients[:, index * 2 * modes:(index + 1) * 2 * modes]
        candidate = predicted[:, index * 2 * modes:(index + 1) * 2 * modes]
        k_grid = dataset.k_grid[index * modes:(index + 1) * modes]
        velocity = checkpoint_layout[key]["velocity_grid_m_s"] if checkpoint_layout else None
        metrics[species] = prediction_metrics(coefficients, candidate, k_grid, velocity=velocity)
        prediction_paths[species].parent.mkdir(parents=True, exist_ok=True)
        _write_predictions(prediction_paths[species], coefficients, candidate, k_grid, velocity=velocity)
        _save_comparison(comparison_paths[species], coefficients, candidate, k_grid, config["plot_samples"], velocity=velocity, species_name=species)
        # This plotting view keeps the full input band layout; it is not revalidated as a full output dataset.
        view = OTSDataset(dataset.spectra, coefficients, dataset.wavelengths_nm, k_grid, dataset.layout)
        _save_sample_detail(detail_paths[species], view, candidate, k_grid, config["detail_index"], velocity=velocity, species_name=species)
    if not per_species:
        metrics = metrics["e"]
    metrics_payload = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_source_run_id": dataset_source_run_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_source_run_id": checkpoint["run_id"],
        "checkpoint_training_dataset_id": checkpoint["dataset_id"],
        "checkpoint": _relative_path(checkpoint_path, metrics_path.parent),
        "test_data": _relative_path(test_data_path, metrics_path.parent),
        "metrics": metrics,
    }

    _write_json(metrics_path, metrics_payload)
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "predictions_path": prediction_paths if per_species else prediction_paths["e"],
        "comparison_path": comparison_paths if per_species else comparison_paths["e"],
        "sample_detail_path": detail_paths if per_species else detail_paths["e"],
    }
