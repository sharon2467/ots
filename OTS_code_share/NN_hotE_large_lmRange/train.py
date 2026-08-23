# -*- coding: utf-8 -*-
"""Canonical train/validation training and checkpoint production."""

import json
import math
import os
import random
import time
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from data import OTSDataset, load_dataset, validate_dataset
from model import (
    FOURIER_MODES,
    INPUT_DIM,
    LEAKY_RELU_SLOPE,
    OUTPUT_DIM,
    OTSRegressor,
    hermitian_project,
    validate_checkpoint,
)


DEFAULT_CONFIG = {
    "seed": 20260710,
    "validation_fraction": 0.10,
    "candidates": "512,256;1024,512",
    "batch_size": 128,
    "max_epochs": 250,
    "patience": 25,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "raw_loss_weight": 0.10,
    "gradient_clip_norm": 5.0,
    "device": "auto",
}

DEFAULT_OPTIMIZER_PARAMETERS = {
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "amsgrad": False,
}

DEFAULT_SCHEDULER_PARAMETERS = {
    "mode": "min",
    "factor": 0.5,
    "patience": 7,
    "threshold": 1.0e-4,
    "threshold_mode": "rel",
    "cooldown": 0,
    "min_lr": 1.0e-6,
    "eps": 1.0e-8,
}


def set_determinism(seed):
    """Seed all local RNGs and enable deterministic PyTorch behavior."""

    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def choose_device(requested):
    """Resolve ``cpu``, ``cuda``, or ``auto`` without silent CUDA fallback."""

    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be one of: cpu, cuda, auto.")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_candidates(value):
    """Parse candidate sizes and resolve the existing v2 dropout policy."""

    if isinstance(value, str):
        raw_candidates = []
        for raw in value.split(";"):
            if not raw.strip():
                continue
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 2:
                raise ValueError("Each candidate must contain two hidden sizes.")
            try:
                raw_candidates.append(tuple(int(part) for part in parts))
            except ValueError as error:
                raise ValueError("Candidate hidden sizes must be integers.") from error
    elif isinstance(value, (tuple, list)):
        raw_candidates = list(value)
    else:
        raise ValueError("candidates must be a string or sequence.")

    parsed = []
    seen = set()
    for candidate in raw_candidates:
        if isinstance(candidate, dict):
            if set(candidate) != {"hidden_sizes", "dropout"}:
                raise ValueError(
                    "Candidate dictionaries require hidden_sizes and dropout only."
                )
            hidden_sizes = candidate["hidden_sizes"]
            dropout = float(candidate["dropout"])
        else:
            hidden_sizes = candidate
            dropout = None
        if (
            not isinstance(hidden_sizes, (tuple, list))
            or len(hidden_sizes) != 2
            or any(
                not isinstance(size, Integral)
                or isinstance(size, bool)
                or int(size) <= 0
                for size in hidden_sizes
            )
        ):
            raise ValueError("Each candidate must contain two positive hidden sizes.")
        hidden_sizes = (int(hidden_sizes[0]), int(hidden_sizes[1]))
        if dropout is None:
            dropout = 0.0 if max(hidden_sizes) <= 512 else 0.02
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("Candidate dropout must satisfy 0 <= dropout < 1.")
        identity = hidden_sizes
        if identity in seen:
            raise ValueError(
                "Duplicate candidate hidden_sizes are not allowed, even when "
                "dropout values differ."
            )
        seen.add(identity)
        parsed.append({"hidden_sizes": list(hidden_sizes), "dropout": float(dropout)})
    if not parsed:
        raise ValueError("At least one model candidate is required.")
    return parsed


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


def _nonnegative_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ValueError("{} must be a non-negative integer.".format(name))
    return int(value)


def _finite_float(value, name, minimum=None, maximum=None):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite.".format(name))
    if minimum is not None and value < minimum:
        raise ValueError("{} must be at least {}.".format(name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be at most {}.".format(name, maximum))
    return value


def resolve_training_config(config):
    """Resolve defaults and record every optimizer/scheduler parameter used."""

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")
    supported = set(DEFAULT_CONFIG) | {"optimizer", "scheduler"}
    unexpected = sorted(set(config) - supported)
    if unexpected:
        raise ValueError("Unsupported training config fields: {}.".format(unexpected))

    merged = dict(DEFAULT_CONFIG)
    merged.update({key: value for key, value in config.items() if key in merged})
    seed = merged["seed"]
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    seed = int(seed)
    validation_fraction = _finite_float(
        merged["validation_fraction"], "validation_fraction"
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    learning_rate = _finite_float(merged["learning_rate"], "learning_rate", minimum=0.0)
    if learning_rate == 0.0:
        raise ValueError("learning_rate must be positive.")
    weight_decay = _finite_float(merged["weight_decay"], "weight_decay", minimum=0.0)
    raw_loss_weight = _finite_float(
        merged["raw_loss_weight"], "raw_loss_weight", minimum=0.0, maximum=1.0
    )
    gradient_clip_norm = _finite_float(
        merged["gradient_clip_norm"], "gradient_clip_norm", minimum=0.0
    )
    if gradient_clip_norm == 0.0:
        raise ValueError("gradient_clip_norm must be positive.")

    optimizer_input = config.get("optimizer", {})
    if not isinstance(optimizer_input, dict):
        raise ValueError("optimizer must be a dictionary.")
    if set(optimizer_input) - {"name", "parameters"}:
        raise ValueError("optimizer contains unsupported fields.")
    optimizer_name = optimizer_input.get("name", "AdamW")
    if optimizer_name != "AdamW":
        raise ValueError("Only AdamW is supported.")
    optimizer_parameters = dict(DEFAULT_OPTIMIZER_PARAMETERS)
    supplied_optimizer_parameters = optimizer_input.get("parameters", {})
    if not isinstance(supplied_optimizer_parameters, dict):
        raise ValueError("optimizer.parameters must be a dictionary.")
    allowed_optimizer_parameters = set(DEFAULT_OPTIMIZER_PARAMETERS) | {
        "lr",
        "weight_decay",
    }
    if set(supplied_optimizer_parameters) - allowed_optimizer_parameters:
        raise ValueError("optimizer.parameters contains unsupported fields.")
    if "lr" in supplied_optimizer_parameters and not math.isclose(
        float(supplied_optimizer_parameters["lr"]),
        learning_rate,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("optimizer lr must match learning_rate.")
    if "weight_decay" in supplied_optimizer_parameters and not math.isclose(
        float(supplied_optimizer_parameters["weight_decay"]),
        weight_decay,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("optimizer weight_decay must match weight_decay.")
    optimizer_parameters.update(supplied_optimizer_parameters)
    betas = optimizer_parameters["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("optimizer betas must contain two values.")
    betas = [_finite_float(value, "optimizer beta") for value in betas]
    if not all(0.0 <= value < 1.0 for value in betas):
        raise ValueError("optimizer betas must satisfy 0 <= beta < 1.")
    optimizer_eps = _finite_float(
        optimizer_parameters["eps"], "optimizer eps", minimum=0.0
    )
    if optimizer_eps == 0.0:
        raise ValueError("optimizer eps must be positive.")
    if not isinstance(optimizer_parameters["amsgrad"], bool):
        raise ValueError("optimizer amsgrad must be boolean.")

    scheduler_input = config.get("scheduler", {})
    if not isinstance(scheduler_input, dict):
        raise ValueError("scheduler must be a dictionary.")
    if set(scheduler_input) - {"name", "parameters"}:
        raise ValueError("scheduler contains unsupported fields.")
    scheduler_name = scheduler_input.get("name", "ReduceLROnPlateau")
    if scheduler_name != "ReduceLROnPlateau":
        raise ValueError("Only ReduceLROnPlateau is supported.")
    scheduler_parameters = dict(DEFAULT_SCHEDULER_PARAMETERS)
    supplied_scheduler_parameters = scheduler_input.get("parameters", {})
    if not isinstance(supplied_scheduler_parameters, dict):
        raise ValueError("scheduler.parameters must be a dictionary.")
    if set(supplied_scheduler_parameters) - set(DEFAULT_SCHEDULER_PARAMETERS):
        raise ValueError("scheduler.parameters contains unsupported fields.")
    scheduler_parameters.update(supplied_scheduler_parameters)
    if scheduler_parameters["mode"] != "min":
        raise ValueError("scheduler mode must be min.")
    if scheduler_parameters["threshold_mode"] not in ("rel", "abs"):
        raise ValueError("scheduler threshold_mode must be rel or abs.")
    scheduler_parameters["factor"] = _finite_float(
        scheduler_parameters["factor"], "scheduler factor"
    )
    if not 0.0 < scheduler_parameters["factor"] < 1.0:
        raise ValueError("scheduler factor must be between 0 and 1.")
    scheduler_parameters["patience"] = _nonnegative_integer(
        scheduler_parameters["patience"], "scheduler patience"
    )
    scheduler_parameters["threshold"] = _finite_float(
        scheduler_parameters["threshold"], "scheduler threshold", minimum=0.0
    )
    scheduler_parameters["cooldown"] = _nonnegative_integer(
        scheduler_parameters["cooldown"], "scheduler cooldown"
    )
    scheduler_parameters["min_lr"] = _finite_float(
        scheduler_parameters["min_lr"], "scheduler min_lr", minimum=0.0
    )
    scheduler_parameters["eps"] = _finite_float(
        scheduler_parameters["eps"], "scheduler eps", minimum=0.0
    )

    requested_device = merged["device"]
    choose_device(requested_device)
    resolved = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "candidates": parse_candidates(merged["candidates"]),
        "batch_size": _positive_integer(merged["batch_size"], "batch_size"),
        "max_epochs": _positive_integer(merged["max_epochs"], "max_epochs"),
        "patience": _positive_integer(merged["patience"], "patience"),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "raw_loss_weight": raw_loss_weight,
        "gradient_clip_norm": gradient_clip_norm,
        "optimizer": {
            "name": optimizer_name,
            "parameters": {
                "lr": learning_rate,
                "betas": betas,
                "eps": optimizer_eps,
                "weight_decay": weight_decay,
                "amsgrad": optimizer_parameters["amsgrad"],
            },
        },
        "scheduler": {
            "name": scheduler_name,
            "parameters": scheduler_parameters,
        },
    }
    return resolved, requested_device


def split_train_validation(dataset, fraction, seed):
    """Make a deterministic disjoint split from the train dataset only."""

    if not isinstance(dataset, OTSDataset):
        raise TypeError("dataset must be an OTSDataset.")
    validate_dataset(dataset)
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    sample_count = len(dataset.spectra)
    if sample_count < 2:
        raise ValueError("At least two train samples are required.")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(sample_count, generator=generator)
    validation_count = max(1, int(round(sample_count * fraction)))
    validation_count = min(sample_count - 1, validation_count)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    return (
        dataset.spectra[training_indices],
        dataset.coefficients[training_indices],
        dataset.spectra[validation_indices],
        dataset.coefficients[validation_indices],
    )


def normalization_statistics(x_train, y_train):
    """Compute population statistics from the training subset only."""

    if (
        not isinstance(x_train, torch.Tensor)
        or not isinstance(y_train, torch.Tensor)
        or x_train.ndim != 2
        or y_train.ndim != 2
        or x_train.shape[0] <= 0
        or x_train.shape[0] != y_train.shape[0]
        or x_train.shape[1] != INPUT_DIM
        or y_train.shape[1] != OUTPUT_DIM
    ):
        raise ValueError("Training tensors must have shapes [N,1224] and [N,160].")
    if not torch.isfinite(x_train).all() or not torch.isfinite(y_train).all():
        raise ValueError("Training tensors contain non-finite values.")
    x_mean = x_train.mean(dim=0)
    x_scale = x_train.std(dim=0, unbiased=False).clamp(min=1.0e-4)
    y_mean = y_train.mean(dim=0)
    y_scale = y_train.std(dim=0, unbiased=False).clamp(min=1.0e-8)
    return x_mean, x_scale, y_mean, y_scale


def coefficient_loss(
    predicted_normalized,
    target_raw,
    y_mean,
    y_scale,
    raw_loss_weight,
):
    """Combine standardized and normalized raw coefficient MSE."""

    if predicted_normalized.shape != target_raw.shape:
        raise ValueError("Prediction and target shapes must match.")
    if predicted_normalized.shape[-1] != OUTPUT_DIM:
        raise ValueError("Coefficient tensors must have final dimension 160.")
    if y_mean.shape != (OUTPUT_DIM,) or y_scale.shape != (OUTPUT_DIM,):
        raise ValueError("y_mean and y_scale must have shape [160].")
    if not torch.all(y_scale > 0):
        raise ValueError("y_scale must be strictly positive.")
    raw_loss_weight = float(raw_loss_weight)
    if not math.isfinite(raw_loss_weight) or not 0.0 <= raw_loss_weight <= 1.0:
        raise ValueError("raw_loss_weight must be between 0 and 1.")

    predicted_raw = predicted_normalized * y_scale + y_mean
    predicted_raw = hermitian_project(predicted_raw)
    predicted_standardized = (predicted_raw - y_mean) / y_scale
    target_standardized = (target_raw - y_mean) / y_scale
    standardized_mse = functional.mse_loss(predicted_standardized, target_standardized)
    raw_scale = y_scale.square().mean().clamp(min=1.0e-16)
    normalized_raw_mse = functional.mse_loss(predicted_raw, target_raw) / raw_scale
    return (
        1.0 - raw_loss_weight
    ) * standardized_mse + raw_loss_weight * normalized_raw_mse


def run_epoch(model, loader, optimizer, device, statistics, config, training):
    """Run one sample-weighted finite training or validation epoch."""

    if not isinstance(training, bool):
        raise ValueError("training must be boolean.")
    if training and optimizer is None:
        raise ValueError("Training epochs require an optimizer.")
    y_mean = statistics["y_mean"].to(device)
    y_scale = statistics["y_scale"].to(device)
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
                predicted,
                target,
                y_mean,
                y_scale,
                config["raw_loss_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("A non-finite coefficient loss was produced.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config["gradient_clip_norm"]
                )
                optimizer.step()
        batch_count = spectra.shape[0]
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count <= 0:
        raise ValueError("Epoch loader must contain at least one sample.")
    mean_loss = total_loss / total_count
    if not math.isfinite(mean_loss):
        raise RuntimeError("Epoch mean loss is non-finite.")
    return mean_loss


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def train_candidate(
    hidden_sizes,
    train_loader,
    validation_loader,
    device,
    statistics,
    config,
):
    """Train one fresh candidate and retain its best validation state."""

    matching_candidates = [
        candidate
        for candidate in config["candidates"]
        if tuple(candidate["hidden_sizes"]) == tuple(hidden_sizes)
    ]
    if len(matching_candidates) != 1:
        raise ValueError("hidden_sizes must identify exactly one configured candidate.")
    dropout = float(matching_candidates[0]["dropout"])
    model = OTSRegressor(tuple(hidden_sizes), dropout).to(device)
    optimizer_parameters = config["optimizer"]["parameters"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_parameters["lr"],
        betas=tuple(optimizer_parameters["betas"]),
        eps=optimizer_parameters["eps"],
        weight_decay=optimizer_parameters["weight_decay"],
        amsgrad=optimizer_parameters["amsgrad"],
    )
    scheduler_parameters = config["scheduler"]["parameters"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_parameters["mode"],
        factor=scheduler_parameters["factor"],
        patience=scheduler_parameters["patience"],
        threshold=scheduler_parameters["threshold"],
        threshold_mode=scheduler_parameters["threshold_mode"],
        cooldown=scheduler_parameters["cooldown"],
        min_lr=scheduler_parameters["min_lr"],
        eps=scheduler_parameters["eps"],
    )

    history = []
    best_epoch = None
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs_reached"
    started = time.time()

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            statistics,
            config,
            training=True,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            None,
            device,
            statistics,
            config,
            training=False,
        )
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        if validation_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config["patience"]:
            stop_reason = "patience_exhausted"
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Candidate did not produce a finite validation checkpoint.")
    return {
        "hidden_sizes": list(hidden_sizes),
        "dropout": dropout,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "state_dict": best_state,
        "elapsed_seconds": time.time() - started,
        "stop_reason": stop_reason,
    }


def _relative_path(path, base):
    return Path(os.path.relpath(str(path.resolve()), str(base.resolve()))).as_posix()


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _save_learning_curves(candidate_results, output_path):
    figure, axes = plt.subplots(
        1,
        len(candidate_results),
        figsize=(6 * len(candidate_results), 4),
        squeeze=False,
    )
    for axis, result in zip(axes[0], candidate_results):
        epochs = [record["epoch"] for record in result["history"]]
        axis.semilogy(
            epochs,
            [record["train_loss"] for record in result["history"]],
            label="train",
        )
        axis.semilogy(
            epochs,
            [record["validation_loss"] for record in result["history"]],
            label="validation",
        )
        axis.axvline(result["best_epoch"], color="black", linestyle=":", linewidth=1)
        axis.set_title(
            "{} (best epoch {})".format(
                "x".join(str(value) for value in result["hidden_sizes"]),
                result["best_epoch"],
            )
        )
        axis.set_xlabel("epoch")
        axis.set_ylabel("coefficient loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(str(temporary), dpi=160)
    plt.close(figure)
    os.replace(str(temporary), str(output_path))


def _serializable_candidate(result):
    return {key: value for key, value in result.items() if key != "state_dict"}


def _same_path(left, right):
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _validate_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string.".format(name))


def train_model(
    run_id,
    dataset_id,
    checkpoint_id,
    train_data_path,
    manifest_path,
    output_dir,
    config,
    overwrite=False,
):
    """Train candidates using train/validation only and save the best model."""

    for value, name in (
        (run_id, "run_id"),
        (dataset_id, "dataset_id"),
        (checkpoint_id, "checkpoint_id"),
    ):
        _validate_id(value, name)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean.")
    resolved_config, requested_device = resolve_training_config(config)
    device = choose_device(requested_device)
    set_determinism(resolved_config["seed"])

    train_data_path = Path(train_data_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.pt"
    history_path = output_dir / "training_history.json"
    curves_path = output_dir / "learning_curves.png"
    outputs = (checkpoint_path, history_path, curves_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing training artifacts: {}".format(existing)
        )

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("run_id") != run_id or manifest.get("dataset_id") != dataset_id:
        raise ValueError(
            "Manifest run_id/dataset_id do not match the training request."
        )
    try:
        manifest_train_path = manifest_path.parent / manifest["splits"]["train"]["file"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Manifest does not contain a valid train file reference."
        ) from error
    if not _same_path(train_data_path, manifest_train_path):
        raise ValueError("train_data_path does not match manifest splits.train.file.")

    dataset = load_dataset(train_data_path)
    x_train, y_train, x_validation, y_validation = split_train_validation(
        dataset,
        resolved_config["validation_fraction"],
        resolved_config["seed"],
    )
    x_mean, x_scale, y_mean, y_scale = normalization_statistics(x_train, y_train)
    statistics = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }
    normalized_train = (x_train - x_mean) / x_scale
    normalized_validation = (x_validation - x_mean) / x_scale
    pin_memory = device.type == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(resolved_config["seed"])
    train_loader = DataLoader(
        TensorDataset(normalized_train, y_train),
        batch_size=resolved_config["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        TensorDataset(normalized_validation, y_validation),
        batch_size=resolved_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    candidate_results = []
    for candidate in resolved_config["candidates"]:
        set_determinism(resolved_config["seed"])
        loader_generator.manual_seed(resolved_config["seed"])
        candidate_results.append(
            train_candidate(
                tuple(candidate["hidden_sizes"]),
                train_loader,
                validation_loader,
                device,
                statistics,
                resolved_config,
            )
        )
    selected = min(candidate_results, key=lambda result: result["best_validation_loss"])

    dtype_name = str(x_train.dtype).replace("torch.", "")
    checkpoint = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "checkpoint_id": checkpoint_id,
        "model_state": selected["state_dict"],
        "hidden_sizes": list(selected["hidden_sizes"]),
        "dropout": selected["dropout"],
        "layer_norm": True,
        "leaky_relu_slope": LEAKY_RELU_SLOPE,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "fourier_modes": FOURIER_MODES,
        "x_mean": x_mean.detach().cpu(),
        "x_scale": x_scale.detach().cpu(),
        "y_mean": y_mean.detach().cpu(),
        "y_scale": y_scale.detach().cpu(),
        "k_grid": dataset.k_grid.detach().cpu(),
        "wavelengths_nm": dataset.wavelengths_nm.detach().cpu(),
        "selected_candidate": {
            "hidden_sizes": list(selected["hidden_sizes"]),
            "dropout": selected["dropout"],
        },
        "best_epoch": selected["best_epoch"],
        "best_validation_loss": selected["best_validation_loss"],
        "device": str(device),
        "dtype": dtype_name,
        "training_config": resolved_config,
    }
    validate_checkpoint(checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary_checkpoint))
    os.replace(str(temporary_checkpoint), str(checkpoint_path))

    history = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "checkpoint_id": checkpoint_id,
        "train_data": _relative_path(train_data_path, history_path.parent),
        "dataset_manifest": _relative_path(manifest_path, history_path.parent),
        "checkpoint": _relative_path(checkpoint_path, history_path.parent),
        "training_config": resolved_config,
        "sample_counts": {
            "train": len(x_train),
            "validation": len(x_validation),
        },
        "candidates": [_serializable_candidate(result) for result in candidate_results],
        "selected_candidate": {
            "hidden_sizes": list(selected["hidden_sizes"]),
            "dropout": selected["dropout"],
            "best_epoch": selected["best_epoch"],
            "best_validation_loss": selected["best_validation_loss"],
        },
    }
    _write_json(history_path, history)
    _save_learning_curves(candidate_results, curves_path)
    return {
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "learning_curves_path": curves_path,
        "selected_candidate": history["selected_candidate"],
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
    }
