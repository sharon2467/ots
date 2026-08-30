# -*- coding: utf-8 -*-
"""Canonical OTS regressor, physical projection, and checkpoint loading."""

import math
from numbers import Integral
from pathlib import Path

import torch
from torch import nn


INPUT_DIM = 1224
OUTPUT_DIM = 160
FOURIER_MODES = 80
LEAKY_RELU_SLOPE = 0.05

CHECKPOINT_FIELDS = {
    "run_id",
    "dataset_id",
    "checkpoint_id",
    "model_state",
    "hidden_sizes",
    "dropout",
    "layer_norm",
    "leaky_relu_slope",
    "input_dim",
    "output_dim",
    "fourier_modes",
    "x_mean",
    "x_scale",
    "y_mean",
    "y_scale",
    "k_grid",
    "wavelengths_nm",
    "selected_candidate",
    "best_epoch",
    "best_validation_loss",
    "device",
    "dtype",
    "training_config",
}

TRAINING_CONFIG_FIELDS = {
    "seed",
    "validation_fraction",
    "candidates",
    "batch_size",
    "max_epochs",
    "patience",
    "learning_rate",
    "weight_decay",
    "raw_loss_weight",
    "gradient_clip_norm",
    "optimizer",
    "scheduler",
}


class OTSRegressor(nn.Module):
    """Map a 1224-point OTS spectrum to 80 complex Fourier coefficients."""

    def __init__(self, hidden_sizes, dropout):
        super().__init__()
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
            raise ValueError("hidden_sizes must contain two positive integers.")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        first, second = (int(hidden_sizes[0]), int(hidden_sizes[1]))
        self.hidden_sizes = (first, second)
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, first),
            nn.LayerNorm(first),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(second, OUTPUT_DIM),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=LEAKY_RELU_SLOPE)
                nn.init.zeros_(module.bias)

    def forward(self, spectra):
        return self.network(spectra)


def hermitian_project(coefficients):
    """Project the final 160 values so ``F(-K) = conjugate(F(K))``."""

    if not isinstance(coefficients, torch.Tensor) or coefficients.ndim < 1:
        raise ValueError("coefficients must be a tensor with a final dimension.")
    if coefficients.shape[-1] != OUTPUT_DIM:
        raise ValueError("coefficients must have final dimension 160.")
    real = coefficients[..., :FOURIER_MODES]
    imaginary = coefficients[..., FOURIER_MODES:]
    projected_real = 0.5 * (real + torch.flip(real, dims=(-1,)))
    projected_imaginary = 0.5 * (imaginary - torch.flip(imaginary, dims=(-1,)))
    return torch.cat((projected_real, projected_imaginary), dim=-1)


def _require_exact_fields(payload, expected, name):
    fields = set(payload)
    missing = sorted(expected - fields)
    unexpected = sorted(fields - expected)
    if missing or unexpected:
        raise ValueError(
            "{} fields do not match the canonical schema; missing={}, "
            "unexpected={}.".format(name, missing, unexpected)
        )


def _require_nonempty_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string.".format(name))


def _validate_hidden_sizes(value, name):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            not isinstance(size, Integral) or isinstance(size, bool) or int(size) <= 0
            for size in value
        )
    ):
        raise ValueError("{} must contain two positive integers.".format(name))


def _validate_training_config(config):
    if not isinstance(config, dict):
        raise ValueError("training_config must be a dictionary.")
    _require_exact_fields(config, TRAINING_CONFIG_FIELDS, "training_config")

    if not isinstance(config["seed"], Integral) or isinstance(config["seed"], bool):
        raise ValueError("training_config.seed must be an integer.")
    if not 0.0 < float(config["validation_fraction"]) < 1.0:
        raise ValueError("training_config.validation_fraction must be between 0 and 1.")
    for name in ("batch_size", "max_epochs", "patience"):
        value = config[name]
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or int(value) <= 0
        ):
            raise ValueError("training_config.{} must be positive.".format(name))
    for name in ("learning_rate", "gradient_clip_norm"):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("training_config.{} must be positive.".format(name))
    weight_decay = float(config["weight_decay"])
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("training_config.weight_decay must be non-negative.")
    raw_loss_weight = float(config["raw_loss_weight"])
    if not math.isfinite(raw_loss_weight) or not 0.0 <= raw_loss_weight <= 1.0:
        raise ValueError("training_config.raw_loss_weight must be between 0 and 1.")

    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("training_config.candidates must be a non-empty list.")
    candidate_identities = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each training candidate must be a dictionary.")
        _require_exact_fields(candidate, {"hidden_sizes", "dropout"}, "candidate")
        _validate_hidden_sizes(candidate["hidden_sizes"], "candidate.hidden_sizes")
        dropout = float(candidate["dropout"])
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("candidate.dropout must satisfy 0 <= dropout < 1.")
        identity = (tuple(int(value) for value in candidate["hidden_sizes"]), dropout)
        if identity in candidate_identities:
            raise ValueError("training_config contains duplicate candidates.")
        candidate_identities.add(identity)

    optimizer = config["optimizer"]
    if not isinstance(optimizer, dict):
        raise ValueError("training_config.optimizer must be a dictionary.")
    _require_exact_fields(optimizer, {"name", "parameters"}, "optimizer")
    if optimizer["name"] != "AdamW":
        raise ValueError("The canonical optimizer is AdamW.")
    optimizer_parameters = optimizer["parameters"]
    if not isinstance(optimizer_parameters, dict):
        raise ValueError("optimizer.parameters must be a dictionary.")
    _require_exact_fields(
        optimizer_parameters,
        {"lr", "betas", "eps", "weight_decay", "amsgrad"},
        "optimizer.parameters",
    )
    optimizer_lr = float(optimizer_parameters["lr"])
    optimizer_eps = float(optimizer_parameters["eps"])
    optimizer_weight_decay = float(optimizer_parameters["weight_decay"])
    if not math.isfinite(optimizer_lr) or optimizer_lr <= 0:
        raise ValueError("optimizer lr must be finite and positive.")
    if not math.isclose(
        optimizer_lr, float(config["learning_rate"]), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("optimizer lr must match training_config.learning_rate.")
    if not math.isfinite(optimizer_eps) or optimizer_eps <= 0:
        raise ValueError("optimizer eps must be finite and positive.")
    if not math.isfinite(optimizer_weight_decay) or optimizer_weight_decay < 0:
        raise ValueError("optimizer weight_decay must be finite and non-negative.")
    if not math.isclose(
        optimizer_weight_decay,
        float(config["weight_decay"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "optimizer weight_decay must match training_config.weight_decay."
        )
    betas = optimizer_parameters["betas"]
    if (
        not isinstance(betas, (tuple, list))
        or len(betas) != 2
        or any(
            not math.isfinite(float(beta)) or not 0.0 <= float(beta) < 1.0
            for beta in betas
        )
    ):
        raise ValueError("optimizer betas must contain two values in [0, 1).")
    if not isinstance(optimizer_parameters["amsgrad"], bool):
        raise ValueError("optimizer amsgrad must be boolean.")

    scheduler = config["scheduler"]
    if not isinstance(scheduler, dict):
        raise ValueError("training_config.scheduler must be a dictionary.")
    _require_exact_fields(scheduler, {"name", "parameters"}, "scheduler")
    if scheduler["name"] != "ReduceLROnPlateau":
        raise ValueError("The canonical scheduler is ReduceLROnPlateau.")
    scheduler_parameters = scheduler["parameters"]
    if not isinstance(scheduler_parameters, dict):
        raise ValueError("scheduler.parameters must be a dictionary.")
    _require_exact_fields(
        scheduler_parameters,
        {
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
        },
        "scheduler.parameters",
    )
    if scheduler_parameters["mode"] != "min":
        raise ValueError("scheduler mode must be min.")
    factor = float(scheduler_parameters["factor"])
    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("scheduler factor must be between 0 and 1.")
    for name in ("patience", "cooldown"):
        value = scheduler_parameters[name]
        if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
            raise ValueError("scheduler {} must be non-negative.".format(name))
    if scheduler_parameters["threshold_mode"] not in ("rel", "abs"):
        raise ValueError("scheduler threshold_mode must be rel or abs.")
    for name in ("threshold", "min_lr", "eps"):
        value = float(scheduler_parameters[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError("scheduler {} must be non-negative.".format(name))


def _validate_checkpoint_tensor(payload, name, shape, positive=False):
    tensor = payload[name]
    if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
        raise ValueError("{} must be a tensor with shape {}.".format(name, shape))
    if not torch.isfinite(tensor).all():
        raise ValueError("{} contains non-finite values.".format(name))
    if positive and not torch.all(tensor > 0):
        raise ValueError("{} must be strictly positive.".format(name))


def validate_checkpoint(payload):
    """Validate the complete canonical checkpoint schema."""

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if "pet1" in payload:
        raise ValueError("Legacy pet1 checkpoints are not supported.")
    _require_exact_fields(payload, CHECKPOINT_FIELDS, "checkpoint")

    for name in ("run_id", "dataset_id", "checkpoint_id"):
        _require_nonempty_id(payload[name], name)
    if payload["input_dim"] != INPUT_DIM or payload["output_dim"] != OUTPUT_DIM:
        raise ValueError("Checkpoint input/output dimensions are invalid.")
    if payload["fourier_modes"] != FOURIER_MODES:
        raise ValueError("Checkpoint Fourier mode count must be 80.")
    _validate_hidden_sizes(payload["hidden_sizes"], "hidden_sizes")
    dropout = float(payload["dropout"])
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("Checkpoint dropout is invalid.")
    if payload["layer_norm"] is not True:
        raise ValueError("Checkpoint must enable LayerNorm.")
    if not math.isclose(
        float(payload["leaky_relu_slope"]),
        LEAKY_RELU_SLOPE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Checkpoint LeakyReLU slope is invalid.")
    if not isinstance(payload["model_state"], dict) or not payload["model_state"]:
        raise ValueError("model_state must be a non-empty dictionary.")

    _validate_checkpoint_tensor(payload, "x_mean", (INPUT_DIM,))
    _validate_checkpoint_tensor(payload, "x_scale", (INPUT_DIM,), positive=True)
    _validate_checkpoint_tensor(payload, "y_mean", (OUTPUT_DIM,))
    _validate_checkpoint_tensor(payload, "y_scale", (OUTPUT_DIM,), positive=True)
    _validate_checkpoint_tensor(payload, "k_grid", (FOURIER_MODES,))
    _validate_checkpoint_tensor(payload, "wavelengths_nm", (INPUT_DIM,))
    _validate_training_config(payload["training_config"])

    selected = payload["selected_candidate"]
    if not isinstance(selected, dict):
        raise ValueError("selected_candidate must be a dictionary.")
    _require_exact_fields(selected, {"hidden_sizes", "dropout"}, "selected_candidate")
    _validate_hidden_sizes(selected["hidden_sizes"], "selected_candidate.hidden_sizes")
    if list(selected["hidden_sizes"]) != list(payload["hidden_sizes"]):
        raise ValueError("selected_candidate hidden sizes do not match the model.")
    if not math.isclose(
        float(selected["dropout"]), float(payload["dropout"]), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("selected_candidate dropout does not match the model.")
    configured_candidates = payload["training_config"].get("candidates", [])
    if selected not in configured_candidates:
        raise ValueError("selected_candidate is absent from training_config.")

    best_epoch = payload["best_epoch"]
    if (
        not isinstance(best_epoch, Integral)
        or isinstance(best_epoch, bool)
        or int(best_epoch) <= 0
    ):
        raise ValueError("best_epoch must be a positive integer.")
    best_loss = float(payload["best_validation_loss"])
    if not math.isfinite(best_loss) or best_loss < 0:
        raise ValueError("best_validation_loss must be finite and non-negative.")
    if not isinstance(payload["device"], str) or not payload["device"]:
        raise ValueError("device must be a non-empty string.")
    if payload["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64.")


def load_model_checkpoint(path, device, checkpoint_id):
    """Load a trusted canonical checkpoint selected by its checkpoint ID."""

    device = torch.device(device)
    payload = torch.load(str(Path(path)), map_location=device)
    validate_checkpoint(payload)
    _require_nonempty_id(checkpoint_id, "checkpoint_id")
    if payload["checkpoint_id"] != checkpoint_id:
        raise ValueError(
            "Checkpoint checkpoint_id {!r} does not match requested {!r}.".format(
                payload["checkpoint_id"], checkpoint_id
            )
        )

    dtype = torch.float64 if payload["dtype"] == "float64" else torch.float32
    regressor = OTSRegressor(
        tuple(payload["hidden_sizes"]), float(payload["dropout"])
    ).to(device=device, dtype=dtype)
    regressor.load_state_dict(payload["model_state"], strict=True)
    regressor.eval()
    return regressor, payload
