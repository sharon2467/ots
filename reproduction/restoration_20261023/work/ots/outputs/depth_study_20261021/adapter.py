#!/usr/bin/env python3
"""Isolated depth study adapter; canonical numerical functions stay untouched.

Run this file with the same arguments as main.py. install_adapter() is usable in an
isolated inference process. Do not import it into a shared production process.
Runtime snapshots are recovery artifacts, not canonical model.pt files; they
preserve current/best weights every ten epochs, on completion, and on a pause.
Every epoch has an atomic JSON progress record. SIGINT/SIGTERM request a pause
at the next complete epoch, as does an experiment-root PAUSE file. Snapshots do
not implement automatic continuation. export_snapshot_best() creates an
independently loadable best checkpoint.
"""

import inspect
import hashlib
import math
import os
from numbers import Integral
from pathlib import Path
import random
import signal
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "OTS_code_share" / "NN_hotE_large_lmRange"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PAUSE_FILE = EXPERIMENT_ROOT / "PAUSE"
SPLIT_SEED = 20261030
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from torch import nn
import model as canonical_model
import train as canonical_train


CanonicalRegressor = canonical_model.OTSRegressor
canonical_parse_candidates = canonical_train.parse_candidates
canonical_validate_hidden_sizes = canonical_model._validate_hidden_sizes
canonical_split_train_validation = canonical_train.split_train_validation


def validate_hidden_sizes(value, name="hidden_sizes"):
    if (not isinstance(value, (tuple, list)) or not value or any(
        not isinstance(size, Integral) or isinstance(size, bool) or int(size) <= 0
        for size in value
    )):
        raise ValueError("{} must contain one or more positive integers.".format(name))


class DepthRegressor(CanonicalRegressor):
    """Same canonical block repeated once per real hidden_sizes entry."""

    def __init__(self, hidden_sizes, dropout, *, input_dim=canonical_model.INPUT_DIM,
                 output_dim=canonical_model.OUTPUT_DIM):
        validate_hidden_sizes(hidden_sizes)
        if len(hidden_sizes) == 2:
            # Preserves construction, RNG consumption, keys, initialization and
            # forward exactly, including any future canonical depth-two change.
            super().__init__(hidden_sizes, dropout, input_dim=input_dim, output_dim=output_dim)
            return
        nn.Module.__init__(self)
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        for dimension in (input_dim, output_dim):
            if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError("Model dimensions must be positive integers.")
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.dropout = dropout
        layers = []
        previous = input_dim
        for width in self.hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width),
                           nn.LeakyReLU(canonical_model.LEAKY_RELU_SLOPE), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()  # inherited canonical Kaiming normal/bias zero


def parse_candidates(value):
    """Extend only the candidate depth; retain original dropout/duplicate policy."""
    if isinstance(value, str):
        candidates = []
        for raw in value.split(";"):
            if raw.strip():
                try:
                    candidates.append(tuple(int(part.strip()) for part in raw.split(",")))
                except ValueError as error:
                    raise ValueError("Candidate hidden sizes must be integers.") from error
    elif isinstance(value, (tuple, list)):
        candidates = list(value)
    else:
        raise ValueError("candidates must be a string or sequence.")
    parsed, seen = [], set()
    for candidate in candidates:
        if isinstance(candidate, dict):
            if set(candidate) != {"hidden_sizes", "dropout"}:
                raise ValueError("Candidate dictionaries require hidden_sizes and dropout only.")
            sizes, dropout = candidate["hidden_sizes"], float(candidate["dropout"])
        else:
            sizes, dropout = candidate, None
        validate_hidden_sizes(sizes, "candidate.hidden_sizes")
        sizes = tuple(int(size) for size in sizes)
        if dropout is None:
            dropout = 0.0 if max(sizes) <= 512 else 0.02
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("Candidate dropout must satisfy 0 <= dropout < 1.")
        if sizes in seen:
            raise ValueError("Duplicate candidate hidden_sizes are not allowed.")
        seen.add(sizes)
        parsed.append({"hidden_sizes": list(sizes), "dropout": float(dropout)})
    if not parsed:
        raise ValueError("At least one model candidate is required.")
    return parsed


def split_train_validation(dataset, fraction, seed):
    """Hold data membership fixed while --seed controls initialization/shuffling."""
    return canonical_split_train_validation(dataset, fraction, SPLIT_SEED)


def install_adapter():
    """Install depth, fixed split seed and isolated run/registry root."""
    canonical_model.OTSRegressor = DepthRegressor
    canonical_model._validate_hidden_sizes = validate_hidden_sizes
    canonical_train.OTSRegressor = DepthRegressor
    canonical_train.parse_candidates = parse_candidates
    canonical_train.split_train_validation = split_train_validation
    import main as canonical_main
    canonical_main._project_root = lambda: EXPERIMENT_ROOT


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def split_control(training):
    """Independent audit of canonical split membership and actual train statistics."""
    config = training["resolved_config"]
    sample_count = len(training["dataset"].spectra)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(sample_count, generator=generator)
    validation_count = min(sample_count - 1, max(1, int(round(sample_count * config["validation_fraction"]))))
    return {
        "split_seed": SPLIT_SEED, "training_seed": config["seed"],
        "index_hash_encoding": "contiguous torch.int64 native-endian bytes",
        "train_index_sha256": tensor_sha256(permutation[validation_count:]),
        "validation_index_sha256": tensor_sha256(permutation[:validation_count]),
        "sample_counts": {"train": sample_count - validation_count, "validation": validation_count},
        "statistics_sha256": {key: tensor_sha256(training[key])
                              for key in ("x_mean", "x_scale", "y_mean", "y_scale")},
        "statistics_hash_encoding": "contiguous tensor native-endian bytes",
        "statistics_dtype": str(training["x_mean"].dtype),
    }


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_metadata(candidate, training):
    """Record canonical metadata without recomputing any training quantities."""
    dataset = training["dataset"]
    metadata = {key: training[key] for key in ("run_id", "dataset_id", "checkpoint_id",
                "input_dim", "output_dim", "fourier_modes")}
    metadata.update({key: value.detach().cpu() for key, value in candidate["statistics"].items()})
    metadata.update({
        "hidden_sizes": list(candidate["hidden_sizes"]), "dropout": candidate["dropout"],
        "layer_norm": True, "leaky_relu_slope": canonical_model.LEAKY_RELU_SLOPE,
        "k_grid": dataset.k_grid.detach().cpu(),
        "wavelengths_nm": dataset.wavelengths_nm.detach().cpu(),
        "selected_candidate": {"hidden_sizes": list(candidate["hidden_sizes"]),
                               "dropout": candidate["dropout"]},
        "device": str(candidate["device"]),
        "dtype": str(training["x_train"].dtype).replace("torch.", ""),
        "training_config": candidate["config"],
    })
    if dataset.layout is not None:
        metadata["layout"] = dataset.layout
    return metadata


def export_snapshot_best(snapshot_path, output_path):
    """Export the actual best completed epoch; never promote partial current weights."""
    install_adapter()
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    if snapshot["best_state"] is None or snapshot["best_epoch"] is None:
        raise ValueError("Snapshot has no completed finite validation epoch.")
    checkpoint = dict(snapshot["checkpoint_metadata"])
    checkpoint.update(model_state=snapshot["best_state"], best_epoch=snapshot["best_epoch"],
                      best_validation_loss=snapshot["best_validation_loss"])
    canonical_model.validate_checkpoint(checkpoint)
    _atomic_save(checkpoint, output_path)
    return Path(output_path)


class EpochSnapshots:
    """Trace only train_candidate Python lines; never wrap numerical operations."""

    def __init__(self):
        self.target = canonical_train.train_candidate.__code__
        source, start = inspect.getsourcelines(canonical_train.train_candidate)
        self.boundaries = {start + index for index, line in enumerate(source)
                           if line.strip().startswith(("for epoch in range(", "if best_state is None"))}
        if len(self.boundaries) != 2:
            raise RuntimeError("Canonical epoch boundary changed; adapter needs review.")
        self.saved = {}
        self.active = None
        self.previous_trace = None
        self.previous_handlers = {}
        self.pause_requested = None
        self.logged = {}
        self.controls_written = set()

    def save(self, frame, reason):
        current = frame.f_locals
        if "model" not in current or "optimizer" not in current or "history" not in current:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not canonical_train.train_model.__code__:
            parent = parent.f_back
        if parent is None:
            # Direct unit-level train_candidate calls have no canonical run metadata.
            return
        training = parent.f_locals
        sizes = tuple(current["hidden_sizes"])
        completed = len(current["history"])
        identity = (str(training["output_dir"]), sizes)
        if self.saved.get(identity) == (completed, reason):
            return
        if reason == "completed_epoch" and completed == 0:
            return
        numpy_rng = np.random.get_state()
        snapshot = {
            "format": "ots_depth_runtime_snapshot_v1", "reason": reason,
            "completed_epochs": completed, "current_epoch": current.get("epoch", 0),
            "checkpoint_metadata": _checkpoint_metadata(current, training),
            "current_state": canonical_train._cpu_state_dict(current["model"]),
            "best_state": current["best_state"], "best_epoch": current["best_epoch"],
            "best_validation_loss": current["best_validation_loss"],
            "history": current["history"],
            "epochs_without_improvement": current["epochs_without_improvement"],
            "optimizer_state": current["optimizer"].state_dict(),
            "scheduler_state": current["scheduler"].state_dict(),
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "loader_generator_rng": training["loader_generator"].get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_data_path": str(training["train_data_path"].resolve()),
            "manifest_path": str(training["manifest_path"].resolve()),
            "continuation": "No automatic resume; snapshot is taken at a complete epoch boundary.",
        }
        path = Path(training["output_dir"]) / "depth_runtime" / ("x".join(map(str, sizes))) / "snapshot.pt"
        _atomic_save(snapshot, path)
        self.saved[identity] = (completed, reason)
        print("depth snapshot: {} epochs={}, reason={}".format(path, completed, reason), flush=True)

    def boundary(self, frame, final=False):
        current = frame.f_locals
        history = current.get("history", [])
        completed = len(history)
        if PAUSE_FILE.exists():
            self.pause_requested = "pause_file"
        parent = frame.f_back
        while parent is not None and parent.f_code is not canonical_train.train_model.__code__:
            parent = parent.f_back
        if parent is not None:
            control_path = Path(parent.f_locals["output_dir"]) / "depth_runtime" / "split_control.json"
            if str(control_path) not in self.controls_written:
                control_path.parent.mkdir(parents=True, exist_ok=True)
                canonical_train._write_json(control_path, split_control(parent.f_locals))
                self.controls_written.add(str(control_path))
        if completed:
            if parent is not None:
                sizes = tuple(current["hidden_sizes"])
                path = Path(parent.f_locals["output_dir"]) / "depth_runtime" / "x".join(map(str, sizes))
                identity = str(path)
                if self.logged.get(identity) != completed:
                    path.mkdir(parents=True, exist_ok=True)
                    canonical_train._write_json(path / "progress.json", {
                        "hidden_sizes": list(sizes), "dropout": current["dropout"],
                        "completed_epochs": completed, "history": history,
                        "best_epoch": current["best_epoch"],
                        "best_validation_loss": current["best_validation_loss"],
                    })
                    self.logged[identity] = completed
        if self.pause_requested:
            self.save(frame, self.pause_requested)
            raise SystemExit(75)
        if completed and (completed % 10 == 0 or final):
            self.save(frame, "candidate_complete" if final else "completed_epoch")

    def local_trace(self, frame, event, arg):
        if event == "line" and frame.f_lineno in self.boundaries:
            self.boundary(frame, final=frame.f_lineno == max(self.boundaries))
        elif event == "return":
            self.active = None
        return self.local_trace

    def global_trace(self, frame, event, arg):
        if frame.f_code is self.target:
            self.active = frame
            return self.local_trace
        return None

    def stop(self, signum, frame):
        self.pause_requested = "signal_{}".format(signum)
        if self.active is None:
            raise SystemExit(75)
        print("Pause requested; saving and stopping at the next complete epoch.", flush=True)

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        if self.previous_trace is not None:
            raise RuntimeError("Refusing to replace an existing Python trace.")
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[signum] = signal.signal(signum, self.stop)
        sys.settrace(self.global_trace)
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.settrace(self.previous_trace)
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)


def main():
    if PAUSE_FILE.exists():
        raise SystemExit("PAUSE exists: refusing to start a new experiment command.")
    install_adapter()
    os.chdir(EXPERIMENT_ROOT)
    import main as canonical_main
    with EpochSnapshots():
        canonical_main.main()


if __name__ == "__main__":
    main()
