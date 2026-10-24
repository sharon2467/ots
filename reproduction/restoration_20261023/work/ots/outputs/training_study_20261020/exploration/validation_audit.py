#!/usr/bin/env python3
"""Independent validation diagnostic; does not modify production evaluation.

Reads only the training Parquet, restores its held-out validation split, and
audits a frozen checkpoint. Never opens the independent test Parquet.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "OTS_code_share/NN_hotE_large_lmRange"
sys.path.insert(0, str(SOURCE))
from data import load_dataset, SPECIES_BLOCKS
from train import split_train_validation
from model import load_model_checkpoint
from evaluation import predict_coefficients, prediction_metrics, reconstruct_real_evdf


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(target, predicted, k_grid, velocity, indices, chunk_size):
    """Use canonical metrics in bounded VDF chunks; combine global R² exactly."""
    sums = {}
    moments = {key: [0., 0., 0., 0] for key in ("coefficient", "evdf")}
    records = []
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        truth, guess = target[start:stop], predicted[start:stop]
        metrics = prediction_metrics(truth, guess, k_grid, velocity=velocity)
        v, true_vdf = reconstruct_real_evdf(truth, k_grid, velocity=velocity)
        _, pred_vdf = reconstruct_real_evdf(guess, k_grid, velocity=velocity)
        for key, value in metrics.items():
            if key.endswith("_r2") or key.endswith("_median"):
                continue
            # RMSE must be combined in squared units.
            sums[key] = sums.get(key, 0.) + (value ** 2 if key.endswith("_rmse") else value) * (stop-start)
        for key, actual, estimate in (("coefficient", truth, guess), ("evdf", true_vdf, pred_vdf)):
            actual, estimate = actual.double(), estimate.double()
            acc = moments[key]
            acc[0] += float(((estimate-actual)**2).sum())
            acc[1] += float(actual.sum())
            acc[2] += float((actual**2).sum())
            acc[3] += actual.numel()
        error = pred_vdf.double() - true_vdf.double()
        relative = ((error**2).sum(1) / (true_vdf.double()**2).sum(1).clamp(min=1.e-30)).sqrt()
        negative = torch.trapz(torch.relu(-pred_vdf), v, dim=1) / torch.trapz(pred_vdf.abs(), v, dim=1).clamp(min=1.e-30)
        for row in range(stop-start):
            records.append({
                "train_parquet_physical_index_zero_based": int(indices[start+row]),
                "parquet_row_including_metadata_zero_based": int(indices[start+row])+1,
                "evdf_relative_l2": float(relative[row]),
                "evdf_rmse": float((error[row]**2).mean().sqrt()),
                "coefficient_rmse": float(((guess[row].double()-truth[row].double())**2).mean().sqrt()),
                "negative_mass_fraction": float(negative[row]),
            })
    result = {key: ((value/len(target))**.5 if key.endswith("_rmse") else value/len(target)) for key, value in sums.items()}
    for key, (sse, total, squares, count) in moments.items():
        result[key + "_r2"] = 1.-sse/max(squares-total*total/count, 1.e-30)
    relative = np.array([record["evdf_relative_l2"] for record in records])
    for label, quantile in (("p50", .5), ("p90", .9), ("p95", .95)):
        result["evdf_relative_l2_"+label] = float(np.quantile(relative, quantile))
    result["evdf_relative_l2_median"] = result["evdf_relative_l2_p50"]
    result["evdf_relative_l2_mean"] = float(relative.mean())
    return {"metrics": result, "per_sample": records}


def audit(run_dir, output_dir, chunk_size):
    run_dir = run_dir.resolve()
    history_path = run_dir / "training/training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint_path = (history_path.parent/history["checkpoint"]).resolve()
    model, checkpoint = load_model_checkpoint(checkpoint_path, "cpu", history["checkpoint_id"])
    config = checkpoint["training_config"]
    manifest_path = (history_path.parent/history["dataset_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = (manifest_path.parent/manifest["splits"]["train"]["file"]).resolve()
    if train_path != (history_path.parent/history["train_data"]).resolve():
        raise ValueError("Training history and manifest disagree on training Parquet")
    if manifest["dataset_id"] != checkpoint["dataset_id"]:
        raise ValueError("Checkpoint and training manifest dataset IDs disagree")
    dataset = load_dataset(train_path, manifest.get("physical_configuration"))
    x_train, y_train, x_validation, y_validation = split_train_validation(dataset, config["validation_fraction"], config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    permutation = torch.randperm(len(dataset.spectra), generator=generator)
    validation_indices = permutation[:len(x_validation)]
    training_indices = permutation[len(x_validation):]
    assert torch.equal(dataset.spectra[validation_indices], x_validation)
    assert torch.equal(dataset.coefficients[validation_indices], y_validation)
    assert not set(training_indices.tolist()).intersection(validation_indices.tolist())
    predicted = predict_coefficients(model, x_validation, checkpoint, chunk_size, "cpu")
    baseline = y_train.mean(0, keepdim=True).expand(len(y_validation), -1)
    layout = checkpoint.get("layout")
    modes = checkpoint["fourier_modes"]
    selected = [(species, key) for species, key in SPECIES_BLOCKS if layout[key]["used_in_model"]] if layout else [("e", "electron_dis")]
    species_results = {}
    for index, (species, key) in enumerate(selected):
        block = slice(index*2*modes, (index+1)*2*modes)
        k_grid = dataset.k_grid[index*modes:(index+1)*modes]
        velocity = layout[key]["velocity_grid_m_s"] if layout else None
        species_results[species] = {
            "train_mean_coefficients": y_train[:, block].mean(0).tolist(),
            "model": summarize(y_validation[:, block], predicted[:, block], k_grid, velocity, validation_indices, chunk_size),
            "train_mean_curve_baseline": summarize(y_validation[:, block], baseline[:, block], k_grid, velocity, validation_indices, chunk_size),
        }
    score = float(np.mean([entry["model"]["metrics"]["evdf_relative_l2_mean"] for entry in species_results.values()]))
    baseline_score = float(np.mean([entry["train_mean_curve_baseline"]["metrics"]["evdf_relative_l2_mean"] for entry in species_results.values()]))
    payload = {
        "diagnostic": "Independent training-Parquet validation audit; production evaluation unchanged",
        "run_id": history["run_id"], "dataset_id": history["dataset_id"], "checkpoint_id": history["checkpoint_id"],
        "test_data_read": False, "cpu_threads": torch.get_num_threads(), "vdf_chunk_size": chunk_size,
        "training_config": config,
        "inputs": {"training_manifest": str(manifest_path), "training_manifest_sha256": file_hash(manifest_path), "train_parquet": str(train_path), "train_parquet_sha256": file_hash(train_path), "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path)},
        "split": {"training_count": len(y_train), "validation_count": len(y_validation), "validation_indices": validation_indices.tolist(), "validation_indices_sha256": hashlib.sha256(np.asarray(validation_indices.tolist(), dtype="<i8").tobytes()).hexdigest(), "index_hash_format": "ordered zero-based physical indices, little-endian int64", "disjoint": True},
        "selection_score": score, "selection_rule": "equal species macro mean of validation VDF relative-L2 mean; electron-only for fixed_ion",
        "baseline_selection_score": baseline_score,
        "worst_species_relative_l2_mean": max(entry["model"]["metrics"]["evdf_relative_l2_mean"] for entry in species_results.values()),
        "r2_definition": "canonical global flattened R2 against one grand mean; accumulated in float64",
        "vdf_truth_definition": "inverse transform of target Fourier coefficients, not original sampled VDF",
        "species": species_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/(history["run_id"]+"_validation_metrics.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "selection_score": score, "baseline_selection_score": baseline_score, "validation_count": len(y_validation)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for run_dir in args.run_dirs:
        audit(run_dir, args.output_dir, args.chunk_size)
