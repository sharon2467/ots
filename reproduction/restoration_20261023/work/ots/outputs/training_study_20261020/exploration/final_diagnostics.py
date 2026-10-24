#!/usr/bin/env python3
"""Independent frozen-run test diagnostics, intended ONLY after tuning freezes.

Reads canonical evaluation predictions plus the training-only validation audit
baseline and manifest moments. Never trains, rewrites canonical outputs, chooses
hyperparameters, or applies postprocessing. Signed VDF values are preserved.
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "OTS_code_share/NN_hotE_large_lmRange"))
from evaluation import reconstruct_real_evdf

E_CHARGE = 1.602176634e-19
SPECIES_KEYS = {"e": "electron_dis", "i1": "ion1_dis", "i2": "ion2_dis"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def trapz(values, velocity):
    return np.trapezoid(values, velocity, axis=-1) if hasattr(np, "trapezoid") else np.trapz(values, velocity, axis=-1)


def moments(vdf, velocity, mass_kg):
    """Signed moments; positive integral alone does not ensure physical VDF."""
    integral = trapz(vdf, velocity)
    positive_mass = integral > 0
    mean = np.full(len(vdf), np.nan)
    mean[positive_mass] = trapz(vdf[positive_mass]*velocity, velocity)/integral[positive_mass]
    variance = np.full(len(vdf), np.nan)
    variance[positive_mass] = trapz(vdf[positive_mass]*(velocity-mean[positive_mass, None])**2, velocity)/integral[positive_mass]
    temperature = mass_kg/E_CHARGE*variance
    return {
        "integral_mass": integral,
        "mean_velocity_m_s": mean,
        "central_second_moment_m2_s2": variance,
        "temperature_ev": temperature,
        "nonpositive_mass_count": int((~positive_mass).sum()),
        "negative_temperature_count": int((temperature < 0).sum()),
        "valid_nonnegative_temperature_count": int((positive_mass & (temperature >= 0) & np.isfinite(temperature)).sum()),
        "mass_absolute_error_from_one_mean": float(np.abs(integral-1).mean()),
    }


def metric_vectors(target, predicted, velocity):
    squared = ((predicted-target)**2).sum(1)
    relative = np.sqrt(squared/np.maximum((target**2).sum(1), 1.e-30))
    negative = trapz(np.maximum(-predicted, 0), velocity)/np.maximum(trapz(np.abs(predicted), velocity), 1.e-30)
    return relative, negative


def summarize(target, predicted, velocity):
    relative, negative = metric_vectors(target, predicted, velocity)
    centered = target-target.mean()
    return {
        "evdf_relative_l2_mean": float(relative.mean()),
        "evdf_relative_l2_median": float(np.median(relative)),
        "evdf_relative_l2_p90": float(np.quantile(relative, .90)),
        "evdf_relative_l2_p95": float(np.quantile(relative, .95)),
        "evdf_global_flattened_r2": float(1-((predicted-target)**2).sum()/max((centered**2).sum(), 1.e-30)),
        "negative_mass_fraction_mean": float(negative.mean()),
    }, relative, negative


def paired_bootstrap(model_relative, baseline_relative, repetitions, seed):
    """Rows are species, columns identical test sample indices across species."""
    rng = np.random.default_rng(seed)
    species_count, sample_count = model_relative.shape
    model_draws = np.empty((repetitions, species_count))
    baseline_draws = np.empty_like(model_draws)
    # Bound temporary bootstrap memory independently of test sample count.
    for start in range(0, repetitions, 100):
        stop = min(repetitions, start+100)
        indices = rng.integers(0, sample_count, size=(stop-start, sample_count))
        model_draws[start:stop] = model_relative[:, indices].mean(2).T
        baseline_draws[start:stop] = baseline_relative[:, indices].mean(2).T
    interval = lambda values: np.quantile(values, [.025, .975], axis=0).T.tolist()
    return {
        "repetitions": repetitions, "seed": seed, "method": "paired percentile bootstrap; same sample indices for model/baseline and every species",
        "species_model_relative_l2_mean_ci95": interval(model_draws),
        "species_baseline_relative_l2_mean_ci95": interval(baseline_draws),
        "species_baseline_minus_model_mean_ci95": interval(baseline_draws-model_draws),
        "macro_model_relative_l2_mean_ci95": interval(model_draws.mean(1)),
        "macro_baseline_relative_l2_mean_ci95": interval(baseline_draws.mean(1)),
        "macro_baseline_minus_model_mean_ci95": interval((baseline_draws-model_draws).mean(1)),
        "interpretation": "conditional uncertainty for the frozen model on iid sampled synthetic states; excludes retraining and hyperparameter-selection uncertainty",
    }


def temperature_error(left, right):
    valid = np.isfinite(left) & np.isfinite(right) & (left >= 0) & (right >= 0)
    difference = left[valid]-right[valid]
    return {"valid_pair_count": int(valid.sum()), "excluded_pair_count": int((~valid).sum()), "bias_ev": float(difference.mean()) if len(difference) else None, "mae_ev": float(np.abs(difference).mean()) if len(difference) else None}


def diagnose(run_dir, output_root, validation_root, repetitions, seed, chunk_size):
    run_dir = run_dir.resolve()
    config = json.loads((run_dir/"pipeline_config.json").read_text())
    run_id = config["run_id"]
    output_dir = output_root/run_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite independent diagnostics: {output_dir}")
    validation_path = validation_root/(run_id+"_validation_metrics.json")
    validation = json.loads(validation_path.read_text())
    manifest_path = run_dir/"data/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if validation["checkpoint_id"] != config["checkpoint_id"] or validation["dataset_id"] != config["dataset_id"] or manifest["dataset_id"] != config["dataset_id"]:
        raise ValueError("Run, baseline validation audit, and manifest IDs must agree")
    names = list(validation["species"])
    output_dir.mkdir(parents=True)
    results, model_relative, baseline_relative, inputs = {}, [], [], []
    expected_count = manifest["splits"]["test"]["physical_samples"]
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 4), squeeze=False)
    for axis, species in zip(axes[0], names):
        prediction_path = run_dir/"evaluation"/species/"predictions.npz"
        if names == ["e"]:
            prediction_path = run_dir/"evaluation/predictions.npz"
        inputs.append({"path": str(prediction_path), "sha256": sha256(prediction_path)})
        with np.load(prediction_path, allow_pickle=False) as predictions:
            velocity = predictions["velocity_m_s"].astype(np.float64)
            target = predictions["target_evdf"].astype(np.float64)
            predicted = predictions["predicted_evdf"].astype(np.float64)
            k_grid = predictions["k_grid"].copy()
        if len(target) != expected_count or target.shape != predicted.shape:
            raise ValueError("Canonical predictions disagree with manifest sample count")
        mean_coefficients = torch.tensor(validation["species"][species]["train_mean_coefficients"], dtype=torch.float64).reshape(1, -1)
        _, baseline_curve = reconstruct_real_evdf(mean_coefficients, torch.tensor(k_grid, dtype=torch.float64), velocity=velocity)
        baseline = np.broadcast_to(baseline_curve.numpy(), target.shape)
        model_metrics, relative, negative = summarize(target, predicted, velocity)
        baseline_metrics, relative_baseline, baseline_negative = summarize(target, baseline, velocity)
        model_relative.append(relative)
        baseline_relative.append(relative_baseline)
        physical = manifest["physical_configuration"]
        mass = physical[SPECIES_KEYS[species]]["mass_kg"]
        moment_results = {}
        for label, values in (("target_fourier", target), ("model", predicted), ("train_mean_baseline", baseline)):
            parts = [moments(values[start:start+chunk_size], velocity, mass) for start in range(0, len(values), chunk_size)]
            moment_results[label] = {}
            for key in parts[0]:
                if isinstance(parts[0][key], np.ndarray):
                    moment_results[label][key] = np.concatenate([part[key] for part in parts])
                elif key.endswith("_count"):
                    moment_results[label][key] = sum(part[key] for part in parts)
                else:
                    moment_results[label][key] = float(np.mean(np.abs(moment_results[label]["integral_mass"]-1)))
        original_temperature = np.array([row["species"][species]["actual_temperature_ev"] for row in manifest["splits"]["test"]["samples"]])
        target_temperature = moment_results["target_fourier"]["temperature_ev"]
        predicted_temperature = moment_results["model"]["temperature_ev"]
        results[species] = {
            "model": model_metrics, "train_mean_baseline": baseline_metrics,
            "baseline_minus_model_relative_l2_mean": float((relative_baseline-relative).mean()),
            "moments": moment_results,
            "temperature_comparisons": {
                "target_fourier_minus_original_vdf_label_truncation": temperature_error(target_temperature, original_temperature),
                "model_minus_target_fourier_model_error": temperature_error(predicted_temperature, target_temperature),
                "model_minus_original_vdf_combined_error": temperature_error(predicted_temperature, original_temperature),
            },
            "per_sample": [{"sample_index_zero_based": i, "model_relative_l2": relative[i], "baseline_relative_l2": relative_baseline[i], "model_negative_mass_fraction": negative[i], "baseline_negative_mass_fraction": baseline_negative[i], "original_vdf_actual_temperature_ev": original_temperature[i]} for i in range(len(relative))],
        }
        for values, label in ((relative, "Model"), (relative_baseline, "Training mean baseline")):
            axis.step(np.sort(values), np.arange(1, len(values)+1)/len(values), where="post", label=label)
        axis.set(xlabel="VDF relative L2 error", ylabel="Empirical CDF", title=species)
        axis.grid(alpha=.25)
        axis.legend()
        ordered = np.argsort(relative)
        representative = [int(ordered[len(ordered)//2]), int(ordered[-1])]
        comparison, panels = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
        for panel, index, label in zip(panels[0], representative, ("Median error", "Worst error")):
            panel.plot(velocity, target[index], "k-", label="Fourier target")
            panel.plot(velocity, predicted[index], label="Prediction")
            panel.plot(velocity, baseline[index], "--", alpha=.7, label="Training mean")
            panel.set(xlabel="Velocity (m/s)", ylabel="Signed VDF", title=f"{species}: {label}; sample {index}; L2={relative[index]:.3g}")
            panel.grid(alpha=.25)
            panel.legend()
        comparison.tight_layout()
        comparison.savefig(output_dir/(species+"_representative.png"), dpi=160)
        plt.close(comparison)
        results[species]["representative_sample_indices"] = representative
    fig.tight_layout()
    fig.savefig(output_dir/"error_cdf.png", dpi=160)
    plt.close(fig)
    model_relative, baseline_relative = np.stack(model_relative), np.stack(baseline_relative)
    payload = {
        "run_id": run_id, "checkpoint_id": config["checkpoint_id"], "dataset_id": config["dataset_id"],
        "purpose": "Final frozen-run diagnostics, never a tuning criterion",
        "cpu_threads": torch.get_num_threads(), "test_sample_count": expected_count,
        "inputs": inputs + [{"path": str(validation_path), "sha256": sha256(validation_path)}, {"path": str(manifest_path), "sha256": sha256(manifest_path)}],
        "species_order": names, "species": results,
        "macro_model_relative_l2_mean": float(model_relative.mean()),
        "macro_baseline_relative_l2_mean": float(baseline_relative.mean()),
        "macro_baseline_minus_model_relative_l2_mean": float((baseline_relative-model_relative).mean()),
        "bootstrap": paired_bootstrap(model_relative, baseline_relative, repetitions, seed),
        "r2_definition": "all sample and velocity coordinates flattened against one grand mean",
        "moment_cautions": ["Signed VDF preserved; no clipping or normalization", "Nonpositive mass yields null velocity/temperature; negative temperatures counted explicitly", "Positive mass and temperature alone do not establish physicality", "Moment comparisons exclude invalid or negative temperatures and report exclusion counts", "Target-Fourier versus original manifest temperature measures label truncation/discretization; model versus Fourier target isolates prediction error"],
        "postprocessing": "No postprocessing performed; any clipping/normalization recommendation must be selected on validation only",
    }
    output_path = output_dir/"detailed_test.json"
    output_path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output_path), "macro_model_relative_l2_mean": payload["macro_model_relative_l2_mean"], "macro_baseline_relative_l2_mean": payload["macro_baseline_relative_l2_mean"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent/"final_diagnostics")
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1 or args.chunk_size < 1:
        parser.error("bootstrap repetitions and chunk size must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for run in args.run_dirs:
        diagnose(run, args.output_root, args.validation_root, args.bootstrap_repetitions, args.bootstrap_seed, args.chunk_size)
