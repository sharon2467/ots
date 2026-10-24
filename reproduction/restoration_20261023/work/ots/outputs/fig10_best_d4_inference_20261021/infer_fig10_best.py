"""Standalone, frozen-checkpoint inference for the three uploaded Fig.10 spectra.

Only new input processing: each aligned spectrum / max(abs(spectrum)).
No training, registry publication, fitting, mask handling or VDF correction.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SOURCE = REPO / "OTS_code_share" / "NN_hotE_large_lmRange"
ALIGNMENT = REPO / "outputs" / "fig10_input_alignment_20261021"
STUDY = REPO / "outputs" / "training_scale_20261020"
DEPTH = REPO / "outputs/depth_study_20261021"
OLD_CHECKPOINT = SOURCE / "runs/scale_fig_26k_wide/training/model.pt"
CHECKPOINT = DEPTH / "runs/depth_fig_d4_s0/training/model.pt"
PLAN = HERE / "inference_plan.json"
CHECKPOINT_SHA = "b822cac9b18878dd45ee324c2bbf2e480f4ece56742e10e1126a571bcc737019"
SELECTION_SHA = "233e0ecb9936f1e41a2ad62c1adc68503bd9cb487247963fda27755c939bf34b"
SPECIES = [("e", "electron_dis"), ("H", "ion1_dis"), ("C", "ion2_dis")]

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
sys.path.insert(0, str(DEPTH))
import adapter
adapter.install_adapter()  # This standalone process only; no training entry is called.
from model import load_model_checkpoint
from evaluation import predict_coefficients, reconstruct_real_evdf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def registry_snapshot():
    return {str(root): {str(p.relative_to(root)): os.readlink(p)
                       for p in sorted(root.rglob("*")) if p.is_symlink()}
            for root in (SOURCE / "registry", DEPTH / "registry")}


def verify_plan(plan):
    for path, expected in plan["protected_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Protected artifact changed: {path}")
    assert registry_snapshot() == plan["registry_snapshots"]


def inputs():
    a = read_json(ALIGNMENT / "fig10a/alignment.json")
    b = read_json(ALIGNMENT / "fig10b/input_audit.json")
    c = read_json(ALIGNMENT / "fig10c/alignment.json")
    return [
        ("fig10a", Path(a["output"]), "intensity_counts", a["output_sha256"], a, 556),
        ("fig10b", Path(b["original_copy"]), "intensity", b["source_sha256"], b, 480),
        ("fig10c", Path(c["output"]), "intensity", c["output_sha256"], c, 480),
    ]


def describe_vdf(velocity, values, time_ns, rows):
    # Diagnostics only: these calculations do not replace or change the VDF.
    v = velocity.astype(np.float64)
    f = values.astype(np.float64)
    signed_area = np.trapezoid(f, v, axis=1)
    absolute_area = np.trapezoid(np.abs(f), v, axis=1)
    negative_area = np.trapezoid(np.maximum(-f, 0), v, axis=1)
    fraction = negative_area / np.maximum(absolute_area, 1e-30)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "velocity_range_m_s": [float(v[0]), float(v[-1])],
        "vdf_range_s_per_m": [float(f.min()), float(f.max())],
        "signed_integral_range": [float(signed_area.min()), float(signed_area.max())],
        "signed_integral_mean": float(signed_area.mean()),
        "rows_with_nonpositive_integral": int((signed_area <= 0).sum()),
        "negative_mass_fraction_definition": "integral(max(-f,0))/integral(abs(f))",
        "negative_mass_fraction_mean": float(fraction.mean()),
        "negative_mass_fraction_p95": float(np.quantile(fraction, .95)),
        "negative_mass_fraction_max": float(fraction.max()),
        "negative_mass_fraction_max_array_row": int(np.argmax(fraction)),
        "negative_mass_fraction_max_source_row_index": int(rows[np.argmax(fraction)]),
        "negative_mass_fraction_max_time_ns": float(time_ns[np.argmax(fraction)]),
        "rows_with_any_negative_vdf": int((f.min(axis=1) < 0).sum()),
        "ground_truth_available": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; existing results are immutable.")
    if out == SOURCE or SOURCE in out.parents:
        raise ValueError("Inference artifacts must stay outside the training source tree.")
    started = time.perf_counter()
    torch.set_num_threads(2)
    plan = read_json(PLAN)
    verify_plan(plan)
    protected = read_json(STUDY / "source_hashes_before.json")
    for name, expected in protected.items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Protected source differs from frozen study: {name}")
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    selection = REPO / "outputs/fig_unified_test_20261021/results.json"
    assert sha(selection) == SELECTION_SHA
    comparison = read_json(selection)
    assert len(comparison["models"]) == 20
    chosen = min(comparison["models"], key=lambda item: item["macro_relative_l2"]["mean"])
    assert chosen["model_id"] == "depth_fig_d4_s0"
    assert chosen["kind"] == "single"
    assert Path(chosen["checkpoint"]) == CHECKPOINT
    assert chosen["checkpoint_sha256"] == CHECKPOINT_SHA
    registry_before = registry_snapshot()
    source_files = inputs()
    for _, path, _, expected, provenance, _ in source_files:
        assert sha(path) == expected
        assert sha(provenance["original_copy"]) == provenance["source_sha256"]

    model, checkpoint = load_model_checkpoint(CHECKPOINT, "cpu", "depth_fig_d4_s0_model")
    _, baseline = load_model_checkpoint(OLD_CHECKPOINT, "cpu", "scale_fig_26k_wide_model")
    assert checkpoint["layout"] == baseline["layout"]
    for key in ("k_grid", "wavelengths_nm"):
        assert torch.equal(checkpoint[key], baseline[key])
    assert checkpoint["hidden_sizes"] == chosen["hidden_sizes"]
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == chosen["parameter_count"]
    assert not model.training
    assert checkpoint["input_dim"] == 319 and checkpoint["output_dim"] == 480
    modes = checkpoint["fourier_modes"]
    assert modes == 80 and all(checkpoint["layout"][key]["used_in_model"] for _, key in SPECIES)
    axis = np.asarray(checkpoint["layout"]["wavelength_bands_nm"][0], dtype=np.float64)
    stored_axis = checkpoint["wavelengths_nm"].cpu().numpy()
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA,
        "selection_file": str(selection), "selection_file_sha256": SELECTION_SHA,
        "model_choice": "Minimum unified test macro relative-L2 across all 20 completed Fig entries, as requested; frozen checkpoint",
        "model_id": chosen["model_id"], "checkpoint_id": chosen["checkpoint_id"],
        "hidden_sizes": checkpoint["hidden_sizes"], "parameter_count": parameter_count,
        "unified_test_macro_relative_l2": chosen["macro_relative_l2"],
        "original_choice": "Depth study selected depth_fig_d4_s0 by validation before opening its test; this inference identifies the same model by the requested unified test minimum.",
        "previous_fig10_checkpoint": str(OLD_CHECKPOINT),
        "inference_plan": str(PLAN), "inference_plan_sha256": sha(PLAN),
        "input_dim": 319, "output_dim": 480, "dtype": checkpoint["dtype"],
        "preprocessing": [
            "Use previously authorized wavelength alignment (a/c linear; b unchanged)",
            "Ignore mask; use every existing value and every row",
            "Divide each aligned spectrum by its own max(abs(intensity))",
            "Apply unchanged checkpoint x_mean/x_scale inside predict_coefficients",
        ],
        "output_procedure": "Canonical y de-standardization and Hermitian projection, "
        "then canonical inverse Fourier on each species checkpoint velocity grid.",
        "excluded_operations": [
            "background subtraction", "smoothing", "instrument correction",
            "mask filling", "row filtering", "additional wavelength/time shifts",
            "VDF clipping", "VDF renormalization", "temperature fitting",
        ],
        "units": {"velocity": "m/s", "vdf": "s/m", "time": "ns", "wavelength": "nm"},
        "ground_truth_available": False,
        "interpretation": "Direct model predictions for experimental/image-derived spectra; "
        "synthetic held-out accuracy does not measure accuracy on these data.",
        "datasets": {},
    }
    for name, path, field, expected_hash, provenance, expected_rows in source_files:
        with np.load(path, allow_pickle=False) as saved:
            raw = np.asarray(saved[field], dtype=np.float64)
            wave = saved["wavelength_nm"]
            time_ns = saved["time_paper_ns"]
            rows = saved["source_row_index"]
        assert raw.shape == (expected_rows, 319)
        assert time_ns.shape == rows.shape == (expected_rows,)
        assert np.isfinite(raw).all() and np.isfinite(time_ns).all()
        assert np.all(np.diff(wave) > 0) and np.all(np.diff(time_ns) > 0)
        assert np.array_equal(wave.astype(stored_axis.dtype), stored_axis)
        assert np.allclose(wave, axis, rtol=0, atol=1e-12)
        scale = np.max(np.abs(raw), axis=1)
        if not np.isfinite(scale).all() or not np.all(scale > 0):
            raise ValueError(f"{name}: no normalization possible for nonpositive/nonfinite peak.")
        normalized64 = raw / scale[:, None]
        normalized = normalized64.astype(np.float32)
        assert np.array_equal(np.max(np.abs(normalized), axis=1), np.ones(expected_rows))
        with np.load(REPO / "outputs/fig10_inference_20261021/results" / name / "predictions.npz", allow_pickle=False) as old:
            for key, value in (("time_paper_ns", time_ns), ("source_row_index", rows),
                               ("wavelength_nm", wave), ("spectra_before_peak_normalization", raw),
                               ("input_peak_scale", scale), ("spectra_peak_normalized", normalized)):
                assert np.array_equal(old[key], value), (name, key)
        with torch.inference_mode():
            coefficients = predict_coefficients(
                model, torch.from_numpy(normalized), checkpoint, batch_size=128, device="cpu"
            )
        assert coefficients.shape == (expected_rows, 480) and torch.isfinite(coefficients).all()
        arrays = {
            "time_paper_ns": time_ns, "source_row_index": rows, "wavelength_nm": wave,
            "spectra_before_peak_normalization": raw,
            "input_peak_scale": scale, "spectra_peak_normalized": normalized,
            "predicted_coefficients": coefficients.numpy(),
        }
        physical = {}
        for index, (species, key) in enumerate(SPECIES):
            block = coefficients[:, index * 2 * modes:(index + 1) * 2 * modes]
            k = checkpoint["k_grid"][index * modes:(index + 1) * modes]
            v, f = reconstruct_real_evdf(
                block, k, velocity=checkpoint["layout"][key]["velocity_grid_m_s"]
            )
            assert f.shape == (expected_rows, 4097)
            arrays[f"{species}_velocity_m_s"] = v.numpy()
            arrays[f"{species}_vdf"] = f.numpy()
            arrays[f"{species}_k_grid"] = k.numpy()
            physical[species] = describe_vdf(v.numpy(), f.numpy(), time_ns, rows)
        folder = out / name
        folder.mkdir()
        output = folder / "predictions.npz"
        with output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(output, allow_pickle=False) as saved:
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value), (name, key)
        record = {
            "input_file": str(path), "input_sha256": expected_hash, "input_array": field,
            "original_file": provenance["original_copy"],
            "original_sha256": provenance["source_sha256"],
            "row_count": expected_rows,
            "time_range_ns": [float(time_ns[0]), float(time_ns[-1])],
            "peak_scale_range": [float(scale.min()), float(scale.max())],
            "normalized_peak_min_max": [1.0, 1.0],
            "normalization_precision": "Divide in float64, cast to model float32",
            "wave_max_abs_difference_from_layout_nm": float(np.abs(wave - axis).max()),
            "output": str(output), "output_sha256": sha(output),
            "species": physical,
        }
        write_json(folder / "diagnostics.json", record)
        summary["datasets"][name] = record
        print(json.dumps({"completed": name, "rows": expected_rows, "output": str(output)}), flush=True)

    for name, expected in protected.items():
        assert sha(REPO / name) == expected
    for _, path, _, expected, provenance, _ in source_files:
        assert sha(path) == expected
        assert sha(provenance["original_copy"]) == provenance["source_sha256"]
    assert sha(CHECKPOINT) == CHECKPOINT_SHA and sha(selection) == SELECTION_SHA
    assert registry_snapshot() == registry_before
    verify_plan(plan)
    summary["verification"] = {
        "protected_source_files_unchanged": len(protected),
        "checkpoint_unchanged": True, "original_and_aligned_inputs_unchanged": True,
        "registry_links_unchanged": {root: len(links) for root, links in registry_before.items()},
        "protected_artifacts_unchanged": len(plan["protected_sha256"]),
        "layout_and_axes_equal_previous_checkpoint": True,
        "preprocessing_arrays_equal_previous_inference": True,
        "saved_arrays_roundtrip_exact": True,
        "all_predictions_finite": True,
    }
    summary["status"] = "completed"
    summary["total_rows"] = sum(record["row_count"] for record in summary["datasets"].values())
    summary["runtime_versions"] = {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}
    summary["inference_and_artifact_wall_seconds"] = time.perf_counter() - started
    summary["inference_script_sha256"] = sha(__file__)
    write_json(out / "summary.json", summary)
    print(json.dumps({"summary": str(out / "summary.json"),
                      "seconds": summary["inference_and_artifact_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
