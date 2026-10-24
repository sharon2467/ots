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
CHECKPOINT = SOURCE / "runs/scale_fig_26k_wide/training/model.pt"
CHECKPOINT_SHA = "df9a43f15c7ac63917a46b8653db03466fca898931d825e626f991d9f2bf1a58"
SELECTION_SHA = "aa53407497f3741a216aa1fb1ab9356b558a9c73cac08be3fed407525e0cd53f"
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
    return {
        str(p.relative_to(SOURCE)): os.readlink(p)
        for p in sorted((SOURCE / "registry").rglob("*")) if p.is_symlink()
    }


def inputs():
    a = read_json(ALIGNMENT / "fig10a/alignment.json")
    b = read_json(ALIGNMENT / "fig10b/input_audit.json")
    c = read_json(ALIGNMENT / "fig10c/alignment.json")
    return [
        ("fig10a", Path(a["output"]), "intensity_counts", a["output_sha256"], a, 556),
        ("fig10b", Path(b["original_copy"]), "intensity", b["source_sha256"], b, 480),
        ("fig10c", Path(c["output"]), "intensity", c["output_sha256"], c, 480),
    ]


def describe_vdf(velocity, values):
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
    protected = read_json(STUDY / "source_hashes_before.json")
    for name, expected in protected.items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Protected source differs from frozen study: {name}")
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    selection = STUDY / "frozen_single_selection.json"
    assert sha(selection) == SELECTION_SHA
    chosen = read_json(selection)["comparisons"]["fig1_fig11"]
    assert Path(chosen["selected_training_run"]) / "training/model.pt" == CHECKPOINT
    registry_before = registry_snapshot()
    source_files = inputs()
    for _, path, _, expected, provenance, _ in source_files:
        assert sha(path) == expected
        assert sha(provenance["original_copy"]) == provenance["source_sha256"]

    model, checkpoint = load_model_checkpoint(CHECKPOINT, "cpu", "scale_fig_26k_wide_model")
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
        "model_choice": "Frozen best native single Fig model; no ensemble or retraining",
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
            physical[species] = describe_vdf(v.numpy(), f.numpy())
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
    summary["verification"] = {
        "protected_source_files_unchanged": len(protected),
        "checkpoint_unchanged": True, "original_and_aligned_inputs_unchanged": True,
        "registry_links_unchanged": len(registry_before),
        "saved_arrays_roundtrip_exact": True,
        "all_predictions_finite": True,
    }
    summary["inference_and_artifact_wall_seconds"] = time.perf_counter() - started
    summary["inference_script_sha256"] = sha(__file__)
    write_json(out / "summary.json", summary)
    print(json.dumps({"summary": str(out / "summary.json"),
                      "seconds": summary["inference_and_artifact_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
