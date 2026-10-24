"""Only align uploaded Fig.10(a) measured_raw to the frozen model's axis."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import shutil

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/fig10_input_alignment_20261021/fig10a/original/saved_spectral_image.npz')
CHECKPOINT = Path(
    "/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange/runs/"
    "scale_fig_26k_wide/training/model.pt"
)
SOURCE_HASH = "1e34392d96e33e97eb2403ddd89b1705f067a2dbd6d78e3e423178f5ed5c2771"
MODEL_HASH = "df9a43f15c7ac63917a46b8653db03466fca898931d825e626f991d9f2bf1a58"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    raw_copy = ROOT / "fig10a" / "original" / "saved_spectral_image.npz"
    aligned = ROOT / "fig10a" / "measured_raw_aligned.npz"
    record = ROOT / "fig10a" / "alignment.json"
    if any(p.exists() for p in (aligned, record)):
        raise FileExistsError("Refusing to overwrite existing alignment artifacts.")
    assert digest(SOURCE) == SOURCE_HASH
    assert digest(CHECKPOINT) == MODEL_HASH
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["input_dim"] == 319
    bands = checkpoint["layout"]["wavelength_bands_nm"]
    assert len(bands) == 1
    target = np.asarray(bands[0], dtype=np.float64)
    stored = checkpoint["wavelengths_nm"].cpu().numpy()
    assert np.array_equal(target.astype(stored.dtype), stored)

    with np.load(SOURCE, allow_pickle=False) as source:
        wavelength = source["wavelength_nm"]
        intensity = source["measured_raw"]
        time = source["time_paper_ns"]
        indices = source["source_row_index"]
        scale = source["scale"]
        measured = source["measured"]
    assert intensity.shape == (time.size, wavelength.size) == (556, 527)
    assert target.shape == (319,) and indices.shape == time.shape
    assert all(np.isfinite(a).all() for a in (wavelength, target, intensity, time))
    assert np.all(np.diff(wavelength) > 0) and np.all(np.diff(target) > 0)
    if target[0] < wavelength[0] or target[-1] > wavelength[-1]:
        raise ValueError("Source does not cover the model axis; extrapolation forbidden.")

    # Wavelength interpolation only: no time shift, rescaling, background
    # removal, smoothing, clipping, mask filling or model inference.
    values = np.stack([np.interp(target, wavelength, row) for row in intensity])
    assert values.shape == (556, 319) and np.isfinite(values).all()

    assert raw_copy.parent.is_dir()
    assert digest(raw_copy) == SOURCE_HASH  # Preserved input is never overwritten.
    with aligned.open("xb") as stream:
        np.savez_compressed(
            stream,
            intensity_counts=values,
            wavelength_nm=target,
            time_paper_ns=time,
            source_row_index=indices,
        )
    with np.load(aligned, allow_pickle=False) as saved:
        assert np.array_equal(saved["wavelength_nm"], target)
        assert np.array_equal(saved["time_paper_ns"], time)
        assert np.array_equal(saved["source_row_index"], indices)
        assert np.array_equal(saved["intensity_counts"], values)
    assert digest(raw_copy) == digest(SOURCE) == SOURCE_HASH
    assert digest(CHECKPOINT) == MODEL_HASH
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "operation": "Piecewise-linear interpolation along wavelength only",
        "source": str(SOURCE),
        "source_sha256": SOURCE_HASH,
        "original_copy": str(raw_copy),
        "source_array": "measured_raw",
        "source_shape_time_wavelength": list(intensity.shape),
        "source_wavelength_nm": [float(wavelength[0]), float(wavelength[-1])],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": MODEL_HASH,
        "target_axis_source": "checkpoint.layout.wavelength_bands_nm[0], float64",
        "target_rounds_exactly_to_checkpoint_float32_wavelengths": True,
        "target_shape_time_wavelength": list(values.shape),
        "target_wavelength_nm": [float(target[0]), float(target[-1])],
        "time_range_paper_ns": [float(time[0]), float(time[-1])],
        "all_target_points_bracketed_by_source": True,
        "extrapolation": False,
        "time_and_source_row_indices_unchanged": True,
        "intensity_units": "Original counts; no new normalization",
        "source_intensity_range": [float(intensity.min()), float(intensity.max())],
        "aligned_intensity_range": [float(values.min()), float(values.max())],
        "existing_measured_equals_raw_divided_by_scale": bool(
            np.array_equal(measured, intensity / scale[:, None])
        ),
        "existing_scale_equals_source_row_max": bool(
            np.array_equal(scale, intensity.max(axis=1))
        ),
        "output": str(aligned),
        "output_sha256": digest(aligned),
        "output_array": "intensity_counts",
        "inference_executed": False,
        "inference_gate": "Alignment retains original counts. Authorized normalization is applied only by the separate inference stage.",
        "production_code_modified": False,
    }
    with record.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
