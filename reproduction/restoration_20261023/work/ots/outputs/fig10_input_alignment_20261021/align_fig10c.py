"""Wavelength interpolation only; user instructed us to ignore the mask."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/fig10_input_alignment_20261021/fig10c/original/digitized_spectra.npz')
CHECKPOINT = Path(
    "/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange/runs/"
    "scale_fig_26k_wide/training/model.pt"
)
SOURCE_SHA = "3937bb3ec1d790576b57e208c8182941d3031a805356514f0cd62b990ea02291"
MODEL_SHA = "df9a43f15c7ac63917a46b8653db03466fca898931d825e626f991d9f2bf1a58"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "fig10c"
    if any((folder / name).exists() for name in ("intensity_aligned.npz", "alignment.json")):
        raise FileExistsError("Refusing to overwrite existing Fig.10(c) artifacts.")
    assert digest(SOURCE) == SOURCE_SHA
    assert digest(CHECKPOINT) == MODEL_SHA
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    bands = checkpoint["layout"]["wavelength_bands_nm"]
    assert len(bands) == 1 and checkpoint["input_dim"] == 319
    target = np.asarray(bands[0], dtype=np.float64)
    assert np.array_equal(
        target.astype(np.float32), checkpoint["wavelengths_nm"].cpu().numpy()
    )
    with np.load(SOURCE, allow_pickle=False) as source:
        wavelength = source["wavelength_nm"]
        intensity = source["intensity"]
        time = source["time_paper_ns"]
        indices = source["source_row_index"]
    assert intensity.shape == (time.size, wavelength.size) == (480, 320)
    assert indices.shape == time.shape and target.shape == (319,)
    assert all(np.isfinite(a).all() for a in (wavelength, intensity, time, target))
    assert np.all(np.diff(wavelength) > 0) and np.all(np.diff(target) > 0)
    if target[0] < wavelength[0] or target[-1] > wavelength[-1]:
        raise ValueError("Extrapolation is not authorized.")

    # Keep every source value, even where valid=False; no mask filling.
    aligned = np.stack([np.interp(target, wavelength, row) for row in intensity])
    assert aligned.shape == (480, 319) and np.isfinite(aligned).all()
    assert (folder / "original").is_dir()
    raw_copy = folder / "original" / "digitized_spectra.npz"
    assert digest(raw_copy) == SOURCE_SHA  # Preserved input is never overwritten.
    output = folder / "intensity_aligned.npz"
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, intensity=aligned, wavelength_nm=target,
            time_paper_ns=time, source_row_index=indices,
        )
    with np.load(output, allow_pickle=False) as saved:
        assert np.array_equal(saved["intensity"], aligned)
        assert np.array_equal(saved["wavelength_nm"], target)
        assert np.array_equal(saved["time_paper_ns"], time)
        assert np.array_equal(saved["source_row_index"], indices)
    assert digest(SOURCE) == digest(raw_copy) == SOURCE_SHA
    assert digest(CHECKPOINT) == MODEL_SHA
    record = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source": str(SOURCE), "source_sha256": SOURCE_SHA,
        "original_copy": str(raw_copy),
        "source_array": "intensity",
        "source_shape_time_wavelength": list(intensity.shape),
        "source_wavelength_range_nm": [float(wavelength[0]), float(wavelength[-1])],
        "source_row_maximum_range": [
            float(intensity.max(axis=1).min()), float(intensity.max(axis=1).max())
        ],
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": MODEL_SHA,
        "target_axis_source": "checkpoint.layout.wavelength_bands_nm[0], float64",
        "target_shape_time_wavelength": list(aligned.shape),
        "target_wavelength_range_nm": [float(target[0]), float(target[-1])],
        "model_float32_axis_exactly_equal": True,
        "operation": "Piecewise-linear interpolation along wavelength only",
        "all_target_points_bracketed": True, "extrapolation": False,
        "mask_policy": "User instructed ignore mask: preserve all rows and use "
        "all existing intensity values. No valid-mask filtering or filling.",
        "time_and_source_row_indices_unchanged": True,
        "intensity_scale": "Original digitized colorbar scale; no normalization",
        "aligned_row_maximum_range": [
            float(aligned.max(axis=1).min()), float(aligned.max(axis=1).max())
        ],
        "output": str(output), "output_sha256": digest(output),
        "inference_executed": False,
        "remaining_blocker": "Alignment retains native scale; authorized normalization is a separate inference stage",
        "production_code_modified": False,
    }
    with (folder / "alignment.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
