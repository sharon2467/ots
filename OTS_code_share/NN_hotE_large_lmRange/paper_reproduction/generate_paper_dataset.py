#!/usr/bin/env python3
"""Generate the 10-Gaussian OTS dataset described around paper Figure 3.

This is deliberately separate from ``genDataSet.py``.  The latter remains the
hot-electron/large-wavelength-range generator; this module creates a versioned
paper-baseline dataset without overwriting it.

The paper defines every EVDF as the normalized sum of ten Gaussians whose
centres, widths, and amplitudes are sampled uniformly from [-0.2c, 0.2c],
[0.02c, 0.08c], and [0.5, 1.5], respectively.  Here "width" is interpreted as
the Gaussian standard deviation because the paper does not state a different
convention.  The paper wavelength bands (349-353, 375-500, and 200-340 nm)
are used with 579/304/341 samples.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from phyEqns.ots_fourier import OTS_spectrum_fft, ft_x_n


SPEED_OF_LIGHT = 2.99792458e8
ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23
PROTON_MASS = 1.67262192369e-27
INPUT_DIM = 1224
FOURIER_MODES = 80
OUTPUT_DIM = 160
VELOCITY_POINTS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="paper_10gaussian_v1_seed20260715")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def trapezoid(values, grid):
    function = getattr(np, "trapezoid", np.trapz)
    return function(values, grid)


def paper_wavelength_grid(dtype=torch.float64):
    ion = torch.linspace(349.0, 353.0, 579, dtype=dtype) * 1.0e-9
    red = torch.linspace(375.0, 500.0, 304, dtype=dtype) * 1.0e-9
    blue = torch.linspace(200.0, 340.0, 341, dtype=dtype) * 1.0e-9
    return torch.cat((ion, red, blue))


def velocity_grid_numpy():
    return np.linspace(-0.8, 0.8, VELOCITY_POINTS, dtype=np.float64) * SPEED_OF_LIGHT


def ten_gaussian_evdf(velocity_m_s, rng):
    centres = rng.uniform(-0.2, 0.2, size=(10, 1)) * SPEED_OF_LIGHT
    sigmas = rng.uniform(0.02, 0.08, size=(10, 1)) * SPEED_OF_LIGHT
    amplitudes = rng.uniform(0.5, 1.5, size=(10, 1))
    profiles = amplitudes * np.exp(
        -0.5 * ((velocity_m_s[None, :] - centres) / sigmas) ** 2
    )
    evdf = profiles.sum(axis=0)
    integral = trapezoid(evdf, velocity_m_s)
    if not np.isfinite(integral) or integral <= 0.0:
        raise RuntimeError("Generated EVDF has an invalid normalization integral.")
    return evdf / integral


def physics_parameters(dtype=torch.float64):
    scattering_angle = math.radians(123.0)
    zero_drift = torch.zeros(3, dtype=dtype)
    return {
        "scattering_direction": torch.tensor(
            [math.cos(scattering_angle), math.sin(scattering_angle), 0.0],
            dtype=dtype,
        ),
        "incident_direction": torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        "incident_polarization": torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        "probe_wavelength_m": 351.0e-9,
        "electron_density_m3": 1.81e26,
        "electron_temperature_k": 1370.0 * ELEMENTARY_CHARGE / BOLTZMANN_CONSTANT,
        "ion_temperature_k": 171.3 * ELEMENTARY_CHARGE / BOLTZMANN_CONSTANT,
        "ion_1_charge": 1.0,
        "ion_1_fraction": 0.55,
        "ion_1_mass_kg": torch.tensor(PROTON_MASS, dtype=dtype),
        "ion_2_charge": 6.0,
        "ion_2_fraction": 0.45,
        "ion_2_mass_kg": torch.tensor(12.0 * PROTON_MASS, dtype=dtype),
        "plasma_volume_m3": 1.0e-9,
        "electron_drift_m_s": zero_drift.clone(),
        "ion_1_drift_m_s": zero_drift.clone(),
        "ion_2_drift_m_s": zero_drift.clone(),
    }


def spectrum_band(wavelengths_m, k_grid, fourier_coefficients, physics):
    return OTS_spectrum_fft(
        wavelengths_m,
        physics["scattering_direction"],
        physics["incident_direction"],
        physics["incident_polarization"],
        physics["probe_wavelength_m"],
        physics["electron_density_m3"],
        physics["electron_temperature_k"],
        physics["ion_1_charge"],
        physics["ion_1_fraction"],
        physics["ion_1_mass_kg"],
        physics["ion_2_charge"],
        physics["ion_2_fraction"],
        physics["ion_2_mass_kg"],
        physics["ion_temperature_k"],
        physics["plasma_volume_m3"],
        k_grid,
        fourier_coefficients,
        physics["electron_drift_m_s"],
        physics["ion_1_drift_m_s"],
        physics["ion_2_drift_m_s"],
    )


def generate_table(sample_count, seed, progress_every):
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    velocity_np = velocity_grid_numpy()
    velocity = torch.from_numpy(velocity_np)
    wavelengths = paper_wavelength_grid()
    physics = physics_parameters()
    values = np.empty((sample_count + 1, INPUT_DIM + OUTPUT_DIM), dtype=np.float32)
    values[0, :INPUT_DIM] = wavelengths.numpy().astype(np.float32) / 1.0e-9
    k_grid_metadata = None
    started = time.time()

    bands = (slice(0, 579), slice(579, 883), slice(883, 1224))
    for index in range(sample_count):
        evdf_np = ten_gaussian_evdf(velocity_np, rng)
        evdf = torch.from_numpy(evdf_np)
        k_grid, coefficients = ft_x_n(velocity, evdf, FOURIER_MODES)
        if k_grid_metadata is None:
            k_grid_metadata = k_grid.detach().cpu().numpy().astype(np.float32)
            values[0, INPUT_DIM:] = np.concatenate(
                (k_grid_metadata, k_grid_metadata)
            )

        spectra = torch.cat(
            tuple(
                spectrum_band(wavelengths[band], k_grid, coefficients, physics)
                for band in bands
            )
        )
        row = index + 1
        values[row, :INPUT_DIM] = spectra.detach().cpu().numpy().astype(np.float32)
        values[row, INPUT_DIM:] = np.concatenate(
            (
                coefficients.real.detach().cpu().numpy(),
                coefficients.imag.detach().cpu().numpy(),
            )
        ).astype(np.float32)

        if not np.isfinite(values[row]).all():
            raise RuntimeError("Non-finite value generated at physical row {}".format(row))
        if progress_every > 0 and ((index + 1) % progress_every == 0 or index == 0):
            elapsed = time.time() - started
            print(
                "generated={}/{} elapsed_seconds={:.1f}".format(
                    index + 1, sample_count, elapsed
                ),
                flush=True,
            )

    columns = ["lm_{}".format(i) for i in range(INPUT_DIM)] + [
        "K_{}".format(i) for i in range(OUTPUT_DIM)
    ]
    return pd.DataFrame(values, columns=columns)


def atomic_write_parquet(frame, output_path):
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, str(temporary), compression="snappy")
    os.replace(str(temporary), str(output_path))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_test():
    rng = np.random.default_rng(7)
    velocity_np = velocity_grid_numpy()
    evdf_np = ten_gaussian_evdf(velocity_np, rng)
    integral = trapezoid(evdf_np, velocity_np)
    assert abs(integral - 1.0) < 1.0e-10
    velocity = torch.from_numpy(velocity_np)
    k_grid, coefficients = ft_x_n(velocity, torch.from_numpy(evdf_np), FOURIER_MODES)
    assert tuple(k_grid.shape) == (FOURIER_MODES,)
    assert tuple(coefficients.shape) == (FOURIER_MODES,)
    wavelengths = paper_wavelength_grid()
    assert tuple(wavelengths.shape) == (INPUT_DIM,)
    assert abs(wavelengths[579].item() / 1.0e-9 - 375.0) < 1.0e-6
    assert abs(wavelengths[883].item() / 1.0e-9 - 200.0) < 1.0e-6
    print("PAPER_DATASET_SELF_TEST_PASSED integral={:.12f}".format(integral))


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("train/test sample counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "dataset_train.parquet"
    test_path = args.output_dir / "dataset_test.parquet"
    manifest_path = args.output_dir / "dataset_manifest.json"
    existing = [path for path in (train_path, test_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing paper dataset files: {}".format(existing)
        )

    run_started = time.time()
    print("Generating paper training data with seed {}".format(args.seed), flush=True)
    train_frame = generate_table(args.train_samples, args.seed, args.progress_every)
    atomic_write_parquet(train_frame, train_path)
    del train_frame
    print("Generating paper test data with seed {}".format(args.seed + 1), flush=True)
    test_frame = generate_table(args.test_samples, args.seed + 1, args.progress_every)
    atomic_write_parquet(test_frame, test_path)
    del test_frame

    script_path = Path(__file__).resolve()
    manifest = {
        "dataset_id": args.dataset_id,
        "family": "paper_10gaussian",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": script_path.name,
        "generator_sha256": sha256_file(script_path),
        "train_seed": args.seed,
        "test_seed": args.seed + 1,
        "train_physical_rows": args.train_samples,
        "test_physical_rows": args.test_samples,
        "table_columns": INPUT_DIM + OUTPUT_DIM,
        "metadata_row": 0,
        "gaussians_per_evdf": 10,
        "gaussian_center_c": [-0.2, 0.2],
        "gaussian_sigma_c": [0.02, 0.08],
        "gaussian_amplitude": [0.5, 1.5],
        "width_interpretation": "standard_deviation_sigma",
        "velocity_grid_c": [-0.8, 0.8, VELOCITY_POINTS],
        "fourier_modes": FOURIER_MODES,
        "wavelength_bands_nm": [
            [349.0, 353.0, 579],
            [375.0, 500.0, 304],
            [200.0, 340.0, 341],
        ],
        "fixed_physics": {
            "probe_wavelength_nm": 351.0,
            "scattering_angle_deg": 123.0,
            "electron_density_m3": 1.81e26,
            "electron_temperature_eV": 1370.0,
            "ion_temperature_eV": 171.3,
            "ions": {"H": {"Z": 1, "fraction": 0.55}, "C": {"Z": 6, "fraction": 0.45}},
            "drift_m_s": 0.0,
        },
        "paper_limitations": [
            "The paper does not publish the random seeds used for Figure 3.",
            "The paper says several plasma parameters were random but the released network has no output heads for them; this reproduction keeps the source-code physics fixed.",
        ],
        "elapsed_seconds": time.time() - run_started,
        "files": {
            train_path.name: {
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
                "shape": [args.train_samples + 1, INPUT_DIM + OUTPUT_DIM],
            },
            test_path.name: {
                "bytes": test_path.stat().st_size,
                "sha256": sha256_file(test_path),
                "shape": [args.test_samples + 1, INPUT_DIM + OUTPUT_DIM],
            },
        },
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary_manifest), str(manifest_path))
    print("PAPER_DATASET_READY {}".format(args.output_dir), flush=True)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
