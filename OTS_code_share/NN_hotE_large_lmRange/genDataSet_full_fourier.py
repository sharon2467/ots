# -*- coding: utf-8 -*-
"""Generate OTS data for inversion of an arbitrary ion-1 distribution.

The Parquet contract intentionally matches ``genDataSet.py``:

* 1224 independently normalized spectral samples;
* 80 real plus 80 imaginary Fourier coefficients;
* row 0 stores wavelength and K-grid metadata.

The physical target differs.  Electron and ion-2 distributions are fixed
Maxwellians represented numerically by Fourier coefficients.  Ion-1 is a
sampled non-Maxwellian Gaussian mixture, and its Fourier transform is the
160-value network label.  The forward model receives Fourier distributions for
all three species and receives no temperature.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from phyEqns.ots_fourier import OTS_spectrum_fft as legacy_ots_spectrum_fft
from phyEqns.ots_fourier import ft_x_n, w_fast
from phyEqns.ots_full_fourier import (
    OTS_spectrum_full_fourier,
    fourier_landau_response,
    inverse_fourier_at,
)


ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23
SPEED_OF_LIGHT = 2.99792458e8
ELECTRON_MASS = 9.1093837015e-31
PROTON_MASS = 1.67262192369e-27
FOURIER_MODES = 80
INPUT_DIM = 1224
OUTPUT_DIM = 160


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/full_fourier_ion1_v1_seed20260716"),
    )
    parser.add_argument(
        "--dataset-id", default="full_fourier_ion1_v1_seed20260716"
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize_distribution(values, velocity):
    integral = torch.trapz(values, velocity)
    if not torch.isfinite(integral) or not bool((integral > 0).item()):
        raise ValueError("Velocity distribution has an invalid normalization.")
    return values / integral


def maxwellian_distribution(velocity, thermal_speed):
    values = torch.exp(-(velocity**2) / (2.0 * thermal_speed**2))
    return normalize_distribution(values, velocity)


def sample_ion1_distribution(velocity, reference_speed, generator):
    """Sample a three-core-plus-one-satellite proton distribution."""

    normalized_velocity = (velocity / reference_speed).cpu().numpy()
    values = np.zeros_like(normalized_velocity)
    for _ in range(3):
        center = generator.uniform(-2.0, 2.0)
        sigma = generator.uniform(0.35, 1.50)
        amplitude = generator.uniform(0.2, 2.5)
        values += amplitude * np.exp(
            -((normalized_velocity - center) ** 2) / (2.0 * sigma**2)
        )

    center = generator.uniform(-7.0, 7.0)
    sigma = generator.uniform(0.25, 1.00)
    amplitude = generator.uniform(0.02, 0.50)
    values += amplitude * np.exp(
        -((normalized_velocity - center) ** 2) / (2.0 * sigma**2)
    )
    distribution = torch.from_numpy(values).to(dtype=velocity.dtype)
    return normalize_distribution(distribution, velocity)


def default_configuration():
    dtype = torch.float64
    electron_temperature_ev = 1370.0
    ion_reference_temperature_ev = 171.3
    electron_thermal_speed = math.sqrt(
        electron_temperature_ev * ELEMENTARY_CHARGE / ELECTRON_MASS
    )
    ion1_reference_speed = math.sqrt(
        ion_reference_temperature_ev * ELEMENTARY_CHARGE / PROTON_MASS
    )
    ion2_thermal_speed = math.sqrt(
        ion_reference_temperature_ev * ELEMENTARY_CHARGE / (12.0 * PROTON_MASS)
    )

    electron_velocity = (
        torch.linspace(-0.8, 0.8, 1000, dtype=dtype) * SPEED_OF_LIGHT
    )
    ion1_velocity = (
        torch.linspace(-12.0, 12.0, 1000, dtype=dtype) * ion1_reference_speed
    )
    ion2_velocity = (
        torch.linspace(-12.0, 12.0, 1000, dtype=dtype) * ion2_thermal_speed
    )
    electron_distribution = maxwellian_distribution(
        electron_velocity, electron_thermal_speed
    )
    ion2_distribution = maxwellian_distribution(
        ion2_velocity, ion2_thermal_speed
    )
    K_e, F_e = ft_x_n(
        electron_velocity, electron_distribution, FOURIER_MODES
    )
    K_i2, F_i2 = ft_x_n(ion2_velocity, ion2_distribution, FOURIER_MODES)

    scattering_angle = math.radians(123.0)
    ion_band = torch.linspace(349.0, 353.0, 579, dtype=dtype) * 1.0e-9
    red_band = (
        torch.linspace(375.3552, 599.7064, 304, dtype=dtype) * 1.0e-9
    )
    blue_band = (
        torch.linspace(100.1144, 339.6504, 341, dtype=dtype) * 1.0e-9
    )
    return {
        "dtype": dtype,
        "electron_temperature_ev": electron_temperature_ev,
        "ion_reference_temperature_ev": ion_reference_temperature_ev,
        "electron_thermal_speed": electron_thermal_speed,
        "ion1_reference_speed": ion1_reference_speed,
        "ion2_thermal_speed": ion2_thermal_speed,
        "electron_velocity": electron_velocity,
        "electron_distribution": electron_distribution,
        "ion1_velocity": ion1_velocity,
        "ion2_velocity": ion2_velocity,
        "ion2_distribution": ion2_distribution,
        "K_e": K_e,
        "F_e": F_e,
        "K_i2": K_i2,
        "F_i2": F_i2,
        "bands": (ion_band, red_band, blue_band),
        "wavelengths": torch.cat((ion_band, red_band, blue_band)),
        "ss": torch.tensor(
            [math.cos(scattering_angle), math.sin(scattering_angle), 0.0],
            dtype=dtype,
        ),
        "k_sh": torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        "E_i0": torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        "lm_0": 351.0e-9,
        "n_e": 1.81e26,
        "Z_i1": 1.0,
        "N_i1": 0.55,
        "m_i1": PROTON_MASS,
        "Z_i2": 6.0,
        "N_i2": 0.45,
        "m_i2": 12.0 * PROTON_MASS,
        "V": (1.0e-3) ** 3,
        "zero_drift": torch.zeros(3, dtype=dtype),
    }


def sample_spectrum_and_label(configuration, generator):
    ion1_distribution = sample_ion1_distribution(
        configuration["ion1_velocity"],
        configuration["ion1_reference_speed"],
        generator,
    )
    K_i1, F_i1 = ft_x_n(
        configuration["ion1_velocity"], ion1_distribution, FOURIER_MODES
    )

    spectra = []
    for wavelengths in configuration["bands"]:
        spectra.append(
            OTS_spectrum_full_fourier(
                wavelengths,
                configuration["ss"],
                configuration["k_sh"],
                configuration["E_i0"],
                configuration["lm_0"],
                configuration["n_e"],
                configuration["Z_i1"],
                configuration["N_i1"],
                configuration["m_i1"],
                configuration["Z_i2"],
                configuration["N_i2"],
                configuration["m_i2"],
                configuration["V"],
                configuration["K_e"],
                configuration["F_e"],
                K_i1,
                F_i1,
                configuration["K_i2"],
                configuration["F_i2"],
                configuration["zero_drift"],
                configuration["zero_drift"],
                configuration["zero_drift"],
            )
        )
    spectrum = torch.cat(spectra)
    label = torch.cat((F_i1.real, F_i1.imag))
    return spectrum, label, K_i1, ion1_distribution


def _write_parquet_atomic(rows, path):
    columns = ["lm_{}".format(index) for index in range(INPUT_DIM)]
    columns += ["K_{}".format(index) for index in range(OUTPUT_DIM)]
    dataframe = pd.DataFrame(rows, columns=columns)
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, str(temporary))
    os.replace(str(temporary), str(path))


def generate_split(configuration, count, seed, path, progress_every):
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    rows = np.empty((count + 1, INPUT_DIM + OUTPUT_DIM), dtype=np.float32)
    rows[0, :INPUT_DIM] = (
        configuration["wavelengths"].cpu().numpy() / 1.0e-9
    ).astype(np.float32)
    generator = np.random.default_rng(seed)
    started = time.time()
    label_k = None

    for sample_index in range(count):
        spectrum, label, K_i1, _ = sample_spectrum_and_label(
            configuration, generator
        )
        if label_k is None:
            label_k = K_i1.detach().cpu().numpy().astype(np.float32)
            rows[0, INPUT_DIM : INPUT_DIM + FOURIER_MODES] = label_k
            rows[0, INPUT_DIM + FOURIER_MODES :] = label_k
        if not torch.isfinite(spectrum).all() or not torch.isfinite(label).all():
            raise ValueError(
                "Non-finite sample generated at index {}.".format(sample_index)
            )
        rows[sample_index + 1, :INPUT_DIM] = (
            spectrum.detach().cpu().numpy().astype(np.float32)
        )
        rows[sample_index + 1, INPUT_DIM:] = (
            label.detach().cpu().numpy().astype(np.float32)
        )
        completed = sample_index + 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == count
        ):
            print(
                "{} progress {}/{} elapsed={:.1f}s".format(
                    path.name, completed, count, time.time() - started
                ),
                flush=True,
            )

    _write_parquet_atomic(rows, path)
    return {
        "rows": int(rows.shape[0]),
        "columns": int(rows.shape[1]),
        "physical_samples": count,
        "seed": seed,
        "elapsed_seconds": time.time() - started,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def relative_l2(reference, candidate):
    return (
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference).clamp(min=1.0e-30)
    ).item()


def run_self_test():
    configuration = default_configuration()
    generator = np.random.default_rng(17)
    spectrum, label, K_i1, ion1_distribution = sample_spectrum_and_label(
        configuration, generator
    )
    assert spectrum.shape == (INPUT_DIM,)
    assert label.shape == (OUTPUT_DIM,)
    assert torch.isfinite(spectrum).all()
    assert torch.isfinite(label).all()
    assert abs(torch.max(torch.abs(spectrum)).item() - 1.0) < 1.0e-10

    reconstructed = inverse_fourier_at(
        K_i1,
        label[:FOURIER_MODES] + 1j * label[FOURIER_MODES:],
        configuration["ion1_velocity"],
    ).real
    round_trip_relative_l2 = relative_l2(ion1_distribution, reconstructed)
    hermitian_error = torch.max(
        torch.abs(
            (label[:FOURIER_MODES] + 1j * label[FOURIER_MODES:])
            - torch.conj(
                torch.flip(
                    label[:FOURIER_MODES] + 1j * label[FOURIER_MODES:],
                    dims=(0,),
                )
            )
        )
    ).item()
    assert round_trip_relative_l2 < 0.08
    assert hermitian_error < 1.0e-8

    electron_temperature_k = (
        configuration["electron_temperature_ev"]
        * ELEMENTARY_CHARGE
        / BOLTZMANN_CONSTANT
    )
    ion_temperature_k = (
        configuration["ion_reference_temperature_ev"]
        * ELEMENTARY_CHARGE
        / BOLTZMANN_CONSTANT
    )
    maxwellian_i1 = maxwellian_distribution(
        configuration["ion1_velocity"], configuration["ion1_reference_speed"]
    )
    K_i1_maxwellian, F_i1_maxwellian = ft_x_n(
        configuration["ion1_velocity"], maxwellian_i1, FOURIER_MODES
    )

    full_bands = []
    legacy_bands = []
    for wavelengths in configuration["bands"]:
        full_bands.append(
            OTS_spectrum_full_fourier(
                wavelengths,
                configuration["ss"],
                configuration["k_sh"],
                configuration["E_i0"],
                configuration["lm_0"],
                configuration["n_e"],
                configuration["Z_i1"],
                configuration["N_i1"],
                configuration["m_i1"],
                configuration["Z_i2"],
                configuration["N_i2"],
                configuration["m_i2"],
                configuration["V"],
                configuration["K_e"],
                configuration["F_e"],
                K_i1_maxwellian,
                F_i1_maxwellian,
                configuration["K_i2"],
                configuration["F_i2"],
                configuration["zero_drift"],
                configuration["zero_drift"],
                configuration["zero_drift"],
            )
        )
        legacy_bands.append(
            legacy_ots_spectrum_fft(
                wavelengths,
                configuration["ss"],
                configuration["k_sh"],
                configuration["E_i0"],
                configuration["lm_0"],
                configuration["n_e"],
                torch.tensor(electron_temperature_k, dtype=torch.float64),
                configuration["Z_i1"],
                configuration["N_i1"],
                torch.tensor(configuration["m_i1"], dtype=torch.float64),
                configuration["Z_i2"],
                configuration["N_i2"],
                torch.tensor(configuration["m_i2"], dtype=torch.float64),
                torch.tensor(ion_temperature_k, dtype=torch.float64),
                configuration["V"],
                configuration["K_e"],
                configuration["F_e"],
                configuration["zero_drift"],
                configuration["zero_drift"],
                configuration["zero_drift"],
            )
        )
    maxwellian_band_relative_l2 = [
        relative_l2(reference, candidate)
        for reference, candidate in zip(legacy_bands, full_bands)
    ]
    ion1_probe_velocity = torch.linspace(
        -6.0,
        6.0,
        601,
        dtype=torch.float64,
    ) * configuration["ion1_reference_speed"]
    ion1_fourier_response = (
        fourier_landau_response(
            K_i1_maxwellian,
            F_i1_maxwellian,
            ion1_probe_velocity,
        )
        * configuration["ion1_reference_speed"] ** 2
    )
    ion1_legacy_response = w_fast(
        ion1_probe_velocity
        / (math.sqrt(2.0) * configuration["ion1_reference_speed"])
    )
    ion1_response_relative_l2 = relative_l2(
        ion1_legacy_response, ion1_fourier_response
    )

    result = {
        "ion1_round_trip_relative_l2": round_trip_relative_l2,
        "ion1_fourier_hermitian_max_abs": hermitian_error,
        "ion1_maxwellian_response_relative_l2": ion1_response_relative_l2,
        "maxwellian_band_relative_l2": maxwellian_band_relative_l2,
    }
    print("FULL_FOURIER_SELF_TEST_DIAGNOSTICS " + json.dumps(result, sort_keys=True))
    assert ion1_response_relative_l2 < 0.05
    assert max(maxwellian_band_relative_l2) < 0.10
    print("FULL_FOURIER_SELF_TEST_OK " + json.dumps(result, sort_keys=True))


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("Training and test sample counts must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "dataset_train.parquet"
    test_path = output_dir / "dataset_test.parquet"
    manifest_path = output_dir / "dataset_manifest.json"
    existing = [
        str(path)
        for path in (train_path, test_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing dataset artifacts: {}".format(existing)
        )

    configuration = default_configuration()
    train_record = generate_split(
        configuration,
        args.train_samples,
        args.seed,
        train_path,
        args.progress_every,
    )
    test_record = generate_split(
        configuration,
        args.test_samples,
        args.seed + 1,
        test_path,
        args.progress_every,
    )
    manifest = {
        "format_version": 1,
        "dataset_id": args.dataset_id,
        "family": "full_fourier_variable_ion1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "genDataSet_full_fourier.py",
        "forward_model": "phyEqns.ots_full_fourier.OTS_spectrum_full_fourier",
        "target": {
            "species": "ion1",
            "charge": configuration["Z_i1"],
            "mass_kg": configuration["m_i1"],
            "fourier_modes": FOURIER_MODES,
            "velocity_grid_points": len(configuration["ion1_velocity"]),
            "velocity_grid_reference_thermal_speeds": [-12.0, 12.0],
            "mixture": {
                "core_components": 3,
                "core_center_reference_speeds": [-2.0, 2.0],
                "core_sigma_reference_speeds": [0.35, 1.50],
                "core_amplitude": [0.2, 2.5],
                "satellite_components": 1,
                "satellite_center_reference_speeds": [-7.0, 7.0],
                "satellite_sigma_reference_speeds": [0.25, 1.00],
                "satellite_amplitude": [0.02, 0.50],
            },
        },
        "fixed_distributions": {
            "electron": {
                "family": "Maxwellian",
                "temperature_ev": configuration["electron_temperature_ev"],
                "velocity_grid_c": [-0.8, 0.8],
                "velocity_grid_points": len(configuration["electron_velocity"]),
            },
            "ion2": {
                "family": "Maxwellian",
                "temperature_ev": configuration["ion_reference_temperature_ev"],
                "velocity_grid_thermal_speeds": [-12.0, 12.0],
                "velocity_grid_points": len(configuration["ion2_velocity"]),
            },
        },
        "plasma": {
            "probe_wavelength_nm": configuration["lm_0"] / 1.0e-9,
            "scattering_angle_deg": 123.0,
            "electron_density_m3": configuration["n_e"],
            "ion1_fraction": configuration["N_i1"],
            "ion2_charge": configuration["Z_i2"],
            "ion2_fraction": configuration["N_i2"],
            "ion2_mass_kg": configuration["m_i2"],
            "drifts_m_s": [0.0, 0.0, 0.0],
        },
        "wavelength_bands_nm": [
            [349.0, 353.0, 579],
            [375.3552, 599.7064, 304],
            [100.1144, 339.6504, 341],
        ],
        "parquet_contract": {
            "metadata_row": 0,
            "input_columns": INPUT_DIM,
            "label_columns": OUTPUT_DIM,
            "label_order": ["real(F_i1)", "imag(F_i1)"],
        },
        "splits": {
            "train": train_record,
            "test": test_record,
        },
        "files": {
            train_path.name: file_record(train_path),
            test_path.name: file_record(test_path),
        },
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary_manifest), str(manifest_path))
    print(
        "FULL_FOURIER_DATASET_COMPLETE "
        + json.dumps(
            {
                "dataset_id": args.dataset_id,
                "train": str(train_path),
                "test": str(test_path),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
