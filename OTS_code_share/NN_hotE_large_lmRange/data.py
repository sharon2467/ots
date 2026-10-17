# -*- coding: utf-8 -*-
"""Canonical data generation and loading for the OTS EVDF pipeline."""

import json
import math
import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from phyEqns import ots_full_fourier


ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 2.99792458e8
ELECTRON_MASS = 9.1093837015e-31
PROTON_MASS = 1.67262192369e-27

FOURIER_MODES = 80
INPUT_DIM = 1224
OUTPUT_DIM = 160
ELECTRON_VELOCITY_POINTS = 1000
ELECTRON_REFERENCE_TEMPERATURE_EV = 1370.0
ION_REFERENCE_TEMPERATURE_EV = 171.3
ION_SPECIES = (("i1", PROTON_MASS), ("i2", 12.0 * PROTON_MASS))

@dataclass(frozen=True)
class OTSDataset:
    spectra: torch.Tensor
    coefficients: torch.Tensor
    wavelengths_nm: torch.Tensor
    k_grid: torch.Tensor


def _real_arrays(v, f, name):
    v = torch.as_tensor(v)
    f = torch.as_tensor(f, device=v.device)
    if torch.is_complex(v) or torch.is_complex(f):
        raise ValueError("{} arrays must be real-valued.".format(name))
    dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    v = v.to(dtype=dtype)
    f = f.to(dtype=dtype)
    if v.ndim != 1 or f.ndim != 1 or len(v) != len(f) or len(v) < 2:
        raise ValueError(
            "{} velocity and distribution must be equal-length one-dimensional "
            "arrays with at least two points.".format(name)
        )
    if not torch.isfinite(v).all() or not torch.isfinite(f).all():
        raise ValueError("{} arrays must contain only finite values.".format(name))
    if not torch.all(v[1:] > v[:-1]):
        raise ValueError("{} velocity grid must be strictly increasing.".format(name))
    return v, f


def normalize_distribution(v, f, name):
    """Return a finite non-negative distribution normalized on ``v``."""

    v, f = _real_arrays(v, f, name)
    if not torch.all(f >= 0):
        raise ValueError("{} distribution must be non-negative.".format(name))
    integral = torch.trapz(f, v)
    if not torch.isfinite(integral) or not bool((integral > 0).item()):
        raise ValueError("{} distribution has an invalid integral.".format(name))
    return f / integral


def sample_electron_evdf(num_points, rng):
    """Sample the current three-background-plus-one-hot electron EVDF."""

    if (
        not isinstance(num_points, Integral)
        or isinstance(num_points, bool)
        or int(num_points) < 3
    ):
        raise ValueError("num_points must be an integer of at least three.")
    num_points = int(num_points)
    velocity_c = np.linspace(-0.8, 0.8, num_points, dtype=np.float64)
    values = np.zeros(num_points, dtype=np.float64)

    for _ in range(3):
        center = rng.uniform(-0.06, 0.06)
        sigma = rng.uniform(0.01, 0.05)
        amplitude = rng.uniform(0.2, 2.5)
        values += amplitude * np.exp(-((velocity_c - center) ** 2) / (2.0 * sigma**2))

    center = rng.uniform(-0.4, 0.4)
    sigma = rng.uniform(0.01, 0.05)
    amplitude = rng.uniform(0.001, 0.045)
    values += amplitude * np.exp(-((velocity_c - center) ** 2) / (2.0 * sigma**2))

    velocity = torch.from_numpy(velocity_c * SPEED_OF_LIGHT)
    distribution = torch.from_numpy(values)
    distribution = normalize_distribution(velocity, distribution, "electron")
    return velocity, distribution


def build_fixed_ion_distributions(ion_species,reference_temperature_ev):
    """Build normalized H and C Maxwellian VDFs on species-specific grids."""

    reference_temperature_ev = float(reference_temperature_ev)
    if not math.isfinite(reference_temperature_ev) or reference_temperature_ev <= 0:
        raise ValueError("reference_temperature_ev must be finite and positive.")

    distributions = {}
    for suffix, mass in ion_species:
        thermal_speed = math.sqrt(reference_temperature_ev * ELEMENTARY_CHARGE / mass)
        velocity = torch.linspace(
            -12.0 * thermal_speed,
            12.0 * thermal_speed,
            1000,
            dtype=torch.float64,
        )
        values = torch.exp(-velocity.square() / (2.0 * thermal_speed**2))
        distributions["v_{}".format(suffix)] = velocity
        distributions["f_{}".format(suffix)] = normalize_distribution(
            velocity, values, "ion-{}".format(suffix[-1])
        )
    return distributions


def electron_moments(v_e, f_e, configured_ev):
    """Compute one-dimensional central moments and effective temperature."""

    v_e, f_e = _real_arrays(v_e, f_e, "electron")
    configured_ev = float(configured_ev)
    if not math.isfinite(configured_ev) or configured_ev <= 0:
        raise ValueError("configured_ev must be finite and positive.")
    if not torch.all(f_e >= 0):
        raise ValueError("electron distribution must be non-negative.")

    norm = torch.trapz(f_e, v_e)
    if not torch.isfinite(norm) or not bool((norm > 0).item()):
        raise ValueError("electron distribution has an invalid integral.")
    mean = torch.trapz(v_e * f_e, v_e) / norm
    variance = torch.trapz((v_e - mean).square() * f_e, v_e) / norm
    effective_ev = ELECTRON_MASS * variance / ELEMENTARY_CHARGE
    ratio = effective_ev / configured_ev
    values = (mean, effective_ev, ratio)
    if not all(torch.isfinite(value).item() for value in values):
        raise ValueError("electron moment calculation produced non-finite values.")
    return {
        "mean_velocity_m_s": float(mean.item()),
        "effective_temperature_ev": float(effective_ev.item()),
        "ratio_to_configured": float(ratio.item()),
    }


def build_wavelength_bands(dtype):
    """Return the fixed ion, red-electron, and blue-electron wavelength bands."""

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    ion = torch.linspace(349.0, 353.0, 579, dtype=dtype) * 1.0e-9
    red = torch.linspace(375.3552, 599.7064, 304, dtype=dtype) * 1.0e-9
    blue = torch.linspace(100.1144, 339.6504, 341, dtype=dtype) * 1.0e-9
    return ion, red, blue


def generate_sample(rng, physical_config, ion_distributions):
    """Generate one spectrum, electron Fourier label, and moment record."""

    v_e, f_e = sample_electron_evdf(physical_config["electron_velocity_points"], rng)
    moments = electron_moments(
        v_e, f_e, physical_config["electron_reference_temperature_ev"]
    )
    k_grid, coefficients = ots_full_fourier.ft_x_n(
        v_e, f_e, physical_config["fourier_modes"]
    )

    spectra = []
    for wavelengths in physical_config["wavelength_bands"]:
        spectra.append(
            ots_full_fourier.OTS_spectrum_PV(
                wavelengths,
                physical_config["ss"],
                physical_config["k_sh"],
                physical_config["E_i0"],
                physical_config["lm_0"],
                physical_config["n_e"],
                physical_config["Z_i1"],
                physical_config["N_i1"],
                physical_config["m_i1"],
                physical_config["Z_i2"],
                physical_config["N_i2"],
                physical_config["m_i2"],
                physical_config["V"],
                v_e=v_e,
                f_e=f_e,
                v_i1=ion_distributions["v_i1"],
                f_i1=ion_distributions["f_i1"],
                v_i2=ion_distributions["v_i2"],
                f_i2=ion_distributions["f_i2"],
                u_e=physical_config["u_e"],
                u_i1=physical_config["u_i1"],
                u_i2=physical_config["u_i2"],
                fourier_modes=physical_config["fourier_modes"],
            )
        )

    spectrum = torch.cat(spectra)
    label = torch.cat((coefficients.real, coefficients.imag))
    if spectrum.shape != (INPUT_DIM,):
        raise ValueError("Generated spectrum must have shape [1224].")
    if label.shape != (OUTPUT_DIM,):
        raise ValueError("Generated label must have shape [160].")
    if k_grid.shape != (FOURIER_MODES,):
        raise ValueError("Generated K grid must have shape [80].")
    if (
        not torch.isfinite(spectrum).all()
        or not torch.isfinite(label).all()
        or not torch.isfinite(k_grid).all()
    ):
        raise ValueError("Generated sample contains non-finite values.")
    return {
        "spectrum": spectrum,
        "label": label,
        "k_grid": k_grid,
        "moments": moments,
    }


def temperature_statistics(records):
    """Return population statistics for the three recorded electron moments."""

    if not records:
        raise ValueError("Temperature records must not be empty.")
    result = {}
    for key in (
        "mean_velocity_m_s",
        "effective_temperature_ev",
        "ratio_to_configured",
    ):
        try:
            values = np.asarray([record[key] for record in records], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Invalid temperature record for {}.".format(key)
            ) from error
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Temperature records for {} must be finite.".format(key))
        result[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return result


def _physical_config():
    dtype = torch.float64
    scattering_angle = math.radians(123.0)
    bands = build_wavelength_bands(dtype)
    zero_drift = torch.zeros(3, dtype=dtype)
    return {
        "fourier_modes": FOURIER_MODES,
        "electron_velocity_points": ELECTRON_VELOCITY_POINTS,
        "electron_reference_temperature_ev": ELECTRON_REFERENCE_TEMPERATURE_EV,
        "wavelength_bands": bands,
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
        "u_e": zero_drift.clone(),
        "u_i1": zero_drift.clone(),
        "u_i2": zero_drift.clone(),
    }


def _generate_split(count, seed, physical_config, ion_distributions):
    if not isinstance(count, Integral) or isinstance(count, bool) or int(count) <= 0:
        raise ValueError("Physical sample count must be a positive integer.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("Seed must be an integer.")
    count = int(count)
    seed = int(seed)

    wavelengths = torch.cat(physical_config["wavelength_bands"]) / 1.0e-9
    rows = np.empty((count + 1, INPUT_DIM + OUTPUT_DIM), dtype=np.float32)
    rows[0, :INPUT_DIM] = wavelengths.cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(seed)
    moment_records = []
    reference_k = None

    for index in range(count):
        sample = generate_sample(rng, physical_config, ion_distributions)
        k_grid = sample["k_grid"].detach().cpu()
        if reference_k is None:
            reference_k = k_grid
            k_values = reference_k.numpy().astype(np.float32)
            rows[0, INPUT_DIM : INPUT_DIM + FOURIER_MODES] = k_values
            rows[0, INPUT_DIM + FOURIER_MODES :] = k_values
        elif not torch.allclose(k_grid, reference_k, rtol=1.0e-7, atol=0.0):
            raise ValueError("Generated samples do not share the same K grid.")

        rows[index + 1, :INPUT_DIM] = (
            sample["spectrum"].detach().cpu().numpy().astype(np.float32)
        )
        rows[index + 1, INPUT_DIM:] = (
            sample["label"].detach().cpu().numpy().astype(np.float32)
        )
        moment_records.append({"row": index + 1, **sample["moments"]})
    return rows, moment_records


def _write_parquet(rows, path):
    names = ["lm_{}".format(index) for index in range(INPUT_DIM)]
    names += ["K_{}".format(index) for index in range(OUTPUT_DIM)]
    arrays = [pa.array(rows[:, index]) for index in range(rows.shape[1])]
    table = pa.Table.from_arrays(arrays, names=names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, str(temporary))
    os.replace(str(temporary), str(path))


def _relative_path(path, base):
    return Path(os.path.relpath(str(path.resolve()), str(base.resolve()))).as_posix()


def _band_record(wavelengths):
    wavelengths_nm = wavelengths / 1.0e-9
    return {
        "start": float(wavelengths_nm[0].item()),
        "stop": float(wavelengths_nm[-1].item()),
        "points": len(wavelengths_nm),
    }


def generate_dataset_pair(
    run_id,
    dataset_id,
    output_dir,
    train_samples,
    test_samples,
    seed,
    overwrite=False,
):
    """Generate reproducible train/test Parquet files and their minimal manifest."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    seed = int(seed)

    output_dir = Path(output_dir)
    train_path = output_dir / "dataset_train.parquet"
    test_path = output_dir / "dataset_test.parquet"
    manifest_path = output_dir / "dataset_manifest.json"
    outputs = (train_path, test_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing dataset artifacts: {}".format(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_config = _physical_config()
    ion_distributions = build_fixed_ion_distributions(ION_SPECIES, ION_REFERENCE_TEMPERATURE_EV)
    train_rows, train_records = _generate_split(
        train_samples, seed, physical_config, ion_distributions
    )
    test_rows, test_records = _generate_split(
        test_samples, seed + 1, physical_config, ion_distributions
    )
    _write_parquet(train_rows, train_path)
    _write_parquet(test_rows, test_path)

    ss = physical_config["ss"] / torch.linalg.vector_norm(physical_config["ss"])
    k_sh = physical_config["k_sh"] / torch.linalg.vector_norm(physical_config["k_sh"])
    cosine = torch.clamp(torch.dot(ss, k_sh), min=-1.0, max=1.0)
    scattering_angle_deg = math.degrees(math.acos(float(cosine.item())))
    velocity_grid, _ = sample_electron_evdf(
        physical_config["electron_velocity_points"], np.random.default_rng(seed)
    )
    manifest = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "physical_configuration": {
            "fourier_modes": physical_config["fourier_modes"],
            "probe_wavelength_nm": physical_config["lm_0"] / 1.0e-9,
            "scattering_angle_deg": scattering_angle_deg,
            "electron_density_m3": physical_config["n_e"],
            "electron_dis": {
                "configured_reference_temperature_ev": physical_config[
                    "electron_reference_temperature_ev"
                ],
                "velocity_points": len(velocity_grid),
                "velocity_range_c": [
                    float((velocity_grid[0] / SPEED_OF_LIGHT).item()),
                    float((velocity_grid[-1] / SPEED_OF_LIGHT).item()),
                ],
            },
            "wavelength_bands_nm": [
                _band_record(band) for band in physical_config["wavelength_bands"]
            ],
        },
        "splits": {
            "train": {
                "file": _relative_path(train_path, manifest_path.parent),
                "seed": seed,
                "physical_samples": len(train_records),
                "samples": train_records,
                "statistics": temperature_statistics(train_records),
            },
            "test": {
                "file": _relative_path(test_path, manifest_path.parent),
                "seed": seed + 1,
                "physical_samples": len(test_records),
                "samples": test_records,
                "statistics": temperature_statistics(test_records),
            },
        },
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary_manifest), str(manifest_path))

    return {
        "train_path": train_path,
        "test_path": test_path,
        "manifest_path": manifest_path,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "manifest": manifest,
    }


def load_dataset(path):
    """Load the metadata-first Parquet contract without dropping its last row."""

    path = Path(path)
    table = pq.read_table(str(path))
    required_columns = INPUT_DIM + OUTPUT_DIM
    if table.num_columns < required_columns:
        raise ValueError(
            "Dataset must contain at least {} columns; got {}.".format(
                required_columns, table.num_columns
            )
        )
    if table.num_rows < 2:
        raise ValueError("Dataset must contain metadata and physical sample rows.")
    frame = table.select(list(range(required_columns))).to_pandas()
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains non-finite values.")

    metadata = values[0]
    first_k = metadata[INPUT_DIM : INPUT_DIM + FOURIER_MODES]
    second_k = metadata[INPUT_DIM + FOURIER_MODES : required_columns]
    if not np.allclose(first_k, second_k, rtol=1.0e-6, atol=0.0):
        raise ValueError("Metadata K grids for real and imaginary labels differ.")
    dataset = OTSDataset(
        spectra=torch.from_numpy(values[1:, :INPUT_DIM].copy()),
        coefficients=torch.from_numpy(values[1:, INPUT_DIM:].copy()),
        wavelengths_nm=torch.from_numpy(metadata[:INPUT_DIM].copy()),
        k_grid=torch.from_numpy(first_k.copy()),
    )
    validate_dataset(dataset)
    return dataset


def _require_uniform_increasing(values, name):
    spacing = values[1:] - values[:-1]
    if not torch.all(spacing > 0):
        raise ValueError("{} must be strictly increasing.".format(name))
    tolerance = torch.abs(spacing.mean()) * 1.0e-4
    if not torch.all(torch.abs(spacing - spacing.mean()) <= tolerance):
        raise ValueError("{} must be approximately uniform.".format(name))


def validate_dataset(dataset):
    """Validate dimensions, finite values, K spacing, and wavelength layout."""

    if not isinstance(dataset, OTSDataset):
        raise TypeError("dataset must be an OTSDataset.")
    if dataset.spectra.ndim != 2 or dataset.spectra.shape[1] != INPUT_DIM:
        raise ValueError("spectra must have shape [N, 1224].")
    if dataset.coefficients.ndim != 2 or dataset.coefficients.shape[1] != OUTPUT_DIM:
        raise ValueError("coefficients must have shape [N, 160].")
    if dataset.wavelengths_nm.shape != (INPUT_DIM,):
        raise ValueError("wavelengths_nm must have shape [1224].")
    if dataset.k_grid.shape != (FOURIER_MODES,):
        raise ValueError("k_grid must have shape [80].")
    sample_count = dataset.spectra.shape[0]
    if sample_count <= 0 or dataset.coefficients.shape[0] != sample_count:
        raise ValueError("spectra and coefficients must have equal positive counts.")
    for tensor, name in (
        (dataset.spectra, "spectra"),
        (dataset.coefficients, "coefficients"),
        (dataset.wavelengths_nm, "wavelengths_nm"),
        (dataset.k_grid, "k_grid"),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError("{} contains non-finite values.".format(name))

    _require_uniform_increasing(dataset.k_grid, "k_grid")
    expected = torch.cat(build_wavelength_bands(dataset.wavelengths_nm.dtype)) / 1.0e-9
    expected = expected.to(device=dataset.wavelengths_nm.device)
    if not torch.allclose(dataset.wavelengths_nm, expected, rtol=1.0e-5, atol=1.0e-4):
        raise ValueError("wavelengths_nm does not match the 579/304/341 layout.")
