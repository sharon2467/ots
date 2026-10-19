# -*- coding: utf-8 -*-
"""Canonical data generation and loading for the OTS EVDF pipeline."""

import csv
import copy
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
ION1_VELOCITY_POINTS = 1000
ION2_VELOCITY_POINTS = 1000
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
    layout: dict | None = None


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





def distribution_moments(v, f, mass_kg):
    """Measure the actual one-dimensional moments of the supplied distribution."""
    v, f = _real_arrays(v, f, "distribution")
    mass_kg = float(mass_kg)
    if not math.isfinite(mass_kg) or mass_kg <= 0 or not torch.all(f >= 0):
        raise ValueError("Mass must be positive and distribution non-negative.")
    norm = torch.trapz(f, v)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Distribution has an invalid integral.")
    mean = torch.trapz(v * f, v) / norm
    temperature = mass_kg * torch.trapz((v - mean).square() * f, v) / norm / ELEMENTARY_CHARGE
    if not torch.isfinite(mean) or not torch.isfinite(temperature):
        raise ValueError("Distribution moments must be finite.")
    return {"actual_temperature_ev": float(temperature), "actual_mean_velocity_m_s": float(mean)}


def electron_moments(v_e, f_e, configured_ev):
    configured_ev = float(configured_ev)
    if not math.isfinite(configured_ev) or configured_ev <= 0:
        raise ValueError("configured_ev must be finite and positive.")
    actual = distribution_moments(v_e, f_e, ELECTRON_MASS)
    return {"mean_velocity_m_s": actual["actual_mean_velocity_m_s"],
            "effective_temperature_ev": actual["actual_temperature_ev"],
            "ratio_to_configured": actual["actual_temperature_ev"] / configured_ev}


def build_wavelength_bands(dtype):
    """Return the fixed ion, red-electron, and blue-electron wavelength bands."""

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    ion = torch.linspace(349.0, 353.0, 579, dtype=dtype) * 1.0e-9
    red = torch.linspace(375.3552, 599.7064, 304, dtype=dtype) * 1.0e-9
    blue = torch.linspace(100.1144, 339.6504, 341, dtype=dtype) * 1.0e-9
    return ion, red, blue

SPECIES_BLOCKS = (("e", "electron_dis"), ("i1", "ion1_dis"), ("i2", "ion2_dis"))


class DistributionSampler:
    """One parameter draw followed by all three physical distributions."""

    def __init__(self, rng, physical_config):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        self.rng = rng
        self.physical_config = physical_config
        self.ion_species = [(s, physical_config[k]["mass_kg"]) for s, k in SPECIES_BLOCKS[1:]]
        self.electron_velocity_grid = torch.as_tensor(physical_config["electron_dis"]["velocity_grid_m_s"], dtype=torch.float64)
        self.ion1_velocity_grid = torch.as_tensor(physical_config["ion1_dis"]["velocity_grid_m_s"], dtype=torch.float64)
        self.ion2_velocity_grid = torch.as_tensor(physical_config["ion2_dis"]["velocity_grid_m_s"], dtype=torch.float64)

    def sample_parameters(self):
        raise NotImplementedError

    def sample(self):
        self.parameters = self.sample_parameters()
        expected = copy.deepcopy(self.parameters["expected"])
        if set(expected) != {"e", "i1", "i2"}:
            raise ValueError("Expected parameters must include all three species.")
        for values in expected.values():
            if (set(values) != {"temperature_ev", "mean_velocity_m_s"}
                    or not all(math.isfinite(float(v)) for v in values.values())
                    or values["temperature_ev"] <= 0):
                raise ValueError("Expected temperatures must be positive and parameters finite.")
        density = float(self.parameters["electron_density_m3"])
        if not math.isfinite(density) or density <= 0:
            raise ValueError("Electron density must be finite and positive.")
        return {"expected": expected, "electron_density_m3": density,
                "distributions": {"e": self.sample_electron_vdf(),
                                  "i1": self.sample_ion1_vdf(), "i2": self.sample_ion2_vdf()}}

    def sample_electron_vdf(self):
        raise NotImplementedError

    def sample_ion1_vdf(self):
        raise NotImplementedError

    def sample_ion2_vdf(self):
        raise NotImplementedError


class Fixed_ion_sampler(DistributionSampler):
    def __init__(self, rng, physical_config):
        super().__init__(rng, physical_config)
        self.reference_temperature_ion_ev = float(physical_config["ion1_dis"]["expected_temperature_ev"])
        self.reference_temperature_electron_ev = float(physical_config["electron_dis"]["expected_temperature_ev"])
        for temperature in (self.reference_temperature_ion_ev, self.reference_temperature_electron_ev):
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Reference temperatures must be finite and positive.")

    def sample_parameters(self):
        return {"expected": {s: {"temperature_ev": self.physical_config[k]["expected_temperature_ev"],
                                 "mean_velocity_m_s": self.physical_config[k]["expected_mean_velocity_m_s"]}
                             for s, k in SPECIES_BLOCKS},
                "electron_density_m3": self.physical_config["n_e"]}

    def sample_ion_vdf(self,idx):
        mass = self.ion_species[idx-1][1]
        thermal_speed = math.sqrt(self.reference_temperature_ion_ev * ELEMENTARY_CHARGE / mass)
        if idx==1:
            velocity = self.ion1_velocity_grid
        elif idx==2:
            velocity = self.ion2_velocity_grid
        values = torch.exp(-velocity.square() / (2.0 * thermal_speed**2))
        values = normalize_distribution(velocity, values, "ion{}".format(idx))
        return velocity, values
    def sample_ion1_vdf(self):
        return self.sample_ion_vdf(1)
    def sample_ion2_vdf(self):
        return self.sample_ion_vdf(2)
    def sample_electron_vdf(self):
        """Sample the current three-background-plus-one-hot electron EVDF."""
        num_points = len(self.electron_velocity_grid)
        velocity = self.electron_velocity_grid
        velocity_c = velocity.detach().cpu().numpy() / SPEED_OF_LIGHT
        values = np.zeros(num_points, dtype=np.float64)
        
        for _ in range(3):
            center = self.rng.uniform(-0.06, 0.06)
            sigma = self.rng.uniform(0.01, 0.05)
            amplitude = self.rng.uniform(0.2, 2.5)
            values += amplitude * np.exp(-((velocity_c - center) ** 2) / (2.0 * sigma**2))
        
        center = self.rng.uniform(-0.4, 0.4)
        sigma = self.rng.uniform(0.01, 0.05)
        amplitude = self.rng.uniform(0.001, 0.045)
        values += amplitude * np.exp(-((velocity_c - center) ** 2) / (2.0 * sigma**2))
        
        distribution = torch.from_numpy(values)
        distribution = normalize_distribution(velocity, distribution, "electron")
        return velocity, distribution


class Fig1_Fig11_sampler(DistributionSampler):
    def __init__(self, rng, physical_config):
        super().__init__(rng, physical_config)
        for name, value in physical_config["sampling"].items():
            values = np.asarray(value, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Sampling parameters must be finite: " + name)
            if name == "shape_probabilities":
                if values.shape != (3,) or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
                    raise ValueError("shape_probabilities must contain three nonnegative weights summing to one.")
            elif name.endswith("floor_ev"):
                if values.ndim != 0 or values <= 0:
                    raise ValueError("Temperature floors must be positive scalars.")
            elif values.shape != (2,) or values[1] < values[0]:
                raise ValueError("Sampling ranges must contain two ordered values: " + name)
            elif ("temperature" in name or "density" in name) and np.any(values <= 0):
                raise ValueError("Temperature factors and density ranges must be positive.")
        self.anchors = self.load_anchors(physical_config["points_csv"])

    def load_anchors(self, points_csv):
        pools = {"T_e_ev": [], "T_i_ev": []}
        with Path(points_csv).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["figure"] == "fig11" and row["shot_id"] == "20170811020" and row["parameter"] in pools:
                    value = float(row["value"])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("Anchor temperatures must be finite and positive.")
                    pools[row["parameter"]].append({"point_id": row["point_id"], "value": value})
        if not all(pools.values()):
            raise ValueError("Both Fig.11 temperature pools must be nonempty.")
        return {"fig1": [(209.0, 78.0), (1392.0, 912.0)], "fig11": pools}

    def sample_parameters(self):
        ranges = self.physical_config["sampling"]
        centers = self.anchors["fig11"]
        index = int(self.rng.integers(2 + len(centers["T_e_ev"])))
        if index < 2:
            te0, ti0 = self.anchors["fig1"][index]
            ne_range = ranges["density_fig1a_cm3" if index == 0 else "density_fig1b_cm3"]
        else:
            te0 = centers["T_e_ev"][index - 2]["value"]
            ti0 = centers["T_i_ev"][int(self.rng.integers(len(centers["T_i_ev"])))]["value"]
            ne_range = ranges["density_fig11_cm3"]
        te = max(ranges["electron_temperature_floor_ev"], te0 * math.exp(self.rng.uniform(*np.log(ranges["electron_temperature_factor"]))))
        ti = [max(ranges["ion_temperature_floor_ev"], ti0 * math.exp(self.rng.uniform(*np.log(ranges["ion_temperature_factor"])))) for _ in range(2)]
        density = self.rng.uniform(*ne_range) * 1e6
        common_u = self.rng.uniform(*ranges["ion_common_velocity_m_s"])
        ui = [common_u + self.rng.uniform(*ranges["ion_velocity_offset_m_s"]) for _ in range(2)]
        ue = self.rng.uniform(*ranges["electron_velocity_m_s"])
        return {"expected": {s: {"temperature_ev": float(t), "mean_velocity_m_s": float(u)}
                             for s, t, u in zip(("e", "i1", "i2"), (te, *ti), (ue, *ui))},
                "electron_density_m3": float(density)}

    def _sample_vdf(self, species):
        block = self.physical_config[dict(SPECIES_BLOCKS)[species]]
        v = getattr(self, {"e": "electron_velocity_grid", "i1": "ion1_velocity_grid", "i2": "ion2_velocity_grid"}[species])
        target = self.parameters["expected"][species]
        sigma = math.sqrt(ELEMENTARY_CHARGE * target["temperature_ev"] / block["mass_kg"])
        q = (v - target["mean_velocity_m_s"]) / sigma
        family = self.rng.choice(3, p=self.physical_config["sampling"]["shape_probabilities"])
        if family == 0:
            values = torch.exp(-q.square() / 2)
        elif family == 1:
            weights = self.rng.dirichlet([2.0, 2.0, 2.0])
            centers = self.rng.uniform(-1, 1, 3)
            widths = self.rng.uniform(.7, 1.4, 3)
            mean = np.sum(weights * centers)
            std = math.sqrt(float(np.sum(weights * (widths**2 + (centers - mean)**2))))
            values = sum(float(a / w) * torch.exp(-.5 * ((q * std + mean - c) / w).square())
                         for a, c, w in zip(weights, centers, widths))
        else:
            exponent = self.rng.uniform(2, 4)
            std = math.sqrt(math.gamma(3 / exponent) / math.gamma(1 / exponent))
            values = torch.exp(-torch.abs(q * std).pow(exponent))
        return v, normalize_distribution(v, values, species)

    def sample_electron_vdf(self):
        return self._sample_vdf("e")

    def sample_ion1_vdf(self):
        return self._sample_vdf("i1")

    def sample_ion2_vdf(self):
        return self._sample_vdf("i2")


def dataset_layout(physical_config):
    """Extract and validate coordinates and species semantics, without sampling."""
    modes = physical_config["fourier_modes"]
    if not isinstance(modes, Integral) or isinstance(modes, bool) or modes < 3:
        raise ValueError("fourier_modes must be an integer of at least 3.")
    bands = []
    for values in physical_config["wavelength_bands_nm"]:
        axis = torch.as_tensor(values, dtype=torch.float64)
        if axis.ndim != 1 or len(axis) < 2 or not torch.isfinite(axis).all() or not torch.all(axis > 0) or not torch.all(axis[1:] > axis[:-1]):
            raise ValueError("Wavelength bands must be positive, finite, increasing arrays.")
        bands.append(axis.tolist())
    if not bands:
        raise ValueError("At least one wavelength band is required.")
    result = {"fourier_modes": int(modes), "wavelength_bands_nm": bands}
    for species, key in SPECIES_BLOCKS:
        block = physical_config[key]
        if block["name"] != species or not isinstance(block["used_in_model"], bool):
            raise ValueError("Species names/order and boolean used_in_model are required.")
        mass, charge = float(block["mass_kg"]), float(block["charge"])
        if not math.isfinite(mass) or mass <= 0 or not math.isfinite(charge) or (charge >= 0 if species == "e" else charge <= 0):
            raise ValueError("Invalid species mass or charge.")
        velocity = torch.as_tensor(block["velocity_grid_m_s"], dtype=torch.float64)
        if velocity.ndim != 1 or len(velocity) < modes or not torch.isfinite(velocity).all():
            raise ValueError("Velocity axes must be finite and contain at least M points.")
        _require_uniform_increasing(velocity, key)
        result[key] = {"name": species, "mass_kg": mass, "charge": charge,
                       "velocity_grid_m_s": velocity.tolist(), "used_in_model": block["used_in_model"]}
    if not any(result[key]["used_in_model"] for _, key in SPECIES_BLOCKS):
        raise ValueError("At least one output species is required.")
    return result


def _physical_config(*, wavelength_bands_nm=None, velocity_grids_m_s=None,
                     used_in_model=None, fourier_modes=80):
    for overrides in (velocity_grids_m_s, used_in_model):
        if overrides is not None and (not isinstance(overrides, dict) or set(overrides) - {"e", "i1", "i2"}):
            raise ValueError("Grid/selection overrides must use e, i1, i2 keys.")
    dtype = torch.float64
    scattering_angle = math.radians(123.0)
    bands = build_wavelength_bands(dtype)
    config = {
        "fourier_modes": FOURIER_MODES,
        "electron_velocity_grid": torch.from_numpy(
            np.linspace(-0.8, 0.8, ELECTRON_VELOCITY_POINTS, dtype=np.float64)
            * SPEED_OF_LIGHT
        ),
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
        "u_e": torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        "u_i1": torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        "u_i2": torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
    }
    for index, points in ((1, ION1_VELOCITY_POINTS), (2, ION2_VELOCITY_POINTS)):
        thermal_speed = math.sqrt(
            ION_REFERENCE_TEMPERATURE_EV * ELEMENTARY_CHARGE / config[f"m_i{index}"]
        )
        config[f"ion{index}_velocity_grid"] = torch.linspace(
            -12.0 * thermal_speed, 12.0 * thermal_speed, points, dtype=dtype
        )
    config["dataset"] = "fixed_ion"
    config["fourier_modes"] = fourier_modes
    config["wavelength_bands_nm"] = [band / 1e-9 for band in bands] if wavelength_bands_nm is None else wavelength_bands_nm
    for species, key, grid_key, mass, charge, target in (
        ("e", "electron_dis", "electron_velocity_grid", ELECTRON_MASS, -1.0, 1370.0),
        ("i1", "ion1_dis", "ion1_velocity_grid", PROTON_MASS, 1.0, 171.3),
        ("i2", "ion2_dis", "ion2_velocity_grid", 12 * PROTON_MASS, 6.0, 171.3),
    ):
        config[key] = {
            "name": species, "mass_kg": mass, "charge": charge,
            "velocity_grid_m_s": (velocity_grids_m_s or {}).get(species, config[grid_key]),
            "used_in_model": (used_in_model or {}).get(species, species == "e"),
            "expected_temperature_ev": target, "expected_mean_velocity_m_s": 0.0,
        }
    for key in ("electron_velocity_grid", "ion1_velocity_grid", "ion2_velocity_grid",
                "wavelength_bands", "electron_reference_temperature_ev"):
        config.pop(key)
    dataset_layout(config)
    return config


def _physical_config_fig1_fig11(*, wavelength_bands_nm=None, velocity_grids_m_s=None,
                              used_in_model=None, fourier_modes=80):
    # The common configuration structure is reused; none of this reads anchors.
    config = _physical_config()
    for overrides in (velocity_grids_m_s, used_in_model):
        if overrides is not None and (not isinstance(overrides, dict) or set(overrides) - {"e", "i1", "i2"}):
            raise ValueError("Grid/selection overrides must use e, i1, i2 keys.")
    config["fourier_modes"] = fourier_modes
    config.update(dataset="fig1_fig11", lm_0=263.3e-9, N_i1=12/17, N_i2=5/17,
                  V=(100e-6)**3, ss=torch.tensor([0., 1., 0.], dtype=torch.float64))
    config.pop("n_e")
    config["points_csv"] = "C:/Users/win/.codex/worktrees/74c1/ots/outputs/ots_experimental_fits/fig6_fig11_shared_ne6e20_20260921/paper_experimental_points.csv"
    config["sampling"] = {
        "electron_temperature_factor": [.6, 1.5], "ion_temperature_factor": [.5, 1.8],
        "electron_temperature_floor_ev": 20., "ion_temperature_floor_ev": 10.,
        "density_fig1a_cm3": [2.8e20, 4.2e20], "density_fig1b_cm3": [3e20, 5e20],
        "density_fig11_cm3": [2.8e20, 5e20], "ion_common_velocity_m_s": [-2e4, 3e4],
        "ion_velocity_offset_m_s": [-5e3, 5e3], "electron_velocity_m_s": [-2.5e5, .5e5],
        "shape_probabilities": [.4, .4, .2],
    }
    config["wavelength_bands_nm"] = [torch.linspace(262.31242236024843, 264.2875776397516, 319, dtype=torch.float64)] if wavelength_bands_nm is None else wavelength_bands_nm
    for species, key in SPECIES_BLOCKS:
        block = config[key]
        limit = (2.5e5 if species == "e" else 3.5e4) + 8 * math.sqrt(ELEMENTARY_CHARGE * (2500 if species == "e" else 1700) / block["mass_kg"])
        block["velocity_grid_m_s"] = (velocity_grids_m_s or {}).get(species, torch.linspace(-limit, limit, 4097, dtype=torch.float64))
        block["used_in_model"] = (used_in_model or {}).get(species, True)
    dataset_layout(config)
    return config


def generate_sample(rng, physical_config, sampler):
    state = sampler.sample()
    v_e, f_e = state["distributions"]["e"]
    v_i1, f_i1 = state["distributions"]["i1"]
    v_i2, f_i2 = state["distributions"]["i2"]
    spectra = []
    for wavelengths in physical_config["wavelength_bands_nm"]:
        spectra.append(
            ots_full_fourier.OTS_spectrum_PV(
                torch.as_tensor(wavelengths, dtype=torch.float64) * 1e-9,
                physical_config["ss"],
                physical_config["k_sh"],
                physical_config["E_i0"],
                physical_config["lm_0"],
                state["electron_density_m3"],
                physical_config["Z_i1"],
                physical_config["N_i1"],
                physical_config["m_i1"],
                physical_config["Z_i2"],
                physical_config["N_i2"],
                physical_config["m_i2"],
                physical_config["V"],
                v_e=v_e,
                f_e=f_e,
                v_i1=v_i1,
                f_i1=f_i1,
                v_i2=v_i2,
                f_i2=f_i2,
                u_e=physical_config["u_e"],
                u_i1=physical_config["u_i1"],
                u_i2=physical_config["u_i2"],
                fourier_modes=physical_config["fourier_modes"],
            )
        )


    labels, grids, records = [], [], {}
    for species, key in SPECIES_BLOCKS:
        v, f = state["distributions"][species]
        block = physical_config[key]
        if block["used_in_model"]:
            k, coefficients = ots_full_fourier.ft_x_n(v, f, physical_config["fourier_modes"])
            grids.append(k)
            labels.append(torch.cat((coefficients.real, coefficients.imag)))
        expected = state["expected"][species]
        records[species] = {"expected_temperature_ev": expected["temperature_ev"],
                            "expected_mean_velocity_m_s": expected["mean_velocity_m_s"],
                            **distribution_moments(v, f, block["mass_kg"])}
    spectrum, label, k_grid = torch.cat(spectra), torch.cat(labels), torch.cat(grids)
    input_dim = sum(len(band) for band in physical_config["wavelength_bands_nm"])
    modes = physical_config["fourier_modes"]
    count = sum(physical_config[key]["used_in_model"] for _, key in SPECIES_BLOCKS)
    if spectrum.shape != (input_dim,) or label.shape != (2 * modes * count,) or k_grid.shape != (modes * count,):
        raise ValueError("Generated sample dimensions disagree with configuration.")
    if not all(torch.isfinite(t).all() for t in (spectrum, label, k_grid)):
        raise ValueError("Generated sample contains non-finite values.")
    return {"spectrum": spectrum, "label": label, "k_grid": k_grid,
            "sample_record": {"electron_density_m3": state["electron_density_m3"], "species": records}}


def temperature_statistics(records):
    if not records:
        raise ValueError("Temperature records must not be empty.")
    result = {}
    for species, _ in SPECIES_BLOCKS:
        result[species] = {}
        for key in ("expected_temperature_ev", "expected_mean_velocity_m_s", "actual_temperature_ev", "actual_mean_velocity_m_s"):
            try:
                values = np.asarray([record["species"][species][key] for record in records], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid species moment record.") from error
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Moment records must be finite.")
            result[species][key] = {"min": float(values.min()), "max": float(values.max()),
                                    "mean": float(values.mean()), "median": float(np.median(values)),
                                    "std": float(values.std(ddof=0))}
    return result


def _generate_split(count, seed, physical_config, sampler_class=Fixed_ion_sampler):
    if not isinstance(count, Integral) or isinstance(count, bool) or int(count) <= 0:
        raise ValueError("Physical sample count must be a positive integer.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("Seed must be an integer.")
    count = int(count)
    seed = int(seed)

    layout = dataset_layout(physical_config)
    wavelengths = torch.cat([torch.tensor(band, dtype=torch.float64) for band in layout["wavelength_bands_nm"]])
    input_dim = len(wavelengths)
    modes = layout["fourier_modes"]
    output_dim = 2 * modes * sum(layout[k]["used_in_model"] for _, k in SPECIES_BLOCKS)
    rows = np.empty((count + 1, input_dim + output_dim), dtype=np.float32)
    rows[0, :input_dim] = wavelengths.numpy().astype(np.float32)
    rng = np.random.default_rng(seed)
    moment_records = []
    reference_k = None
    sampler = sampler_class(rng, physical_config)
    for index in range(count):
        sample = generate_sample(rng, physical_config,sampler)
        k_grid = sample["k_grid"].detach().cpu()
        if reference_k is None:
            reference_k = k_grid
            k_values = reference_k.numpy().astype(np.float32)
            rows[0, input_dim:] = np.repeat(k_values.reshape(-1, modes), 2, axis=0).reshape(-1)
        elif not torch.allclose(k_grid, reference_k, rtol=1.0e-7, atol=0.0):
            raise ValueError("Generated samples do not share the same K grid.")

        rows[index + 1, :input_dim] = (
            sample["spectrum"].detach().cpu().numpy().astype(np.float32)
        )
        rows[index + 1, input_dim:] = (
            sample["label"].detach().cpu().numpy().astype(np.float32)
        )
        moment_records.append({"row": index + 1, **sample["sample_record"]})
    return rows, moment_records


def _write_parquet(rows, path, *, input_dim=INPUT_DIM):
    names = ["lm_{}".format(index) for index in range(input_dim)]
    names += ["K_{}".format(index) for index in range(rows.shape[1] - input_dim)]
    arrays = [pa.array(rows[:, index]) for index in range(rows.shape[1])]
    table = pa.Table.from_arrays(arrays, names=names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, str(temporary))
    os.replace(str(temporary), str(path))


def _relative_path(path, base):
    return Path(os.path.relpath(str(path.resolve()), str(base.resolve()))).as_posix()


def generate_dataset_pair(
    run_id,
    dataset_id,
    output_dir,
    train_samples,
    test_samples,
    seed,
    overwrite=False,
    *, dataset="fixed_ion", generation_config=None,
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

    if generation_config is not None and not isinstance(generation_config, dict):
        raise ValueError("generation_config must be a dictionary.")
    options = dict(generation_config or {})
    axes = {key: options.pop(key) for key in ("wavelength_bands_nm", "velocity_grids_m_s", "used_in_model", "fourier_modes") if key in options}
    if dataset == "fixed_ion":
        physical_config, sampler_class = _physical_config(**axes), Fixed_ion_sampler
        if options:
            raise ValueError("Unsupported fixed_ion generation options: {}".format(sorted(options)))
    elif dataset == "fig1_fig11":
        physical_config, sampler_class = _physical_config_fig1_fig11(**axes), Fig1_Fig11_sampler
        if set(options) - {"points_csv", "sampling"}:
            raise ValueError("Unsupported fig1_fig11 generation options.")
        if "points_csv" in options:
            physical_config["points_csv"] = str(options["points_csv"])
        if "sampling" in options:
            if not isinstance(options["sampling"], dict):
                raise ValueError("sampling must be a dictionary of partial overrides.")
            if set(options["sampling"]) - set(physical_config["sampling"]):
                raise ValueError("Unsupported sampling options.")
            physical_config["sampling"].update(options["sampling"])
    else:
        raise ValueError("dataset must be fixed_ion or fig1_fig11.")
    layout = dataset_layout(physical_config)
    train_rows, train_records = _generate_split(train_samples, seed, physical_config, sampler_class)
    test_rows, test_records = _generate_split(test_samples, seed + 1, physical_config, sampler_class)
    input_dim = sum(map(len, layout["wavelength_bands_nm"]))
    _write_parquet(train_rows, train_path, input_dim=input_dim)
    _write_parquet(test_rows, test_path, input_dim=input_dim)
    # Serialize the common layout once; sampling fields remain ordinary JSON values.
    physical = dict(layout)
    for _, key in SPECIES_BLOCKS:
        physical[key].update({k: physical_config[key][k] for k in ("expected_temperature_ev", "expected_mean_velocity_m_s")})
    physical.update(dataset=dataset, probe_wavelength_nm=physical_config["lm_0"] / 1e-9,
                    scattering_angle_deg=math.degrees(math.acos(float(torch.dot(physical_config["ss"], physical_config["k_sh"])))),
                    ion1_fraction=physical_config["N_i1"], ion2_fraction=physical_config["N_i2"])
    for key in ("sampling", "points_csv"):
        if key in physical_config:
            physical[key] = physical_config[key]
    if "n_e" in physical_config:
        physical["electron_density_m3"] = physical_config["n_e"]
    manifest = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "physical_configuration": physical,
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


def load_dataset(path, physical_configuration=None):
    """Load the metadata-first Parquet contract without dropping its last row."""

    path = Path(path)
    if physical_configuration is None and (path.parent / "dataset_manifest.json").exists():
        with (path.parent / "dataset_manifest.json").open(encoding="utf-8") as stream:
            physical_configuration = json.load(stream).get("physical_configuration")
    layout = None
    if physical_configuration and any(
        set(physical_configuration.get(key, {})) & {"name", "mass_kg", "velocity_grid_m_s"}
        for _, key in SPECIES_BLOCKS
    ):
        try:
            layout = dataset_layout(physical_configuration)
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete dataset layout in physical_configuration.") from error
    modes = layout["fourier_modes"] if layout else FOURIER_MODES
    input_dim = sum(map(len, layout["wavelength_bands_nm"])) if layout else INPUT_DIM
    species_count = sum(layout[k]["used_in_model"] for _, k in SPECIES_BLOCKS) if layout else 1
    required_columns = input_dim + 2 * modes * species_count
    table = pq.read_table(str(path))
    if table.num_columns != required_columns:
        raise ValueError(
            "Dataset must contain exactly {} columns; got {}.".format(
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
    blocks = metadata[input_dim:].reshape(species_count, 2, modes)
    if not np.allclose(blocks[:, 0], blocks[:, 1], rtol=1e-6, atol=0):
        raise ValueError("Metadata K grids for real and imaginary labels differ.")
    first_k = blocks[:, 0].reshape(-1)
    dataset = OTSDataset(
        spectra=torch.from_numpy(values[1:, :input_dim].copy()),
        coefficients=torch.from_numpy(values[1:, input_dim:].copy()),
        wavelengths_nm=torch.from_numpy(metadata[:input_dim].copy()),
        k_grid=torch.from_numpy(first_k.copy()),
        layout=layout,
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
    """Check shape, physical coordinates, and independent Fourier blocks."""
    if not isinstance(dataset, OTSDataset):
        raise TypeError("dataset must be an OTSDataset.")
    layout = dataset_layout(dataset.layout) if dataset.layout is not None else None
    modes = layout["fourier_modes"] if layout else FOURIER_MODES
    selected = [key for _, key in SPECIES_BLOCKS if layout[key]["used_in_model"]] if layout else ["electron_dis"]
    input_dim = sum(map(len, layout["wavelength_bands_nm"])) if layout else INPUT_DIM
    output_dim = 2 * modes * len(selected)
    if dataset.spectra.ndim != 2 or dataset.spectra.shape[1] != input_dim:
        raise ValueError("Spectrum dimensions disagree with layout.")
    if dataset.coefficients.shape != (len(dataset.spectra), output_dim) or len(dataset.spectra) == 0:
        raise ValueError("Coefficient dimensions disagree with layout.")
    if dataset.wavelengths_nm.shape != (input_dim,) or dataset.k_grid.shape != (modes * len(selected),):
        raise ValueError("Coordinate dimensions disagree with layout.")
    for tensor in (dataset.spectra, dataset.coefficients, dataset.wavelengths_nm, dataset.k_grid):
        if not torch.isfinite(tensor).all():
            raise ValueError("Dataset contains non-finite values.")
    expected_wavelengths = (torch.tensor(sum(layout["wavelength_bands_nm"], []), dtype=dataset.wavelengths_nm.dtype)
                            if layout else torch.cat(build_wavelength_bands(dataset.wavelengths_nm.dtype)) / 1e-9)
    if not torch.allclose(dataset.wavelengths_nm, expected_wavelengths.to(dataset.wavelengths_nm.device), rtol=1e-7 if layout else 1e-5, atol=1e-5 if layout else 1e-4):
        raise ValueError("Wavelengths do not match layout.")
    for index, key in enumerate(selected):
        k = dataset.k_grid[index * modes:(index + 1) * modes]
        _require_uniform_increasing(k, "k_grid")
        if layout:
            v = torch.tensor(layout[key]["velocity_grid_m_s"], dtype=torch.float64)
            expected_k, _ = ots_full_fourier.ft_x_n(v, torch.zeros_like(v), modes)
            if not torch.allclose(k, expected_k.to(k), rtol=1e-6, atol=0):
                raise ValueError("Fourier K grid disagrees with species velocity axis.")
