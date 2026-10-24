# -*- coding: utf-8 -*-
"""Compare the canonical full-Fourier and analytic Maxwellian OTS paths.

This is a deliberately standalone diagnostic rather than a unittest-discovery
module.  It imports the physical configuration and constants from ``data.py``,
constructs Maxwellians at those temperatures, compares all three canonical
wavelength segments, and exits non-zero when the canonical-resolution spectra
are not numerically consistent under the declared tolerances.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path

import torch

import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The comparison only needs configuration/distribution helpers from data.py.
# Permit the known torch-only local environment to import that module without
# pretending to provide any usable Parquet API.
if importlib.util.find_spec("pyarrow") is None:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__path__ = []
    parquet_stub = types.ModuleType("pyarrow.parquet")
    sys.modules["pyarrow"] = pyarrow_stub
    sys.modules["pyarrow.parquet"] = parquet_stub

import data  # noqa: E402
from phyEqns import ots_full_fourier as full  # noqa: E402


BAND_NAMES = ("ion_peak", "red_electron", "blue_electron")
MAX_ABS_TOLERANCE = 1.0e-2
RELATIVE_L2_TOLERANCE = 1.0e-2
PEAK_VALUE_TOLERANCE = 1.0e-2
PEAK_SAMPLE_TOLERANCE = 1.0
REFINEMENT_FACTOR = 8


def _maxwellian(velocity, temperature_ev, mass, name):
    """Return a project-normalized 1D Maxwellian on ``velocity``."""

    thermal_speed = math.sqrt(
        float(temperature_ev) * data.ELEMENTARY_CHARGE / float(mass)
    )
    values = torch.exp(-velocity.square() / (2.0 * thermal_speed**2))
    return data.normalize_distribution(velocity, values, name)


def _canonical_distributions(config):
    """Build Maxwellian e/H/C arrays from the canonical config/constants."""

    electron_velocity = torch.linspace(
        -0.8 * data.SPEED_OF_LIGHT,
        0.8 * data.SPEED_OF_LIGHT,
        data.ELECTRON_VELOCITY_POINTS,
        dtype=torch.float64,
    )
    electron_distribution = _maxwellian(
        electron_velocity,
        config["electron_reference_temperature_ev"],
        data.ELECTRON_MASS,
        "comparison-electron",
    )
    ions = data.build_fixed_ion_distributions(
        data.ION_SPECIES, data.ION_REFERENCE_TEMPERATURE_EV
    )
    return {
        "v_e": electron_velocity,
        "f_e": electron_distribution,
        **ions,
    }


def _refined_distributions(config, factor):
    """Expand computational support and K sampling for source diagnosis.

    Temperatures, densities, species and observation parameters remain fixed.
    The nonrelativistic Maxwellian integration domain and sample count grow by
    ``factor`` while preserving nearly the same velocity-grid spacing; the
    Fourier mode count grows by the same factor, keeping similar K_max but
    reducing delta-K.
    """

    points = factor * (data.ELECTRON_VELOCITY_POINTS - 1) + 1
    canonical = _canonical_distributions(config)
    refined = {}
    for suffix, mass in (
        ("e", data.ELECTRON_MASS),
        ("i1", config["m_i1"]),
        ("i2", config["m_i2"]),
    ):
        source_velocity = canonical["v_{}".format(suffix)]
        support = factor * float(torch.max(torch.abs(source_velocity)).item())
        velocity = torch.linspace(-support, support, points, dtype=torch.float64)
        temperature = (
            config["electron_reference_temperature_ev"]
            if suffix == "e"
            else data.ION_REFERENCE_TEMPERATURE_EV
        )
        refined["v_{}".format(suffix)] = velocity
        refined["f_{}".format(suffix)] = _maxwellian(
            velocity, temperature, mass, "refined-{}".format(suffix)
        )
    return refined


def _full_spectrum(config, distributions, wavelengths, modes):
    """Call the public full-Fourier path with one explicit configuration."""
    candidate1 =full.OTS_spectrum_PV(
        wavelengths,
        config["ss"],
        config["k_sh"],
        config["E_i0"],
        config["lm_0"],
        config["n_e"],
        config["Z_i1"],
        config["N_i1"],
        config["m_i1"],
        config["Z_i2"],
        config["N_i2"],
        config["m_i2"],
        config["V"],
        v_e=distributions["v_e"],
        f_e=distributions["f_e"],
        v_i1=distributions["v_i1"],
        f_i1=distributions["f_i1"],
        v_i2=distributions["v_i2"],
        f_i2=distributions["f_i2"],
        u_e=config["u_e"],
        u_i1=config["u_i1"],
        u_i2=config["u_i2"],
        fourier_modes=modes,
    )
    candidate2 = full.OTS_spectrum_full_fourier(
        wavelengths,
        config["ss"],
        config["k_sh"],
        config["E_i0"],
        config["lm_0"],
        config["n_e"],
        config["Z_i1"],
        config["N_i1"],
        config["m_i1"],
        config["Z_i2"],
        config["N_i2"],
        config["m_i2"],
        config["V"],
        v_e=distributions["v_e"],
        f_e=distributions["f_e"],
        v_i1=distributions["v_i1"],
        f_i1=distributions["f_i1"],
        v_i2=distributions["v_i2"],
        f_i2=distributions["f_i2"],
        u_e=config["u_e"],
        u_i1=config["u_i1"],
        u_i2=config["u_i2"],
        fourier_modes=modes,
    )
    return candidate2


def _analytic_spectrum(config, wavelengths):
    """Call the public temperature-driven two-ion Maxwellian path."""

    signal= full.OTS_spectrum_maxwellian_two_ion(
        wavelengths,
        config["ss"],
        config["k_sh"],
        config["E_i0"],
        config["lm_0"],
        config["n_e"],
        config["electron_reference_temperature_ev"],
        config["Z_i1"],
        config["N_i1"],
        config["m_i1"],
        data.ION_REFERENCE_TEMPERATURE_EV,
        config["Z_i2"],
        config["N_i2"],
        config["m_i2"],
        data.ION_REFERENCE_TEMPERATURE_EV,
        config["V"],
        u_e=config["u_e"],
        u_i1=config["u_i1"],
        u_i2=config["u_i2"],
    )
    return signal

def _metrics(wavelengths, candidate, reference):
    """Return normalized shape and peak metrics for one wavelength band."""

    difference = candidate - reference
    candidate_peak_index = int(torch.argmax(torch.abs(candidate)).item())
    reference_peak_index = int(torch.argmax(torch.abs(reference)).item())
    wavelength_nm = wavelengths * 1.0e9
    grid_step_nm = float(torch.max(torch.diff(wavelength_nm)).item())
    candidate_peak_nm = float(wavelength_nm[candidate_peak_index].item())
    reference_peak_nm = float(wavelength_nm[reference_peak_index].item())
    candidate_peak_value = float(candidate[candidate_peak_index].item())
    reference_peak_value = float(reference[reference_peak_index].item())
    maximum_absolute = float(torch.max(torch.abs(difference)).item())
    relative_l2 = float(
        (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()
    )
    peak_shift_nm = abs(candidate_peak_nm - reference_peak_nm)
    peak_value_difference = abs(candidate_peak_value - reference_peak_value)
    passed = (
        maximum_absolute <= MAX_ABS_TOLERANCE
        and relative_l2 <= RELATIVE_L2_TOLERANCE
        and peak_shift_nm <= PEAK_SAMPLE_TOLERANCE * grid_step_nm + 1.0e-12
        and peak_value_difference <= PEAK_VALUE_TOLERANCE
    )
    return {
        "max_abs": maximum_absolute,
        "relative_l2": relative_l2,
        "candidate_peak_nm": candidate_peak_nm,
        "reference_peak_nm": reference_peak_nm,
        "peak_shift_nm": peak_shift_nm,
        "candidate_peak_value": candidate_peak_value,
        "reference_peak_value": reference_peak_value,
        "peak_value_difference": peak_value_difference,
        "grid_step_nm": grid_step_nm,
        "passed": passed,
    }


def _phase_velocity(config, wavelengths, drift):
    """Reproduce the shared k_s-k_i, omega_s-omega_i phase mapping."""

    ss = config["ss"] / torch.linalg.vector_norm(config["ss"])
    k_sh = config["k_sh"] / torch.linalg.vector_norm(config["k_sh"])
    omega_0 = 2.0 * math.pi * full.SPEED_OF_LIGHT / config["lm_0"]
    omega_pe_squared = (
        config["n_e"]
        * full.ELEMENTARY_CHARGE**2
        / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
    )
    k_0 = (
        math.sqrt(omega_0**2 - omega_pe_squared)
        / full.SPEED_OF_LIGHT
        * k_sh
    )
    omega = 2.0 * math.pi * full.SPEED_OF_LIGHT / wavelengths
    scattered_k = torch.sqrt(omega.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
    components = tuple(-k_0[index] + ss[index] * scattered_k for index in range(3))
    magnitude = torch.sqrt(sum(component.square() for component in components))
    drift_projection = sum(
        components[index] * drift[index] for index in range(3)
    ) / magnitude
    return (omega - omega_0) / magnitude - drift_projection


def _response_errors(config, distributions, wavelengths, modes):
    """Compare sampled Fourier response kernels with exact Eq. (9) kernels."""

    result = {}
    samples={}
    exacts={}
    species = (
        (
            "e",
            data.ELECTRON_MASS,
            config["electron_reference_temperature_ev"],
            config["u_e"],
        ),
        (
            "i1",
            config["m_i1"],
            data.ION_REFERENCE_TEMPERATURE_EV,
            config["u_i1"],
        ),
        (
            "i2",
            config["m_i2"],
            data.ION_REFERENCE_TEMPERATURE_EV,
            config["u_i2"],
        ),
    )
    for suffix, mass, temperature_ev, drift in species:
        thermal_speed = math.sqrt(
            temperature_ev * data.ELEMENTARY_CHARGE / mass
        )
        velocity = _phase_velocity(config, wavelengths, drift)
        k_grid, coefficients = full.ft_x_n(
            distributions["v_{}".format(suffix)],
            distributions["f_{}".format(suffix)],
            modes,
        )
        sampled1 = (
            full._pv_fd_landau_response(distributions["v_{}".format(suffix)], distributions["f_{}".format(suffix)],velocity,name='a')
            * thermal_speed**2
        )
        sampled2 = full.fourier_landau_response(
            k_grid, coefficients, velocity,
        )* thermal_speed**2
        sampled =sampled2
        samples[suffix] = sampled
        exact = full._maxwellian_response_eq9(
            velocity / (math.sqrt(2.0) * thermal_speed)
        )
        exacts[suffix] = exact
        difference = sampled - exact
        result[suffix] = {
            "max_abs": float(torch.max(torch.abs(difference)).item()),
            "relative_l2": float(
                (
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(exact)
                ).item()
            ),
            "max_abs_x": float(
                torch.max(torch.abs(velocity / (math.sqrt(2.0) * thermal_speed))).item()
            ),
        }
    return result,samples,exacts


def _shared_implementation_conventions():
    """Check source-level sign and normalization markers in both public paths."""

    sources = (
        inspect.getsource(full.OTS_spectrum_full_fourier),
        inspect.getsource(full.OTS_spectrum_maxwellian_two_ion),
    )
    sign_markers = (
        "-k_0[0] + ss[0] * k_scattered_magnitude",
        "omega_shift = omega - omega_0",
    )
    normalization_markers = (
        "scale = torch.max(torch.abs(power))",
        "intensity = power / scale",
    )
    return {
        "paired_sign": all(marker in source for source in sources for marker in sign_markers),
        "segment_normalization": all(
            marker in source for source in sources for marker in normalization_markers
        ),
    }


def _validate_config_source(config):
    """Reject accidental drift away from data.py's declared globals."""

    checks = (
        config["fourier_modes"] == data.FOURIER_MODES,
        config["electron_velocity_points"] == data.ELECTRON_VELOCITY_POINTS,
        config["electron_reference_temperature_ev"]
        == data.ELECTRON_REFERENCE_TEMPERATURE_EV,
        config["m_i1"] == data.ION_SPECIES[0][1],
        config["m_i2"] == data.ION_SPECIES[1][1],
        tuple(len(band) for band in config["wavelength_bands"]) == (579, 304, 341),
    )
    if not all(checks):
        raise RuntimeError("data._physical_config() and data.py globals are inconsistent.")


def main():
    """Run canonical comparison, refinement diagnosis, and print a report."""

    config = data._physical_config()
    _validate_config_source(config)
    canonical_distributions = _canonical_distributions(config)
    refined_distributions = _refined_distributions(config, REFINEMENT_FACTOR)
    conventions = _shared_implementation_conventions()

    print("CONFIG_SOURCE=data._physical_config() + data.py global constants")
    print(
        "CONFIG Te_eV={:.6g} Ti1_eV={:.6g} Ti2_eV={:.6g} ne_m^-3={:.6g} "
        "modes={} electron_points={} bands={}".format(
            config["electron_reference_temperature_ev"],
            data.ION_REFERENCE_TEMPERATURE_EV,
            data.ION_REFERENCE_TEMPERATURE_EV,
            config["n_e"],
            config["fourier_modes"],
            config["electron_velocity_points"],
            tuple(len(band) for band in config["wavelength_bands"]),
        )
    )
    print(
        "TOLERANCE max_abs<=1% rel_l2<=1% peak_shift<=1_sample "
        "peak_value_diff<=1%"
    )
    print(
        "SOURCE_CHECK paired_sign={} segment_max_abs_normalization={}".format(
            conventions["paired_sign"], conventions["segment_normalization"]
        )
    )

    all_passed = True
    plt.figure(figsize=(12, 8))
    for name, wavelengths in zip(BAND_NAMES, config["wavelength_bands"]):
        analytic = _analytic_spectrum(config, wavelengths)
        canonical_full = _full_spectrum(
            config, canonical_distributions, wavelengths, config["fourier_modes"]
        )
        refined_full = _full_spectrum(
            config,
            refined_distributions,
            wavelengths,
            REFINEMENT_FACTOR * config["fourier_modes"],
        )
        canonical_metrics = _metrics(wavelengths, canonical_full, analytic)
        refined_metrics = _metrics(wavelengths, refined_full, analytic)
        canonical_response_errors, sampled, exact = _response_errors(
            config,
            canonical_distributions,
            wavelengths,
            config["fourier_modes"],
        )
        refined_response_errors, sampled_refined, _ = _response_errors(
            config,
            refined_distributions,
            wavelengths,
            REFINEMENT_FACTOR * config["fourier_modes"],
        )
        all_passed = all_passed and canonical_metrics["passed"]

        if canonical_metrics["passed"]:
            diagnosis = "consistent_within_declared_tolerance"
        elif (
            conventions["paired_sign"]
            and conventions["segment_normalization"]
            and max(
                canonical_response_errors["i1"]["relative_l2"],
                canonical_response_errors["i2"]["relative_l2"],
            )
            > 1.0
        ):
            diagnosis = "finite_Fourier_velocity_grid_aliasing_in_ion_response"
        elif (
            refined_metrics["max_abs"] < 0.5 * canonical_metrics["max_abs"]
            and refined_metrics["relative_l2"]
            < 0.5 * canonical_metrics["relative_l2"]
        ):
            diagnosis = "finite_Fourier_or_velocity_grid_discretization"
        elif not conventions["paired_sign"]:
            diagnosis = "sign_convention_mismatch"
        else:
            diagnosis = "persistent_formula_level_difference"

        print(
            "BAND {} canonical max_abs={:.9e} rel_l2={:.9e} "
            "peak_nm(full/analytic)={:.9f}/{:.9f} peak_shift_nm={:.9e} "
            "peak_value(full/analytic)={:.9f}/{:.9f} peak_value_diff={:.9e} "
            "grid_step_nm={:.9e} pass={}".format(
                name,
                canonical_metrics["max_abs"],
                canonical_metrics["relative_l2"],
                canonical_metrics["candidate_peak_nm"],
                canonical_metrics["reference_peak_nm"],
                canonical_metrics["peak_shift_nm"],
                canonical_metrics["candidate_peak_value"],
                canonical_metrics["reference_peak_value"],
                canonical_metrics["peak_value_difference"],
                canonical_metrics["grid_step_nm"],
                canonical_metrics["passed"],
            )
        )
        print(
            "BAND {} refined_diagnostic factor={} modes={} max_abs={:.9e} "
            "rel_l2={:.9e} peak_shift_nm={:.9e} diagnosis={}".format(
                name,
                REFINEMENT_FACTOR,
                REFINEMENT_FACTOR * config["fourier_modes"],
                refined_metrics["max_abs"],
                refined_metrics["relative_l2"],
                refined_metrics["peak_shift_nm"],
                diagnosis,
            )
        )
        print(
            "BAND {} response_kernel canonical dimensionless_rel_l2(e/i1/i2)="
            "{:.9e}/{:.9e}/{:.9e} max_abs_x(e/i1/i2)={:.6g}/{:.6g}/{:.6g}".format(
                name,
                canonical_response_errors["e"]["relative_l2"],
                canonical_response_errors["i1"]["relative_l2"],
                canonical_response_errors["i2"]["relative_l2"],
                canonical_response_errors["e"]["max_abs_x"],
                canonical_response_errors["i1"]["max_abs_x"],
                canonical_response_errors["i2"]["max_abs_x"],
            )
        )
        plt.subplot(len(BAND_NAMES),len(BAND_NAMES),BAND_NAMES.index(name)*len(BAND_NAMES)+1)
        plt.plot(wavelengths * 1.0e9, analytic, label="analytic", color="black")
        plt.plot(wavelengths * 1.0e9, canonical_full, label="canonical_full", color="#467A22")
        plt.subplot(len(BAND_NAMES),len(BAND_NAMES),BAND_NAMES.index(name)*len(BAND_NAMES)+2)
        plt.plot(wavelengths * 1.0e9, analytic, label="analytic", color="black")
        plt.plot(wavelengths * 1.0e9, refined_full, label="refined_full", color="#FF7F00")
        plt.subplot(len(BAND_NAMES),len(BAND_NAMES),BAND_NAMES.index(name)*len(BAND_NAMES)+3)
        plt.plot(wavelengths * 1.0e9, sampled['i2'], label="K response", color= "#C615C6")
        plt.plot(wavelengths * 1.0e9, exact['i2'], label="Eq(9) response", color= "#000000")
        plt.plot(wavelengths * 1.0e9, sampled_refined['i2'], label="K response refined", color= "#934720")

    print("OVERALL canonical_numerically_consistent={}".format(all_passed))
    plt.show()
    if not conventions["segment_normalization"]:
        print("WARNING normalization implementations do not share expected markers")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
