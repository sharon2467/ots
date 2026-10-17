# -*- coding: utf-8 -*-
"""Fit an ion-acoustic-wave OTS spectrum following Zhao et al. (2019).

The public :func:`fit_spectrum` function extends the least-squares model in
Eq. (10) to eight parameters: electron temperature, one temperature for each
ion species, ion/electron projected velocities, signal amplitude, and the two
endpoint background levels.  The electron density is fixed by the caller, as
described for IAW fitting in the paper.

The public :func:`plot_fit_result` function adds a spectrum comparison and
paper-style ``Delta R`` contours in the ``T_e-T_i1`` and ``T_e-T_i2`` planes.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares

from phyEqns.ots_full_fourier import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    OTS_spectrum_maxwellian_two_ion,
)


_PROBE_WAVELENGTH_NM = 263.3
_SCATTERING_ANGLE_DEG = 90.0
_ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
_PARAMETER_NAMES = (
    "T_e_ev",
    "T_i1_ev",
    "T_i2_ev",
    "u_ik_cm_s",
    "u_ek_cm_s",
    "A",
    "B1",
    "B2",
)
_LOWER_BOUNDS = np.array(
    [10.0, 10.0, 10.0, -1.5e8, -3.0e8, 0.0, -2.0, -2.0], dtype=np.float64
)
_UPPER_BOUNDS = np.array(
    [5000.0, 6000.0, 6000.0, 1.5e8, 3.0e8, 10.0, 2.0, 2.0], dtype=np.float64
)


def fit_spectrum(
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
    initial_guess=None,
    *,
    probe_wavelength_nm=_PROBE_WAVELENGTH_NM,
    scattering_angle_deg=_SCATTERING_ANGLE_DEG,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
) -> dict[str, float]:
    """Fit one IAW spectrum and return the eight fitted parameters.

    Parameters
    ----------
    wavelength_nm : array-like
        Strictly increasing absolute wavelengths in nm.
    measured_power : array-like
        Finite relative spectral power at each wavelength.
    electron_density_cm3 : float
        Fixed electron number density in cm^-3.
    instrument_fwhm_nm : float
        Gaussian instrumental spectral resolution (FWHM) in nm.
    initial_guess : array-like, optional
        Initial values ordered as
        ``(T_e, T_i1, T_i2, u_ik, u_ek, A, B1, B2)``. Temperatures are in
        eV and velocities are in cm/s.
    probe_wavelength_nm, scattering_angle_deg : float, keyword-only
        Probe wavelength and scattering angle used by the forward model.
    ion1_charge, ion1_mass_amu, ion1_number_fraction : float, keyword-only
        Charge, mass, and number fraction of the first ion species.
    ion2_charge, ion2_mass_amu : float, keyword-only
        Charge and mass of the second ion species.  Its number fraction is
        ``1 - ion1_number_fraction``.  The ion species have independent
        fitted temperatures and share the fitted ``u_ik``.

    Returns
    -------
    dict[str, float]
        Best-fit ``T_e_ev``, ``T_i1_ev``, ``T_i2_ev``, ``u_ik_cm_s``,
        ``u_ek_cm_s``, ``A``, ``B1``, and ``B2``.
    """

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    measured = np.asarray(measured_power, dtype=np.float64)
    if wavelength.ndim != 1 or measured.ndim != 1:
        raise ValueError("wavelength_nm and measured_power must be one-dimensional.")
    if wavelength.shape != measured.shape:
        raise ValueError("wavelength_nm and measured_power must have equal length.")
    if wavelength.size < 16:
        raise ValueError("The input spectrum must contain at least 16 points.")
    if not np.isfinite(wavelength).all():
        raise ValueError("wavelength_nm contains non-finite values.")
    if not np.isfinite(measured).all():
        raise ValueError("measured_power contains non-finite values.")
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_nm must contain positive absolute wavelengths.")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be strictly increasing.")

    density = np.asarray(electron_density_cm3, dtype=np.float64)
    fwhm = np.asarray(instrument_fwhm_nm, dtype=np.float64)
    if density.ndim != 0 or not np.isfinite(density) or density <= 0.0:
        raise ValueError("electron_density_cm3 must be a finite positive scalar.")
    if fwhm.ndim != 0 or not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("instrument_fwhm_nm must be a finite positive scalar.")
    density_value = float(density)
    fwhm_value = float(fwhm)

    physical_inputs = {
        "probe_wavelength_nm": probe_wavelength_nm,
        "scattering_angle_deg": scattering_angle_deg,
        "ion1_charge": ion1_charge,
        "ion1_mass_amu": ion1_mass_amu,
        "ion1_number_fraction": ion1_number_fraction,
        "ion2_charge": ion2_charge,
        "ion2_mass_amu": ion2_mass_amu,
    }
    physical_values = {}
    for name, value in physical_inputs.items():
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError("{} must be a finite scalar.".format(name))
        physical_values[name] = float(scalar)
    for name in (
        "probe_wavelength_nm",
        "ion1_charge",
        "ion1_mass_amu",
        "ion2_charge",
        "ion2_mass_amu",
    ):
        if physical_values[name] <= 0.0:
            raise ValueError("{} must be positive.".format(name))
    if not 0.0 < physical_values["scattering_angle_deg"] < 180.0:
        raise ValueError("scattering_angle_deg must lie between 0 and 180 degrees.")
    if not 0.0 < physical_values["ion1_number_fraction"] < 1.0:
        raise ValueError("ion1_number_fraction must lie strictly between 0 and 1.")

    if initial_guess is None:
        guess = 0.5 * (_LOWER_BOUNDS + _UPPER_BOUNDS)
    else:
        guess = np.asarray(initial_guess, dtype=np.float64)
        if guess.shape != (8,):
            raise ValueError("initial_guess must contain exactly eight values.")
        if not np.isfinite(guess).all():
            raise ValueError("initial_guess contains non-finite values.")
        if np.any(guess <= _LOWER_BOUNDS) or np.any(guess >= _UPPER_BOUNDS):
            raise ValueError("initial_guess must lie strictly inside the fit bounds.")

    result = least_squares(
        _residuals,
        guess,
        args=(
            wavelength,
            measured,
            density_value,
            fwhm_value,
            physical_values["probe_wavelength_nm"],
            physical_values["scattering_angle_deg"],
            physical_values["ion1_charge"],
            physical_values["ion1_mass_amu"],
            physical_values["ion1_number_fraction"],
            physical_values["ion2_charge"],
            physical_values["ion2_mass_amu"],
        ),
        method="trf",
        bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS),
        x_scale="jac",
        max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError("Spectrum fitting failed: {}".format(result.message))
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        raise RuntimeError("Spectrum fitting produced non-finite results.")

    return {name: float(value) for name, value in zip(_PARAMETER_NAMES, result.x)}


def _residuals(
    parameters,
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
    probe_wavelength_nm,
    scattering_angle_deg,
    ion1_charge,
    ion1_mass_amu,
    ion1_number_fraction,
    ion2_charge,
    ion2_mass_amu,
) -> np.ndarray:
    """Return the unweighted residual ``P_fit - P_exp``."""

    fitted = _fit_model(
        parameters,
        wavelength_nm,
        electron_density_cm3,
        instrument_fwhm_nm,
        probe_wavelength_nm,
        scattering_angle_deg,
        ion1_charge,
        ion1_mass_amu,
        ion1_number_fraction,
        ion2_charge,
        ion2_mass_amu,
    )
    return fitted - measured_power


def _fit_model(
    parameters,
    wavelength_nm,
    electron_density_cm3,
    instrument_fwhm_nm,
    probe_wavelength_nm=_PROBE_WAVELENGTH_NM,
    scattering_angle_deg=_SCATTERING_ANGLE_DEG,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
) -> np.ndarray:
    """Evaluate the eight-parameter two-ion fitted spectrum."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("parameters must contain eight finite values.")
    (
        te_ev,
        ti1_ev,
        ti2_ev,
        u_ik_cm_s,
        u_ek_cm_s,
        amplitude,
        background_1,
        background_2,
    ) = values

    electron_density_m3 = float(electron_density_cm3) * 1.0e6
    ion2_number_fraction = 1.0 - float(ion1_number_fraction)
    mean_charge = (
        float(ion1_number_fraction) * float(ion1_charge)
        + ion2_number_fraction * float(ion2_charge)
    )
    total_ion_density_m3 = electron_density_m3 / mean_charge
    ion1_density_m3 = float(ion1_number_fraction) * total_ion_density_m3
    ion2_density_m3 = ion2_number_fraction * total_ion_density_m3

    theta = math.radians(float(scattering_angle_deg))
    incident_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    scattered_direction = np.array(
        [math.cos(theta), math.sin(theta), 0.0], dtype=np.float64
    )
    incident_polarization = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nominal_k = scattered_direction - incident_direction
    nominal_k /= np.linalg.norm(nominal_k)
    ion_drift_m_s = float(u_ik_cm_s) * 1.0e-2 * nominal_k
    electron_drift_m_s = float(u_ek_cm_s) * 1.0e-2 * nominal_k

    wavelengths_m = torch.as_tensor(
        np.asarray(wavelength_nm, dtype=np.float64) * 1.0e-9,
        dtype=torch.float64,
    )
    signal_tensor = OTS_spectrum_maxwellian_two_ion(
        wavelengths_m,
        scattered_direction,
        incident_direction,
        incident_polarization,
        float(probe_wavelength_nm) * 1.0e-9,
        electron_density_m3,
        te_ev,
        float(ion1_charge),
        ion1_density_m3,
        float(ion1_mass_amu) * _ATOMIC_MASS_UNIT_KG,
        ti1_ev,
        float(ion2_charge),
        ion2_density_m3,
        float(ion2_mass_amu) * _ATOMIC_MASS_UNIT_KG,
        ti2_ev,
        1.0,
        u_e=electron_drift_m_s,
        u_i1=ion_drift_m_s,
        u_i2=ion_drift_m_s,
    )
    signal = signal_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    signal = _gaussian_convolve(wavelength_nm, signal, instrument_fwhm_nm)
    scale = np.max(np.abs(signal))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The instrument-broadened signal has invalid normalization.")
    signal = signal / scale

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    fraction = (wavelength - wavelength[0]) / (wavelength[-1] - wavelength[0])
    background = background_1 + (background_2 - background_1) * fraction
    fitted = amplitude * signal + background
    if not np.isfinite(fitted).all():
        raise ValueError("The fitted spectrum contains non-finite values.")
    return fitted


def _gaussian_convolve(wavelength_nm, signal, fwhm_nm) -> np.ndarray:
    """Apply a row-normalized Gaussian line-spread function."""
    if fwhm_nm <= np.finfo(np.float64).tiny:
        return signal
    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    sigma_nm = float(fwhm_nm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    separation = wavelength[:, None] - wavelength[None, :]
    kernel = np.exp(-0.5 * (separation / sigma_nm) ** 2)
    row_sum = kernel.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sum).all() or np.any(row_sum <= 0.0):
        raise ValueError("The Gaussian instrument kernel is invalid.")
    broadened = (kernel / row_sum) @ values
    if not np.isfinite(broadened).all():
        raise ValueError("The instrument-broadened signal contains non-finite values.")
    return broadened


def plot_fit_result(
    wavelength_nm,
    measured_power,
    fitted_result,
    electron_density_cm3,
    instrument_fwhm_nm,
    *,
    true_result=None,
    title=None,
    output_path=None,
    contour_points=21,
    probe_wavelength_nm=_PROBE_WAVELENGTH_NM,
    scattering_angle_deg=_SCATTERING_ANGLE_DEG,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
):
    """Plot the spectrum fit and paper-style temperature contours.

    The two contour panels show ``Delta R = R - R_min`` in the
    ``T_e-T_i1`` and ``T_e-T_i2`` planes, where
    ``R = sqrt(mean((P_fit-P_measured)**2)) / A_fit``.  Parameters outside
    the displayed pair are fixed at their best-fit values.  The optional
    true result is used only to mark the synthetic truth on the plots.
    """

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    measured = np.asarray(measured_power, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or measured.ndim != 1
        or wavelength.shape != measured.shape
    ):
        raise ValueError("wavelength_nm and measured_power must be equal-length 1D arrays.")
    if not np.isfinite(wavelength).all() or not np.isfinite(measured).all():
        raise ValueError("Plot inputs must contain only finite values.")

    try:
        fitted_parameters = np.array(
            [fitted_result[name] for name in _PARAMETER_NAMES], dtype=np.float64
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("fitted_result must contain all eight fitted parameters.") from exc
    if not np.isfinite(fitted_parameters).all():
        raise ValueError("fitted_result contains non-finite values.")

    true_parameters = None
    if true_result is not None:
        try:
            true_parameters = np.array(
                [true_result[name] for name in _PARAMETER_NAMES], dtype=np.float64
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("true_result must contain all eight fitted parameters.") from exc
        if not np.isfinite(true_parameters).all():
            raise ValueError("true_result contains non-finite values.")

    if isinstance(contour_points, bool) or not isinstance(
        contour_points, (int, np.integer)
    ):
        raise ValueError("contour_points must be an integer.")
    if contour_points < 11:
        raise ValueError("contour_points must be at least 11.")

    amplitude_fit = fitted_parameters[5]
    if amplitude_fit <= 0.0:
        raise ValueError("The fitted amplitude must be positive to calculate R.")
    fitted_power = _fit_model(
        fitted_parameters,
        wavelength,
        electron_density_cm3,
        instrument_fwhm_nm,
        probe_wavelength_nm,
        scattering_angle_deg,
        ion1_charge,
        ion1_mass_amu,
        ion1_number_fraction,
        ion2_charge,
        ion2_mass_amu,
    )
    r_min = np.sqrt(np.mean((fitted_power - measured) ** 2)) / amplitude_fit

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    wavelength_shift = wavelength - float(probe_wavelength_nm)
    axes[0].plot(
        wavelength_shift,
        measured,
        color="#000000",
        linewidth=1.2,
        label="True spectrum",
    )
    axes[0].plot(
        wavelength_shift,
        fitted_power,
        color="#1565c0",
        linestyle="--",
        linewidth=2.0,
        label="Best fit",
    )
    
    
    axes[0].set_xlabel(r"Wavelength shift $\Delta\lambda$ [nm]")
    axes[0].set_ylabel("Relative spectral power")
    axes[0].set_title("Spectrum")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    te_fit = fitted_parameters[0]
    te_values = np.linspace(
        max(_LOWER_BOUNDS[0], 0.7 * te_fit),
        min(_UPPER_BOUNDS[0], 1.3 * te_fit),
        contour_points,
    )
    contour_pairs = ((1, r"$T_{i1}$ [eV]"), (2, r"$T_{i2}$ [eV]"))
    for axis, (temperature_index, y_label) in zip(axes[1:], contour_pairs):
        ti_fit = fitted_parameters[temperature_index]
        ti_values = np.linspace(
            max(_LOWER_BOUNDS[temperature_index], 0.5 * ti_fit),
            min(_UPPER_BOUNDS[temperature_index], 1.5 * ti_fit),
            contour_points,
        )
        delta_r = np.empty((contour_points, contour_points), dtype=np.float64)
        trial = fitted_parameters.copy()
        for row, ti_value in enumerate(ti_values):
            trial[temperature_index] = ti_value
            for column, te_value in enumerate(te_values):
                trial[0] = te_value
                trial_power = _fit_model(
                    trial,
                    wavelength,
                    electron_density_cm3,
                    instrument_fwhm_nm,
                    probe_wavelength_nm,
                    scattering_angle_deg,
                    ion1_charge,
                    ion1_mass_amu,
                    ion1_number_fraction,
                    ion2_charge,
                    ion2_mass_amu,
                )
                r_value = np.sqrt(np.mean((trial_power - measured) ** 2)) / amplitude_fit
                delta_r[row, column] = max(0.0, r_value - r_min)

        contours = axis.contour(
                te_values,
                ti_values,
                delta_r,
                cmap="viridis",
        )
        axis.clabel(contours, inline=True, fontsize=8, fmt="%.3g")
        axis.plot(
            te_fit,
            ti_fit,
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.5,
            linestyle="none",
            label="Best fit",
        )
        if true_parameters is not None:
            axis.plot(
                true_parameters[0],
                true_parameters[temperature_index],
                marker="x",
                markersize=7,
                color="black",
                linestyle="none",
                label="True",
            )
        axis.set_xlabel(r"$T_e$ [eV]")
        axis.set_ylabel(y_label)
        axis.set_title(r"$\Delta R = R-R_{min}$")
        axis.text(
            0.97,
            0.96,
            r"$R_{min}$=" + "{:.3g}".format(r_min),
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
        )
        axis.legend(frameon=False, loc="lower right")

    if title:
        figure.suptitle(str(title))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_ion_composition_spectra(
    wavelength_nm,
    fitted_result,
    electron_density_cm3,
    *,
    ion1_label,
    ion2_label,
    title=None,
    output_path=None,
    probe_wavelength_nm=_PROBE_WAVELENGTH_NM,
    scattering_angle_deg=_SCATTERING_ANGLE_DEG,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
):
    """Plot pure-ion, average-ion, and two-ion theoretical OTS spectra."""

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("wavelength_nm must be a non-empty one-dimensional array.")
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_nm must contain finite positive values.")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be strictly increasing.")

    try:
        fitted_parameters = np.array(
            [fitted_result[name] for name in _PARAMETER_NAMES], dtype=np.float64
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("fitted_result must contain all eight fitted parameters.") from exc
    if not np.isfinite(fitted_parameters).all():
        raise ValueError("fitted_result contains non-finite values.")

    if not isinstance(ion1_label, str) or not ion1_label.strip():
        raise ValueError("ion1_label must be a non-empty string.")
    if not isinstance(ion2_label, str) or not ion2_label.strip():
        raise ValueError("ion2_label must be a non-empty string.")
    ion1_label = ion1_label.strip()
    ion2_label = ion2_label.strip()

    physical_inputs = {
        "electron_density_cm3": electron_density_cm3,
        "probe_wavelength_nm": probe_wavelength_nm,
        "scattering_angle_deg": scattering_angle_deg,
        "ion1_charge": ion1_charge,
        "ion1_mass_amu": ion1_mass_amu,
        "ion1_number_fraction": ion1_number_fraction,
        "ion2_charge": ion2_charge,
        "ion2_mass_amu": ion2_mass_amu,
    }
    physical_values = {}
    for name, value in physical_inputs.items():
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError("{} must be a finite scalar.".format(name))
        physical_values[name] = float(scalar)
    for name in (
        "electron_density_cm3",
        "probe_wavelength_nm",
        "ion1_charge",
        "ion1_mass_amu",
        "ion2_charge",
        "ion2_mass_amu",
    ):
        if physical_values[name] <= 0.0:
            raise ValueError("{} must be positive.".format(name))
    if not 0.0 < physical_values["scattering_angle_deg"] < 180.0:
        raise ValueError("scattering_angle_deg must lie between 0 and 180 degrees.")
    if not 0.0 < physical_values["ion1_number_fraction"] < 1.0:
        raise ValueError("ion1_number_fraction must lie strictly between 0 and 1.")

    (
        te_ev,
        ti1_ev,
        ti2_ev,
        u_ik_cm_s,
        u_ek_cm_s,
        _,
        _,
        _,
    ) = fitted_parameters
    electron_density_m3 = physical_values["electron_density_cm3"] * 1.0e6
    ion1_fraction = physical_values["ion1_number_fraction"]
    ion2_fraction = 1.0 - ion1_fraction
    average_charge = (
        ion1_fraction * physical_values["ion1_charge"]
        + ion2_fraction * physical_values["ion2_charge"]
    )
    average_mass_amu = (
        ion1_fraction * physical_values["ion1_mass_amu"]
        + ion2_fraction * physical_values["ion2_mass_amu"]
    )
    average_temperature_ev = ion1_fraction * ti1_ev + ion2_fraction * ti2_ev

    theta = math.radians(physical_values["scattering_angle_deg"])
    incident_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    scattered_direction = np.array(
        [math.cos(theta), math.sin(theta), 0.0], dtype=np.float64
    )
    incident_polarization = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nominal_k = scattered_direction - incident_direction
    nominal_k /= np.linalg.norm(nominal_k)
    ion_drift_m_s = u_ik_cm_s * 1.0e-2 * nominal_k
    electron_drift_m_s = u_ek_cm_s * 1.0e-2 * nominal_k
    wavelengths_m = torch.as_tensor(wavelength * 1.0e-9, dtype=torch.float64)

    mixed_total_ion_density = electron_density_m3 / average_charge
    spectra_inputs = (
        (
            "{} = 1".format(ion1_label),
            physical_values["ion1_charge"],
            electron_density_m3 / physical_values["ion1_charge"],
            physical_values["ion1_mass_amu"],
            ti1_ev,
            physical_values["ion2_charge"],
            0.0,
            physical_values["ion2_mass_amu"],
            ti2_ev,
        ),
        (
            "{} = 1".format(ion2_label),
            physical_values["ion1_charge"],
            0.0,
            physical_values["ion1_mass_amu"],
            ti1_ev,
            physical_values["ion2_charge"],
            electron_density_m3 / physical_values["ion2_charge"],
            physical_values["ion2_mass_amu"],
            ti2_ev,
        ),
        (
            "Average ion: Z = {:.3g}".format(average_charge),
            average_charge,
            electron_density_m3 / (2.0 * average_charge),
            average_mass_amu,
            average_temperature_ev,
            average_charge,
            electron_density_m3 / (2.0 * average_charge),
            average_mass_amu,
            average_temperature_ev,
        ),
        (
            "{}-{}: x_{} = {:.3g}".format(
                ion1_label, ion2_label, ion1_label, ion1_fraction
            ),
            physical_values["ion1_charge"],
            ion1_fraction * mixed_total_ion_density,
            physical_values["ion1_mass_amu"],
            ti1_ev,
            physical_values["ion2_charge"],
            ion2_fraction * mixed_total_ion_density,
            physical_values["ion2_mass_amu"],
            ti2_ev,
        ),
    )

    spectra = []
    scales= []
    for (
        panel_title,
        charge_1,
        density_1,
        mass_1_amu,
        temperature_1_ev,
        charge_2,
        density_2,
        mass_2_amu,
        temperature_2_ev,
    ) in spectra_inputs:
        signal_tensor,original_scale = OTS_spectrum_maxwellian_two_ion(
            wavelengths_m,
            scattered_direction,
            incident_direction,
            incident_polarization,
            physical_values["probe_wavelength_nm"] * 1.0e-9,
            electron_density_m3,
            te_ev,
            charge_1,
            density_1,
            mass_1_amu * _ATOMIC_MASS_UNIT_KG,
            temperature_1_ev,
            charge_2,
            density_2,
            mass_2_amu * _ATOMIC_MASS_UNIT_KG,
            temperature_2_ev,
            1.0,
            u_e=electron_drift_m_s,
            u_i1=ion_drift_m_s,
            u_i2=ion_drift_m_s,
            switch_scale_output=True
        )
        signal = signal_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        original_scale = float(original_scale.detach().cpu().numpy())
        if signal.shape != wavelength.shape or not np.isfinite(signal).all():
            raise ValueError("The theoretical composition spectrum is invalid.")
        spectra.append((panel_title, signal))
        scales.append(original_scale)

    wavelength_shift = wavelength - physical_values["probe_wavelength_nm"]
    figure, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True, constrained_layout=True
    )
    colors = ("#1565c0", "#ef6c00", "#2e7d32", "#8e24aa")
    composition_scale=scales[-1]
    for axis, (panel_title, signal), scale, color in zip(axes.flat, spectra, scales, colors):
        axis.plot(wavelength_shift, signal*scale/composition_scale, color=color, linewidth=2.0)
        axis.set_title(panel_title)
        axis.set_xlabel(r"Wavelength shift $\Delta\lambda$ [nm]")
        axis.set_ylabel("OTS power")
        axis.grid(alpha=0.2)
    if title:
        figure.suptitle(str(title))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def main() -> int:
    """Verify forward/inverse consistency for the report's two-ion plasmas."""

    probe_wavelength_nm = 351.0
    probe_wavelength_m = probe_wavelength_nm * 1.0e-9
    probe_omega = 2.0 * math.pi * SPEED_OF_LIGHT / probe_wavelength_m
    critical_density_m3 = (
        VACUUM_PERMITTIVITY
        * ELECTRON_MASS
        * probe_omega**2
        / ELEMENTARY_CHARGE**2
    )
    electron_density_cm3 = 0.3 * critical_density_m3 / 1.0e6
    instrument_fwhm_nm = 0.065
    wavelength_nm = np.linspace(
        probe_wavelength_nm - 3.0,
        probe_wavelength_nm + 3.0,
        241,
        dtype=np.float64,
    )

    states = []
    for hydrogen_fraction in (0.3, 0.5, 0.7):
        states.append(
            (
                "H-He x_H={:.2f}".format(hydrogen_fraction),
                "H",
                "He",
                2000.0,
                1000.0,
                1.0,
                1.0,
                hydrogen_fraction,
                2.0,
                4.0,
            )
        )
    for hydrogen_fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        states.append(
            (
                "H-C x_H={:.2f}".format(hydrogen_fraction),
                "H",
                "C",
                4800.0,
                600.0,
                1.0,
                1.0,
                hydrogen_fraction,
                6.0,
                12.0,
            )
        )
    states.append(
        ("Au-B x_Au=0.6", "Au", "B", 3000.0, 1000.0, 50.0, 197.0, 0.6, 5.0, 11.0)
    )

    true_nuisance = np.array([0.0, 0.0, 0.7, 0.10, 0.12], dtype=np.float64)
    print(
        "Two-ion OTS forward/inverse check: lambda0={:.1f} nm, ne=0.3 nc="
        "{:.6e} cm^-3".format(probe_wavelength_nm, electron_density_cm3)
    )
    print(
        "state                 Te_fit/true   Ti1_fit/true  Ti2_fit/true  "
        "relative_spectrum_L2"
    )
    plot_directory = Path(__file__).resolve().parent / "spectral_fitting_plots2"
    composition_plot_directory = (
        Path(__file__).resolve().parent / "spectral_fitting_composition_plots"
    )

    for state_index, (
        name,
        ion1_label,
        ion2_label,
        te_ev,
        ti_ev,
        ion1_charge,
        ion1_mass_amu,
        ion1_fraction,
        ion2_charge,
        ion2_mass_amu,
    ) in enumerate(states, start=1):
        ti1_ev = 0.8 * ti_ev
        ti2_ev = 1.2 * ti_ev
        true_parameters = np.concatenate(
            (np.array([te_ev, ti1_ev, ti2_ev], dtype=np.float64), true_nuisance)
        )
        measured_power = _fit_model(
            true_parameters,
            wavelength_nm,
            electron_density_cm3,
            instrument_fwhm_nm,
            probe_wavelength_nm,
            _SCATTERING_ANGLE_DEG,
            ion1_charge,
            ion1_mass_amu,
            ion1_fraction,
            ion2_charge,
            ion2_mass_amu,
        )
        initial_guess = np.array(
            [
                0.85 * te_ev,
                1.15 * ti1_ev,
                0.85 * ti2_ev,
                1.0e6,
                -5.0e6,
                0.62,
                0.12,
                0.10,
            ],
            dtype=np.float64,
        )
        fitted = fit_spectrum(
            wavelength_nm,
            measured_power,
            electron_density_cm3,
            instrument_fwhm_nm,
            initial_guess,
            probe_wavelength_nm=probe_wavelength_nm,
            scattering_angle_deg=_SCATTERING_ANGLE_DEG,
            ion1_charge=ion1_charge,
            ion1_mass_amu=ion1_mass_amu,
            ion1_number_fraction=ion1_fraction,
            ion2_charge=ion2_charge,
            ion2_mass_amu=ion2_mass_amu,
        )
        fitted_parameters = np.array(
            [fitted[key] for key in _PARAMETER_NAMES], dtype=np.float64
        )
        reconstructed_power = _fit_model(
            fitted_parameters,
            wavelength_nm,
            electron_density_cm3,
            instrument_fwhm_nm,
            probe_wavelength_nm,
            _SCATTERING_ANGLE_DEG,
            ion1_charge,
            ion1_mass_amu,
            ion1_fraction,
            ion2_charge,
            ion2_mass_amu,
        )
        relative_l2 = np.linalg.norm(reconstructed_power - measured_power) / max(
            np.linalg.norm(measured_power), np.finfo(np.float64).tiny
        )
        te_relative_error = abs(fitted["T_e_ev"] - te_ev) / te_ev
        ti1_relative_error = abs(fitted["T_i1_ev"] - ti1_ev) / ti1_ev
        ti2_relative_error = abs(fitted["T_i2_ev"] - ti2_ev) / ti2_ev
        velocity_absolute_error = np.max(
            np.abs(fitted_parameters[3:5] - true_parameters[3:5])
        )
        amplitude_background_error = np.max(
            np.abs(fitted_parameters[5:8] - true_parameters[5:8])
        )
        if max(te_relative_error, ti1_relative_error, ti2_relative_error) > 1.0e-4:
            raise RuntimeError("{} temperature inversion is not self-consistent.".format(name))
        if velocity_absolute_error > 1.0e3:
            raise RuntimeError("{} velocity inversion is not self-consistent.".format(name))
        if amplitude_background_error > 1.0e-5:
            raise RuntimeError(
                "{} amplitude/background inversion is not self-consistent.".format(name)
            )
        if relative_l2 > 1.0e-6:
            raise RuntimeError("{} spectrum inversion is not self-consistent.".format(name))
        true_result = {
            key: float(value) for key, value in zip(_PARAMETER_NAMES, true_parameters)
        }
        safe_name = name.replace(" ", "_").replace("=", "_").replace(".", "p")
        plot_path = plot_directory / "{:02d}_{}.png".format(state_index, safe_name)
        figure = plot_fit_result(
            wavelength_nm,
            measured_power,
            fitted,
            electron_density_cm3,
            instrument_fwhm_nm,
            true_result=true_result,
            title=name,
            output_path=plot_path,
            probe_wavelength_nm=probe_wavelength_nm,
            scattering_angle_deg=_SCATTERING_ANGLE_DEG,
            ion1_charge=ion1_charge,
            ion1_mass_amu=ion1_mass_amu,
            ion1_number_fraction=ion1_fraction,
            ion2_charge=ion2_charge,
            ion2_mass_amu=ion2_mass_amu,
        )
        plt.close(figure)
        composition_plot_path = composition_plot_directory / "{:02d}_{}.png".format(
            state_index, safe_name
        )
        composition_figure = plot_ion_composition_spectra(
            wavelength_nm,
            fitted,
            electron_density_cm3,
            ion1_label=ion1_label,
            ion2_label=ion2_label,
            title=name,
            output_path=composition_plot_path,
            probe_wavelength_nm=probe_wavelength_nm,
            scattering_angle_deg=_SCATTERING_ANGLE_DEG,
            ion1_charge=ion1_charge,
            ion1_mass_amu=ion1_mass_amu,
            ion1_number_fraction=ion1_fraction,
            ion2_charge=ion2_charge,
            ion2_mass_amu=ion2_mass_amu,
        )
        plt.close(composition_figure)
        print(
            "{:<21} {:9.3f}/{:<7.1f} {:9.3f}/{:<7.1f} "
            "{:9.3f}/{:<7.1f} {:.3e}".format(
                name,
                fitted["T_e_ev"],
                te_ev,
                fitted["T_i1_ev"],
                ti1_ev,
                fitted["T_i2_ev"],
                ti2_ev,
                relative_l2,
            )
        )

    print("All {} two-ion states passed.".format(len(states)))
    print("Plots saved to: {}".format(plot_directory))
    print("Composition plots saved to: {}".format(composition_plot_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
