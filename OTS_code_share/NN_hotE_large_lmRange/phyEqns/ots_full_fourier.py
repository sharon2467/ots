# -*- coding: utf-8 -*-
"""OTS forward model with Fourier velocity distributions for every species.

This module extends :mod:`phyEqns.ots_fourier`.  The electron, ion-1, and
ion-2 one-dimensional velocity distributions are all supplied as normalized
Fourier transforms,

    F_s(K) = integral f_s(v) exp(-i K v) dv,

and are evaluated with

    f_s(v) = (1 / 2 pi) integral F_s(K) exp(i K v) dK.

The susceptibility of each species uses the same positive-K Landau response,

    chi_s = (omega_ps / k)^2
            integral_0^infinity K F_s(K) exp(i K v_s) dK.

Consequently the forward-model interface contains no electron or ion
temperature.  Temperatures may still be used upstream to construct a fixed
Maxwellian distribution, but the OTS calculation only receives its Fourier
representation.

For a uniformly sampled K grid, ``Delta K`` also fixes the non-aliased
velocity interval ``|v| <= pi / Delta K``.  Values outside that interval are
set to zero instead of allowing the discrete representation to repeat
periodically into another species' spectral feature.
"""

from __future__ import division

import math

import torch


ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 2.99792458e8
ELECTRON_MASS = 9.1093837015e-31
VACUUM_PERMITTIVITY = 8.8541878128e-12


def _real_dtype(tensor):
    if tensor.dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def _prepare_fourier_pair(k_grid, coefficients, device, dtype, name):
    """Validate and move one Fourier grid/coefficient pair."""

    k_grid = torch.as_tensor(k_grid, device=device, dtype=dtype)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    coefficients = torch.as_tensor(coefficients, device=device).to(complex_dtype)

    if k_grid.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("{} K/F arrays must both be one-dimensional.".format(name))
    if len(k_grid) < 3 or len(k_grid) != len(coefficients):
        raise ValueError(
            "{} K/F arrays must have the same length and at least three modes.".format(
                name
            )
        )
    if not torch.isfinite(k_grid).all():
        raise ValueError("{} K grid contains non-finite values.".format(name))
    if not torch.isfinite(coefficients.real).all() or not torch.isfinite(
        coefficients.imag
    ).all():
        raise ValueError("{} Fourier coefficients contain non-finite values.".format(name))
    if not torch.all(k_grid[1:] > k_grid[:-1]):
        raise ValueError("{} K grid must be strictly increasing.".format(name))
    if not bool((k_grid[0] < 0).item()) or not bool((k_grid[-1] > 0).item()):
        raise ValueError("{} K grid must span negative and positive modes.".format(name))
    return k_grid, coefficients


def reciprocal_velocity_support(k_grid):
    """Return the non-aliased velocity half-width implied by a uniform K grid."""

    spacing = k_grid[1:] - k_grid[:-1]
    mean_spacing = spacing.mean()
    tolerance = torch.abs(mean_spacing) * 1.0e-4 + 1.0e-14
    if not torch.all(torch.abs(spacing - mean_spacing) <= tolerance):
        raise ValueError(
            "Fourier K grid must be uniform to define its valid velocity domain."
        )
    return math.pi / mean_spacing


def _zero_outside_reciprocal_domain(values, k_grid, velocity):
    """Suppress periodic aliases outside the velocity domain encoded by ΔK."""

    support = reciprocal_velocity_support(k_grid)
    mask = torch.abs(velocity) <= support * (1.0 + 1.0e-7)
    return torch.where(mask, values, torch.zeros_like(values))


def inverse_fourier_at(k_grid, coefficients, velocity):
    """Evaluate a Fourier VDF inside the non-aliased domain implied by ΔK."""

    velocity = torch.as_tensor(
        velocity, device=k_grid.device, dtype=_real_dtype(k_grid)
    )
    phase = k_grid.reshape(-1, 1) * velocity.reshape(1, -1)
    integrand = coefficients.reshape(-1, 1) * torch.exp(1j * phase)
    values = torch.trapz(integrand, k_grid, dim=0) / (2.0 * math.pi)
    values = values.reshape(velocity.shape)
    return _zero_outside_reciprocal_domain(values, k_grid, velocity)


def fourier_landau_response(k_grid, coefficients, velocity):
    """Evaluate the positive-K response inside the K grid's velocity domain."""

    velocity = torch.as_tensor(
        velocity, device=k_grid.device, dtype=_real_dtype(k_grid)
    )
    positive_k = torch.clamp(k_grid, min=0.0)
    phase = k_grid.reshape(-1, 1) * velocity.reshape(1, -1)
    integrand = (
        positive_k.reshape(-1, 1)
        * coefficients.reshape(-1, 1)
        * torch.exp(1j * phase)
    )
    response = torch.trapz(integrand, k_grid, dim=0)
    response = response.reshape(velocity.shape)
    return _zero_outside_reciprocal_domain(response, k_grid, velocity)


def _projected_phase_velocity(omega_shift, k_magnitude, k_components, drift):
    drift_projection = (
        k_components[0] * drift[0]
        + k_components[1] * drift[1]
        + k_components[2] * drift[2]
    ) / k_magnitude
    return omega_shift / k_magnitude - drift_projection


def OTS_spectrum_full_fourier(
    lms,
    ss,
    k_sh,
    E_i0,
    lm_0,
    n_e,
    Z_i1,
    N_i1,
    m_i1,
    Z_i2,
    N_i2,
    m_i2,
    V,
    K_e,
    F_e,
    K_i1,
    F_i1,
    K_i2,
    F_i2,
    u_e,
    u_i1,
    u_i2,
):
    """Calculate a normalized OTS spectrum using three arbitrary VDFs.

    ``F_e``, ``F_i1``, and ``F_i2`` must describe distributions normalized to
    unit velocity integral.  ``N_i1`` and ``N_i2`` are ion number fractions.
    Quasi-neutrality gives

        n_i,j = n_e * N_i,j / (Z_i1*N_i1 + Z_i2*N_i2).

    The distributions are interpreted in their species rest frames; the
    optional three-component drift vectors shift the phase velocities.
    ``E_i0`` and ``V`` are retained for interface compatibility.  As in the
    existing training pipeline, the absolute Thomson prefactor is omitted
    because every wavelength segment is normalized independently.
    """

    lms = torch.as_tensor(lms)
    if not lms.dtype.is_floating_point:
        lms = lms.to(torch.float64)
    dtype = _real_dtype(lms)
    device = lms.device
    lms = lms.to(dtype=dtype)

    if lms.ndim != 1 or len(lms) == 0:
        raise ValueError("lms must be a non-empty one-dimensional wavelength grid.")
    if not torch.isfinite(lms).all() or not torch.all(lms > 0):
        raise ValueError("lms contains invalid wavelengths.")

    ss = torch.as_tensor(ss, device=device, dtype=dtype)
    k_sh = torch.as_tensor(k_sh, device=device, dtype=dtype)
    u_e = torch.as_tensor(u_e, device=device, dtype=dtype)
    u_i1 = torch.as_tensor(u_i1, device=device, dtype=dtype)
    u_i2 = torch.as_tensor(u_i2, device=device, dtype=dtype)
    for vector, name in (
        (ss, "ss"),
        (k_sh, "k_sh"),
        (u_e, "u_e"),
        (u_i1, "u_i1"),
        (u_i2, "u_i2"),
    ):
        if vector.shape != (3,) or not torch.isfinite(vector).all():
            raise ValueError("{} must be a finite three-component vector.".format(name))

    K_e, F_e = _prepare_fourier_pair(K_e, F_e, device, dtype, "electron")
    K_i1, F_i1 = _prepare_fourier_pair(K_i1, F_i1, device, dtype, "ion-1")
    K_i2, F_i2 = _prepare_fourier_pair(K_i2, F_i2, device, dtype, "ion-2")

    n_e = torch.as_tensor(n_e, device=device, dtype=dtype)
    lm_0 = torch.as_tensor(lm_0, device=device, dtype=dtype)
    Z_i1 = torch.as_tensor(Z_i1, device=device, dtype=dtype)
    N_i1 = torch.as_tensor(N_i1, device=device, dtype=dtype)
    m_i1 = torch.as_tensor(m_i1, device=device, dtype=dtype)
    Z_i2 = torch.as_tensor(Z_i2, device=device, dtype=dtype)
    N_i2 = torch.as_tensor(N_i2, device=device, dtype=dtype)
    m_i2 = torch.as_tensor(m_i2, device=device, dtype=dtype)

    scalar_values = (
        (n_e, "n_e"),
        (lm_0, "lm_0"),
        (Z_i1, "Z_i1"),
        (N_i1, "N_i1"),
        (m_i1, "m_i1"),
        (Z_i2, "Z_i2"),
        (N_i2, "N_i2"),
        (m_i2, "m_i2"),
    )
    for value, name in scalar_values:
        if value.numel() != 1 or not torch.isfinite(value):
            raise ValueError("{} must be a finite scalar.".format(name))
    if not bool((n_e > 0).item()) or not bool((lm_0 > 0).item()):
        raise ValueError("n_e and lm_0 must be positive.")
    if not bool((m_i1 > 0).item()) or not bool((m_i2 > 0).item()):
        raise ValueError("Ion masses must be positive.")
    if not bool((N_i1 >= 0).item()) or not bool((N_i2 >= 0).item()):
        raise ValueError("Ion fractions must be non-negative.")

    charge_weight = Z_i1 * N_i1 + Z_i2 * N_i2
    if not bool((charge_weight > 0).item()):
        raise ValueError("The charge-weighted ion fraction must be positive.")

    ss_norm = torch.linalg.vector_norm(ss)
    k_sh_norm = torch.linalg.vector_norm(k_sh)
    if not bool((ss_norm > 0).item()) or not bool((k_sh_norm > 0).item()):
        raise ValueError("Scattering and incident directions must be non-zero.")
    ss = ss / ss_norm
    k_sh = k_sh / k_sh_norm

    omega_0 = 2.0 * math.pi * SPEED_OF_LIGHT / lm_0
    omega_pe_squared = (
        n_e
        * ELEMENTARY_CHARGE**2
        / (ELECTRON_MASS * VACUUM_PERMITTIVITY)
    )
    incident_argument = omega_0.square() - omega_pe_squared
    if not bool((incident_argument > 0).item()):
        raise ValueError("Probe frequency must exceed the electron plasma frequency.")
    k_0 = torch.sqrt(incident_argument) / SPEED_OF_LIGHT * k_sh

    omega = 2.0 * math.pi * SPEED_OF_LIGHT / lms
    k_components = (
        k_0[0] - ss[0] * omega / SPEED_OF_LIGHT,
        k_0[1] - ss[1] * omega / SPEED_OF_LIGHT,
        k_0[2] - ss[2] * omega / SPEED_OF_LIGHT,
    )
    k_magnitude = torch.sqrt(
        k_components[0].square()
        + k_components[1].square()
        + k_components[2].square()
    )
    omega_shift = omega - omega_0

    v_e = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_e
    )
    v_i1 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i1
    )
    v_i2 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i2
    )

    f_e = inverse_fourier_at(K_e, F_e, v_e)
    f_i1 = inverse_fourier_at(K_i1, F_i1, v_i1)
    f_i2 = inverse_fourier_at(K_i2, F_i2, v_i2)
    response_e = fourier_landau_response(K_e, F_e, v_e)
    response_i1 = fourier_landau_response(K_i1, F_i1, v_i1)
    response_i2 = fourier_landau_response(K_i2, F_i2, v_i2)

    ion_1_density = n_e * N_i1 / charge_weight
    ion_2_density = n_e * N_i2 / charge_weight
    omega_pi1_squared = (
        ion_1_density
        * (Z_i1 * ELEMENTARY_CHARGE).square()
        / (m_i1 * VACUUM_PERMITTIVITY)
    )
    omega_pi2_squared = (
        ion_2_density
        * (Z_i2 * ELEMENTARY_CHARGE).square()
        / (m_i2 * VACUUM_PERMITTIVITY)
    )

    chi_e = omega_pe_squared / k_magnitude.square() * response_e
    chi_i1 = omega_pi1_squared / k_magnitude.square() * response_i1
    chi_i2 = omega_pi2_squared / k_magnitude.square() * response_i2
    dielectric = 1.0 + chi_e + chi_i1 + chi_i2

    electron_screening = torch.abs(1.0 - chi_e / dielectric).square()
    ion_screening = torch.abs(chi_e / dielectric).square()
    ion_1_weight = Z_i1.square() * N_i1 / charge_weight
    ion_2_weight = Z_i2.square() * N_i2 / charge_weight
    structure_factor = 2.0 * math.pi / k_magnitude * (
        electron_screening * f_e
        + ion_screening * ion_1_weight * f_i1
        + ion_screening * ion_2_weight * f_i2
    )

    power = torch.real((1.0 + 2.0 * omega_shift / omega_0) * structure_factor)
    scale = torch.max(torch.abs(power))
    if not torch.isfinite(scale) or not bool((scale > 0).item()):
        raise ValueError("OTS spectrum has an invalid normalization scale.")
    intensity = power / scale
    if not torch.isfinite(intensity).all():
        raise ValueError("OTS spectrum contains non-finite values.")

    # Retained only to make intentional compatibility explicit.
    _ = E_i0, V
    return intensity


# The familiar name is available when callers import it from this module.
OTS_spectrum_fft = OTS_spectrum_full_fourier
