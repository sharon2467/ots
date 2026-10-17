# -*- coding: utf-8 -*-
"""OTS forward model with explicit velocity distributions for all species.

Each electron or ion distribution is supplied on its own uniformly sampled
one-dimensional velocity grid.  The forward model validates the normalized
distributions, constructs the requested low-pass Fourier representation, and
uses the original velocity grids for phase-velocity interpolation.  Values
outside a distribution's supplied velocity interval are zero; they are never
periodically wrapped or extrapolated.

The Fourier convention is

    F_s(K) = integral f_s(v) exp(-i K v) dv.

The corresponding positive-K Landau response is used for every species.
"""

from __future__ import division

import math
from numbers import Integral

import numpy as np
import torch


ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 2.99792458e8
ELECTRON_MASS = 9.1093837015e-31
VACUUM_PERMITTIVITY = 8.8541878128e-12


_EQ9_GAUSS_NODES, _EQ9_GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)
_EQ9_GAUSS_NODES = 0.5 * (_EQ9_GAUSS_NODES + 1.0)
_EQ9_GAUSS_WEIGHTS = 0.5 * _EQ9_GAUSS_WEIGHTS
_EQ9_ASYMPTOTIC_THRESHOLD = 8.0
_EQ9_ASYMPTOTIC_TERMS = 10


def _real_dtype(tensor):
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _as_real_tensor(value, device, dtype, name):
    tensor = torch.as_tensor(value, device=device)
    if torch.is_complex(tensor):
        raise ValueError("{} must be real-valued.".format(name))
    return tensor.to(dtype=dtype)


def _validate_grid_and_values(v, f, name):
    """Validate a real finite distribution grid without requiring unit mass."""

    v_tensor = torch.as_tensor(v)
    if torch.is_complex(v_tensor):
        raise ValueError("{} velocity grid must be real-valued.".format(name))
    if v_tensor.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32
    v_tensor = v_tensor.to(dtype=dtype)

    f_tensor = torch.as_tensor(f, device=v_tensor.device)
    if torch.is_complex(f_tensor):
        raise ValueError("{} distribution must be real-valued.".format(name))
    f_tensor = f_tensor.to(dtype=dtype)

    if v_tensor.ndim != 1 or f_tensor.ndim != 1:
        raise ValueError(
            "{} velocity and distribution arrays must be one-dimensional.".format(name)
        )
    if len(v_tensor) != len(f_tensor) or len(v_tensor) < 3:
        raise ValueError(
            "{} velocity and distribution arrays must have equal length and at "
            "least three points.".format(name)
        )
    if not torch.isfinite(v_tensor).all():
        raise ValueError("{} velocity grid contains non-finite values.".format(name))
    if not torch.isfinite(f_tensor).all():
        raise ValueError("{} distribution contains non-finite values.".format(name))

    spacing = v_tensor[1:] - v_tensor[:-1]
    if not torch.all(spacing > 0):
        raise ValueError("{} velocity grid must be strictly increasing.".format(name))
    mean_spacing = spacing.mean()
    tolerance = torch.abs(mean_spacing) * 1.0e-4
    if not torch.all(torch.abs(spacing - mean_spacing) <= tolerance):
        raise ValueError("{} velocity grid must be approximately uniform.".format(name))
    return v_tensor, f_tensor


def ft_x_n(v, f, n):
    """Return ``n`` low-pass Fourier modes for a sampled real distribution.

    ``v`` and ``f`` must be equal-length, finite one-dimensional arrays on a
    strictly increasing approximately uniform grid.  This transform does not
    require ``f`` to be non-negative or normalized; the OTS public interface
    applies those stronger physical checks separately.
    """

    v, f = _validate_grid_and_values(v, f, "Fourier input")
    sample_count = len(v)
    if (
        not isinstance(n, Integral)
        or isinstance(n, bool)
        or int(n) <= 0
        or int(n) > sample_count
    ):
        raise ValueError(
            "Fourier mode count n={} must be an integer satisfying 0 < n <= N "
            "with N={}.".format(n, sample_count)
        )
    n = int(n)

    dx = v[2] - v[1]
    domain_length = sample_count * dx
    dk = 2.0 * math.pi / domain_length
    k_grid = torch.linspace(-1.0, 1.0, n, device=v.device, dtype=v.dtype) * dk * n / 2.0
    phase = k_grid.reshape(-1, 1) * v.reshape(1, -1)
    integrand = f.reshape(1, -1) * torch.exp(-1j * phase)
    coefficients = torch.trapz(integrand, v, dim=1)
    return k_grid, coefficients


def _validate_distribution(v, f, name):
    """Validate a non-negative, unit-normalized sampled velocity distribution."""

    v, f = _validate_grid_and_values(v, f, name)
    if not torch.all(f >= 0):
        raise ValueError("{} distribution must be non-negative.".format(name))

    integral = torch.trapz(f, v)
    if not torch.isfinite(integral):
        raise ValueError("{} distribution integral is non-finite.".format(name))
    if not torch.isclose(
        integral,
        torch.ones((), device=integral.device, dtype=integral.dtype),
        rtol=1.0e-5,
        atol=1.0e-8,
    ):
        raise ValueError(
            "{} distribution must integrate to one; got {:.9g}.".format(
                name, integral.item()
            )
        )
    return v, f


def interpolate_distribution(v, f, query_velocity):
    """Linearly interpolate a distribution and return zero outside its grid."""

    v, f = _validate_grid_and_values(v, f, "Interpolation input")
    query_tensor = torch.as_tensor(query_velocity, device=v.device)
    if torch.is_complex(query_tensor):
        raise ValueError("query_velocity must be real-valued.")
    query_tensor = query_tensor.to(dtype=v.dtype)
    if not torch.isfinite(query_tensor).all():
        raise ValueError("query_velocity contains non-finite values.")

    original_shape = query_tensor.shape
    query_flat = query_tensor.reshape(-1)
    upper = torch.searchsorted(v, query_flat, right=False).clamp(1, len(v) - 1)
    lower = upper - 1
    v_lower = v[lower]
    v_upper = v[upper]
    fraction = (query_flat - v_lower) / (v_upper - v_lower)
    interpolated = f[lower] + fraction * (f[upper] - f[lower])
    outside = (query_flat < v[0]) | (query_flat > v[-1])
    interpolated = torch.where(outside, torch.zeros_like(interpolated), interpolated)
    interpolated = torch.where(query_flat == v[0], f[0], interpolated)
    interpolated = torch.where(query_flat == v[-1], f[-1], interpolated)
    return interpolated.reshape(original_shape)


def fourier_landau_response(k_grid, coefficients, velocity):
    """Evaluate the positive-K Landau response for one species."""

    velocity = torch.as_tensor(
        velocity, device=k_grid.device, dtype=_real_dtype(k_grid)
    )
    positive_k = torch.clamp(k_grid, min=0.0)
    phase = k_grid.reshape(-1, 1) * velocity.reshape(1, -1)
    integrand = (
        positive_k.reshape(-1, 1) * coefficients.reshape(-1, 1) * torch.exp(1j * phase)
    )
    response = torch.trapz(integrand, k_grid, dim=0)
    return response.reshape(velocity.shape)


def _projected_phase_velocity(omega_shift, k_magnitude, k_components, drift):
    drift_projection = (
        k_components[0] * drift[0]
        + k_components[1] * drift[1]
        + k_components[2] * drift[2]
    ) / k_magnitude
    return omega_shift / k_magnitude - drift_projection


def _dawson_integral_eq9(x):
    """Evaluate the Dawson integral used by Zhao et al. Eq. (9).

    ``x`` must be a finite float32 or float64 tensor.  For ``|x| <= 8`` the
    mathematically equivalent, overflow-free integral

        D(x) = x * integral_0^1 exp[-x^2 (1 - t^2)] dt

    is evaluated by fixed 64-point Gauss-Legendre quadrature.  Larger
    magnitudes use the ten-term asymptotic expansion of the same integral.
    The result preserves the input shape, dtype, device, and odd symmetry.
    """

    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch.Tensor.")
    if torch.is_complex(x) or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x must be a real float32 or float64 tensor.")
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values.")

    nodes = torch.as_tensor(_EQ9_GAUSS_NODES, device=x.device, dtype=x.dtype)
    weights = torch.as_tensor(_EQ9_GAUSS_WEIGHTS, device=x.device, dtype=x.dtype)
    x_column = x.unsqueeze(-1)
    quadrature = x * torch.sum(
        weights * torch.exp(-x_column.square() * (1.0 - nodes.square())), dim=-1
    )

    large = torch.abs(x) > _EQ9_ASYMPTOTIC_THRESHOLD
    safe_x = torch.where(large, x, torch.ones_like(x))
    term = torch.ones_like(safe_x)
    series = torch.ones_like(safe_x)
    for order in range(1, _EQ9_ASYMPTOTIC_TERMS):
        term = term * (2 * order - 1) / (2.0 * safe_x.square())
        series = series + term
    asymptotic = series / (2.0 * safe_x)
    return torch.where(large, asymptotic, quadrature)


def _maxwellian_response_eq9(x):
    """Return the complex Maxwellian response in Zhao et al. Eq. (9).

    The implemented convention is exactly
    ``1 - 2*x*D(x) + i*sqrt(pi)*x*exp(-x^2)``; in particular, the Landau
    imaginary part has the positive sign used by the cited equation.
    """

    dawson = _dawson_integral_eq9(x)
    real_part = 1.0 - 2.0 * x * dawson
    large = torch.abs(x) > _EQ9_ASYMPTOTIC_THRESHOLD
    safe_x = torch.where(large, x, torch.ones_like(x))
    term = 1.0 / (2.0 * safe_x.square())
    asymptotic_tail = term
    for order in range(2, _EQ9_ASYMPTOTIC_TERMS):
        term = term * (2 * order - 1) / (2.0 * safe_x.square())
        asymptotic_tail = asymptotic_tail + term
    real_part = torch.where(large, -asymptotic_tail, real_part)
    imaginary_part = math.sqrt(math.pi) * x * torch.exp(-x.square())
    response = torch.complex(real_part, imaginary_part)
    if not torch.isfinite(response).all():
        raise ValueError("Equation (9) response is non-finite.")
    return response


def OTS_spectrum_maxwellian_two_ion(
    lms,
    ss,
    k_sh,
    E_i0,
    lm_0,
    n_e,
    T_e_ev,
    Z_i1,
    N_i1,
    m_i1,
    T_i1_ev,
    Z_i2,
    N_i2,
    m_i2,
    T_i2_ev,
    V,
    *,
    u_e,
    u_i1,
    u_i2,
    switch_scale_output=False
):
    """Calculate an analytic two-ion Maxwellian OTS wavelength segment.

    Temperatures are independent per species and expressed in eV.  The
    function evaluates the one-dimensional Maxwellians at their projected
    phase velocities, Zhao et al. (2019) Eqs. (7)-(9) for the susceptibilities,
    and Eq. (2) for the dynamic structure factor.  The paired convention is
    ``k = k_s - k_i`` and ``omega_shift = omega_s - omega_i``; ``u_e``,
    ``u_i1``, and ``u_i2`` are keyword-only three-component drift velocities
    in m/s.

    The returned signal describes one uniform volume element and is normalized
    by ``max(abs(power))``.  Consequently the validated compatibility inputs
    ``E_i0`` and ``V`` do not contribute an absolute-power prefactor.  This
    function does not include amplitude/background fitting, instrument
    convolution, volume integration, gradients, k broadening, collisions,
    magnetization, relativistic effects, or non-Maxwellian corrections.
    """

    lms = torch.as_tensor(lms)
    if torch.is_complex(lms):
        raise ValueError("lms must be real-valued.")
    if not lms.dtype.is_floating_point:
        lms = lms.to(torch.float64)
    dtype = _real_dtype(lms)
    device = lms.device
    lms = lms.to(dtype=dtype)

    if lms.ndim != 1 or len(lms) == 0:
        raise ValueError("lms must be a non-empty one-dimensional wavelength grid.")
    if not torch.isfinite(lms).all() or not torch.all(lms > 0):
        raise ValueError("lms contains invalid wavelengths.")

    vectors = {}
    for value, name in (
        (ss, "ss"),
        (k_sh, "k_sh"),
        (E_i0, "E_i0"),
        (u_e, "u_e"),
        (u_i1, "u_i1"),
        (u_i2, "u_i2"),
    ):
        vector = _as_real_tensor(value, device, dtype, name)
        if vector.shape != (3,) or not torch.isfinite(vector).all():
            raise ValueError("{} must be a finite three-component vector.".format(name))
        vectors[name] = vector
    ss = vectors["ss"]
    k_sh = vectors["k_sh"]
    u_e = vectors["u_e"]
    u_i1 = vectors["u_i1"]
    u_i2 = vectors["u_i2"]

    scalars = {}
    for value, name in (
        (lm_0, "lm_0"),
        (n_e, "n_e"),
        (T_e_ev, "T_e_ev"),
        (Z_i1, "Z_i1"),
        (N_i1, "N_i1"),
        (m_i1, "m_i1"),
        (T_i1_ev, "T_i1_ev"),
        (Z_i2, "Z_i2"),
        (N_i2, "N_i2"),
        (m_i2, "m_i2"),
        (T_i2_ev, "T_i2_ev"),
        (V, "V"),
    ):
        scalar = _as_real_tensor(value, device, dtype, name)
        if scalar.numel() != 1 or not torch.isfinite(scalar).all():
            raise ValueError("{} must be a finite scalar.".format(name))
        scalars[name] = scalar.reshape(())

    lm_0 = scalars["lm_0"]
    n_e = scalars["n_e"]
    T_e_ev = scalars["T_e_ev"]
    Z_i1 = scalars["Z_i1"]
    N_i1 = scalars["N_i1"]
    m_i1 = scalars["m_i1"]
    T_i1_ev = scalars["T_i1_ev"]
    Z_i2 = scalars["Z_i2"]
    N_i2 = scalars["N_i2"]
    m_i2 = scalars["m_i2"]
    T_i2_ev = scalars["T_i2_ev"]
    V = scalars["V"]

    for value, name in (
        (lm_0, "lm_0"),
        (n_e, "n_e"),
        (T_e_ev, "T_e_ev"),
        (m_i1, "m_i1"),
        (T_i1_ev, "T_i1_ev"),
        (m_i2, "m_i2"),
        (T_i2_ev, "T_i2_ev"),
        (V, "V"),
    ):
        if not bool((value > 0).item()):
            raise ValueError("{} must be positive.".format(name))
    for value, name in ((N_i1, "N_i1"), (N_i2, "N_i2")):
        if not bool((value >= 0).item()):
            raise ValueError("{} must be non-negative.".format(name))

    charge_weight = Z_i1 * N_i1 + Z_i2 * N_i2
    if not torch.isfinite(charge_weight) or not bool((charge_weight > 0).item()):
        raise ValueError("Z_i1*N_i1 + Z_i2*N_i2 must be finite and positive.")

    ss_norm = torch.linalg.vector_norm(ss)
    k_sh_norm = torch.linalg.vector_norm(k_sh)
    if not bool((ss_norm > 0).item()):
        raise ValueError("ss must be non-zero.")
    if not bool((k_sh_norm > 0).item()):
        raise ValueError("k_sh must be non-zero.")
    ss = ss / ss_norm
    k_sh = k_sh / k_sh_norm

    omega_0 = 2.0 * math.pi * SPEED_OF_LIGHT / lm_0
    omega_pe_squared = (
        n_e * ELEMENTARY_CHARGE**2 / (ELECTRON_MASS * VACUUM_PERMITTIVITY)
    )
    incident_argument = omega_0.square() - omega_pe_squared
    if not torch.isfinite(incident_argument) or not bool((incident_argument > 0).item()):
        raise ValueError("Probe frequency must exceed the electron plasma frequency.")
    k_0 = torch.sqrt(incident_argument) / SPEED_OF_LIGHT * k_sh

    omega = 2.0 * math.pi * SPEED_OF_LIGHT / lms
    scattered_argument = omega.square() - omega_pe_squared
    if not torch.isfinite(scattered_argument).all() or not torch.all(
        scattered_argument > 0
    ):
        raise ValueError(
            "Scattered-light frequencies must exceed the electron plasma frequency."
        )
    k_scattered_magnitude = torch.sqrt(scattered_argument) / SPEED_OF_LIGHT
    k_components = (
        -k_0[0] + ss[0] * k_scattered_magnitude,
        -k_0[1] + ss[1] * k_scattered_magnitude,
        -k_0[2] + ss[2] * k_scattered_magnitude,
    )
    k_magnitude = torch.sqrt(
        k_components[0].square() + k_components[1].square() + k_components[2].square()
    )
    if not torch.isfinite(k_magnitude).all() or not torch.all(k_magnitude > 0):
        raise ValueError(
            "Scattering wave-vector magnitude must be finite and positive."
        )
    omega_shift = omega - omega_0

    phase_velocity_e = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_e
    )
    phase_velocity_i1 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i1
    )
    phase_velocity_i2 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i2
    )
    for value, name in (
        (phase_velocity_e, "electron phase velocity"),
        (phase_velocity_i1, "ion-1 phase velocity"),
        (phase_velocity_i2, "ion-2 phase velocity"),
    ):
        if not torch.isfinite(value).all():
            raise ValueError("{} is non-finite.".format(name))

    electron_energy = T_e_ev * ELEMENTARY_CHARGE
    ion_1_energy = T_i1_ev * ELEMENTARY_CHARGE
    ion_2_energy = T_i2_ev * ELEMENTARY_CHARGE
    v_te = torch.sqrt(electron_energy / ELECTRON_MASS)
    v_ti1 = torch.sqrt(ion_1_energy / m_i1)
    v_ti2 = torch.sqrt(ion_2_energy / m_i2)
    for value, name in (
        (v_te, "electron thermal speed"),
        (v_ti1, "ion-1 thermal speed"),
        (v_ti2, "ion-2 thermal speed"),
    ):
        if not torch.isfinite(value) or not bool((value > 0).item()):
            raise ValueError("{} must be finite and positive.".format(name))

    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    f_e_phase = torch.exp(-0.5 * (phase_velocity_e / v_te).square()) / (
        sqrt_two_pi * v_te
    )
    f_i1_phase = torch.exp(-0.5 * (phase_velocity_i1 / v_ti1).square()) / (
        sqrt_two_pi * v_ti1
    )
    f_i2_phase = torch.exp(-0.5 * (phase_velocity_i2 / v_ti2).square()) / (
        sqrt_two_pi * v_ti2
    )

    debye_length = torch.sqrt(
        VACUUM_PERMITTIVITY
        * electron_energy
        / (n_e * ELEMENTARY_CHARGE**2)
    )
    alpha = 1.0 / (k_magnitude * debye_length)
    if not torch.isfinite(alpha).all() or not torch.all(alpha > 0):
        raise ValueError("The scattering parameter alpha must be finite and positive.")

    sqrt_two = math.sqrt(2.0)
    x_e = phase_velocity_e / (sqrt_two * v_te)
    x_i1 = phase_velocity_i1 / (sqrt_two * v_ti1)
    x_i2 = phase_velocity_i2 / (sqrt_two * v_ti2)
    response_e = _maxwellian_response_eq9(x_e)
    response_i1 = _maxwellian_response_eq9(x_i1)
    response_i2 = _maxwellian_response_eq9(x_i2)

    alpha_squared = alpha.square()
    ion_1_weight = Z_i1.square() * N_i1 / charge_weight
    ion_2_weight = Z_i2.square() * N_i2 / charge_weight
    chi_e = alpha_squared * response_e
    chi_i1 = (
        alpha_squared
        * ion_1_weight
        * (T_e_ev / T_i1_ev)
        * response_i1
    )
    chi_i2 = (
        alpha_squared
        * ion_2_weight
        * (T_e_ev / T_i2_ev)
        * response_i2
    )
    dielectric = 1.0 + chi_e + chi_i1 + chi_i2
    if not torch.isfinite(dielectric).all() or not torch.all(torch.abs(dielectric) > 0):
        raise ValueError("The dielectric response epsilon is non-finite or zero.")

    electron_screening = torch.abs(1.0 - chi_e / dielectric).square()
    ion_screening = torch.abs(chi_e / dielectric).square()
    structure_factor = (
        2.0
        * math.pi
        / k_magnitude
        * (
            electron_screening * f_e_phase
            + ion_screening * ion_1_weight * f_i1_phase
            + ion_screening * ion_2_weight * f_i2_phase
        )
    )
    if not torch.isfinite(structure_factor).all():
        raise ValueError("The dynamic structure factor is non-finite.")

    power = torch.real((1.0 + 2.0 * omega_shift / omega_0) * structure_factor)
    if not torch.isfinite(power).all():
        raise ValueError("OTS spectrum power is non-finite.")
    scale = torch.max(torch.abs(power))
    if not torch.isfinite(scale) or not bool((scale > 0).item()):
        raise ValueError("OTS spectrum has an invalid normalization scale.")
    intensity = power / scale
    if not torch.isfinite(intensity).all():
        raise ValueError("OTS spectrum contains non-finite values.")

    _ = vectors["E_i0"], V
    if switch_scale_output:
        return intensity, scale
    else:
        return intensity


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
    *,
    v_e,
    f_e,
    v_i1,
    f_i1,
    v_i2,
    f_i2,
    u_e,
    u_i1,
    u_i2,
    fourier_modes=80,
):
    """Calculate one independently normalized OTS wavelength segment.

    All three species are supplied as normalized sampled velocity
    distributions.  Their grids may have different lengths.  The function
    constructs a separate Fourier representation for each species while
    obtaining phase-velocity distribution values by non-periodic interpolation
    of the original samples.
    """

    lms = torch.as_tensor(lms)
    if torch.is_complex(lms):
        raise ValueError("lms must be real-valued.")
    if not lms.dtype.is_floating_point:
        lms = lms.to(torch.float64)
    dtype = _real_dtype(lms)
    device = lms.device
    lms = lms.to(dtype=dtype)

    if lms.ndim != 1 or len(lms) == 0:
        raise ValueError("lms must be a non-empty one-dimensional wavelength grid.")
    if not torch.isfinite(lms).all() or not torch.all(lms > 0):
        raise ValueError("lms contains invalid wavelengths.")

    vectors = {}
    for value, name in (
        (ss, "ss"),
        (k_sh, "k_sh"),
        (E_i0, "E_i0"),
        (u_e, "u_e"),
        (u_i1, "u_i1"),
        (u_i2, "u_i2"),
    ):
        vector = _as_real_tensor(value, device, dtype, name)
        if vector.shape != (3,) or not torch.isfinite(vector).all():
            raise ValueError("{} must be a finite three-component vector.".format(name))
        vectors[name] = vector
    ss = vectors["ss"]
    k_sh = vectors["k_sh"]
    u_e = vectors["u_e"]
    u_i1 = vectors["u_i1"]
    u_i2 = vectors["u_i2"]

    scalars = {}
    for value, name in (
        (n_e, "n_e"),
        (lm_0, "lm_0"),
        (Z_i1, "Z_i1"),
        (N_i1, "N_i1"),
        (m_i1, "m_i1"),
        (Z_i2, "Z_i2"),
        (N_i2, "N_i2"),
        (m_i2, "m_i2"),
        (V, "V"),
    ):
        scalar = _as_real_tensor(value, device, dtype, name)
        if scalar.numel() != 1 or not torch.isfinite(scalar).all():
            raise ValueError("{} must be a finite scalar.".format(name))
        scalars[name] = scalar.reshape(())
    n_e = scalars["n_e"]
    lm_0 = scalars["lm_0"]
    Z_i1 = scalars["Z_i1"]
    N_i1 = scalars["N_i1"]
    m_i1 = scalars["m_i1"]
    Z_i2 = scalars["Z_i2"]
    N_i2 = scalars["N_i2"]
    m_i2 = scalars["m_i2"]
    V = scalars["V"]

    if not bool((n_e > 0).item()) or not bool((lm_0 > 0).item()):
        raise ValueError("n_e and lm_0 must be positive.")
    if not bool((m_i1 > 0).item()) or not bool((m_i2 > 0).item()):
        raise ValueError("Ion masses must be positive.")
    if not bool((N_i1 >= 0).item()) or not bool((N_i2 >= 0).item()):
        raise ValueError("Ion fractions must be non-negative.")
    if not bool((V > 0).item()):
        raise ValueError("V must be positive.")

    charge_weight = Z_i1 * N_i1 + Z_i2 * N_i2
    if not bool((charge_weight > 0).item()):
        raise ValueError("The charge-weighted ion fraction must be positive.")

    ss_norm = torch.linalg.vector_norm(ss)
    k_sh_norm = torch.linalg.vector_norm(k_sh)
    if not bool((ss_norm > 0).item()) or not bool((k_sh_norm > 0).item()):
        raise ValueError("Scattering and incident directions must be non-zero.")
    ss = ss / ss_norm
    k_sh = k_sh / k_sh_norm

    v_e = _as_real_tensor(v_e, device, dtype, "electron velocity grid")
    f_e = _as_real_tensor(f_e, device, dtype, "electron distribution")
    v_i1 = _as_real_tensor(v_i1, device, dtype, "ion-1 velocity grid")
    f_i1 = _as_real_tensor(f_i1, device, dtype, "ion-1 distribution")
    v_i2 = _as_real_tensor(v_i2, device, dtype, "ion-2 velocity grid")
    f_i2 = _as_real_tensor(f_i2, device, dtype, "ion-2 distribution")
    v_e, f_e = _validate_distribution(v_e, f_e, "electron")
    v_i1, f_i1 = _validate_distribution(v_i1, f_i1, "ion-1")
    v_i2, f_i2 = _validate_distribution(v_i2, f_i2, "ion-2")

    minimum_samples = min(len(v_e), len(v_i1), len(v_i2))
    if (
        not isinstance(fourier_modes, Integral)
        or isinstance(fourier_modes, bool)
        or int(fourier_modes) <= 0
        or int(fourier_modes) > minimum_samples
    ):
        raise ValueError(
            "Fourier mode count n={} must be an integer satisfying 0 < n <= N "
            "with N={} (the smallest species grid).".format(
                fourier_modes, minimum_samples
            )
        )
    fourier_modes = int(fourier_modes)
    K_e, F_e = ft_x_n(v_e, f_e, fourier_modes)
    K_i1, F_i1 = ft_x_n(v_i1, f_i1, fourier_modes)
    K_i2, F_i2 = ft_x_n(v_i2, f_i2, fourier_modes)

    omega_0 = 2.0 * math.pi * SPEED_OF_LIGHT / lm_0
    omega_pe_squared = (
        n_e * ELEMENTARY_CHARGE**2 / (ELECTRON_MASS * VACUUM_PERMITTIVITY)
    )
    incident_argument = omega_0.square() - omega_pe_squared
    if not bool((incident_argument > 0).item()):
        raise ValueError("Probe frequency must exceed the electron plasma frequency.")
    k_0 = torch.sqrt(incident_argument) / SPEED_OF_LIGHT * k_sh

    omega = 2.0 * math.pi * SPEED_OF_LIGHT / lms
    scattered_argument = omega.square() - omega_pe_squared
    if not torch.isfinite(scattered_argument).all() or not torch.all(
        scattered_argument > 0
    ):
        raise ValueError(
            "Scattered-light frequencies must exceed the electron plasma frequency."
        )
    k_scattered_magnitude = torch.sqrt(scattered_argument) / SPEED_OF_LIGHT
    k_components = (
        -k_0[0] + ss[0] * k_scattered_magnitude,
        -k_0[1] + ss[1] * k_scattered_magnitude,
        -k_0[2] + ss[2] * k_scattered_magnitude,
    )
    k_magnitude = torch.sqrt(
        k_components[0].square() + k_components[1].square() + k_components[2].square()
    )
    if not torch.isfinite(k_magnitude).all() or not torch.all(k_magnitude > 0):
        raise ValueError(
            "Scattering wave-vector magnitude must be finite and positive."
        )
    omega_shift = omega - omega_0

    phase_velocity_e = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_e
    )
    phase_velocity_i1 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i1
    )
    phase_velocity_i2 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i2
    )

    f_e_phase = interpolate_distribution(v_e, f_e, phase_velocity_e)
    f_i1_phase = interpolate_distribution(v_i1, f_i1, phase_velocity_i1)
    f_i2_phase = interpolate_distribution(v_i2, f_i2, phase_velocity_i2)
    response_e = fourier_landau_response(K_e, F_e, phase_velocity_e)
    response_i1 = fourier_landau_response(K_i1, F_i1, phase_velocity_i1)
    response_i2 = fourier_landau_response(K_i2, F_i2, phase_velocity_i2)

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
    structure_factor = (
        2.0
        * math.pi
        / k_magnitude
        * (
            electron_screening * f_e_phase
            + ion_screening * ion_1_weight * f_i1_phase
            + ion_screening * ion_2_weight * f_i2_phase
        )
    )

    power = torch.real((1.0 + 2.0 * omega_shift / omega_0) * structure_factor)
    scale = torch.max(torch.abs(power))
    if not torch.isfinite(scale) or not bool((scale > 0).item()):
        raise ValueError("OTS spectrum has an invalid normalization scale.")
    intensity = power / scale
    if not torch.isfinite(intensity).all():
        raise ValueError("OTS spectrum contains non-finite values.")

    # These validated inputs are retained by the public interface even though
    # segment normalization intentionally omits the absolute-power prefactor.
    _ = vectors["E_i0"], V
    return intensity


def _pv_fd_landau_response(v, f, velocity, *, name):
    """Evaluate a hard-truncated sampled distribution by same-grid PV-FD.

    Interior differences are centered, endpoint differences one-sided. For a
    non-node query, its two bracketing quotients are exactly the slope of the
    linearly interpolated discrete derivative. No analytic derivatives, query
    snapping, smoothing, or imaginary displacement are used.
    """
    v, f = _validate_grid_and_values(v, f, name)
    velocity = _as_real_tensor(velocity, v.device, v.dtype, name + ' query velocity')
    if not torch.isfinite(velocity).all():
        raise ValueError('{} query velocity contains non-finite values.'.format(name))
    shape = velocity.shape
    query = velocity.reshape(-1)
    complex_dtype = torch.complex128 if v.dtype == torch.float64 else torch.complex64
    g = torch.cat(((f[1:2] - f[:1]) / (v[1:2] - v[:1]),
                   (f[2:] - f[:-2]) / (v[2:] - v[:-2]),
                   (f[-1:] - f[-2:-1]) / (v[-1:] - v[-2:-1])))
    if not torch.isfinite(g).all():
        raise ValueError('{} discrete derivative is non-finite.'.format(name))
    slopes = (g[1:] - g[:-1]) / (v[1:] - v[:-1])
    merged = torch.cat((slopes[:1], (g[2:] - g[:-2]) / (v[2:] - v[:-2]), slopes[-1:]))
    if query.numel() == 0:
        return torch.empty(shape, device=v.device, dtype=complex_dtype)
    blocks = []
    nodes = torch.arange(len(v), device=v.device).reshape(1, -1)
    for start in range(0, len(query), 256):
        z = query[start:start + 256]
        value = torch.empty(z.shape, device=v.device, dtype=complex_dtype)
        left = z == v[0]
        right = z == v[-1]
        for mask, endpoint in ((left, 0), (right, -1)):
            if torch.any(mask) and (f[endpoint] != 0 or g[endpoint] != 0):
                raise ValueError('{} has a PV representation singularity at endpoint {}.'.format(name, v[endpoint].item()))
        inside = (z > v[0]) & (z < v[-1])
        if torch.any(inside):
            zz = z[inside]
            upper = torch.searchsorted(v, zz, right=False)
            lower = upper - 1
            exact = zz == v[upper]
            gz = interpolate_distribution(v, g, zz)
            gz = torch.where(exact, g[upper], gz)
            equal = exact[:, None] & (nodes == upper[:, None])
            bracket = (~exact[:, None]) & ((nodes == lower[:, None]) | (nodes == upper[:, None]))
            regular = ~(equal | bracket)
            delta = v[None, :] - zz[:, None]
            quotient = torch.empty_like(delta)
            gg = g[None, :].expand_as(delta)
            gzz = gz[:, None].expand_as(delta)
            quotient[regular] = (gg[regular] - gzz[regular]) / delta[regular]
            quotient[bracket] = slopes[lower][:, None].expand_as(delta)[bracket]
            quotient[equal] = merged[upper][:, None].expand_as(delta)[equal]
            real = -torch.trapz(quotient, v, dim=1) - gz * torch.log(torch.abs((v[-1] - zz) / (v[0] - zz)))
            if f[0] != 0:
                real = real - f[0] / (v[0] - zz)
            if f[-1] != 0:
                real = real + f[-1] / (v[-1] - zz)
            value[inside] = torch.complex(real, -math.pi * gz)
        outside = ~inside
        if torch.any(outside):
            zz = z[outside]
            delta = v[None, :] - zz[:, None]
            equal = delta == 0
            regular = ~equal
            quotient = torch.empty_like(delta)
            gg = g[None, :].expand_as(delta)
            quotient[regular] = gg[regular] / delta[regular]
            quotient[equal] = merged[None, :].expand_as(delta)[equal]
            real = -torch.trapz(quotient, v, dim=1)
            if f[0] != 0:
                real = real - f[0] / (v[0] - zz)
            if f[-1] != 0:
                real = real + f[-1] / (v[-1] - zz)
            value[outside] = torch.complex(real, torch.zeros_like(real))
        if not torch.isfinite(value).all():
            raise ValueError('{} PV-FD response is non-finite.'.format(name))
        blocks.append(value)
    return torch.cat(blocks).reshape(shape)

def OTS_spectrum_PV(
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
    *,
    v_e,
    f_e,
    v_i1,
    f_i1,
    v_i2,
    f_i2,
    u_e,
    u_i1,
    u_i2,
    fourier_modes=80,
):
    """Return normalized three-species OTS intensity using same-grid PV-FD.

    The signature and validation match OTS_spectrum_full_fourier. The validated
    fourier_modes argument is compatibility-only and has no numerical effect.
    All species use sampled densities; representation singularities fail the
    entire call rather than yielding a partially normalized spectrum.
    """

    lms = torch.as_tensor(lms)
    if torch.is_complex(lms):
        raise ValueError("lms must be real-valued.")
    if not lms.dtype.is_floating_point:
        lms = lms.to(torch.float64)
    dtype = _real_dtype(lms)
    device = lms.device
    lms = lms.to(dtype=dtype)

    if lms.ndim != 1 or len(lms) == 0:
        raise ValueError("lms must be a non-empty one-dimensional wavelength grid.")
    if not torch.isfinite(lms).all() or not torch.all(lms > 0):
        raise ValueError("lms contains invalid wavelengths.")

    vectors = {}
    for value, name in (
        (ss, "ss"),
        (k_sh, "k_sh"),
        (E_i0, "E_i0"),
        (u_e, "u_e"),
        (u_i1, "u_i1"),
        (u_i2, "u_i2"),
    ):
        vector = _as_real_tensor(value, device, dtype, name)
        if vector.shape != (3,) or not torch.isfinite(vector).all():
            raise ValueError("{} must be a finite three-component vector.".format(name))
        vectors[name] = vector
    ss = vectors["ss"]
    k_sh = vectors["k_sh"]
    u_e = vectors["u_e"]
    u_i1 = vectors["u_i1"]
    u_i2 = vectors["u_i2"]

    scalars = {}
    for value, name in (
        (n_e, "n_e"),
        (lm_0, "lm_0"),
        (Z_i1, "Z_i1"),
        (N_i1, "N_i1"),
        (m_i1, "m_i1"),
        (Z_i2, "Z_i2"),
        (N_i2, "N_i2"),
        (m_i2, "m_i2"),
        (V, "V"),
    ):
        scalar = _as_real_tensor(value, device, dtype, name)
        if scalar.numel() != 1 or not torch.isfinite(scalar).all():
            raise ValueError("{} must be a finite scalar.".format(name))
        scalars[name] = scalar.reshape(())
    n_e = scalars["n_e"]
    lm_0 = scalars["lm_0"]
    Z_i1 = scalars["Z_i1"]
    N_i1 = scalars["N_i1"]
    m_i1 = scalars["m_i1"]
    Z_i2 = scalars["Z_i2"]
    N_i2 = scalars["N_i2"]
    m_i2 = scalars["m_i2"]
    V = scalars["V"]

    if not bool((n_e > 0).item()) or not bool((lm_0 > 0).item()):
        raise ValueError("n_e and lm_0 must be positive.")
    if not bool((m_i1 > 0).item()) or not bool((m_i2 > 0).item()):
        raise ValueError("Ion masses must be positive.")
    if not bool((N_i1 >= 0).item()) or not bool((N_i2 >= 0).item()):
        raise ValueError("Ion fractions must be non-negative.")
    if not bool((V > 0).item()):
        raise ValueError("V must be positive.")

    charge_weight = Z_i1 * N_i1 + Z_i2 * N_i2
    if not bool((charge_weight > 0).item()):
        raise ValueError("The charge-weighted ion fraction must be positive.")

    ss_norm = torch.linalg.vector_norm(ss)
    k_sh_norm = torch.linalg.vector_norm(k_sh)
    if not bool((ss_norm > 0).item()) or not bool((k_sh_norm > 0).item()):
        raise ValueError("Scattering and incident directions must be non-zero.")
    ss = ss / ss_norm
    k_sh = k_sh / k_sh_norm

    v_e = _as_real_tensor(v_e, device, dtype, "electron velocity grid")
    f_e = _as_real_tensor(f_e, device, dtype, "electron distribution")
    v_i1 = _as_real_tensor(v_i1, device, dtype, "ion-1 velocity grid")
    f_i1 = _as_real_tensor(f_i1, device, dtype, "ion-1 distribution")
    v_i2 = _as_real_tensor(v_i2, device, dtype, "ion-2 velocity grid")
    f_i2 = _as_real_tensor(f_i2, device, dtype, "ion-2 distribution")
    v_e, f_e = _validate_distribution(v_e, f_e, "electron")
    v_i1, f_i1 = _validate_distribution(v_i1, f_i1, "ion-1")
    v_i2, f_i2 = _validate_distribution(v_i2, f_i2, "ion-2")

    minimum_samples = min(len(v_e), len(v_i1), len(v_i2))
    if (
        not isinstance(fourier_modes, Integral)
        or isinstance(fourier_modes, bool)
        or int(fourier_modes) <= 0
        or int(fourier_modes) > minimum_samples
    ):
        raise ValueError(
            "Fourier mode count n={} must be an integer satisfying 0 < n <= N "
            "with N={} (the smallest species grid).".format(
                fourier_modes, minimum_samples
            )
        )
    fourier_modes = int(fourier_modes)

    omega_0 = 2.0 * math.pi * SPEED_OF_LIGHT / lm_0
    omega_pe_squared = (
        n_e * ELEMENTARY_CHARGE**2 / (ELECTRON_MASS * VACUUM_PERMITTIVITY)
    )
    incident_argument = omega_0.square() - omega_pe_squared
    if not bool((incident_argument > 0).item()):
        raise ValueError("Probe frequency must exceed the electron plasma frequency.")
    k_0 = torch.sqrt(incident_argument) / SPEED_OF_LIGHT * k_sh

    omega = 2.0 * math.pi * SPEED_OF_LIGHT / lms
    scattered_argument = omega.square() - omega_pe_squared
    if not torch.isfinite(scattered_argument).all() or not torch.all(
        scattered_argument > 0
    ):
        raise ValueError(
            "Scattered-light frequencies must exceed the electron plasma frequency."
        )
    k_scattered_magnitude = torch.sqrt(scattered_argument) / SPEED_OF_LIGHT
    k_components = (
        -k_0[0] + ss[0] * k_scattered_magnitude,
        -k_0[1] + ss[1] * k_scattered_magnitude,
        -k_0[2] + ss[2] * k_scattered_magnitude,
    )
    k_magnitude = torch.sqrt(
        k_components[0].square() + k_components[1].square() + k_components[2].square()
    )
    if not torch.isfinite(k_magnitude).all() or not torch.all(k_magnitude > 0):
        raise ValueError(
            "Scattering wave-vector magnitude must be finite and positive."
        )
    omega_shift = omega - omega_0

    phase_velocity_e = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_e
    )
    phase_velocity_i1 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i1
    )
    phase_velocity_i2 = _projected_phase_velocity(
        omega_shift, k_magnitude, k_components, u_i2
    )

    f_e_phase = interpolate_distribution(v_e, f_e, phase_velocity_e)
    f_i1_phase = interpolate_distribution(v_i1, f_i1, phase_velocity_i1)
    f_i2_phase = interpolate_distribution(v_i2, f_i2, phase_velocity_i2)
    response_e = _pv_fd_landau_response(v_e, f_e, phase_velocity_e, name="electron")
    response_i1 = _pv_fd_landau_response(v_i1, f_i1, phase_velocity_i1, name="ion-1")
    response_i2 = _pv_fd_landau_response(v_i2, f_i2, phase_velocity_i2, name="ion-2")

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
    structure_factor = (
        2.0
        * math.pi
        / k_magnitude
        * (
            electron_screening * f_e_phase
            + ion_screening * ion_1_weight * f_i1_phase
            + ion_screening * ion_2_weight * f_i2_phase
        )
    )

    power = torch.real((1.0 + 2.0 * omega_shift / omega_0) * structure_factor)
    scale = torch.max(torch.abs(power))
    if not torch.isfinite(scale) or not bool((scale > 0).item()):
        raise ValueError("OTS spectrum has an invalid normalization scale.")
    intensity = power / scale
    if not torch.isfinite(intensity).all():
        raise ValueError("OTS spectrum contains non-finite values.")

    # These validated inputs are retained by the public interface even though
    # segment normalization intentionally omits the absolute-power prefactor.
    _ = vectors["E_i0"], V
    return intensity

if __name__ == "__main__":
    """
    测试与演示代码：验证 OTS 光谱计算正向模型

    流程：
    1. 设定等离子体参数（温度、密度、离子组成）
    2. 生成超高斯速度分布（参数化后进行傅里叶变换）
    3. 调用正向模型计算光谱
    4. 绘图对比两个版本（复数模式 vs 分离实部虚部）
    """
    import math
    import matplotlib.pyplot as plt

    # 常数定义（物理单位）
    e = torch.tensor(1.602176634e-19)              # 基本电荷 (C)
    k_B = torch.tensor(1.380649e-23)               # 玻尔兹曼常数 (J/K)
    c = torch.tensor(2.99792458e8)                 # 光速 (m/s)
    m_e = torch.tensor(9.10938356e-31)             # 电子质量 (kg)
    m_p = torch.tensor(1.67262192369e-27)          # 质子质量 (kg)
    epsilon_0 = torch.tensor(8.854187817e-12)      # 真空介电常数 (F/m)
    # ─────────────────────────────────────────────────────────────
    # 物理常数与参数设置
    # ─────────────────────────────────────────────────────────────
    k_sh = torch.tensor([1., 0., 0.])                # 入射波矢方向（沿x轴）
    ss = torch.tensor([0., 1., 0.])                 # 散射方向（沿y轴）
    E_i0 = torch.tensor([0., 0., 1.])               # 入射光极化（沿z轴）
    lm_0 = 263e-9                                    # 入射光波长 (263 nm)

    # 等离子体参数：以 eV 设定，再转换为 K 供正向模型使用
    T_e = 1.4e3 * e / k_B                           # 电子温度 1400 eV -> K
    T_i = 0.5e3 * e / k_B                           # 离子温度 500 eV -> K
    n_e = 1e26                                       # 电子密度 (m⁻³)
    m = 2.                                           # 超高斯指数

    # 测量参数
    lms = torch.linspace(-1, 1, 5000) * 120e-9 + lm_0  # 测量波长范围

    # 离子种类（C1H1 混合物）
    Z_i1 = 1                                         # 氢离子：核电荷数 = 1
    N_i1 = 1                                         # 相对浓度
    Z_i2 = 6                                         # 碳离子：核电荷数 = 6
    N_i2 = 0                                         # 不存在
    ZN = Z_i1 * N_i1 + Z_i2 * N_i2

    # 离子质量
    m_i1 = m_p                                       # 质子
    m_i2 = m_p * 12                                  # 碳12

    # 参数化速度分布
    R = 1e-3                                         # 等离子体尺度
    V = R**3                                         # 体积
    I_0 = 1
    v_te = torch.sqrt(k_B * T_e / m_e)              # 电子热速度
    v_ti1 = torch.sqrt(k_B * T_i / m_i1)              # 离子热速度
    v_ti2 = torch.sqrt(k_B * T_i / m_i2)              # 离子热速度
    # ─────────────────────────────────────────────────────────────
    # 生成超高斯速度分布
    # ─────────────────────────────────────────────────────────────

    u_e = torch.tensor([0., 0., 0.]) * c
    u_i1 = torch.tensor([0., 0., 0.]) * c
    u_i2 = torch.tensor([0., 0., 0.]) * c

    # 超高斯分布参数
    a = lambda m: (2 * math.gamma(5 / m)) / (3 * math.gamma(3 / m))
    g = lambda x, m: torch.exp(-(a(m) * x**2)**(m / 2))
    C = lambda v_te, m: m / (4 * math.pi * math.gamma(3 / m)) * (a(m) / 2 / v_te**2)**(3 / 2)
    f_ev = lambda v: C(v_te, m) * g(v / math.sqrt(2) / v_te, m)

    # 在实空间进行傅里叶变换
    vxs = torch.linspace(-1, 1, 1000) * 10 * v_te * math.sqrt(2)  # 速度网格
    vrs = torch.linspace(0, 1, 1000) * 10 * v_te * math.sqrt(2)   # 径向速度
    vi1 = torch.linspace(-1, 1, 1000) * 10 * v_ti1 * math.sqrt(2)  # 离子1速度网格
    vi2 = torch.linspace(-1, 1, 1000) * 10 * v_ti2 * math.sqrt(2)  # 离子2速度网格
    # 二维积分得到一维速度分布
    f_es1 = torch.zeros_like(vxs)
    f_i1 = torch.zeros_like(vxs)
    f_i2 = torch.zeros_like(vxs)
    for i in range(len(vxs)):
        vx = vxs[i]
        v = torch.sqrt(vx**2 + vrs**2)
        f_es1[i] = 2 * math.pi * torch.trapz(vrs * f_ev(v), vrs)
    f_i1 = torch.exp(-vi1**2 / (2 * v_ti1**2))
    f_i2 = torch.exp(-vi2**2 / (2 * v_ti2**2))
    f_es1 = f_es1 / torch.trapz(f_es1, vxs)
    f_i1 = f_i1 / torch.trapz(f_i1, vi1)
    f_i2 = f_i2 / torch.trapz(f_i2, vi2)
    # ─────────────────────────────────────────────────────────────
    # 计算光谱并对比两个版本
    # ─────────────────────────────────────────────────────────────


    # 版本1：复数傅里叶系数
    P,chi_e,chi_i1,chi_i2 = OTS_spectrum_full_fourier(lms, ss, k_sh, E_i0, lm_0, n_e, Z_i1, N_i1, m_i1,
                                Z_i2, N_i2, m_i2, V, v_e=vxs, f_e=f_es1, v_i1=vi1, f_i1=f_i1, v_i2=vi2, f_i2=f_i2,
                                u_e=u_e, u_i1=u_i1, u_i2=u_i2)
    # 绘制对比图
    plt.plot(lms, P, label='Complex FFT')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Spectral Intensity')
    plt.legend()
    plt.show()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(lms, chi_e.real, label='Electron Chi Real')
    plt.plot(lms, chi_e.imag, label='Electron Chi Imag')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Susceptibility')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(lms, chi_i1.real, label='Ion 1 Chi')
    plt.plot(lms, chi_i1.imag, label='Ion 1 Chi Imag')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Susceptibility')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(lms, (1+chi_e + chi_i1 + chi_i2).real, label='Dielectric Function Real')
    plt.plot(lms, (1+chi_e + chi_i1 + chi_i2).imag, label='Dielectric Function Imag')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Susceptibility')
    plt.legend()
    plt.show()
