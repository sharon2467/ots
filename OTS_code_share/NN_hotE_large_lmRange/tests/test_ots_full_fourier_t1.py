"""Targeted acceptance tests for canonical-pipeline T1.

These tests intentionally cover only ``phyEqns.ots_full_fourier``.  They do
not exercise data generation, training, evaluation, CLI, or Slurm behavior.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phyEqns import ots_full_fourier as full  # noqa: E402


def normalized_gaussian(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    values = torch.exp(-0.5 * (grid / sigma) ** 2)
    return values / torch.trapz(values, grid)


def make_forward_inputs(dtype: torch.dtype = torch.float64) -> tuple:
    """Build normalized species distributions on deliberately unequal grids."""

    electron_grid = torch.linspace(
        -0.8 * full.SPEED_OF_LIGHT,
        0.8 * full.SPEED_OF_LIGHT,
        257,
        dtype=dtype,
    )
    electron_distribution = normalized_gaussian(
        electron_grid, 0.04 * full.SPEED_OF_LIGHT
    )

    proton_mass = 1.67262192369e-27
    ion_temperature_ev = 171.3
    ion1_speed = math.sqrt(
        ion_temperature_ev * full.ELEMENTARY_CHARGE / proton_mass
    )
    ion2_speed = math.sqrt(
        ion_temperature_ev * full.ELEMENTARY_CHARGE / (12.0 * proton_mass)
    )
    ion1_grid = torch.linspace(-12.0 * ion1_speed, 12.0 * ion1_speed, 193, dtype=dtype)
    ion2_grid = torch.linspace(-12.0 * ion2_speed, 12.0 * ion2_speed, 161, dtype=dtype)
    ion1_distribution = normalized_gaussian(ion1_grid, ion1_speed)
    ion2_distribution = normalized_gaussian(ion2_grid, ion2_speed)

    angle = math.radians(123.0)
    positional = (
        torch.linspace(349.0e-9, 353.0e-9, 17, dtype=dtype),
        torch.tensor([math.cos(angle), math.sin(angle), 0.0], dtype=dtype),
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        351.0e-9,
        1.81e26,
        1.0,
        0.55,
        proton_mass,
        6.0,
        0.45,
        12.0 * proton_mass,
        1.0e-9,
    )
    keyword = {
        "v_e": electron_grid,
        "f_e": electron_distribution,
        "v_i1": ion1_grid,
        "f_i1": ion1_distribution,
        "v_i2": ion2_grid,
        "f_i2": ion2_distribution,
        "u_e": torch.zeros(3, dtype=dtype),
        "u_i1": torch.zeros(3, dtype=dtype),
        "u_i2": torch.zeros(3, dtype=dtype),
        "fourier_modes": 32,
    }
    return positional, keyword


def make_maxwellian_inputs(dtype: torch.dtype = torch.float64) -> tuple:
    """Build independent-temperature inputs for the analytic two-ion path."""

    proton_mass = 1.67262192369e-27
    angle = math.radians(123.0)
    positional = (
        torch.linspace(350.5e-9, 351.5e-9, 33, dtype=dtype),
        torch.tensor([math.cos(angle), math.sin(angle), 0.0], dtype=dtype),
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        torch.tensor(351.0e-9, dtype=dtype),
        torch.tensor(1.81e26, dtype=dtype),
        torch.tensor(1370.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(0.55, dtype=dtype),
        torch.tensor(proton_mass, dtype=dtype),
        torch.tensor(171.3, dtype=dtype),
        torch.tensor(6.0, dtype=dtype),
        torch.tensor(0.45, dtype=dtype),
        torch.tensor(12.0 * proton_mass, dtype=dtype),
        torch.tensor(295.0, dtype=dtype),
        torch.tensor(1.0e-9, dtype=dtype),
    )
    keyword = {
        "u_e": torch.zeros(3, dtype=dtype),
        "u_i1": torch.zeros(3, dtype=dtype),
        "u_i2": torch.zeros(3, dtype=dtype),
    }
    return positional, keyword


def reference_maxwellian_spectrum(positional: tuple, keyword: dict) -> torch.Tensor:
    """Independent NumPy scalar reference for Eqs. (2) and (7)-(9)."""

    (
        lms,
        ss,
        k_sh,
        _E_i0,
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
        _V,
    ) = positional
    wavelengths = lms.detach().cpu().double().numpy()
    scattered_direction = np.asarray(ss, dtype=np.float64)
    incident_direction = np.asarray(k_sh, dtype=np.float64)
    scattered_direction /= np.linalg.norm(scattered_direction)
    incident_direction /= np.linalg.norm(incident_direction)
    drifts = [np.asarray(keyword[name], dtype=np.float64) for name in ("u_e", "u_i1", "u_i2")]

    omega_0 = 2.0 * math.pi * full.SPEED_OF_LIGHT / float(lm_0)
    omega_pe_squared = (
        float(n_e)
        * full.ELEMENTARY_CHARGE**2
        / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
    )
    k_incident = (
        math.sqrt(omega_0**2 - omega_pe_squared)
        / full.SPEED_OF_LIGHT
        * incident_direction
    )
    thermal_speeds = (
        math.sqrt(float(T_e_ev) * full.ELEMENTARY_CHARGE / full.ELECTRON_MASS),
        math.sqrt(float(T_i1_ev) * full.ELEMENTARY_CHARGE / float(m_i1)),
        math.sqrt(float(T_i2_ev) * full.ELEMENTARY_CHARGE / float(m_i2)),
    )
    debye_length = math.sqrt(
        full.VACUUM_PERMITTIVITY
        * float(T_e_ev)
        * full.ELEMENTARY_CHARGE
        / (float(n_e) * full.ELEMENTARY_CHARGE**2)
    )
    charge_weight = float(Z_i1) * float(N_i1) + float(Z_i2) * float(N_i2)
    ion_weights = (
        float(Z_i1) ** 2 * float(N_i1) / charge_weight,
        float(Z_i2) ** 2 * float(N_i2) / charge_weight,
    )
    nodes, weights = np.polynomial.legendre.leggauss(256)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    def response(x_value: float) -> complex:
        dawson = x_value * np.sum(
            weights * np.exp(-(x_value**2) * (1.0 - nodes**2))
        )
        return complex(
            1.0 - 2.0 * x_value * dawson,
            math.sqrt(math.pi) * x_value * math.exp(-(x_value**2)),
        )

    powers = []
    for wavelength in wavelengths:
        omega = 2.0 * math.pi * full.SPEED_OF_LIGHT / wavelength
        k_scattered = (
            math.sqrt(omega**2 - omega_pe_squared)
            / full.SPEED_OF_LIGHT
            * scattered_direction
        )
        k_vector = k_scattered - k_incident
        k_magnitude = np.linalg.norm(k_vector)
        omega_shift = omega - omega_0
        phase_velocities = tuple(
            omega_shift / k_magnitude - np.dot(k_vector, drift) / k_magnitude
            for drift in drifts
        )
        maxwellians = tuple(
            math.exp(-0.5 * (velocity / thermal_speed) ** 2)
            / (math.sqrt(2.0 * math.pi) * thermal_speed)
            for velocity, thermal_speed in zip(phase_velocities, thermal_speeds)
        )
        x_values = tuple(
            velocity / (math.sqrt(2.0) * thermal_speed)
            for velocity, thermal_speed in zip(phase_velocities, thermal_speeds)
        )
        alpha_squared = (1.0 / (k_magnitude * debye_length)) ** 2
        chi_e = alpha_squared * response(x_values[0])
        chi_i1 = (
            alpha_squared
            * ion_weights[0]
            * (float(T_e_ev) / float(T_i1_ev))
            * response(x_values[1])
        )
        chi_i2 = (
            alpha_squared
            * ion_weights[1]
            * (float(T_e_ev) / float(T_i2_ev))
            * response(x_values[2])
        )
        dielectric = 1.0 + chi_e + chi_i1 + chi_i2
        electron_screening = abs(1.0 - chi_e / dielectric) ** 2
        ion_screening = abs(chi_e / dielectric) ** 2
        structure_factor = 2.0 * math.pi / k_magnitude * (
            electron_screening * maxwellians[0]
            + ion_screening * ion_weights[0] * maxwellians[1]
            + ion_screening * ion_weights[1] * maxwellians[2]
        )
        powers.append(
            (1.0 + 2.0 * omega_shift / omega_0) * structure_factor
        )
    powers = np.asarray(powers, dtype=np.float64)
    return torch.from_numpy(powers / np.max(np.abs(powers))).to(dtype=lms.dtype)


class TestFtXN(unittest.TestCase):
    def setUp(self) -> None:
        self.v = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)
        self.f = torch.exp(-self.v.square())

    def test_formula_shapes_dtype_and_n_equal_sample_count(self) -> None:
        for n in (1, 8, len(self.v)):
            with self.subTest(n=n):
                k_grid, coefficients = full.ft_x_n(self.v, self.f, n)
                dx = self.v[2] - self.v[1]
                dk = 2.0 * math.pi / (len(self.v) * dx)
                expected_k = torch.linspace(-1.0, 1.0, n, dtype=self.v.dtype) * dk * n / 2.0
                expected_coefficients = torch.stack(
                    [
                        torch.trapz(self.f * torch.exp(-1j * k_value * self.v), self.v)
                        for k_value in expected_k
                    ]
                )
                self.assertEqual(k_grid.shape, (n,))
                self.assertEqual(coefficients.shape, (n,))
                self.assertEqual(k_grid.dtype, torch.float64)
                self.assertEqual(coefficients.dtype, torch.complex128)
                torch.testing.assert_close(k_grid, expected_k)
                torch.testing.assert_close(coefficients, expected_coefficients)

    def test_rejects_invalid_n_and_reports_n_and_N(self) -> None:
        for invalid_n in (0, -1, len(self.v) + 1, 3.0, True):
            with self.subTest(n=invalid_n):
                with self.assertRaisesRegex(ValueError, r"n=.*N=17"):
                    full.ft_x_n(self.v, self.f, invalid_n)

    def test_rejects_shape_length_finite_monotonic_and_uniformity_errors(self) -> None:
        cases = {
            "two-dimensional velocity": (self.v.reshape(1, -1), self.f),
            "two-dimensional values": (self.v, self.f.reshape(1, -1)),
            "unequal length": (self.v, self.f[:-1]),
            "fewer than three": (self.v[:2], self.f[:2]),
            "nonfinite velocity": (self.v.clone(), self.f),
            "nonfinite values": (self.v, self.f.clone()),
            "not increasing": (self.v.clone(), self.f),
            "nonuniform": (self.v.clone(), self.f),
        }
        cases["nonfinite velocity"][0][4] = torch.inf
        cases["nonfinite values"][1][4] = torch.nan
        cases["not increasing"][0][8] = cases["not increasing"][0][7]
        cases["nonuniform"][0][8] += 0.02
        for name, (velocity, values) in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                full.ft_x_n(velocity, values, 3)

    def test_rejects_complex_velocity_or_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "real-valued"):
            full.ft_x_n(self.v.to(torch.complex128), self.f, 3)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            full.ft_x_n(self.v, self.f.to(torch.complex128), 3)


class TestDistributionValidationAndInterpolation(unittest.TestCase):
    def setUp(self) -> None:
        self.v = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64)
        raw = torch.tensor(
            [0.0, 0.2, 0.5, 1.0, 1.5, 1.0, 0.5, 0.2, 0.0],
            dtype=torch.float64,
        )
        self.f = raw / torch.trapz(raw, self.v)

    def test_distribution_accepts_only_real_nonnegative_unit_mass(self) -> None:
        validated_v, validated_f = full._validate_distribution(self.v, self.f, "test")
        torch.testing.assert_close(validated_v, self.v)
        torch.testing.assert_close(validated_f, self.f)

        invalid_cases = []
        negative = self.f.clone()
        negative[4] = -0.01
        invalid_cases.append((self.v, negative))
        invalid_cases.append((self.v, self.f * 2.0))
        invalid_cases.append((self.v, torch.full_like(self.f, torch.nan)))
        invalid_cases.append((self.v.reshape(1, -1), self.f))
        invalid_cases.append((self.v, self.f[:-1]))
        nonmonotonic = self.v.clone()
        nonmonotonic[4] = nonmonotonic[3]
        invalid_cases.append((nonmonotonic, self.f))
        nonuniform = self.v.clone()
        nonuniform[4] += 0.01
        invalid_cases.append((nonuniform, self.f))
        for velocity, values in invalid_cases:
            with self.subTest(shape=(velocity.shape, values.shape)), self.assertRaises(ValueError):
                full._validate_distribution(velocity, values, "test")

    def test_interpolation_endpoints_inside_outside_and_shape(self) -> None:
        query = torch.tensor(
            [[-3.0, -2.0, -1.75], [0.25, 2.0, 3.0]], dtype=torch.float64
        )
        actual = full.interpolate_distribution(self.v, self.f, query)
        expected_inside_left = self.f[0] + 0.5 * (self.f[1] - self.f[0])
        expected_inside_center = self.f[4] + 0.5 * (self.f[5] - self.f[4])
        expected = torch.tensor(
            [
                [0.0, self.f[0].item(), expected_inside_left.item()],
                [expected_inside_center.item(), self.f[-1].item(), 0.0],
            ],
            dtype=torch.float64,
        )
        self.assertEqual(actual.shape, query.shape)
        torch.testing.assert_close(actual, expected)

    def test_interpolation_rejects_nonfinite_or_complex_query(self) -> None:
        with self.assertRaises(ValueError):
            full.interpolate_distribution(self.v, self.f, torch.tensor([torch.nan]))
        with self.assertRaisesRegex(ValueError, "real-valued"):
            full.interpolate_distribution(
                self.v, self.f, torch.tensor([0.0 + 1.0j], dtype=torch.complex128)
            )


class TestPublicForwardModel(unittest.TestCase):
    def _capture_first_projected_geometry(self, positional, keyword):
        original_projected_velocity = full._projected_phase_velocity
        with mock.patch.object(
            full,
            "_projected_phase_velocity",
            wraps=original_projected_velocity,
        ) as phase_velocity_spy:
            full.OTS_spectrum_full_fourier(*positional, **keyword)

        self.assertEqual(phase_velocity_spy.call_count, 3)
        first_call = phase_velocity_spy.call_args_list[0]
        return first_call.args[1], first_call.args[2]

    def test_signature_is_keyword_only_and_old_alias_is_absent(self) -> None:
        signature = inspect.signature(full.OTS_spectrum_full_fourier)
        keyword_only = (
            "v_e",
            "f_e",
            "v_i1",
            "f_i1",
            "v_i2",
            "f_i2",
            "u_e",
            "u_i1",
            "u_i2",
            "fourier_modes",
        )
        for name in keyword_only:
            self.assertEqual(
                signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
            )
        self.assertFalse(hasattr(full, "OTS_spectrum_fft"))

        positional, keyword = make_forward_inputs()
        old_style_tail = tuple(keyword[name] for name in keyword_only[:-1])
        with self.assertRaises(TypeError):
            full.OTS_spectrum_full_fourier(*positional, *old_style_tail)

    def test_different_species_grids_use_three_fourier_and_interpolation_paths(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        original_ft = full.ft_x_n
        original_interpolate = full.interpolate_distribution
        with mock.patch.object(full, "ft_x_n", wraps=original_ft) as ft_spy, mock.patch.object(
            full, "interpolate_distribution", wraps=original_interpolate
        ) as interpolation_spy:
            intensity = full.OTS_spectrum_full_fourier(*positional, **keyword)

        self.assertEqual(ft_spy.call_count, 3)
        self.assertEqual(interpolation_spy.call_count, 3)
        self.assertEqual([len(call.args[0]) for call in ft_spy.call_args_list], [257, 193, 161])
        self.assertEqual(
            [len(call.args[0]) for call in interpolation_spy.call_args_list],
            [257, 193, 161],
        )
        self.assertEqual(intensity.shape, positional[0].shape)
        self.assertEqual(intensity.dtype, torch.float64)
        self.assertEqual(intensity.device.type, "cpu")
        self.assertTrue(torch.isfinite(intensity).all())
        torch.testing.assert_close(
            torch.max(torch.abs(intensity)), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_scattered_wavevector_uses_plasma_dispersion_at_default_density(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        captured_magnitude, captured_components = self._capture_first_projected_geometry(
            positional, keyword
        )

        lms, ss, k_sh = positional[:3]
        lm_0 = torch.as_tensor(positional[4], device=lms.device).to(dtype=lms.dtype)
        n_e = torch.as_tensor(positional[5], device=lms.device).to(dtype=lms.dtype)
        ss = ss / torch.linalg.vector_norm(ss)
        k_sh = k_sh / torch.linalg.vector_norm(k_sh)
        omega_0 = 2.0 * math.pi * full.SPEED_OF_LIGHT / lm_0
        omega_s = 2.0 * math.pi * full.SPEED_OF_LIGHT / lms
        omega_pe_squared = (
            n_e
            * full.ELEMENTARY_CHARGE**2
            / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
        )
        k_incident = torch.sqrt(omega_0.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
        k_scattered = torch.sqrt(omega_s.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
        expected_components = tuple(
            k_scattered * ss[index] - k_incident * k_sh[index]
            for index in range(3)
        )
        expected_magnitude = torch.sqrt(
            sum(component.square() for component in expected_components)
        )

        for actual, expected in zip(captured_components, expected_components):
            torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-6)
        torch.testing.assert_close(
            captured_magnitude, expected_magnitude, rtol=1.0e-12, atol=1.0e-6
        )

        vacuum_components = tuple(
            (omega_s / full.SPEED_OF_LIGHT) * ss[index]
            - k_incident * k_sh[index]
            for index in range(3)
        )
        vacuum_magnitude = torch.sqrt(
            sum(component.square() for component in vacuum_components)
        )
        self.assertGreater(
            torch.max(torch.abs(captured_magnitude - vacuum_magnitude)).item(),
            1.0e4,
        )

    def test_elastic_scattering_has_equal_medium_wave_numbers(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        elastic_wavelengths = torch.as_tensor(positional[4]).to(
            dtype=torch.float64
        ).reshape(1)
        elastic_positional = (elastic_wavelengths,) + positional[1:]
        captured_magnitude, captured_components = self._capture_first_projected_geometry(
            elastic_positional, keyword
        )

        ss = positional[1] / torch.linalg.vector_norm(positional[1])
        k_sh = positional[2] / torch.linalg.vector_norm(positional[2])
        lm_0 = torch.as_tensor(positional[4]).to(dtype=torch.float64)
        n_e = torch.as_tensor(positional[5]).to(dtype=torch.float64)
        omega = 2.0 * math.pi * full.SPEED_OF_LIGHT / lm_0
        omega_s = 2.0 * math.pi * full.SPEED_OF_LIGHT / elastic_wavelengths
        omega_pe_squared = (
            n_e
            * full.ELEMENTARY_CHARGE**2
            / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
        )
        k_incident = torch.sqrt(omega.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
        k_scattered = torch.sqrt(omega_s.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
        torch.testing.assert_close(
            k_scattered,
            k_incident.expand_as(k_scattered),
            rtol=0.0,
            atol=0.0,
        )

        expected_components = tuple(
            k_scattered * ss[index] - k_incident * k_sh[index]
            for index in range(3)
        )
        for actual, expected in zip(captured_components, expected_components):
            torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-6)

        angle = torch.acos(torch.clamp(torch.dot(k_sh, ss), min=-1.0, max=1.0))
        expected_magnitude = 2.0 * k_incident * torch.sin(angle / 2.0)
        torch.testing.assert_close(
            captured_magnitude,
            expected_magnitude.expand_as(captured_magnitude),
            rtol=1.0e-12,
            atol=1.0e-6,
        )

    def test_rejects_scattered_wavelength_below_plasma_cutoff(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        omega_pe = math.sqrt(
            positional[5]
            * full.ELEMENTARY_CHARGE**2
            / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
        )
        omega_0 = 2.0 * math.pi * full.SPEED_OF_LIGHT / positional[4]
        self.assertGreater(omega_0, omega_pe)
        cutoff_wavelength = 2.0 * math.pi * full.SPEED_OF_LIGHT / omega_pe
        invalid_positional = (
            torch.tensor([1.01 * cutoff_wavelength], dtype=torch.float64),
        ) + positional[1:]

        with self.assertRaisesRegex(ValueError, "Scattered-light"):
            full.OTS_spectrum_full_fourier(*invalid_positional, **keyword)

    def test_float32_precision_is_inherited_from_lms(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        positional = (positional[0].float(),) + positional[1:]
        intensity = full.OTS_spectrum_full_fourier(*positional, **keyword)
        self.assertEqual(intensity.dtype, torch.float32)
        self.assertTrue(torch.isfinite(intensity).all())

    def test_drift_changes_normalized_spectrum(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        baseline = full.OTS_spectrum_full_fourier(*positional, **keyword)
        drifted_keyword = dict(keyword)
        drifted_keyword.update(
            {
                "u_e": torch.tensor([2.0e6, -1.0e6, 0.0], dtype=torch.float64),
                "u_i1": torch.tensor([5.0e4, 0.0, 0.0], dtype=torch.float64),
                "u_i2": torch.tensor([-3.0e4, 2.0e4, 0.0], dtype=torch.float64),
            }
        )
        drifted = full.OTS_spectrum_full_fourier(*positional, **drifted_keyword)
        self.assertTrue(torch.isfinite(drifted).all())
        self.assertGreater(torch.max(torch.abs(drifted - baseline)).item(), 1.0e-8)

    def test_three_wavelength_segments_are_each_finite_and_independently_normalized(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        bands = (
            torch.linspace(349.0e-9, 353.0e-9, 17, dtype=torch.float64),
            torch.linspace(375.3552e-9, 599.7064e-9, 17, dtype=torch.float64),
            torch.linspace(100.1144e-9, 339.6504e-9, 17, dtype=torch.float64),
        )
        for band in bands:
            with self.subTest(bounds=(band[0].item(), band[-1].item())):
                band_positional = (band,) + positional[1:]
                intensity = full.OTS_spectrum_full_fourier(
                    *band_positional, **keyword
                )
                self.assertEqual(intensity.shape, band.shape)
                self.assertTrue(torch.isfinite(intensity).all())
                torch.testing.assert_close(
                    torch.max(torch.abs(intensity)),
                    torch.tensor(1.0, dtype=torch.float64),
                )

    def test_direction_vectors_are_normalized_internally(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        baseline = full.OTS_spectrum_full_fourier(*positional, **keyword)
        scaled = list(positional)
        scaled[1] = positional[1] * 7.0
        scaled[2] = positional[2] * 0.125
        actual = full.OTS_spectrum_full_fourier(*scaled, **keyword)
        torch.testing.assert_close(actual, baseline, rtol=1.0e-11, atol=1.0e-12)

    def test_public_api_rejects_invalid_species_distributions_and_mode_count(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        cases = []

        unnormalized = keyword["f_e"] * 2.0
        cases.append({"f_e": unnormalized})
        negative = keyword["f_i1"].clone()
        negative[len(negative) // 2] = -1.0
        cases.append({"f_i1": negative})
        nonfinite = keyword["f_i2"].clone()
        nonfinite[2] = torch.nan
        cases.append({"f_i2": nonfinite})
        bad_shape = keyword["f_e"].reshape(1, -1)
        cases.append({"f_e": bad_shape})
        nonmonotonic = keyword["v_i1"].clone()
        nonmonotonic[10] = nonmonotonic[9]
        cases.append({"v_i1": nonmonotonic})
        nonuniform = keyword["v_i2"].clone()
        nonuniform[10] += 0.01 * (nonuniform[1] - nonuniform[0])
        cases.append({"v_i2": nonuniform})
        cases.append({"fourier_modes": 0})
        cases.append({"fourier_modes": len(keyword["v_i2"]) + 1})

        for replacement in cases:
            with self.subTest(replacement=next(iter(replacement))), self.assertRaises(ValueError):
                invalid_keyword = dict(keyword)
                invalid_keyword.update(replacement)
                full.OTS_spectrum_full_fourier(*positional, **invalid_keyword)

    def test_public_api_rejects_invalid_geometry_and_physical_scalars(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        cases = []

        invalid_lms = positional[0].clone()
        invalid_lms[0] = 0.0
        cases.append((0, invalid_lms))
        cases.append((1, torch.zeros(3, dtype=torch.float64)))
        cases.append((2, torch.zeros(3, dtype=torch.float64)))
        cases.append((3, torch.tensor([0.0, torch.nan, 1.0], dtype=torch.float64)))
        cases.append((5, -1.0))
        cases.append((7, -0.1))
        cases.append((8, 0.0))
        cases.append((10, -0.1))
        cases.append((11, 0.0))
        cases.append((12, 0.0))

        for index, invalid_value in cases:
            with self.subTest(positional_index=index), self.assertRaises(ValueError):
                invalid_positional = list(positional)
                invalid_positional[index] = invalid_value
                full.OTS_spectrum_full_fourier(*invalid_positional, **keyword)

        invalid_drift = dict(keyword)
        invalid_drift["u_e"] = torch.tensor([0.0, torch.inf, 0.0])
        with self.assertRaises(ValueError):
            full.OTS_spectrum_full_fourier(*positional, **invalid_drift)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_lms_device_controls_all_computation_and_output(self) -> None:
        positional, keyword = make_forward_inputs(torch.float64)
        positional = (positional[0].cuda(),) + positional[1:]
        intensity = full.OTS_spectrum_full_fourier(*positional, **keyword)
        self.assertEqual(intensity.device.type, "cuda")
        self.assertEqual(intensity.dtype, torch.float64)
        self.assertTrue(torch.isfinite(intensity).all())


class TestEquation9Response(unittest.TestCase):
    @staticmethod
    def _quadrature_reference(values: np.ndarray) -> np.ndarray:
        nodes, weights = np.polynomial.legendre.leggauss(256)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        return values * np.sum(
            weights * np.exp(-(values[:, None] ** 2) * (1.0 - nodes[None, :] ** 2)),
            axis=1,
        )

    @staticmethod
    def _asymptotic_reference(values: np.ndarray) -> np.ndarray:
        term = np.ones_like(values)
        series = np.ones_like(values)
        for order in range(1, 30):
            term = term * (2 * order - 1) / (2.0 * values**2)
            series = series + term
        return series / (2.0 * values)

    def test_zero_parity_and_equation_9_imaginary_sign(self) -> None:
        x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64)
        dawson = full._dawson_integral_eq9(x)
        response = full._maxwellian_response_eq9(x)
        self.assertEqual(dawson.dtype, torch.float64)
        self.assertEqual(response.dtype, torch.complex128)
        torch.testing.assert_close(dawson, -torch.flip(dawson, dims=(0,)))
        torch.testing.assert_close(response.real, torch.flip(response.real, dims=(0,)))
        torch.testing.assert_close(response.imag, -torch.flip(response.imag, dims=(0,)))
        torch.testing.assert_close(dawson[2], torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(
            response[2], torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        )
        self.assertGreater(response.imag[-1].item(), 0.0)

    def test_quadrature_matches_independent_dense_reference(self) -> None:
        values = np.linspace(-8.0, 8.0, 4097, dtype=np.float64)
        expected = self._quadrature_reference(values)
        for dtype, tolerance in ((torch.float64, 1.0e-11), (torch.float32, 2.0e-5)):
            with self.subTest(dtype=dtype):
                actual = full._dawson_integral_eq9(torch.tensor(values, dtype=dtype))
                error = np.max(np.abs(actual.double().numpy() - expected))
                self.assertLessEqual(error, tolerance)

    def test_large_arguments_match_higher_order_asymptotic_reference(self) -> None:
        positive = np.array([10.0, 20.0, 100.0, 1000.0], dtype=np.float64)
        values = np.concatenate((-positive[::-1], positive))
        expected = self._asymptotic_reference(values)
        response_term = 1.0 / (2.0 * values**2)
        response_tail = response_term.copy()
        for order in range(2, 30):
            response_term = response_term * (2 * order - 1) / (2.0 * values**2)
            response_tail = response_tail + response_term
        expected_response = -response_tail + 1j * (
            math.sqrt(math.pi) * values * np.exp(-(values**2))
        )
        for dtype, tolerance in ((torch.float64, 1.0e-10), (torch.float32, 2.0e-5)):
            with self.subTest(dtype=dtype):
                tensor_values = torch.tensor(values, dtype=dtype)
                actual = full._dawson_integral_eq9(tensor_values)
                relative_error = np.max(
                    np.abs((actual.double().numpy() - expected) / expected)
                )
                self.assertLessEqual(relative_error, tolerance)
                actual_response = full._maxwellian_response_eq9(tensor_values)
                response_relative_error = np.max(
                    np.abs(
                        (actual_response.to(torch.complex128).numpy() - expected_response)
                        / expected_response
                    )
                )
                self.assertLessEqual(response_relative_error, tolerance)

    def test_switch_boundary_matches_the_appropriate_reference(self) -> None:
        values = np.array([8.0 * (1.0 - 1.0e-6), 8.0 * (1.0 + 1.0e-6)])
        expected = np.array(
            [
                self._quadrature_reference(values[:1])[0],
                self._asymptotic_reference(values[1:])[0],
            ]
        )
        actual = full._dawson_integral_eq9(
            torch.tensor(values, dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-11)
        self.assertLess(abs(actual[1] - actual[0]), 1.0e-6)

    def test_finite_extremes_and_invalid_inputs(self) -> None:
        extremes = torch.tensor(
            [-1000.0, -100.0, -8.0, 0.0, 8.0, 100.0, 1000.0],
            dtype=torch.float64,
        )
        self.assertTrue(torch.isfinite(full._maxwellian_response_eq9(extremes)).all())
        invalid = (
            [0.0],
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1.0 + 1.0j]),
            torch.tensor([torch.nan]),
            torch.tensor([torch.inf]),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "x"):
                full._dawson_integral_eq9(value)


class TestMaxwellianTwoIonForwardModel(unittest.TestCase):
    def test_signature_and_independent_equation_reference(self) -> None:
        signature = inspect.signature(full.OTS_spectrum_maxwellian_two_ion)
        self.assertEqual(signature.parameters["u_e"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["u_i1"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["u_i2"].kind, inspect.Parameter.KEYWORD_ONLY)
        for forbidden in ("fourier_modes", "v_e", "f_e", "K", "F_e"):
            self.assertNotIn(forbidden, signature.parameters)

        positional, keyword = make_maxwellian_inputs(torch.float64)
        actual = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        expected = reference_maxwellian_spectrum(positional, keyword)
        self.assertEqual(actual.shape, positional[0].shape)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(actual.device.type, "cpu")
        torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-12)
        torch.testing.assert_close(
            torch.max(torch.abs(actual)), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_analytic_path_does_not_call_fourier_helpers(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        forbidden = ("ft_x_n", "fourier_landau_response", "OTS_spectrum_full_fourier")
        patches = [
            mock.patch.object(full, name, side_effect=AssertionError(name))
            for name in forbidden
        ]
        with patches[0], patches[1], patches[2]:
            intensity = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        self.assertTrue(torch.isfinite(intensity).all())

    def test_three_temperatures_independently_change_two_ion_spectrum(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        baseline = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        for index, factor in ((6, 1.1), (10, 1.2), (14, 1.2)):
            with self.subTest(index=index):
                changed = list(positional)
                changed[index] *= factor
                spectrum = full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)
                self.assertTrue(torch.isfinite(spectrum).all())
                self.assertGreater(
                    torch.max(torch.abs(spectrum - baseline)).item(), 1.0e-8
                )

    def test_zero_fraction_species_temperature_is_inert(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        one_ion = list(positional)
        one_ion[12] = 0.0
        baseline = full.OTS_spectrum_maxwellian_two_ion(*one_ion, **keyword)
        changed = list(one_ion)
        changed[14] = 2000.0
        actual = full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)
        torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)

        one_ion = list(positional)
        one_ion[8] = 0.0
        baseline = full.OTS_spectrum_maxwellian_two_ion(*one_ion, **keyword)
        changed = list(one_ion)
        changed[10] = 2000.0
        actual = full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)
        torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)

    def test_swapping_complete_ion_parameter_sets_is_invariant(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        keyword = dict(keyword)
        keyword["u_i1"] = torch.tensor([2.0e4, -1.0e4, 0.0], dtype=torch.float64)
        keyword["u_i2"] = torch.tensor([-3.0e4, 1.5e4, 0.0], dtype=torch.float64)
        baseline = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)

        swapped = list(positional)
        first = tuple(positional[index] for index in (7, 8, 9, 10))
        second = tuple(positional[index] for index in (11, 12, 13, 14))
        for index, value in zip((7, 8, 9, 10), second):
            swapped[index] = value
        for index, value in zip((11, 12, 13, 14), first):
            swapped[index] = value
        swapped_keyword = dict(keyword)
        swapped_keyword["u_i1"], swapped_keyword["u_i2"] = (
            keyword["u_i2"],
            keyword["u_i1"],
        )
        actual = full.OTS_spectrum_maxwellian_two_ion(
            *swapped, **swapped_keyword
        )
        torch.testing.assert_close(actual, baseline, rtol=2.0e-12, atol=2.0e-13)

    def test_nonzero_drift_uses_paired_k_s_minus_k_i_sign(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        keyword = dict(keyword)
        keyword["u_e"] = torch.tensor([1.0e6, -2.0e6, 0.0], dtype=torch.float64)
        original = full._projected_phase_velocity
        with mock.patch.object(
            full, "_projected_phase_velocity", wraps=original
        ) as spy:
            full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        captured_components = spy.call_args_list[0].args[2]

        lms, ss, k_sh = positional[:3]
        ss = ss / torch.linalg.vector_norm(ss)
        k_sh = k_sh / torch.linalg.vector_norm(k_sh)
        omega_0 = 2.0 * math.pi * full.SPEED_OF_LIGHT / positional[4]
        omega_s = 2.0 * math.pi * full.SPEED_OF_LIGHT / lms
        omega_pe_squared = (
            positional[5]
            * full.ELEMENTARY_CHARGE**2
            / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
        )
        k_incident = math.sqrt(omega_0**2 - omega_pe_squared) / full.SPEED_OF_LIGHT
        k_scattered = torch.sqrt(omega_s.square() - omega_pe_squared) / full.SPEED_OF_LIGHT
        expected = tuple(
            k_scattered * ss[index] - k_incident * k_sh[index]
            for index in range(3)
        )
        for actual_component, expected_component in zip(captured_components, expected):
            torch.testing.assert_close(
                actual_component, expected_component, rtol=1.0e-12, atol=1.0e-6
            )

    def test_float32_and_representative_probe_geometries(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float32)
        current = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        self.assertEqual(current.dtype, torch.float32)
        self.assertTrue(torch.isfinite(current).all())

        zhao_style = list(positional)
        zhao_style[0] = torch.linspace(262.8e-9, 263.8e-9, 33, dtype=torch.float32)
        zhao_style[1] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        zhao_style[2] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        zhao_style[4] = 263.3e-9
        zhao_style[5] = 1.0e26
        zhao = full.OTS_spectrum_maxwellian_two_ion(*zhao_style, **keyword)
        self.assertTrue(torch.isfinite(zhao).all())
        torch.testing.assert_close(
            torch.max(torch.abs(zhao)), torch.tensor(1.0, dtype=torch.float32)
        )

    def test_temperature_mass_and_geometry_errors_name_the_field(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        for index, name in ((6, "T_e_ev"), (10, "T_i1_ev"), (14, "T_i2_ev")):
            for invalid in (0.0, -1.0, float("nan"), float("inf"), 1.0 + 1.0j, [1.0, 2.0]):
                with self.subTest(name=name, invalid=invalid):
                    changed = list(positional)
                    changed[index] = invalid
                    with self.assertRaisesRegex(ValueError, name):
                        full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)
        for index, name in ((9, "m_i1"), (13, "m_i2")):
            changed = list(positional)
            changed[index] = 0.0
            with self.assertRaisesRegex(ValueError, name):
                full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)

        for index, name in ((1, "ss"), (2, "k_sh")):
            changed = list(positional)
            changed[index] = torch.zeros(3, dtype=torch.float64)
            with self.assertRaisesRegex(ValueError, name):
                full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)

    def test_other_invalid_inputs_and_zero_power_are_rejected(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        cases = (
            (0, torch.tensor([0.0], dtype=torch.float64), "lms"),
            (3, torch.tensor([0.0, torch.nan, 1.0]), "E_i0"),
            (5, 0.0, "n_e"),
            (8, -0.1, "N_i1"),
            (12, -0.1, "N_i2"),
            (15, 0.0, "V"),
        )
        for index, invalid, message in cases:
            with self.subTest(index=index):
                changed = list(positional)
                changed[index] = invalid
                with self.assertRaisesRegex(ValueError, message):
                    full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)

        changed = list(positional)
        changed[8] = 0.0
        changed[12] = 0.0
        with self.assertRaisesRegex(ValueError, r"Z_i1\*N_i1"):
            full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)

        invalid_keyword = dict(keyword)
        invalid_keyword["u_i1"] = torch.tensor([0.0, torch.inf, 0.0])
        with self.assertRaisesRegex(ValueError, "u_i1"):
            full.OTS_spectrum_maxwellian_two_ion(*positional, **invalid_keyword)

        zero_k = list(positional)
        zero_k[0] = positional[4].reshape(1)
        zero_k[1] = positional[2].clone()
        with self.assertRaisesRegex(ValueError, "wave-vector"):
            full.OTS_spectrum_maxwellian_two_ion(*zero_k, **keyword)

        huge_drift = {
            name: torch.full((3,), 1.0e290, dtype=torch.float64)
            for name in ("u_e", "u_i1", "u_i2")
        }
        with self.assertRaisesRegex(ValueError, "normalization"):
            full.OTS_spectrum_maxwellian_two_ion(*positional, **huge_drift)

    def test_equal_ion_temperatures_still_use_species_specific_thermal_speeds(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        equal_temperature = list(positional)
        equal_temperature[14] = equal_temperature[10].clone()
        original = full._maxwellian_response_eq9
        with mock.patch.object(
            full, "_maxwellian_response_eq9", wraps=original
        ) as spy:
            full.OTS_spectrum_maxwellian_two_ion(
                *equal_temperature, **keyword
            )
        x_i1 = spy.call_args_list[1].args[0]
        x_i2 = spy.call_args_list[2].args[0]
        nonzero = torch.abs(x_i1) > 1.0e-12
        expected_ratio = math.sqrt(
            float(equal_temperature[13] / equal_temperature[9])
        )
        torch.testing.assert_close(
            x_i2[nonzero] / x_i1[nonzero],
            torch.full_like(x_i1[nonzero], expected_ratio),
            rtol=2.0e-12,
            atol=2.0e-13,
        )

    def test_rejects_scattered_wavelength_below_plasma_cutoff(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        omega_pe = math.sqrt(
            positional[5]
            * full.ELEMENTARY_CHARGE**2
            / (full.ELECTRON_MASS * full.VACUUM_PERMITTIVITY)
        )
        cutoff = 2.0 * math.pi * full.SPEED_OF_LIGHT / omega_pe
        changed = list(positional)
        changed[0] = torch.tensor([1.01 * cutoff], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "Scattered-light"):
            full.OTS_spectrum_maxwellian_two_ion(*changed, **keyword)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_output_and_equation_9_stay_on_device(self) -> None:
        positional, keyword = make_maxwellian_inputs(torch.float64)
        positional = (positional[0].cuda(),) + positional[1:]
        intensity = full.OTS_spectrum_maxwellian_two_ion(*positional, **keyword)
        self.assertEqual(intensity.device.type, "cuda")
        self.assertEqual(intensity.dtype, torch.float64)
        self.assertTrue(torch.isfinite(intensity).all())
        response = full._maxwellian_response_eq9(
            torch.tensor([-10.0, 0.0, 10.0], device="cuda", dtype=torch.float64)
        )
        self.assertEqual(response.device.type, "cuda")


class TestSourceContract(unittest.TestCase):
    def test_module_does_not_import_legacy_ots_fourier(self) -> None:
        source_path = PROJECT_ROOT / "phyEqns" / "ots_full_fourier.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        self.assertFalse(
            any(name == "phyEqns.ots_fourier" or name.endswith(".ots_fourier") for name in imported_modules),
            imported_modules,
        )
        self.assertNotIn("scipy", imported_modules)

    def test_analytic_source_does_not_call_fourier_or_legacy_response(self) -> None:
        source = inspect.getsource(full.OTS_spectrum_maxwellian_two_ion)
        for forbidden in (
            "ft_x_n(",
            "fourier_landau_response(",
            "OTS_spectrum_full_fourier(",
            "w_fast(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
