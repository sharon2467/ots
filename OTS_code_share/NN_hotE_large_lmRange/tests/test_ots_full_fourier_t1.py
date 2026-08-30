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
            k_incident * k_sh[index] - k_scattered * ss[index]
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
            k_incident * k_sh[index]
            - (omega_s / full.SPEED_OF_LIGHT) * ss[index]
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
            k_incident * k_sh[index] - k_scattered * ss[index]
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
