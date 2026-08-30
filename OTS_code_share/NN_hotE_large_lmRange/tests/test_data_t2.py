"""Targeted acceptance tests for canonical-pipeline T2 only."""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeFrame:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, dtype=None, copy=False):
        return np.array(self._values, dtype=dtype, copy=copy)


class _FakeTable:
    def __init__(self, values, names):
        self.values = np.asarray(values)
        self.names = list(names)

    @property
    def num_columns(self):
        return self.values.shape[1]

    @property
    def num_rows(self):
        return self.values.shape[0]

    def select(self, indices):
        indices = list(indices)
        return _FakeTable(self.values[:, indices], [self.names[index] for index in indices])

    def to_pandas(self):
        return _FakeFrame(self.values)


class _FakeTableFactory:
    @staticmethod
    def from_arrays(arrays, names):
        return _FakeTable(np.column_stack([np.asarray(array) for array in arrays]), names)


def _install_arrow_fixture_if_needed():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return False
    except ModuleNotFoundError:
        pa = types.ModuleType("pyarrow")
        pq = types.ModuleType("pyarrow.parquet")
        pa.__path__ = []
        pa.array = lambda values: np.asarray(values)
        pa.Table = _FakeTableFactory

        def write_table(table, path):
            with open(path, "wb") as stream:
                np.savez(stream, values=table.values, names=np.asarray(table.names))

        def read_table(path):
            with np.load(path, allow_pickle=False) as payload:
                return _FakeTable(payload["values"], payload["names"].tolist())

        pq.write_table = write_table
        pq.read_table = read_table
        pa.parquet = pq
        sys.modules["pyarrow"] = pa
        sys.modules["pyarrow.parquet"] = pq
        return True


USING_ARROW_FIXTURE = _install_arrow_fixture_if_needed()
REAL_ARROW_PYTHON = Path(
    os.environ.get(
        "OTS_TEST_REAL_ARROW_PYTHON",
        r"D:\anaconda1\python.exe" if USING_ARROW_FIXTURE else sys.executable,
    )
)

import data  # noqa: E402

# Preserve the unmocked writer.  T3/T4 route writes through
# ``_real_arrow_write`` so the same tests can run both with the in-process
# fake Arrow fixture and with an actual ``pyarrow.Table`` implementation.
_NATIVE_ARROW_WRITE_TABLE = data.pq.write_table


def _fake_sample(rng, physical_config, ion_distributions):
    token = float(rng.random())
    spectrum = torch.full((data.INPUT_DIM,), token, dtype=torch.float64)
    label = torch.arange(data.OUTPUT_DIM, dtype=torch.float64) + token
    k_grid = torch.linspace(-2.0, 2.0, data.FOURIER_MODES, dtype=torch.float64)
    return {
        "spectrum": spectrum,
        "label": label,
        "k_grid": k_grid,
        "moments": {
            "mean_velocity_m_s": token * 100.0,
            "effective_temperature_ev": token * 1000.0,
            "ratio_to_configured": token,
        },
    }


def _valid_rows(physical_samples=3):
    bands = data.build_wavelength_bands(torch.float32)
    wavelengths_nm = (torch.cat(bands) / 1.0e-9).numpy()
    k_grid = np.linspace(-2.0, 2.0, data.FOURIER_MODES, dtype=np.float32)
    rows = np.zeros(
        (physical_samples + 1, data.INPUT_DIM + data.OUTPUT_DIM), dtype=np.float32
    )
    rows[0, : data.INPUT_DIM] = wavelengths_nm
    rows[0, data.INPUT_DIM : data.INPUT_DIM + data.FOURIER_MODES] = k_grid
    rows[0, data.INPUT_DIM + data.FOURIER_MODES :] = k_grid
    for row in range(1, physical_samples + 1):
        rows[row, : data.INPUT_DIM] = row
        rows[row, data.INPUT_DIM :] = row * 10.0
    return rows


def _real_arrow_write(table, path):
    path = Path(path)
    if not hasattr(table, "values") or not hasattr(table, "names"):
        # A real pyarrow.Table intentionally has neither of the fake fixture's
        # convenience attributes.  Use the writer captured before mock.patch
        # replaced ``data.pq.write_table``.
        _NATIVE_ARROW_WRITE_TABLE(table, str(path))
        return
    values_path = Path(str(path) + ".values.npy")
    names_path = Path(str(path) + ".names.json")
    np.save(values_path, table.values)
    names_path.write_text(json.dumps(table.names), encoding="utf-8")
    code = (
        "import json,numpy as np,pyarrow as pa,pyarrow.parquet as pq,sys;"
        "a=np.load(sys.argv[1]);names=json.load(open(sys.argv[2],encoding='utf-8'));"
        "t=pa.Table.from_arrays([pa.array(a[:,i]) for i in range(a.shape[1])],names=names);"
        "pq.write_table(t,sys.argv[3])"
    )
    try:
        subprocess.run(
            [str(REAL_ARROW_PYTHON), "-c", code, str(values_path), str(names_path), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        values_path.unlink(missing_ok=True)
        names_path.unlink(missing_ok=True)


def _write_raw_width_parquet(rows, path):
    """Write a deliberately non-contract-width table with matching schema."""

    path = Path(path)
    names = ["raw_{}".format(index) for index in range(rows.shape[1])]
    arrays = [data.pa.array(rows[:, index]) for index in range(rows.shape[1])]
    table = data.pa.Table.from_arrays(arrays, names=names)
    data.pq.write_table(table, str(path))


def _real_arrow_read(path):
    path = Path(path)
    values_path = Path(str(path) + ".read.npy")
    names_path = Path(str(path) + ".read.json")
    code = (
        "import json,numpy as np,pyarrow.parquet as pq,sys;"
        "t=pq.read_table(sys.argv[1]);"
        "a=np.column_stack([t.column(i).to_numpy(zero_copy_only=False) for i in range(t.num_columns)]);"
        "np.save(sys.argv[2],a);json.dump(t.column_names,open(sys.argv[3],'w',encoding='utf-8'))"
    )
    try:
        subprocess.run(
            [str(REAL_ARROW_PYTHON), "-c", code, str(path), str(values_path), str(names_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        values = np.load(values_path)
        names = json.loads(names_path.read_text(encoding="utf-8"))
        return _FakeTable(values, names)
    finally:
        values_path.unlink(missing_ok=True)
        names_path.unlink(missing_ok=True)


def _assert_no_placeholders(test_case, value, path="manifest"):
    if isinstance(value, dict):
        test_case.assertTrue(value, "empty object at {}".format(path))
        for key, nested in value.items():
            test_case.assertNotEqual(key, "")
            _assert_no_placeholders(test_case, nested, "{}.{}".format(path, key))
    elif isinstance(value, list):
        test_case.assertTrue(value, "empty list at {}".format(path))
        for index, nested in enumerate(value):
            _assert_no_placeholders(test_case, nested, "{}[{}]".format(path, index))
    else:
        test_case.assertIsNotNone(value, "null at {}".format(path))
        if isinstance(value, str):
            test_case.assertTrue(value.strip(), "blank string at {}".format(path))


class TestImportAndSourceContract(unittest.TestCase):
    def test_import_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_cwd = Path.cwd()
            os.chdir(temporary)
            try:
                before = list(Path(temporary).iterdir())
                importlib.reload(data)
                after = list(Path(temporary).iterdir())
            finally:
                os.chdir(original_cwd)
        self.assertEqual(before, [])
        self.assertEqual(after, [])

    def test_data_imports_full_fourier_but_not_legacy_ots_fourier(self):
        source_path = PROJECT_ROOT / "data.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
                imported.extend(
                    "{}.{}".format(node.module, alias.name) for alias in node.names
                )
        self.assertIn("phyEqns.ots_full_fourier", imported)
        self.assertFalse(
            any(name == "phyEqns.ots_fourier" or name.endswith(".ots_fourier") for name in imported),
            imported,
        )


class TestDistributionConstruction(unittest.TestCase):
    def test_normalize_distribution_contract_and_rejections(self):
        velocity = torch.linspace(-2.0, 2.0, 101, dtype=torch.float64)
        values = torch.exp(-velocity.square())
        normalized = data.normalize_distribution(velocity, values, "test")
        torch.testing.assert_close(
            torch.trapz(normalized, velocity), torch.tensor(1.0, dtype=torch.float64)
        )

        invalid = []
        negative = values.clone()
        negative[0] = -1.0
        invalid.append((velocity, negative))
        invalid.append((velocity, torch.zeros_like(values)))
        invalid.append((velocity, torch.full_like(values, torch.nan)))
        invalid.append((velocity.reshape(1, -1), values))
        invalid.append((velocity, values[:-1]))
        nonmonotonic = velocity.clone()
        nonmonotonic[10] = nonmonotonic[9]
        invalid.append((nonmonotonic, values))
        for v_grid, distribution in invalid:
            with self.subTest(shape=(v_grid.shape, distribution.shape)), self.assertRaises(ValueError):
                data.normalize_distribution(v_grid, distribution, "test")

    def test_electron_sampler_reproducibility_ranges_and_component_calls(self):
        first_v, first_f = data.sample_electron_evdf(1000, np.random.default_rng(1234))
        second_v, second_f = data.sample_electron_evdf(1000, np.random.default_rng(1234))
        different_v, different_f = data.sample_electron_evdf(1000, np.random.default_rng(1235))
        torch.testing.assert_close(first_v, second_v)
        torch.testing.assert_close(first_f, second_f)
        torch.testing.assert_close(first_v, different_v)
        self.assertFalse(torch.equal(first_f, different_f))
        self.assertEqual(first_v.shape, (1000,))
        self.assertEqual(first_f.shape, (1000,))
        torch.testing.assert_close(
            first_v[[0, -1]] / data.SPEED_OF_LIGHT,
            torch.tensor([-0.8, 0.8], dtype=torch.float64),
        )
        self.assertTrue(torch.isfinite(first_f).all())
        self.assertTrue(torch.all(first_f >= 0))
        torch.testing.assert_close(
            torch.trapz(first_f, first_v), torch.tensor(1.0, dtype=torch.float64)
        )

        class RecordingRng:
            def __init__(self):
                self.calls = []

            def uniform(self, low, high):
                self.calls.append((low, high))
                return (low + high) / 2.0

        recorder = RecordingRng()
        data.sample_electron_evdf(101, recorder)
        expected_background = [(-0.06, 0.06), (0.01, 0.05), (0.2, 2.5)] * 3
        expected_hot = [(-0.4, 0.4), (0.01, 0.05), (0.001, 0.045)]
        self.assertEqual(recorder.calls, expected_background + expected_hot)

        for invalid_points in (0, 2, 3.0, True):
            with self.subTest(num_points=invalid_points), self.assertRaises(ValueError):
                data.sample_electron_evdf(invalid_points, np.random.default_rng(1))

    def test_fixed_ion_vdfs_have_species_grids_and_unit_mass(self):
        ions = data.build_fixed_ion_distributions(data.ION_SPECIES, 171.3)
        self.assertEqual(set(ions), {"v_i1", "f_i1", "v_i2", "f_i2"})
        self.assertEqual(ions["v_i1"].shape, (1000,))
        self.assertEqual(ions["v_i2"].shape, (1000,))
        self.assertGreater(ions["v_i1"][-1].item(), ions["v_i2"][-1].item())
        for suffix, mass in (("i1", data.PROTON_MASS), ("i2", 12.0 * data.PROTON_MASS)):
            velocity = ions["v_{}".format(suffix)]
            distribution = ions["f_{}".format(suffix)]
            thermal_speed = math.sqrt(171.3 * data.ELEMENTARY_CHARGE / mass)
            torch.testing.assert_close(
                velocity[[0, -1]],
                torch.tensor([-12.0, 12.0], dtype=torch.float64) * thermal_speed,
            )
            self.assertTrue(torch.all(distribution >= 0))
            torch.testing.assert_close(
                torch.trapz(distribution, velocity),
                torch.tensor(1.0, dtype=torch.float64),
            )
        for invalid_temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=invalid_temperature), self.assertRaises(ValueError):
                data.build_fixed_ion_distributions(
                    data.ION_SPECIES, invalid_temperature
                )

    def test_wavelength_contract(self):
        bands = data.build_wavelength_bands(torch.float64)
        self.assertEqual(tuple(len(band) for band in bands), (579, 304, 341))
        self.assertEqual(sum(len(band) for band in bands), data.INPUT_DIM)
        expected_bounds = ((349.0, 353.0), (375.3552, 599.7064), (100.1144, 339.6504))
        for band, bounds in zip(bands, expected_bounds):
            torch.testing.assert_close(
                band[[0, -1]] / 1.0e-9,
                torch.tensor(bounds, dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            data.build_wavelength_bands(torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_normalization_and_moments_accept_cuda_tensors(self):
        velocity = torch.linspace(-5.0, 5.0, 1001, dtype=torch.float64, device="cuda")
        distribution = data.normalize_distribution(
            velocity, torch.exp(-velocity.square()), "cuda"
        )
        self.assertEqual(distribution.device.type, "cuda")
        moments = data.electron_moments(velocity, distribution, 1370.0)
        self.assertTrue(all(math.isfinite(value) for value in moments.values()))


class TestMomentsAndStatistics(unittest.TestCase):
    def test_centered_second_moment_and_ratio_without_sample_mutation(self):
        mean_velocity = 2.5e6
        sigma = 4.0e6
        velocity = torch.linspace(
            mean_velocity - 12.0 * sigma,
            mean_velocity + 12.0 * sigma,
            10001,
            dtype=torch.float64,
        )
        distribution = torch.exp(-0.5 * ((velocity - mean_velocity) / sigma) ** 2)
        distribution = data.normalize_distribution(velocity, distribution, "electron")
        original = distribution.clone()
        configured_ev = 1.0
        moments = data.electron_moments(velocity, distribution, configured_ev)
        expected_temperature = data.ELECTRON_MASS * sigma**2 / data.ELEMENTARY_CHARGE
        self.assertAlmostEqual(moments["mean_velocity_m_s"], mean_velocity, delta=1.0e-8 * mean_velocity)
        self.assertAlmostEqual(
            moments["effective_temperature_ev"], expected_temperature, delta=1.0e-8 * expected_temperature
        )
        self.assertAlmostEqual(moments["ratio_to_configured"], expected_temperature, delta=1.0e-8 * expected_temperature)
        torch.testing.assert_close(distribution, original)
        self.assertGreater(moments["ratio_to_configured"], 10.0)
        for invalid in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(configured_ev=invalid), self.assertRaises(ValueError):
                data.electron_moments(velocity, distribution, invalid)

    def test_temperature_statistics_are_population_statistics(self):
        records = [
            {"mean_velocity_m_s": 1.0, "effective_temperature_ev": 10.0, "ratio_to_configured": 0.5},
            {"mean_velocity_m_s": 3.0, "effective_temperature_ev": 30.0, "ratio_to_configured": 1.5},
            {"mean_velocity_m_s": 8.0, "effective_temperature_ev": 80.0, "ratio_to_configured": 4.0},
        ]
        statistics = data.temperature_statistics(records)
        self.assertEqual(
            set(statistics),
            {"mean_velocity_m_s", "effective_temperature_ev", "ratio_to_configured"},
        )
        self.assertEqual(set(statistics["mean_velocity_m_s"]), {"min", "max", "mean", "median", "std"})
        self.assertAlmostEqual(statistics["mean_velocity_m_s"]["std"], float(np.std([1.0, 3.0, 8.0], ddof=0)))
        for invalid in ([], [{}], [{"mean_velocity_m_s": math.nan, "effective_temperature_ev": 1.0, "ratio_to_configured": 1.0}]):
            with self.subTest(records=invalid), self.assertRaises(ValueError):
                data.temperature_statistics(invalid)


class TestRealFullFourierSample(unittest.TestCase):
    def test_one_real_sample_uses_three_full_fourier_calls_and_contract_shapes(self):
        config = data._physical_config()
        ions = data.build_fixed_ion_distributions(
            data.ION_SPECIES, data.ION_REFERENCE_TEMPERATURE_EV
        )
        original_forward = data.ots_full_fourier.OTS_spectrum_full_fourier
        with mock.patch.object(
            data.ots_full_fourier,
            "OTS_spectrum_full_fourier",
            wraps=original_forward,
        ) as forward_spy:
            sample = data.generate_sample(np.random.default_rng(20260710), config, ions)
        self.assertEqual(forward_spy.call_count, 3)
        self.assertEqual(sample["spectrum"].shape, (1224,))
        self.assertEqual(sample["label"].shape, (160,))
        self.assertEqual(sample["k_grid"].shape, (80,))
        self.assertTrue(torch.isfinite(sample["spectrum"]).all())
        self.assertTrue(torch.isfinite(sample["label"]).all())
        for segment in (sample["spectrum"][:579], sample["spectrum"][579:883], sample["spectrum"][883:]):
            torch.testing.assert_close(
                torch.max(torch.abs(segment)), torch.tensor(1.0, dtype=segment.dtype)
            )
        for call in forward_spy.call_args_list:
            self.assertEqual(call.kwargs["fourier_modes"], 80)
            self.assertIs(call.kwargs["v_i1"], ions["v_i1"])
            self.assertIs(call.kwargs["f_i1"], ions["f_i1"])
            self.assertIs(call.kwargs["v_i2"], ions["v_i2"])
            self.assertIs(call.kwargs["f_i2"], ions["f_i2"])

    @unittest.skipUnless(
        USING_ARROW_FIXTURE and REAL_ARROW_PYTHON.exists(),
        "separate real-PyArrow runtime is unavailable or unnecessary",
    )
    def test_real_one_plus_one_dataset_pair_full_path(self):
        seed = 20260710
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            data.pq, "write_table", side_effect=_real_arrow_write
        ), mock.patch.object(data.pq, "read_table", side_effect=_real_arrow_read):
            result = data.generate_dataset_pair(
                "REAL_RUN", "REAL_DATA", temporary, 1, 1, seed
            )
            train = data.load_dataset(result["train_path"])
            test = data.load_dataset(result["test_path"])
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(train.spectra.shape, (1, 1224))
        self.assertEqual(test.spectra.shape, (1, 1224))
        self.assertEqual(train.coefficients.shape, (1, 160))
        self.assertEqual(test.coefficients.shape, (1, 160))
        self.assertTrue(torch.isfinite(train.spectra).all())
        self.assertTrue(torch.isfinite(test.spectra).all())
        self.assertFalse(torch.equal(train.coefficients, test.coefficients))
        for split_name, split_seed in (("train", seed), ("test", seed + 1)):
            velocity, distribution = data.sample_electron_evdf(
                data.ELECTRON_VELOCITY_POINTS, np.random.default_rng(split_seed)
            )
            expected = data.electron_moments(
                velocity, distribution, data.ELECTRON_REFERENCE_TEMPERATURE_EV
            )
            recorded = manifest["splits"][split_name]["samples"][0]
            self.assertEqual(recorded["row"], 1)
            for key, value in expected.items():
                self.assertAlmostEqual(recorded[key], value, delta=max(abs(value) * 1.0e-12, 1.0e-12))


class TestDatasetPairAndManifest(unittest.TestCase):
    def test_reproducible_seed_isolation_manifest_contract_and_loading(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            with mock.patch.object(data, "generate_sample", side_effect=_fake_sample):
                first = data.generate_dataset_pair("RUN", "DATA", first_dir, 2, 2, 1234)
                second = data.generate_dataset_pair("RUN", "DATA", second_dir, 2, 2, 1234)

            first_train = data.load_dataset(first["train_path"])
            first_test = data.load_dataset(first["test_path"])
            second_train = data.load_dataset(second["train_path"])
            second_test = data.load_dataset(second["test_path"])
            torch.testing.assert_close(first_train.spectra, second_train.spectra)
            torch.testing.assert_close(first_train.coefficients, second_train.coefficients)
            torch.testing.assert_close(first_test.spectra, second_test.spectra)
            torch.testing.assert_close(first_test.coefficients, second_test.coefficients)
            self.assertFalse(torch.equal(first_train.spectra, first_test.spectra))

            manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(set(manifest), {"run_id", "dataset_id", "physical_configuration", "splits"})
            self.assertEqual(manifest["run_id"], "RUN")
            self.assertEqual(manifest["dataset_id"], "DATA")
            physical = manifest["physical_configuration"]
            self.assertEqual(
                set(physical),
                {
                    "fourier_modes",
                    "probe_wavelength_nm",
                    "scattering_angle_deg",
                    "electron_density_m3",
                    "electron_dis",
                    "wavelength_bands_nm",
                },
            )
            self.assertEqual(
                set(physical["electron_dis"]),
                {"configured_reference_temperature_ev", "velocity_points", "velocity_range_c"},
            )
            self.assertEqual(tuple(item["points"] for item in physical["wavelength_bands_nm"]), (579, 304, 341))
            for split_name, expected_seed in (("train", 1234), ("test", 1235)):
                split = manifest["splits"][split_name]
                self.assertEqual(
                    set(split),
                    {"file", "seed", "physical_samples", "samples", "statistics"},
                )
                self.assertEqual(split["seed"], expected_seed)
                self.assertEqual(split["physical_samples"], 2)
                self.assertEqual([sample["row"] for sample in split["samples"]], [1, 2])
                self.assertEqual(
                    set(split["samples"][0]),
                    {"row", "mean_velocity_m_s", "effective_temperature_ev", "ratio_to_configured"},
                )
                self.assertEqual(split["file"], "dataset_{}.parquet".format(split_name))
                self.assertEqual(
                    split["statistics"], data.temperature_statistics(split["samples"])
                )
            serialized = json.dumps(manifest, sort_keys=True)
            for forbidden in (
                "checkpoint_id",
                "format_version",
                "family",
                "temperature_definition",
                "source_path",
                "forward_model",
                "target_kind",
                "target-kind",
            ):
                self.assertNotIn(forbidden, serialized)
            _assert_no_placeholders(self, manifest)

    def test_preflight_refuses_overwrite_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            existing = output_dir / "dataset_train.parquet"
            existing.write_bytes(b"user-data")
            with mock.patch.object(data, "_physical_config") as config_spy:
                with self.assertRaises(FileExistsError):
                    data.generate_dataset_pair("RUN", "DATA", output_dir, 1, 1, 1)
            config_spy.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"user-data")

    def test_atomic_paths_and_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_write = data.pq.write_table
            written_paths = []

            def recording_write(table, path):
                written_paths.append(Path(path))
                return original_write(table, path)

            with mock.patch.object(data, "generate_sample", side_effect=_fake_sample), mock.patch.object(
                data.pq, "write_table", side_effect=recording_write
            ):
                data.generate_dataset_pair("RUN", "DATA", temporary, 1, 1, 9)
            self.assertEqual(len(written_paths), 2)
            self.assertTrue(all(path.name.endswith(".parquet.tmp") for path in written_paths))
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

            with mock.patch.object(data, "generate_sample", side_effect=_fake_sample):
                data.generate_dataset_pair("RUN", "DATA", temporary, 1, 1, 10, overwrite=True)
            manifest = json.loads((Path(temporary) / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["splits"]["train"]["seed"], 10)

    def test_invalid_ids_counts_seed_and_overwrite(self):
        invalid_calls = (
            ("", "DATA", 1, 1, 1, False),
            ("RUN", " ", 1, 1, 1, False),
            ("RUN", "DATA", 0, 1, 1, False),
            ("RUN", "DATA", 1, 0, 1, False),
            ("RUN", "DATA", 1, 1, 1.5, False),
            ("RUN", "DATA", 1, 1, 1, "yes"),
        )
        for index, arguments in enumerate(invalid_calls):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(index=index):
                with self.assertRaises((ValueError, TypeError)):
                    data.generate_dataset_pair(*arguments[:2], temporary, *arguments[2:])
                self.assertEqual(list(Path(temporary).glob("dataset_*")), [])


class TestMetadataFirstLoading(unittest.TestCase):
    def test_loads_all_physical_rows_including_last_as_cpu_float32(self):
        rows = _valid_rows(3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.parquet"
            data._write_parquet(rows, path)
            dataset = data.load_dataset(path)
        self.assertEqual(dataset.spectra.shape, (3, 1224))
        self.assertEqual(dataset.coefficients.shape, (3, 160))
        self.assertEqual(dataset.wavelengths_nm.shape, (1224,))
        self.assertEqual(dataset.k_grid.shape, (80,))
        self.assertEqual(dataset.spectra.dtype, torch.float32)
        self.assertEqual(dataset.spectra.device.type, "cpu")
        torch.testing.assert_close(dataset.spectra[-1], torch.full((1224,), 3.0))
        torch.testing.assert_close(dataset.coefficients[-1], torch.full((160,), 30.0))

    @unittest.skipUnless(
        USING_ARROW_FIXTURE and REAL_ARROW_PYTHON.exists(),
        "separate real-PyArrow runtime is unavailable or unnecessary",
    )
    def test_real_pyarrow_codec_round_trip_through_production_paths(self):
        rows = _valid_rows(2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real_dataset.parquet"
            with mock.patch.object(data.pq, "write_table", side_effect=_real_arrow_write), mock.patch.object(
                data.pq, "read_table", side_effect=_real_arrow_read
            ):
                data._write_parquet(rows, path)
                dataset = data.load_dataset(path)
            payload = path.read_bytes()

        self.assertEqual(payload[:4], b"PAR1")
        self.assertEqual(payload[-4:], b"PAR1")
        self.assertEqual(dataset.spectra.shape, (2, 1224))
        self.assertEqual(dataset.coefficients.shape, (2, 160))
        torch.testing.assert_close(dataset.spectra[-1], torch.full((1224,), 2.0))

    def test_rejects_missing_physical_row_short_columns_nonfinite_and_bad_metadata(self):
        cases = []
        cases.append(_valid_rows(0))
        cases.append(_valid_rows(1)[:, :-1])
        nonfinite = _valid_rows(1)
        nonfinite[1, 0] = np.nan
        cases.append(nonfinite)
        mismatched_k = _valid_rows(1)
        mismatched_k[0, data.INPUT_DIM + data.FOURIER_MODES] += 1.0
        cases.append(mismatched_k)
        bad_wavelength = _valid_rows(1)
        bad_wavelength[0, 0] += 10.0
        cases.append(bad_wavelength)

        for index, rows in enumerate(cases):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(index=index):
                path = Path(temporary) / "dataset.parquet"
                if index == 1:
                    # Production _write_parquet is a valid-contract writer and
                    # correctly fixes its schema at 1384 names.  Bypass it to
                    # create a genuine 1383-column input for the reader test.
                    _write_raw_width_parquet(rows, path)
                    with self.assertRaisesRegex(ValueError, "at least 1384"):
                        data.load_dataset(path)
                else:
                    data._write_parquet(rows, path)
                    with self.assertRaises(ValueError):
                        data.load_dataset(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
