"""Targeted acceptance tests for canonical-pipeline T4 only."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_model_train_t3 as t3_fixture  # noqa: E402

data = t3_fixture.data
model_module = t3_fixture.model

import evaluation  # noqa: E402


class RecordingAxis:
    def __init__(self):
        self.plots = []
        self.titles = []
        self.axis_calls = []

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def set_title(self, value):
        self.titles.append(value)

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def axis(self, value):
        self.axis_calls.append(value)


class RecordingFigure:
    def __init__(self):
        self.suptitles = []

    def suptitle(self, value):
        self.suptitles.append(value)

    def tight_layout(self):
        return None

    def savefig(self, path, dpi=None):
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\nT4-fixture")


class RecordingPyplot:
    def __init__(self):
        self.figures = []
        self.axes = []

    def subplots(self, rows, columns, figsize=None, squeeze=True):
        figure = RecordingFigure()
        class AxesGrid(list):
            def __getitem__(self, index):
                if isinstance(index, tuple):
                    row, column = index
                    return list.__getitem__(self, row)[column]
                return list.__getitem__(self, index)

        axes = AxesGrid([[RecordingAxis() for _ in range(columns)] for _ in range(rows)])
        self.figures.append(figure)
        self.axes.append(axes)
        if squeeze and rows == 1:
            axes = axes[0]
        return figure, axes

    def close(self, figure):
        return None


PLOT_FIXTURE = RecordingPyplot()
evaluation.plt = PLOT_FIXTURE


def setUpModule():
    # T2 intentionally reloads data during the combined suite. Refresh the
    # evaluation from-import after that isolated test before T4 starts.
    evaluation.load_dataset = data.load_dataset
    evaluation.plt = PLOT_FIXTURE


def make_dataset(sample_count=3):
    rows = t3_fixture.t2_fixture._valid_rows(sample_count)
    values = torch.from_numpy(rows)
    return data.OTSDataset(
        spectra=values[1:, :1224].clone(),
        coefficients=values[1:, 1224:].clone(),
        wavelengths_nm=values[0, :1224].clone(),
        k_grid=values[0, 1224:1304].clone(),
    )


def write_real_test_parquet(directory, sample_count=3):
    rows = t3_fixture.t2_fixture._valid_rows(sample_count)
    path = Path(directory) / "dataset_test.parquet"
    with mock.patch.object(
        data.pq, "write_table", side_effect=t3_fixture.t2_fixture._real_arrow_write
    ):
        data._write_parquet(rows, path)
    return path


def write_evaluation_manifest(directory, test_path, physical_samples, run_id="RUN", dataset_id="DATA"):
    path = Path(directory) / "dataset_manifest.json"
    payload = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "physical_configuration": {"fixture": True},
        "splits": {
            "train": {"file": "MUST_NOT_BE_OPENED.parquet"},
            "test": {
                "file": os.path.relpath(test_path, path.parent),
                "physical_samples": physical_samples,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_checkpoint(directory, payload=None):
    if payload is None:
        payload = t3_fixture.valid_checkpoint_payload()
    path = Path(directory) / "model.pt"
    torch.save(payload, path)
    return path


class TestReconstruction(unittest.TestCase):
    def test_explicit_real_inverse_matches_independent_trapezoid(self):
        generator = torch.Generator().manual_seed(44)
        coefficients = torch.randn(3, 160, generator=generator, dtype=torch.float64)
        k_grid = torch.linspace(-2.0e-6, 2.0e-6, 80, dtype=torch.float64)
        velocity, actual = evaluation.reconstruct_real_evdf(coefficients, k_grid)
        real = coefficients[:, :80]
        imaginary = coefficients[:, 80:]
        phase = k_grid[:, None] * velocity[None, :]
        integrand = (
            real[:, :, None] * torch.cos(phase)[None, :, :]
            - imaginary[:, :, None] * torch.sin(phase)[None, :, :]
        )
        expected = torch.trapz(integrand, k_grid, dim=1) / (2.0 * math.pi)
        self.assertEqual(actual.shape, (3, 80))
        self.assertEqual(velocity.shape, (80,))
        self.assertFalse(torch.is_complex(actual))
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_reconstruction_rejects_invalid_coefficients_and_k(self):
        valid_coefficients = torch.zeros(1, 160)
        valid_k = torch.linspace(-2.0, 2.0, 80)
        invalid = [
            (torch.zeros(160), valid_k),
            (torch.zeros(0, 160), valid_k),
            (torch.zeros(1, 159), valid_k),
            (torch.full((1, 160), torch.nan), valid_k),
            (valid_coefficients, valid_k[:-1]),
            (valid_coefficients, torch.flip(valid_k, (0,))),
        ]
        nonuniform = valid_k.clone()
        nonuniform[10] += 0.1
        invalid.append((valid_coefficients, nonuniform))
        for coefficients, k_grid in invalid:
            with self.subTest(shapes=(coefficients.shape, k_grid.shape)), self.assertRaises(ValueError):
                evaluation.reconstruct_real_evdf(coefficients, k_grid)


class SpyNormalizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_enabled = []
        self.training_seen = []

    def forward(self, spectra):
        self.inputs.append(spectra.detach().cpu().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_seen.append(self.training)
        return spectra[:, :160]


class TestPrediction(unittest.TestCase):
    def _checkpoint(self, dtype="float64"):
        tensor_dtype = torch.float64 if dtype == "float64" else torch.float32
        return {
            "dtype": dtype,
            "x_mean": torch.full((1224,), 2.0, dtype=tensor_dtype),
            "x_scale": torch.full((1224,), 4.0, dtype=tensor_dtype),
            "y_mean": torch.linspace(-1.0, 1.0, 160, dtype=tensor_dtype),
            "y_scale": torch.linspace(0.5, 1.5, 160, dtype=tensor_dtype),
        }

    def test_eval_no_grad_batching_standardize_inverse_then_project(self):
        spy = SpyNormalizedModel()
        spy.train()
        spectra = torch.arange(5 * 1224, dtype=torch.float64).reshape(5, 1224) / 1000.0
        checkpoint = self._checkpoint()
        actual = evaluation.predict_coefficients(spy, spectra, checkpoint, 2, "cpu")
        normalized = (spectra - checkpoint["x_mean"]) / checkpoint["x_scale"]
        raw = normalized[:, :160] * checkpoint["y_scale"] + checkpoint["y_mean"]
        expected = model_module.hermitian_project(raw)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([len(batch) for batch in spy.inputs], [2, 2, 1])
        torch.testing.assert_close(torch.cat(spy.inputs), normalized)
        self.assertTrue(all(value is False for value in spy.grad_enabled))
        self.assertTrue(all(value is False for value in spy.training_seen))
        self.assertFalse(spy.training)
        self.assertEqual(actual.device.type, "cpu")
        torch.testing.assert_close(actual[:, :80], torch.flip(actual[:, :80], (1,)))
        torch.testing.assert_close(actual[:, 80:], -torch.flip(actual[:, 80:], (1,)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_prediction_path_returns_cpu_projection(self):
        spy = SpyNormalizedModel().cuda()
        spectra = torch.randn(3, 1224)
        actual = evaluation.predict_coefficients(spy, spectra, self._checkpoint("float32"), 2, "cuda")
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(all(batch.dtype == torch.float32 for batch in spy.inputs))


class TestMetrics(unittest.TestCase):
    def test_fixed_metric_set_perfect_and_zero_denominator_are_finite(self):
        k_grid = torch.linspace(-2.0e-6, 2.0e-6, 80, dtype=torch.float64)
        real_half = torch.linspace(0.1, 1.0, 80, dtype=torch.float64)
        real_half = 0.5 * (real_half + torch.flip(real_half, (0,)))
        target = torch.cat((real_half, torch.zeros(80, dtype=torch.float64)))[None, :]
        metrics = evaluation.prediction_metrics(target, target.clone(), k_grid)
        self.assertEqual(
            set(metrics),
            {
                "coefficient_mae",
                "coefficient_rmse",
                "coefficient_r2",
                "hermitian_symmetry_mae",
                "evdf_mae",
                "evdf_rmse",
                "evdf_r2",
                "evdf_relative_l2_mean",
                "evdf_relative_l2_median",
                "negative_mass_fraction_mean",
            },
        )
        self.assertEqual(metrics["coefficient_mae"], 0.0)
        self.assertEqual(metrics["coefficient_rmse"], 0.0)
        self.assertEqual(metrics["coefficient_r2"], 1.0)
        self.assertEqual(metrics["evdf_mae"], 0.0)
        self.assertEqual(metrics["evdf_rmse"], 0.0)
        self.assertEqual(metrics["evdf_r2"], 1.0)
        self.assertEqual(metrics["evdf_relative_l2_mean"], 0.0)
        self.assertEqual(metrics["evdf_relative_l2_median"], 0.0)
        self.assertEqual(metrics["hermitian_symmetry_mae"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

        zeros = torch.zeros(2, 160, dtype=torch.float64)
        zero_metrics = evaluation.prediction_metrics(zeros, zeros, k_grid)
        self.assertTrue(all(math.isfinite(value) for value in zero_metrics.values()))
        self.assertEqual(zero_metrics["coefficient_r2"], 1.0)
        self.assertEqual(zero_metrics["evdf_r2"], 1.0)
        self.assertEqual(zero_metrics["negative_mass_fraction_mean"], 0.0)

    def test_relative_l2_median_uses_conventional_even_sample_definition(self):
        k_grid = torch.linspace(-2.0e-6, 2.0e-6, 80, dtype=torch.float64)
        base = torch.cat((torch.ones(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)))
        target = torch.stack((base, base))
        # Relative-L2 errors are exactly 1 and 3, whose conventional median is 2.
        predicted = torch.stack((torch.zeros_like(base), -2.0 * base))
        metrics = evaluation.prediction_metrics(target, predicted, k_grid)
        self.assertAlmostEqual(metrics["evdf_relative_l2_mean"], 2.0, places=12)
        self.assertAlmostEqual(metrics["evdf_relative_l2_median"], 2.0, places=12)

    def test_metrics_reject_bad_shapes_and_nonfinite(self):
        k_grid = torch.linspace(-2.0, 2.0, 80)
        with self.assertRaises(ValueError):
            evaluation.prediction_metrics(torch.zeros(1, 160), torch.zeros(2, 160), k_grid)
        with self.assertRaises(ValueError):
            evaluation.prediction_metrics(torch.zeros(1, 160), torch.full((1, 160), torch.nan), k_grid)


class TestPredictionArtifactAndPlots(unittest.TestCase):
    def test_predictions_npz_exact_contract_and_atomic_name(self):
        target = torch.randn(2, 160)
        predicted = model_module.hermitian_project(torch.randn(2, 160))
        k_grid = torch.linspace(-2.0, 2.0, 80)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            evaluation._write_predictions(path, target, predicted, k_grid)
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(temporary).glob("*.tmp*")), [])
            with np.load(path) as payload:
                self.assertEqual(
                    set(payload.files),
                    {"target_coefficients", "predicted_coefficients", "k_grid", "velocity_m_s", "target_evdf", "predicted_evdf"},
                )
                self.assertEqual(payload["target_coefficients"].shape, (2, 160))
                self.assertEqual(payload["predicted_coefficients"].shape, (2, 160))
                self.assertEqual(payload["k_grid"].shape, (80,))
                self.assertEqual(payload["velocity_m_s"].shape, (80,))
                self.assertEqual(payload["target_evdf"].shape, (2, 80))
                self.assertEqual(payload["predicted_evdf"].shape, (2, 80))

    def test_comparison_caps_count_and_detail_uses_requested_index(self):
        target = torch.randn(3, 160)
        predicted = model_module.hermitian_project(torch.randn(3, 160))
        dataset = make_dataset(3)
        k_grid = dataset.k_grid
        local_plot = RecordingPyplot()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(evaluation, "plt", local_plot):
            comparison = Path(temporary) / "comparison.png"
            detail = Path(temporary) / "sample_detail.png"
            evaluation._save_comparison(comparison, target, predicted, k_grid, 99)
            evaluation._save_sample_detail(detail, dataset, predicted, k_grid, 1)
            self.assertTrue(comparison.is_file())
            self.assertTrue(detail.is_file())
            self.assertEqual(list(Path(temporary).glob("*.tmp.png")), [])
        comparison_axes = local_plot.axes[0]
        titled = [axis.titles[0] for row in comparison_axes for axis in row if axis.titles]
        self.assertEqual(titled, ["sample 1", "sample 2", "sample 3"])
        self.assertEqual(local_plot.figures[1].suptitles, ["sample 2 detail"])


class TestEvaluateCheckpointFullPath(unittest.TestCase):
    def test_real_checkpoint_independent_test_artifacts_paths_and_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_path = write_real_test_parquet(root, 3)
            manifest_path = write_evaluation_manifest(
                root, test_path, 3, run_id="DATA_SOURCE_RUN", dataset_id="EVAL_DATA"
            )
            checkpoint_payload = t3_fixture.valid_checkpoint_payload()
            checkpoint_payload["run_id"] = "TRAIN_SOURCE_RUN"
            checkpoint_payload["dataset_id"] = "TRAIN_DATA"
            checkpoint_path = write_checkpoint(root, checkpoint_payload)
            history_path = root / "training_history.json"
            history_path.write_bytes(b'{"training":"sentinel"}\n')
            checkpoint_before = checkpoint_path.read_bytes()
            history_before = history_path.read_bytes()
            output_dir = root / "evaluation"
            config = {"batch_size": 2, "device": "cpu", "plot_samples": 2, "detail_index": 1}

            with mock.patch.object(data.pq, "read_table", side_effect=t3_fixture.t2_fixture._real_arrow_read), mock.patch.object(
                evaluation, "plt", RecordingPyplot()
            ):
                result = evaluation.evaluate_checkpoint(
                    "EVAL_RUN",
                    "EVAL_DATA",
                    "CHECKPOINT",
                    checkpoint_path,
                    test_path,
                    manifest_path,
                    output_dir,
                    config,
                )

            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(history_path.read_bytes(), history_before)
            for key in ("metrics_path", "predictions_path", "comparison_path", "sample_detail_path"):
                self.assertTrue(result[key].is_file())
                self.assertGreater(result[key].stat().st_size, 0)
            self.assertEqual(list(output_dir.glob("*.tmp*")), [])
            metrics_payload = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(
                set(metrics_payload),
                {
                    "run_id",
                    "dataset_id",
                    "dataset_source_run_id",
                    "checkpoint_id",
                    "checkpoint_source_run_id",
                    "checkpoint_training_dataset_id",
                    "checkpoint",
                    "test_data",
                    "metrics",
                },
            )
            self.assertEqual(
                (metrics_payload["run_id"], metrics_payload["dataset_id"], metrics_payload["checkpoint_id"]),
                ("EVAL_RUN", "EVAL_DATA", "CHECKPOINT"),
            )
            self.assertEqual(metrics_payload["dataset_source_run_id"], "DATA_SOURCE_RUN")
            self.assertEqual(metrics_payload["checkpoint_source_run_id"], "TRAIN_SOURCE_RUN")
            self.assertEqual(metrics_payload["checkpoint_training_dataset_id"], "TRAIN_DATA")
            self.assertEqual(metrics_payload["checkpoint"], "../model.pt")
            self.assertEqual(metrics_payload["test_data"], "../dataset_test.parquet")
            self.assertEqual(metrics_payload["metrics"], result["metrics"])
            with np.load(result["predictions_path"]) as predictions:
                self.assertEqual(predictions["target_coefficients"].shape, (3, 160))
                projected = torch.from_numpy(predictions["predicted_coefficients"])
                torch.testing.assert_close(projected[:, :80], torch.flip(projected[:, :80], (1,)))
                torch.testing.assert_close(projected[:, 80:], -torch.flip(projected[:, 80:], (1,)))

            output_snapshot = {path.name: path.read_bytes() for path in output_dir.iterdir()}
            with self.assertRaises(FileExistsError):
                evaluation.evaluate_checkpoint(
                    "EVAL_RUN", "EVAL_DATA", "CHECKPOINT", checkpoint_path, test_path, manifest_path, output_dir, config
                )
            self.assertEqual(output_snapshot, {path.name: path.read_bytes() for path in output_dir.iterdir()})
            with mock.patch.object(data.pq, "read_table", side_effect=t3_fixture.t2_fixture._real_arrow_read), mock.patch.object(
                evaluation, "plt", RecordingPyplot()
            ):
                overwritten = evaluation.evaluate_checkpoint(
                    "EVAL_RUN", "EVAL_DATA", "CHECKPOINT", checkpoint_path, test_path, manifest_path, output_dir, config, overwrite=True
                )
            self.assertTrue(overwritten["metrics_path"].is_file())
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(history_path.read_bytes(), history_before)

    def test_ids_paths_counts_grids_and_detail_index_fail_before_outputs(self):
        dataset = make_dataset(3)
        checkpoint = t3_fixture.valid_checkpoint_payload()
        regressor = model_module.OTSRegressor((8, 4), 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "model.pt"
            checkpoint_path.write_bytes(b"mocked")
            test_path = root / "dataset_test.parquet"
            test_path.write_bytes(b"mocked")
            manifest_path = write_evaluation_manifest(root, test_path, 3)

            def call(output_name="out", requested_test_path=None, config=None):
                with mock.patch.object(evaluation, "load_model_checkpoint", return_value=(regressor, checkpoint)), mock.patch.object(
                    evaluation, "load_dataset", return_value=dataset
                ):
                    return evaluation.evaluate_checkpoint(
                        "RUN",
                        "DATA",
                        "CHECKPOINT",
                        checkpoint_path,
                        requested_test_path or test_path,
                        manifest_path,
                        root / output_name,
                        config or {"device": "cpu"},
                    )

            with self.assertRaises(ValueError):
                evaluation.evaluate_checkpoint("", "DATA", "CHECKPOINT", checkpoint_path, test_path, manifest_path, root / "id", {})
            wrong_path = root / "other.parquet"
            wrong_path.write_bytes(b"mocked")
            with self.assertRaises(ValueError):
                call("path", wrong_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["test"]["physical_samples"] = 4
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                call("count")
            manifest["splits"]["test"]["physical_samples"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            bad_k = copy.deepcopy(checkpoint)
            bad_k["k_grid"] = checkpoint["k_grid"] + 1.0
            with mock.patch.object(evaluation, "load_model_checkpoint", return_value=(regressor, bad_k)), mock.patch.object(
                evaluation, "load_dataset", return_value=dataset
            ), self.assertRaises(ValueError):
                evaluation.evaluate_checkpoint("RUN", "DATA", "CHECKPOINT", checkpoint_path, test_path, manifest_path, root / "k", {"device": "cpu"})

            bad_wavelength = copy.deepcopy(checkpoint)
            bad_wavelength["wavelengths_nm"] = checkpoint["wavelengths_nm"].clone()
            bad_wavelength["wavelengths_nm"][0] += 10.0
            with mock.patch.object(evaluation, "load_model_checkpoint", return_value=(regressor, bad_wavelength)), mock.patch.object(
                evaluation, "load_dataset", return_value=dataset
            ), self.assertRaises(ValueError):
                evaluation.evaluate_checkpoint("RUN", "DATA", "CHECKPOINT", checkpoint_path, test_path, manifest_path, root / "wavelength", {"device": "cpu"})

            with self.assertRaises(ValueError):
                call("detail", config={"device": "cpu", "detail_index": 3})
            for name in ("path", "count", "k", "wavelength", "detail"):
                self.assertFalse((root / name).exists())


class TestEvaluationSourceIsolation(unittest.TestCase):
    def test_source_has_no_train_legacy_or_synthetic_dependency(self):
        signature = inspect.signature(evaluation.evaluate_checkpoint)
        self.assertIn("test_data_path", signature.parameters)
        source_path = PROJECT_ROOT / "evaluation.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name == "train" or name.endswith(".train") for name in imported), imported)
        self.assertNotIn("legacy", source.lower())
        self.assertNotIn("synthetic", source.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
