"""Targeted acceptance tests for canonical-pipeline T3 only."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the T2 dependency/real-Parquet fixture so T3 can run in the existing
# PyTorch environment, which does not itself contain PyArrow.
from tests import test_data_t2 as t2_fixture  # noqa: E402

data = t2_fixture.data


def _install_matplotlib_fixture_if_needed():
    try:
        import matplotlib.pyplot  # noqa: F401
        return False
    except ModuleNotFoundError:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib.__path__ = []

        class Axis:
            def semilogy(self, *args, **kwargs):
                return None

            def axvline(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def grid(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class Figure:
            def tight_layout(self):
                return None

            def savefig(self, path, dpi=None):
                # The curve contents are exercised through the Axis calls; this
                # fixture only provides an atomic, non-empty image artifact.
                Path(path).write_bytes(b"\x89PNG\r\n\x1a\nT3-fixture")

        def subplots(rows, columns, figsize=None, squeeze=False):
            return Figure(), [[Axis() for _ in range(columns)] for _ in range(rows)]

        pyplot.subplots = subplots
        pyplot.close = lambda figure: None
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
        return True


USING_MATPLOTLIB_FIXTURE = _install_matplotlib_fixture_if_needed()

import model  # noqa: E402
import train  # noqa: E402


def setUpModule():
    # T2 deliberately reloads ``data`` to verify import safety.  Discovery
    # imports all test modules before running them, so refresh train.py's
    # from-import bindings after that isolated reload and before T3 starts.
    train.OTSDataset = data.OTSDataset
    train.load_dataset = data.load_dataset
    train.validate_dataset = data.validate_dataset


def make_dataset(sample_count=10):
    generator = torch.Generator().manual_seed(321)
    spectra = torch.randn(sample_count, data.INPUT_DIM, generator=generator)
    # Preserve an exact row identity for split-disjointness assertions.
    spectra[:, 0] = torch.arange(sample_count, dtype=torch.float32)
    coefficients = torch.randn(sample_count, data.OUTPUT_DIM, generator=generator)
    wavelengths = torch.cat(data.build_wavelength_bands(torch.float32)) / 1.0e-9
    k_grid = torch.linspace(-2.0, 2.0, data.FOURIER_MODES)
    dataset = data.OTSDataset(spectra, coefficients, wavelengths, k_grid)
    data.validate_dataset(dataset)
    return dataset


def resolved_config(**overrides):
    config = {
        "seed": 17,
        "validation_fraction": 0.25,
        "candidates": [{"hidden_sizes": [8, 4], "dropout": 0.0}],
        "batch_size": 4,
        "max_epochs": 3,
        "patience": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "raw_loss_weight": 0.25,
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "scheduler": {"parameters": {"patience": 1}},
    }
    config.update(overrides)
    return train.resolve_training_config(config)[0]


def valid_checkpoint_payload(device="cpu", dtype="float32"):
    training_config = resolved_config()
    candidate = training_config["candidates"][0]
    regressor = model.OTSRegressor(tuple(candidate["hidden_sizes"]), candidate["dropout"])
    tensor_dtype = torch.float64 if dtype == "float64" else torch.float32
    return {
        "run_id": "RUN",
        "dataset_id": "DATA",
        "checkpoint_id": "CHECKPOINT",
        "model_state": copy.deepcopy(regressor.state_dict()),
        "hidden_sizes": list(candidate["hidden_sizes"]),
        "dropout": candidate["dropout"],
        "layer_norm": True,
        "leaky_relu_slope": model.LEAKY_RELU_SLOPE,
        "input_dim": model.INPUT_DIM,
        "output_dim": model.OUTPUT_DIM,
        "fourier_modes": model.FOURIER_MODES,
        "x_mean": torch.zeros(model.INPUT_DIM, dtype=tensor_dtype),
        "x_scale": torch.ones(model.INPUT_DIM, dtype=tensor_dtype),
        "y_mean": torch.zeros(model.OUTPUT_DIM, dtype=tensor_dtype),
        "y_scale": torch.ones(model.OUTPUT_DIM, dtype=tensor_dtype),
        "k_grid": torch.linspace(-2.0, 2.0, model.FOURIER_MODES, dtype=tensor_dtype),
        "wavelengths_nm": torch.cat(data.build_wavelength_bands(tensor_dtype)) / 1.0e-9,
        "selected_candidate": copy.deepcopy(candidate),
        "best_epoch": 1,
        "best_validation_loss": 0.5,
        "device": device,
        "dtype": dtype,
        "training_config": training_config,
    }


def write_real_train_parquet(directory, sample_count=8):
    rows = t2_fixture._valid_rows(sample_count)
    path = Path(directory) / "dataset_train.parquet"
    with mock.patch.object(data.pq, "write_table", side_effect=t2_fixture._real_arrow_write):
        data._write_parquet(rows, path)
    return path


def write_manifest(directory, train_path, run_id="RUN", dataset_id="DATA"):
    manifest_path = Path(directory) / "dataset_manifest.json"
    payload = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "physical_configuration": {"fixture": True},
        "splits": {
            "train": {"file": os.path.relpath(train_path, manifest_path.parent)},
            # This deliberately absent file proves train_model never opens test.
            "test": {"file": "MUST_NOT_BE_OPENED.parquet"},
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


class TestOTSRegressor(unittest.TestCase):
    def test_layer_order_shapes_forward_and_initialization(self):
        regressor = model.OTSRegressor((16, 7), 0.2)
        with mock.patch.object(
            torch.nn.init, "kaiming_normal_", wraps=torch.nn.init.kaiming_normal_
        ) as kaiming_spy, mock.patch.object(
            torch.nn.init, "zeros_", wraps=torch.nn.init.zeros_
        ) as zeros_spy:
            regressor.reset_parameters()
        layers = list(regressor.network)
        self.assertEqual(
            [type(layer) for layer in layers],
            [
                nn.Linear,
                nn.LayerNorm,
                nn.LeakyReLU,
                nn.Dropout,
                nn.Linear,
                nn.LayerNorm,
                nn.LeakyReLU,
                nn.Dropout,
                nn.Linear,
            ],
        )
        self.assertEqual((layers[0].in_features, layers[0].out_features), (1224, 16))
        self.assertEqual((layers[4].in_features, layers[4].out_features), (16, 7))
        self.assertEqual((layers[8].in_features, layers[8].out_features), (7, 160))
        self.assertTrue(all(layer.negative_slope == 0.05 for layer in (layers[2], layers[6])))
        self.assertTrue(all(layer.p == 0.2 for layer in (layers[3], layers[7])))
        self.assertEqual(kaiming_spy.call_count, 3)
        self.assertEqual(zeros_spy.call_count, 3)
        self.assertTrue(all(call.kwargs["a"] == 0.05 for call in kaiming_spy.call_args_list))
        self.assertTrue(all(torch.count_nonzero(layer.bias) == 0 for layer in (layers[0], layers[4], layers[8])))
        output = regressor(torch.randn(5, 1224))
        self.assertEqual(output.shape, (5, 160))

    def test_invalid_architecture_inputs(self):
        for hidden in ((1,), (1, 2, 3), (0, 2), (True, 2), "1,2"):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                model.OTSRegressor(hidden, 0.0)
        for dropout in (-0.1, 1.0, math.nan, math.inf):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                model.OTSRegressor((2, 2), dropout)


class TestHermitianProjection(unittest.TestCase):
    def test_projection_symmetry_idempotence_shape_gradient_and_no_mutation(self):
        coefficients = torch.randn(3, 2, 160, dtype=torch.float64, requires_grad=True)
        original = coefficients.detach().clone()
        projected = model.hermitian_project(coefficients)
        self.assertEqual(projected.shape, coefficients.shape)
        torch.testing.assert_close(projected[..., :80], torch.flip(projected[..., :80], (-1,)))
        torch.testing.assert_close(projected[..., 80:], -torch.flip(projected[..., 80:], (-1,)))
        torch.testing.assert_close(model.hermitian_project(projected), projected)
        torch.testing.assert_close(coefficients.detach(), original)
        projected.square().sum().backward()
        self.assertIsNotNone(coefficients.grad)
        self.assertTrue(torch.isfinite(coefficients.grad).all())

    def test_projection_rejects_wrong_inputs(self):
        for value in (None, [0.0] * 160, torch.zeros(159), torch.zeros(2, 161)):
            with self.subTest(type=type(value)), self.assertRaises(ValueError):
                model.hermitian_project(value)


class TestCheckpointContract(unittest.TestCase):
    def test_complete_schema_and_training_hyperparameters(self):
        payload = valid_checkpoint_payload()
        model.validate_checkpoint(payload)
        self.assertEqual(set(payload), model.CHECKPOINT_FIELDS)
        self.assertEqual(set(payload["training_config"]), model.TRAINING_CONFIG_FIELDS)
        self.assertEqual(
            set(payload["training_config"]["optimizer"]["parameters"]),
            {"lr", "betas", "eps", "weight_decay", "amsgrad"},
        )
        self.assertEqual(
            set(payload["training_config"]["scheduler"]["parameters"]),
            {"mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr", "eps"},
        )

    def test_schema_rejects_legacy_test_fields_missing_and_incomplete_config(self):
        cases = []
        cases.append({"pet1": {}})
        extra_metrics = valid_checkpoint_payload()
        extra_metrics["test_metrics"] = {"r2": 1.0}
        cases.append(extra_metrics)
        extra_prediction = valid_checkpoint_payload()
        extra_prediction["test_predictions"] = torch.zeros(1, 160)
        cases.append(extra_prediction)
        missing = valid_checkpoint_payload()
        del missing["checkpoint_id"]
        cases.append(missing)
        incomplete_config = valid_checkpoint_payload()
        del incomplete_config["training_config"]["scheduler"]
        cases.append(incomplete_config)
        for index, payload in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.validate_checkpoint(payload)

    def test_strict_load_matches_checkpoint_id_and_preserves_training_source(self):
        payload = valid_checkpoint_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(payload, path)
            loaded, returned = model.load_model_checkpoint(
                path, torch.device("cpu"), "CHECKPOINT"
            )
            self.assertFalse(loaded.training)
            self.assertEqual(returned["checkpoint_id"], "CHECKPOINT")
            for name, expected in payload["model_state"].items():
                torch.testing.assert_close(loaded.state_dict()[name], expected)
            self.assertEqual(returned["run_id"], "RUN")
            self.assertEqual(returned["dataset_id"], "DATA")
            with self.assertRaises(ValueError):
                model.load_model_checkpoint(path, "cpu", "WRONG")

            broken = valid_checkpoint_payload()
            del broken["model_state"][next(iter(broken["model_state"]))]
            broken_path = Path(temporary) / "broken.pt"
            torch.save(broken, broken_path)
            with self.assertRaises(RuntimeError):
                model.load_model_checkpoint(broken_path, "cpu", "CHECKPOINT")


class TestSeedDeviceAndCandidates(unittest.TestCase):
    def test_determinism_seeds_python_numpy_torch_and_cuda(self):
        train.set_determinism(99)
        first = (random.random(), np.random.random(), torch.rand(3))
        first_cuda = torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None
        train.set_determinism(99)
        second = (random.random(), np.random.random(), torch.rand(3))
        second_cuda = torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        if first_cuda is not None:
            torch.testing.assert_close(first_cuda, second_cuda)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        with self.assertRaises(ValueError):
            train.set_determinism(True)

    def test_device_resolution(self):
        self.assertEqual(train.choose_device("cpu"), torch.device("cpu"))
        self.assertEqual(
            train.choose_device("auto"),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        if torch.cuda.is_available():
            self.assertEqual(train.choose_device("cuda"), torch.device("cuda"))
        with self.assertRaises(ValueError):
            train.choose_device("gpu")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                train.choose_device("cuda")

    def test_candidate_parsing_and_complete_config_resolution(self):
        candidates = train.parse_candidates("512,256;1024,512")
        self.assertEqual(candidates[0], {"hidden_sizes": [512, 256], "dropout": 0.0})
        self.assertEqual(candidates[1], {"hidden_sizes": [1024, 512], "dropout": 0.02})
        config, requested = train.resolve_training_config({"device": "cpu"})
        self.assertEqual(requested, "cpu")
        self.assertEqual(set(config), model.TRAINING_CONFIG_FIELDS)
        model._validate_training_config(config)
        invalid = ("", "1", "1,2;1,2", [(1, 0)], [{"hidden_sizes": [1, 2], "dropout": 1.0}])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                train.parse_candidates(value)

    def test_resolver_rejects_candidates_ambiguous_to_train_candidate_interface(self):
        ambiguous = [
            {"hidden_sizes": [8, 4], "dropout": 0.0},
            {"hidden_sizes": [8, 4], "dropout": 0.2},
        ]
        # train_candidate's SPEC interface receives hidden_sizes, not dropout;
        # every accepted candidate therefore has to be uniquely addressable by
        # hidden_sizes alone.
        with self.assertRaises(ValueError):
            train.resolve_training_config({"device": "cpu", "candidates": ambiguous})


class TestSplitAndNormalizationIsolation(unittest.TestCase):
    def test_split_is_seeded_disjoint_fixed_and_local(self):
        dataset = make_dataset(11)
        torch.manual_seed(12345)
        first = train.split_train_validation(dataset, 0.27, 777)
        torch.manual_seed(1)
        second = train.split_train_validation(dataset, 0.27, 777)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)
        x_train, y_train, x_validation, y_validation = first
        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(x_validation), 3)
        train_ids = set(x_train[:, 0].tolist())
        validation_ids = set(x_validation[:, 0].tolist())
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(train_ids | validation_ids, set(range(11)))
        for row, label in zip(x_train, y_train):
            index = int(row[0].item())
            torch.testing.assert_close(label, dataset.coefficients[index])
        for row, label in zip(x_validation, y_validation):
            index = int(row[0].item())
            torch.testing.assert_close(label, dataset.coefficients[index])
        third = train.split_train_validation(dataset, 0.27, 778)
        self.assertFalse(torch.equal(first[0], third[0]))

    def test_normalization_uses_only_supplied_training_rows_and_floors_scales(self):
        x_train = torch.zeros(3, 1224)
        x_train[:, 0] = torch.tensor([1.0, 2.0, 6.0])
        y_train = torch.zeros(3, 160)
        y_train[:, 0] = torch.tensor([2.0, 4.0, 8.0])
        stats = train.normalization_statistics(x_train, y_train)
        self.assertAlmostEqual(stats[0][0].item(), 3.0)
        self.assertAlmostEqual(stats[2][0].item(), 14.0 / 3.0, places=6)
        self.assertTrue(torch.all(stats[1] >= 1.0e-4))
        self.assertTrue(torch.all(stats[3] >= 1.0e-8))
        # A validation-only outlier is never passed and cannot affect the result.
        validation_outlier = torch.full((1, 1224), 1.0e30)
        self.assertNotAlmostEqual(stats[0][0].item(), validation_outlier.mean().item())


class TestLossAndEpoch(unittest.TestCase):
    def test_mixed_loss_endpoints_match_independent_formula_and_project_prediction(self):
        generator = torch.Generator().manual_seed(8)
        predicted_normalized = torch.randn(4, 160, generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.randn(4, 160, generator=generator, dtype=torch.float64)
        y_mean = torch.randn(160, generator=generator, dtype=torch.float64)
        y_scale = torch.rand(160, generator=generator, dtype=torch.float64) + 0.2
        predicted_raw = model.hermitian_project(predicted_normalized * y_scale + y_mean)
        expected_standardized = torch.mean(((predicted_raw - y_mean) / y_scale - (target - y_mean) / y_scale) ** 2)
        expected_raw = torch.mean((predicted_raw - target) ** 2) / torch.mean(y_scale**2)
        zero = train.coefficient_loss(predicted_normalized, target, y_mean, y_scale, 0.0)
        one = train.coefficient_loss(predicted_normalized, target, y_mean, y_scale, 1.0)
        mixed = train.coefficient_loss(predicted_normalized, target, y_mean, y_scale, 0.25)
        torch.testing.assert_close(zero, expected_standardized)
        torch.testing.assert_close(one, expected_raw)
        torch.testing.assert_close(mixed, 0.75 * expected_standardized + 0.25 * expected_raw)
        mixed.backward()
        self.assertTrue(torch.isfinite(predicted_normalized.grad).all())

    def test_validation_epoch_does_not_update_and_training_clips_and_updates(self):
        regressor = model.OTSRegressor((8, 4), 0.0)
        spectra = torch.randn(6, 1224)
        target = torch.randn(6, 160)
        loader = DataLoader(TensorDataset(spectra, target), batch_size=2, shuffle=False)
        statistics = {"y_mean": torch.zeros(160), "y_scale": torch.ones(160)}
        config = {"raw_loss_weight": 0.5, "gradient_clip_norm": 0.1}
        optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01)
        before_validation = {name: value.clone() for name, value in regressor.state_dict().items()}
        validation_loss = train.run_epoch(regressor, loader, None, torch.device("cpu"), statistics, config, False)
        self.assertFalse(regressor.training)
        self.assertTrue(math.isfinite(validation_loss))
        for name, value in regressor.state_dict().items():
            torch.testing.assert_close(value, before_validation[name])
        with mock.patch.object(
            torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
        ) as clip_spy:
            training_loss = train.run_epoch(regressor, loader, optimizer, torch.device("cpu"), statistics, config, True)
        self.assertTrue(regressor.training)
        self.assertTrue(math.isfinite(training_loss))
        self.assertEqual(clip_spy.call_count, len(loader))
        self.assertTrue(any(not torch.equal(value, before_validation[name]) for name, value in regressor.state_dict().items()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_epoch_path(self):
        device = torch.device("cuda")
        regressor = model.OTSRegressor((8, 4), 0.0).to(device)
        loader = DataLoader(TensorDataset(torch.randn(4, 1224), torch.randn(4, 160)), batch_size=2)
        statistics = {"y_mean": torch.zeros(160), "y_scale": torch.ones(160)}
        config = {"raw_loss_weight": 0.5, "gradient_clip_norm": 1.0}
        optimizer = torch.optim.AdamW(regressor.parameters(), lr=1.0e-3)
        loss = train.run_epoch(regressor, loader, optimizer, device, statistics, config, True)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(next(regressor.parameters()).device.type, "cuda")


class TestCandidateTraining(unittest.TestCase):
    def test_best_state_is_cloned_and_early_stopping_is_validation_driven(self):
        config = resolved_config(max_epochs=8, patience=2)
        train_loader = DataLoader(TensorDataset(torch.zeros(2, 1224), torch.zeros(2, 160)), batch_size=1)
        validation_loader = DataLoader(TensorDataset(torch.ones(2, 1224), torch.ones(2, 160)), batch_size=1)
        validation_values = iter([3.0, 1.0, 2.0, 2.5])

        def fake_epoch(regressor, loader, optimizer, device, statistics, epoch_config, training, *, fourier_modes=80):
            if training:
                with torch.no_grad():
                    next(regressor.parameters()).fill_(len(fake_epoch.training_calls) + 1.0)
                fake_epoch.training_calls.append(True)
                return 100.0 - len(fake_epoch.training_calls)
            return next(validation_values)

        fake_epoch.training_calls = []
        statistics = {"y_mean": torch.zeros(160), "y_scale": torch.ones(160)}
        with mock.patch.object(train, "run_epoch", side_effect=fake_epoch):
            result = train.train_candidate((8, 4), train_loader, validation_loader, torch.device("cpu"), statistics, config)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_validation_loss"], 1.0)
        self.assertEqual(result["stop_reason"], "patience_exhausted")
        self.assertEqual(len(result["history"]), 4)
        first_weight_name = next(name for name in result["state_dict"] if name.endswith("weight"))
        torch.testing.assert_close(result["state_dict"][first_weight_name], torch.full_like(result["state_dict"][first_weight_name], 2.0))


class TestTrainModelFullPath(unittest.TestCase):
    def test_small_real_parquet_training_artifacts_no_leakage_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = write_real_train_parquet(root, 8)
            manifest_path = write_manifest(root, train_path)
            output_dir = root / "training"
            config = {
                "seed": 123,
                "validation_fraction": 0.25,
                "candidates": [{"hidden_sizes": [8, 4], "dropout": 0.0}],
                "batch_size": 3,
                "max_epochs": 2,
                "patience": 2,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "raw_loss_weight": 0.5,
                "gradient_clip_norm": 1.0,
                "device": "cpu",
            }
            with mock.patch.object(data.pq, "read_table", side_effect=t2_fixture._real_arrow_read):
                result = train.train_model("RUN", "DATA", "CKPT", train_path, manifest_path, output_dir, config)
                loaded_dataset = data.load_dataset(train_path)
            for path_key in ("checkpoint_path", "history_path", "learning_curves_path"):
                self.assertTrue(result[path_key].is_file())
                self.assertGreater(result[path_key].stat().st_size, 0)
            self.assertEqual(list(output_dir.glob("*.tmp")), [])
            checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
            model.validate_checkpoint(checkpoint)
            history = json.loads(result["history_path"].read_text(encoding="utf-8"))
            self.assertEqual((checkpoint["run_id"], checkpoint["dataset_id"], checkpoint["checkpoint_id"]), ("RUN", "DATA", "CKPT"))
            self.assertEqual(set(checkpoint["training_config"]), model.TRAINING_CONFIG_FIELDS)
            self.assertEqual(history["training_config"], checkpoint["training_config"])
            self.assertEqual(history["sample_counts"], {"train": 6, "validation": 2})
            self.assertEqual(history["dataset_source_run_id"], "RUN")
            self.assertNotIn("test", history)
            self.assertNotIn("test_metrics", checkpoint)
            expected_split = train.split_train_validation(loaded_dataset, 0.25, 123)
            expected_stats = train.normalization_statistics(expected_split[0], expected_split[1])
            for name, expected in zip(("x_mean", "x_scale", "y_mean", "y_scale"), expected_stats):
                torch.testing.assert_close(checkpoint[name], expected)

            original_checkpoint = result["checkpoint_path"].read_bytes()
            with self.assertRaises(FileExistsError):
                train.train_model("RUN", "DATA", "CKPT", train_path, manifest_path, output_dir, config)
            self.assertEqual(result["checkpoint_path"].read_bytes(), original_checkpoint)
            with mock.patch.object(data.pq, "read_table", side_effect=t2_fixture._real_arrow_read):
                overwritten = train.train_model("RUN", "DATA", "CKPT", train_path, manifest_path, output_dir, config, overwrite=True)
            self.assertTrue(overwritten["checkpoint_path"].is_file())

    def test_candidate_selection_uses_only_best_validation_loss_and_dataloaders(self):
        dataset = make_dataset(8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "dataset_train.parquet"
            train_path.write_bytes(b"fixture path only")
            manifest_path = write_manifest(root, train_path, run_id="SOURCE_RUN")
            output_dir = root / "training"
            config = {
                "seed": 5,
                "validation_fraction": 0.25,
                "candidates": [
                    {"hidden_sizes": [8, 4], "dropout": 0.0},
                    {"hidden_sizes": [6, 3], "dropout": 0.0},
                ],
                "batch_size": 3,
                "max_epochs": 1,
                "patience": 1,
                "device": "cpu",
            }
            observed_loaders = []

            def fake_candidate(hidden_sizes, train_loader, validation_loader, device, statistics, candidate_config, *, input_dim=1224, output_dim=160, fourier_modes=80):
                observed_loaders.append((train_loader, validation_loader, statistics))
                regressor = model.OTSRegressor(hidden_sizes, 0.0)
                validation_loss = 5.0 if tuple(hidden_sizes) == (8, 4) else 1.0
                train_loss = 0.1 if tuple(hidden_sizes) == (8, 4) else 100.0
                return {
                    "hidden_sizes": list(hidden_sizes),
                    "dropout": 0.0,
                    "parameter_count": sum(value.numel() for value in regressor.parameters()),
                    "history": [{"epoch": 1, "train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": 1.0e-3}],
                    "best_epoch": 1,
                    "best_validation_loss": validation_loss,
                    "state_dict": train._cpu_state_dict(regressor),
                    "elapsed_seconds": 0.0,
                    "stop_reason": "max_epochs_reached",
                }

            with mock.patch.object(train, "load_dataset", return_value=dataset), mock.patch.object(
                train, "train_candidate", side_effect=fake_candidate
            ):
                result = train.train_model(
                    "RUN", "DATA", "CKPT", train_path, manifest_path, output_dir, config
                )
            self.assertEqual(result["selected_candidate"]["hidden_sizes"], [6, 3])
            self.assertEqual(result["selected_candidate"]["best_validation_loss"], 1.0)
            history = json.loads(result["history_path"].read_text(encoding="utf-8"))
            self.assertEqual(history["dataset_source_run_id"], "SOURCE_RUN")
            self.assertEqual(len(observed_loaders), 2)
            for train_loader, validation_loader, statistics in observed_loaders:
                self.assertIsInstance(train_loader, DataLoader)
                self.assertIsInstance(validation_loader, DataLoader)
                self.assertIsInstance(train_loader.sampler, RandomSampler)
                self.assertIsInstance(validation_loader.sampler, SequentialSampler)
                self.assertEqual(train_loader.batch_size, 3)
                self.assertEqual(validation_loader.batch_size, 3)
                self.assertEqual(len(train_loader.dataset), 6)
                self.assertEqual(len(validation_loader.dataset), 2)
                normalized_train = train_loader.dataset.tensors[0]
                torch.testing.assert_close(normalized_train.mean(0), torch.zeros(1224), atol=2.0e-4, rtol=0.0)

    def test_ids_manifest_and_train_path_must_match(self):
        dataset = make_dataset(4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "dataset_train.parquet"
            train_path.write_bytes(b"fixture")
            manifest_path = write_manifest(root, train_path)
            config = {"candidates": [{"hidden_sizes": [4, 2], "dropout": 0.0}], "max_epochs": 1, "patience": 1, "device": "cpu"}
            for ids in (("", "DATA", "CKPT"), ("RUN", "", "CKPT"), ("RUN", "DATA", ""), ("RUN", "WRONG", "CKPT")):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    train.train_model(*ids, train_path, manifest_path, root / "out", config)
            other_path = root / "other.parquet"
            other_path.write_bytes(b"fixture")
            with mock.patch.object(train, "load_dataset", return_value=dataset):
                with self.assertRaises(ValueError):
                    train.train_model("RUN", "DATA", "CKPT", other_path, manifest_path, root / "out", config)


class TestTrainingSourceIsolation(unittest.TestCase):
    def test_train_source_and_interface_have_no_test_or_evaluation_dependency(self):
        signature = inspect.signature(train.train_model)
        self.assertFalse(any("test" in name.lower() for name in signature.parameters))
        source_path = PROJECT_ROOT / "train.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name == "evaluation" or name.endswith(".evaluation") for name in imported), imported)
        self.assertNotIn("test_data", source.lower())
        self.assertNotIn("test_metrics", source.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
