"""Targeted acceptance tests for canonical-pipeline T5 only."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the dependency fixtures from the already accepted T2--T4 tests.  The
# available Torch runtime has no PyArrow or Matplotlib, while T5 itself must be
# importable and testable without changing production dependencies.
from tests import test_evaluation_t4 as t4_fixture  # noqa: E402,F401

import main as canonical_main  # noqa: E402


def _valid_argv(command):
    common = ["--run-id", "RUN", "--dataset-id", "DATA"]
    if command == "generate":
        return [command] + common + ["--output-dir", "out/data"]
    if command == "train":
        return [command] + common + [
            "--checkpoint-id",
            "CKPT",
            "--train-data",
            "in/train.parquet",
            "--dataset-manifest",
            "in/dataset_manifest.json",
            "--output-dir",
            "out/training",
        ]
    if command == "evaluate":
        return [command] + common + [
            "--checkpoint-id",
            "CKPT",
            "--checkpoint",
            "in/model.pt",
            "--test-data",
            "in/test.parquet",
            "--dataset-manifest",
            "in/dataset_manifest.json",
            "--output-dir",
            "out/evaluation",
        ]
    if command == "all":
        return [command] + common + [
            "--checkpoint-id",
            "CKPT",
            "--run-dir",
            "out/run",
        ]
    raise AssertionError("unknown command")


class TestParserContract(unittest.TestCase):
    def setUp(self):
        self.parser = canonical_main.build_parser()

    def test_exact_four_subcommands_and_required_ids(self):
        subparser_actions = [
            action
            for action in self.parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        ]
        self.assertEqual(len(subparser_actions), 1)
        self.assertEqual(
            set(subparser_actions[0].choices), {"generate", "train", "evaluate", "all"}
        )

        for command in ("generate", "train", "evaluate", "all"):
            with self.subTest(command=command, missing="run_id"):
                argv = _valid_argv(command)
                index = argv.index("--run-id")
                del argv[index : index + 2]
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self.parser.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)
            with self.subTest(command=command, missing="dataset_id"):
                argv = _valid_argv(command)
                index = argv.index("--dataset-id")
                del argv[index : index + 2]
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self.parser.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

        for command in ("train", "evaluate", "all"):
            with self.subTest(command=command, missing="checkpoint_id"):
                argv = _valid_argv(command)
                index = argv.index("--checkpoint-id")
                del argv[index : index + 2]
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self.parser.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

        generated = self.parser.parse_args(_valid_argv("generate"))
        self.assertFalse(hasattr(generated, "checkpoint_id"))

    def test_paths_defaults_handlers_and_all_evaluation_batch_name(self):
        for command, handler in (
            ("generate", canonical_main.run_generate),
            ("train", canonical_main.run_train),
            ("evaluate", canonical_main.run_evaluate),
            ("all", canonical_main.run_all),
        ):
            args = self.parser.parse_args(_valid_argv(command))
            self.assertIs(args.handler, handler)
            self.assertEqual(args.run_id, "RUN")
            self.assertEqual(args.dataset_id, "DATA")

        all_args = self.parser.parse_args(
            _valid_argv("all") + [
                "--seed",
                "11",
                "--batch-size",
                "7",
                "--evaluation-batch-size",
                "5",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(all_args.seed, 11)
        self.assertEqual(all_args.batch_size, 7)
        self.assertEqual(all_args.evaluation_batch_size, 5)
        self.assertEqual(all_args.device, "cpu")
        self.assertIsInstance(all_args.run_dir, Path)

    def test_interface_exposes_no_target_kind_or_slurm_behavior(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("target-kind", source)
        self.assertNotIn("target_kind", source)
        self.assertNotIn("slurm", source)
        for command in ("generate", "train", "evaluate", "all"):
            help_text = self.parser._subparsers._group_actions[0].choices[command].format_help().lower()
            self.assertNotIn("target-kind", help_text)
            self.assertNotIn("slurm", help_text)


class TestJsonWriter(unittest.TestCase):
    def test_exact_path_sorted_json_and_overwrite_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "record.json"
            canonical_main.write_json(path, {"z": 1, "a": {"b": 2}})
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertLess(text.index('"a"'), text.index('"z"'))
            before = path.read_bytes()

            with self.assertRaises(FileExistsError):
                canonical_main.write_json(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

            canonical_main.write_json(path, {"replacement": True}, overwrite=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"replacement": True})

    def test_failed_serialization_does_not_replace_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_bytes(b"original")
            with self.assertRaises(ValueError):
                canonical_main.write_json(path, {"bad": float("nan")}, overwrite=True)
            self.assertEqual(path.read_bytes(), b"original")


class TestStandaloneDispatch(unittest.TestCase):
    def setUp(self):
        self.parser = canonical_main.build_parser()

    def test_generate_forwards_explicit_ids_and_generation_values(self):
        args = self.parser.parse_args(
            _valid_argv("generate")
            + ["--train-samples", "9", "--test-samples", "4", "--seed", "123", "--overwrite"]
        )
        expected = {
            "train_path": Path("train"),
            "test_path": Path("test"),
            "manifest_path": Path("manifest"),
        }
        with mock.patch.object(canonical_main, "generate_dataset_pair", return_value=expected) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                result = canonical_main.run_generate(args)
        self.assertIs(result, expected)
        call.assert_called_once_with(
            "RUN", "DATA", Path("out/data"), 9, 4, 123, overwrite=True
        )

    def test_train_forwards_only_train_manifest_ids_and_complete_cli_config(self):
        args = self.parser.parse_args(
            _valid_argv("train")
            + [
                "--seed",
                "123",
                "--validation-fraction",
                "0.2",
                "--candidates",
                "8,4@0.1",
                "--batch-size",
                "7",
                "--max-epochs",
                "3",
                "--patience",
                "2",
                "--learning-rate",
                "0.01",
                "--weight-decay",
                "0.02",
                "--raw-loss-weight",
                "0.3",
                "--gradient-clip-norm",
                "1.5",
                "--device",
                "cpu",
            ]
        )
        result = {
            "checkpoint_path": Path("model.pt"),
            "history_path": Path("history.json"),
            "learning_curves_path": Path("curves.png"),
            "selected_candidate": {"hidden_sizes": [8, 4], "best_validation_loss": 1.0},
            "train_samples": 8,
            "validation_samples": 2,
        }
        with mock.patch.object(canonical_main, "train_model", return_value=result) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                canonical_main.run_train(args)
        positional = call.call_args.args
        self.assertEqual(
            positional[:6],
            (
                "RUN",
                "DATA",
                "CKPT",
                Path("in/train.parquet"),
                Path("in/dataset_manifest.json"),
                Path("out/training"),
            ),
        )
        self.assertEqual(
            positional[6],
            {
                "seed": 123,
                "validation_fraction": 0.2,
                "candidates": "8,4@0.1",
                "batch_size": 7,
                "max_epochs": 3,
                "patience": 2,
                "learning_rate": 0.01,
                "weight_decay": 0.02,
                "raw_loss_weight": 0.3,
                "gradient_clip_norm": 1.5,
                "device": "cpu",
            },
        )
        self.assertNotIn("test", " ".join(str(value).lower() for value in positional))
        self.assertFalse(call.call_args.kwargs["overwrite"])

    def test_evaluate_forwards_frozen_test_inputs_and_evaluation_config(self):
        args = self.parser.parse_args(
            _valid_argv("evaluate")
            + ["--batch-size", "5", "--plot-samples", "3", "--detail-index", "2", "--device", "cpu"]
        )
        result = {
            "metrics": {"coefficient_mae": 1.0},
            "metrics_path": Path("metrics.json"),
            "predictions_path": Path("predictions.npz"),
            "comparison_path": Path("comparison.png"),
            "sample_detail_path": Path("detail.png"),
        }
        with mock.patch.object(canonical_main, "evaluate_checkpoint", return_value=result) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                canonical_main.run_evaluate(args)
        self.assertEqual(
            call.call_args.args,
            (
                "RUN",
                "DATA",
                "CKPT",
                Path("in/model.pt"),
                Path("in/test.parquet"),
                Path("in/dataset_manifest.json"),
                Path("out/evaluation"),
                {"batch_size": 5, "device": "cpu", "plot_samples": 3, "detail_index": 2},
            ),
        )
        self.assertFalse(call.call_args.kwargs["overwrite"])


class TestAllPipeline(unittest.TestCase):
    def _args(self, run_dir, extra=None):
        argv = [
            "all",
            "--run-id",
            "RUN-5",
            "--dataset-id",
            "DATA-5",
            "--checkpoint-id",
            "CKPT-5",
            "--run-dir",
            str(run_dir),
            "--train-samples",
            "6",
            "--test-samples",
            "3",
            "--seed",
            "77",
            "--validation-fraction",
            "0.25",
            "--candidates",
            "8,4@0.0",
            "--batch-size",
            "2",
            "--max-epochs",
            "2",
            "--patience",
            "1",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.001",
            "--raw-loss-weight",
            "0.4",
            "--gradient-clip-norm",
            "2.5",
            "--evaluation-batch-size",
            "3",
            "--plot-samples",
            "2",
            "--detail-index",
            "1",
            "--device",
            "cpu",
        ]
        if extra:
            argv.extend(extra)
        return canonical_main.build_parser().parse_args(argv)

    def _collaborators(self, run_dir, calls):
        paths = canonical_main._all_artifact_paths(Path(run_dir))

        def generate(run_id, dataset_id, output_dir, train_samples, test_samples, seed, overwrite=False):
            calls.append("generate")
            self.assertTrue(paths["pipeline_config"].is_file())
            self.assertFalse(paths["pipeline_summary"].exists())
            self.assertEqual((run_id, dataset_id, train_samples, test_samples, seed), ("RUN-5", "DATA-5", 6, 3, 77))
            self.assertEqual(Path(output_dir), paths["train_data"].parent)
            for key in ("train_data", "test_data"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(key.encode("ascii"))
            paths["dataset_manifest"].write_text(
                json.dumps({"run_id": run_id, "dataset_id": dataset_id}), encoding="utf-8"
            )
            return {
                "train_path": paths["train_data"],
                "test_path": paths["test_data"],
                "manifest_path": paths["dataset_manifest"],
            }

        def train(run_id, dataset_id, checkpoint_id, train_path, manifest_path, output_dir, config, overwrite=False):
            calls.append("train")
            self.assertEqual((run_id, dataset_id, checkpoint_id), ("RUN-5", "DATA-5", "CKPT-5"))
            self.assertEqual(Path(train_path), paths["train_data"])
            self.assertEqual(Path(manifest_path), paths["dataset_manifest"])
            self.assertNotIn("test", config)
            self.assertEqual(config["seed"], 77)
            self.assertEqual(config["batch_size"], 2)
            for key in ("checkpoint", "training_history", "learning_curves"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(key.encode("ascii"))
            selected = {
                "hidden_sizes": [8, 4],
                "dropout": 0.0,
                "best_epoch": 1,
                "best_validation_loss": 0.125,
            }
            return {
                "checkpoint_path": paths["checkpoint"],
                "history_path": paths["training_history"],
                "learning_curves_path": paths["learning_curves"],
                "selected_candidate": selected,
            }

        def evaluate(run_id, dataset_id, checkpoint_id, checkpoint_path, test_path, manifest_path, output_dir, config, overwrite=False):
            calls.append("evaluate")
            self.assertEqual((run_id, dataset_id, checkpoint_id), ("RUN-5", "DATA-5", "CKPT-5"))
            self.assertEqual(Path(checkpoint_path), paths["checkpoint"])
            self.assertEqual(Path(test_path), paths["test_data"])
            self.assertEqual(Path(manifest_path), paths["dataset_manifest"])
            self.assertEqual(config, {"batch_size": 3, "device": "cpu", "plot_samples": 2, "detail_index": 1})
            for key in ("metrics", "predictions", "comparison", "sample_detail"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(key.encode("ascii"))
            return {
                "metrics": {"coefficient_mae": 999.0},
                "metrics_path": paths["metrics"],
                "predictions_path": paths["predictions"],
                "comparison_path": paths["comparison"],
                "sample_detail_path": paths["sample_detail"],
            }

        return generate, train, evaluate, paths

    def test_all_writes_config_first_chains_roles_and_writes_minimal_actual_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            calls = []
            generate, train, evaluate, paths = self._collaborators(run_dir, calls)
            args = self._args(run_dir)
            with mock.patch.object(canonical_main, "generate_dataset_pair", side_effect=generate), mock.patch.object(
                canonical_main, "train_model", side_effect=train
            ), mock.patch.object(canonical_main, "evaluate_checkpoint", side_effect=evaluate):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = canonical_main.run_all(args)

            self.assertEqual(calls, ["generate", "train", "evaluate"])
            config = json.loads(paths["pipeline_config"].read_text(encoding="utf-8"))
            self.assertEqual(
                (config["run_id"], config["dataset_id"], config["checkpoint_id"]),
                ("RUN-5", "DATA-5", "CKPT-5"),
            )
            self.assertEqual(config["directories"], {"run_dir": ".", "data": "data", "training": "training", "evaluation": "evaluation"})
            self.assertEqual(config["generation"], {"train_samples": 6, "test_samples": 3, "seed": 77})
            self.assertEqual(config["training"]["validation_fraction"], 0.25)
            self.assertEqual(config["evaluation"]["batch_size"], 3)

            saved_summary = json.loads(paths["pipeline_summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary, saved_summary)
            self.assertEqual(set(saved_summary), {
                "run_id", "dataset_id", "checkpoint_id", "config", "dataset_manifest",
                "train_data", "test_data", "training_history", "checkpoint",
                "learning_curves", "metrics", "predictions", "comparison",
                "sample_detail", "selected_candidate", "best_validation_loss",
            })
            self.assertEqual(saved_summary["best_validation_loss"], 0.125)
            self.assertNotIn("test_metrics", saved_summary)
            self.assertNotIn("coefficient_mae", saved_summary)
            summary_to_artifact = {
                "config": "pipeline_config",
                "dataset_manifest": "dataset_manifest",
                "train_data": "train_data",
                "test_data": "test_data",
                "training_history": "training_history",
                "checkpoint": "checkpoint",
                "learning_curves": "learning_curves",
                "metrics": "metrics",
                "predictions": "predictions",
                "comparison": "comparison",
                "sample_detail": "sample_detail",
            }
            for summary_key, artifact_key in summary_to_artifact.items():
                resolved = (run_dir / saved_summary[summary_key]).resolve()
                self.assertEqual(resolved, paths[artifact_key].resolve())

    def test_preflight_rejects_any_existing_artifact_before_writes_or_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            paths = canonical_main._all_artifact_paths(run_dir)
            paths["metrics"].parent.mkdir(parents=True)
            paths["metrics"].write_bytes(b"user-owned")
            args = self._args(run_dir)
            with mock.patch.object(canonical_main, "generate_dataset_pair") as generate, mock.patch.object(
                canonical_main, "train_model"
            ) as train, mock.patch.object(canonical_main, "evaluate_checkpoint") as evaluate:
                with self.assertRaises(FileExistsError):
                    canonical_main.run_all(args)
            generate.assert_not_called()
            train.assert_not_called()
            evaluate.assert_not_called()
            self.assertFalse(paths["pipeline_config"].exists())
            self.assertFalse(paths["pipeline_summary"].exists())
            self.assertEqual(paths["metrics"].read_bytes(), b"user-owned")

    def test_failure_after_config_never_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            paths = canonical_main._all_artifact_paths(run_dir)
            args = self._args(run_dir)
            with mock.patch.object(canonical_main, "generate_dataset_pair", side_effect=RuntimeError("boom")), mock.patch.object(
                canonical_main, "train_model"
            ) as train, mock.patch.object(canonical_main, "evaluate_checkpoint") as evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        canonical_main.run_all(args)
            self.assertTrue(paths["pipeline_config"].is_file())
            self.assertFalse(paths["pipeline_summary"].exists())
            train.assert_not_called()
            evaluate.assert_not_called()


class TestMainEntryPoint(unittest.TestCase):
    def test_dispatches_exactly_once_and_completion_line_only_after_success(self):
        handler = mock.Mock(return_value={})
        parser = mock.Mock()
        parser.parse_args.return_value = mock.Mock(command="generate", handler=handler)
        output = io.StringIO()
        with mock.patch.object(canonical_main, "build_parser", return_value=parser):
            with contextlib.redirect_stdout(output):
                canonical_main.main()
        handler.assert_called_once_with(parser.parse_args.return_value)
        self.assertEqual(output.getvalue().strip(), "generate completed successfully.")

        failing_handler = mock.Mock(side_effect=RuntimeError("failed"))
        parser.parse_args.return_value = mock.Mock(command="all", handler=failing_handler)
        output = io.StringIO()
        with mock.patch.object(canonical_main, "build_parser", return_value=parser):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    canonical_main.main()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
