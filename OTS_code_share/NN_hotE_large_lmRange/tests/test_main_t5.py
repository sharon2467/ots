"""Acceptance tests for the canonical CLI and ID-based dispatch."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_evaluation_t4 as t4_fixture  # noqa: E402,F401

import main as canonical_main  # noqa: E402


def _valid_argv(command, run_name="output"):
    common = [
        command,
        "--run-id",
        "RUN",
        "--run-dir",
        "runs/{}".format(run_name),
        "--dataset-id",
        "DATA",
    ]
    if command in ("train", "evaluate", "all"):
        common.extend(["--checkpoint-id", "CKPT"])
    return common


def _make_dataset_registration(root, dataset_id="DATA", run_name="data_source"):
    data_dir = root / "runs" / run_name / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "dataset_train.parquet").write_bytes(b"train")
    (data_dir / "dataset_test.parquet").write_bytes(b"test")
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps({"run_id": "GENERATION_RUN", "dataset_id": dataset_id}),
        encoding="utf-8",
    )
    entry = root / "registry" / "datasets" / dataset_id
    entry.parent.mkdir(parents=True)
    entry.symlink_to("../../runs/{}/data".format(run_name), target_is_directory=True)
    return data_dir


def _make_checkpoint_registration(root, checkpoint_id="CKPT", run_name="train_source"):
    training_dir = root / "runs" / run_name / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "model.pt").write_bytes(b"checkpoint")
    entry = root / "registry" / "checkpoints" / checkpoint_id
    entry.parent.mkdir(parents=True)
    entry.symlink_to("../../runs/{}/training".format(run_name), target_is_directory=True)
    return training_dir


class TestParserContract(unittest.TestCase):
    def setUp(self):
        self.parser = canonical_main.build_parser()

    def test_exact_four_subcommands_and_required_output_run(self):
        action = next(
            action
            for action in self.parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        )
        self.assertEqual(set(action.choices), {"generate", "train", "evaluate", "all"})
        for command in action.choices:
            for option in ("--run-id", "--run-dir", "--dataset-id"):
                argv = _valid_argv(command)
                index = argv.index(option)
                del argv[index : index + 2]
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.parser.parse_args(argv)

    def test_checkpoint_id_roles_and_removed_path_options(self):
        generate = self.parser.parse_args(_valid_argv("generate"))
        self.assertFalse(hasattr(generate, "checkpoint_id"))
        for command in ("train", "evaluate", "all"):
            args = self.parser.parse_args(_valid_argv(command))
            self.assertEqual(args.checkpoint_id, "CKPT")
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for option in (
            "--output-dir",
            "--train-data",
            "--test-data",
            "--dataset-manifest",
            '"--checkpoint"',
        ):
            self.assertNotIn(option, source)

    def test_all_uses_separate_training_and_evaluation_batch_options(self):
        args = self.parser.parse_args(
            _valid_argv("all")
            + ["--batch-size", "7", "--evaluation-batch-size", "5", "--device", "cpu"]
        )
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(args.evaluation_batch_size, 5)
        self.assertEqual(args.run_dir, Path("runs/output"))


class TestJsonWriter(unittest.TestCase):
    def test_sorted_atomic_json_and_overwrite_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            canonical_main.write_json(path, {"z": 1, "a": 2})
            self.assertLess(
                path.read_text(encoding="utf-8").index('"a"'),
                path.read_text(encoding="utf-8").index('"z"'),
            )
            with self.assertRaises(FileExistsError):
                canonical_main.write_json(path, {"replacement": True})
            canonical_main.write_json(path, {"replacement": True}, overwrite=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"replacement": True})


class TestStandaloneDispatch(unittest.TestCase):
    def test_generate_writes_current_run_and_publishes_two_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = canonical_main.build_parser().parse_args(
                _valid_argv("generate")
                + ["--train-samples", "3", "--test-samples", "2", "--seed", "9"]
            )

            def generate(run_id, dataset_id, output_dir, *unused, **kwargs):
                output_dir.mkdir(parents=True)
                train_path = output_dir / "dataset_train.parquet"
                test_path = output_dir / "dataset_test.parquet"
                manifest_path = output_dir / "dataset_manifest.json"
                train_path.write_bytes(b"train")
                test_path.write_bytes(b"test")
                manifest_path.write_text("{}", encoding="utf-8")
                return {
                    "train_path": train_path,
                    "test_path": test_path,
                    "manifest_path": manifest_path,
                }

            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                canonical_main, "generate_dataset_pair", side_effect=generate
            ):
                canonical_main.run_generate(args)
            self.assertEqual((root / "registry/runs/RUN").resolve(), root / "runs/output")
            self.assertEqual(
                (root / "registry/datasets/DATA").resolve(), root / "runs/output/data"
            )

    def test_train_resolves_only_dataset_input_and_publishes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = _make_dataset_registration(root)
            args = canonical_main.build_parser().parse_args(_valid_argv("train"))

            def train(*positional, **kwargs):
                self.assertEqual(positional[0:3], ("RUN", "DATA", "CKPT"))
                self.assertEqual(Path(positional[3]), data_dir / "dataset_train.parquet")
                self.assertEqual(Path(positional[4]), data_dir / "dataset_manifest.json")
                output_dir = Path(positional[5])
                output_dir.mkdir(parents=True)
                for name in ("model.pt", "training_history.json", "learning_curves.png"):
                    (output_dir / name).write_bytes(b"artifact")
                return {
                    "checkpoint_path": output_dir / "model.pt",
                    "history_path": output_dir / "training_history.json",
                    "learning_curves_path": output_dir / "learning_curves.png",
                    "selected_candidate": {"best_validation_loss": 1.0},
                }

            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                canonical_main, "train_model", side_effect=train
            ):
                canonical_main.run_train(args)
            self.assertEqual(
                (root / "registry/checkpoints/CKPT").resolve(), root / "runs/output/training"
            )

    def test_evaluate_resolves_dataset_and_checkpoint_but_publishes_only_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = _make_dataset_registration(root)
            training_dir = _make_checkpoint_registration(root)
            args = canonical_main.build_parser().parse_args(_valid_argv("evaluate"))

            def evaluate(*positional, **kwargs):
                self.assertEqual(Path(positional[3]), training_dir / "model.pt")
                self.assertEqual(Path(positional[4]), data_dir / "dataset_test.parquet")
                output_dir = Path(positional[6])
                output_dir.mkdir(parents=True)
                outputs = {
                    "metrics_path": output_dir / "metrics.json",
                    "predictions_path": output_dir / "predictions.npz",
                    "comparison_path": output_dir / "comparison.png",
                    "sample_detail_path": output_dir / "sample_detail.png",
                }
                for path in outputs.values():
                    path.write_bytes(b"artifact")
                return {"metrics": {"ok": 1}, **outputs}

            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                canonical_main, "evaluate_checkpoint", side_effect=evaluate
            ):
                canonical_main.run_evaluate(args)
            self.assertTrue((root / "registry/runs/RUN").is_symlink())
            self.assertEqual(
                {path.name for path in (root / "registry/checkpoints").iterdir()}, {"CKPT"}
            )


class TestAllPipeline(unittest.TestCase):
    def test_all_publishes_only_after_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = canonical_main.build_parser().parse_args(
                _valid_argv("all")
                + [
                    "--train-samples",
                    "3",
                    "--test-samples",
                    "2",
                    "--candidates",
                    "8,4",
                    "--max-epochs",
                    "1",
                ]
            )
            run_dir = root / "runs/output"
            paths = canonical_main._all_artifact_paths(run_dir)

            def generate(*unused, **kwargs):
                paths["train_data"].parent.mkdir(parents=True)
                paths["train_data"].write_bytes(b"train")
                paths["test_data"].write_bytes(b"test")
                paths["dataset_manifest"].write_text("{}", encoding="utf-8")
                return {
                    "train_path": paths["train_data"],
                    "test_path": paths["test_data"],
                    "manifest_path": paths["dataset_manifest"],
                }

            def train(*unused, **kwargs):
                paths["checkpoint"].parent.mkdir(parents=True)
                for name in ("checkpoint", "training_history", "learning_curves"):
                    paths[name].write_bytes(b"artifact")
                return {
                    "checkpoint_path": paths["checkpoint"],
                    "history_path": paths["training_history"],
                    "learning_curves_path": paths["learning_curves"],
                    "selected_candidate": {"best_validation_loss": 0.5},
                }

            def evaluate(*unused, **kwargs):
                paths["metrics"].parent.mkdir(parents=True)
                for name in ("metrics", "predictions", "comparison", "sample_detail"):
                    paths[name].write_bytes(b"artifact")
                return {
                    "metrics_path": paths["metrics"],
                    "predictions_path": paths["predictions"],
                    "comparison_path": paths["comparison"],
                    "sample_detail_path": paths["sample_detail"],
                    "metrics": {"ok": 1},
                }

            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                canonical_main, "generate_dataset_pair", side_effect=generate
            ), mock.patch.object(canonical_main, "train_model", side_effect=train), mock.patch.object(
                canonical_main, "evaluate_checkpoint", side_effect=evaluate
            ):
                summary = canonical_main.run_all(args)
            self.assertTrue(paths["pipeline_summary"].is_file())
            self.assertIn("registry", summary)
            self.assertEqual((root / "registry/runs/RUN").resolve(), run_dir)
            self.assertEqual((root / "registry/datasets/DATA").resolve(), run_dir / "data")
            self.assertEqual(
                (root / "registry/checkpoints/CKPT").resolve(), run_dir / "training"
            )

    def test_failure_never_publishes_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = canonical_main.build_parser().parse_args(_valid_argv("all"))
            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                canonical_main, "generate_dataset_pair", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    canonical_main.run_all(args)
            self.assertFalse((root / "registry").exists())


class TestMainEntryPoint(unittest.TestCase):
    def test_completion_line_only_after_success(self):
        handler = mock.Mock(return_value={})
        parser = mock.Mock()
        parser.parse_args.return_value = mock.Mock(command="generate", handler=handler)
        output = io.StringIO()
        with mock.patch.object(canonical_main, "build_parser", return_value=parser):
            with contextlib.redirect_stdout(output):
                canonical_main.main()
        self.assertEqual(output.getvalue().strip(), "generate completed successfully.")


if __name__ == "__main__":
    unittest.main()
