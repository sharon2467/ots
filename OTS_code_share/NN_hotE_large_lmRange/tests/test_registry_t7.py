"""Registry safety, publication and retry tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock


from tests import test_main_t5 as t5_fixture  # noqa: F401

import main as canonical_main


class TestRegistryIdentifiers(unittest.TestCase):
    def test_only_safe_single_segments_are_accepted(self):
        for valid in ("RUN", "run-1", "dataset_2", "checkpoint.v1"):
            self.assertEqual(canonical_main._validate_registry_id(valid, "id"), valid)
        for invalid in ("", ".", "..", "/abs", "a/b", "a\\b", " spaced", "a b"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    canonical_main._validate_registry_id(invalid, "id")

    def test_run_dir_is_exactly_one_direct_runs_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                self.assertEqual(
                    canonical_main._validate_run_dir(Path("runs/example")),
                    root / "runs/example",
                )
                for invalid in ("example", "runs", "runs/a/b", "../runs/a", root / "runs/a"):
                    with self.subTest(invalid=invalid):
                        with self.assertRaises(ValueError):
                            canonical_main._validate_run_dir(Path(invalid))


class TestRegistryResolution(unittest.TestCase):
    def test_dataset_link_must_exist_and_stay_inside_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            outside = root / "outside/data"
            outside.mkdir(parents=True)
            entry = root / "registry/datasets/DATA"
            entry.parent.mkdir(parents=True)
            entry.symlink_to(outside, target_is_directory=True)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    canonical_main._resolve_dataset("DATA")

    def test_broken_and_non_link_dataset_entries_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "registry/datasets/DATA"
            entry.parent.mkdir(parents=True)
            entry.symlink_to("../../runs/missing/data", target_is_directory=True)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "broken"):
                    canonical_main._resolve_dataset("DATA")
            entry.unlink()
            entry.mkdir()
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    canonical_main._resolve_dataset("DATA")

    def test_historical_checkpoint_directory_preserves_original_model_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "runs/legacy/model-v2.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"legacy")
            registration = root / "registry/checkpoints/job_1"
            registration.mkdir(parents=True)
            (registration / "model.pt").symlink_to("../../../runs/legacy/model-v2.pt")
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                resolved = canonical_main._resolve_checkpoint("job_1")
            self.assertEqual(resolved["checkpoint_path"], model)
            self.assertEqual(resolved["source_run_dir"], "legacy")


class TestRegistryPublication(unittest.TestCase):
    def test_relative_links_are_published_and_never_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/run_a"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                canonical_main._publish_registry_entries(
                    [("runs", "RUN", run_dir), ("datasets", "DATA", data_dir)]
                )
                self.assertEqual((root / "registry/runs/RUN").readlink(), Path("../../runs/run_a"))
                self.assertEqual(
                    (root / "registry/datasets/DATA").readlink(),
                    Path("../../runs/run_a/data"),
                )
                with self.assertRaises(FileExistsError):
                    canonical_main._publish_registry_entries([("runs", "RUN", run_dir)])

    def test_partial_publication_failure_rolls_back_only_new_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/run_a"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True)
            original = Path.symlink_to
            calls = {"count": 0}

            def fail_second(path, target, target_is_directory=False):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected publication failure")
                return original(path, target, target_is_directory=target_is_directory)

            with mock.patch.object(canonical_main, "_project_root", return_value=root), mock.patch.object(
                Path, "symlink_to", new=fail_second
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    canonical_main._publish_registry_entries(
                        [("runs", "RUN", run_dir), ("datasets", "DATA", data_dir)]
                    )
            self.assertFalse((root / "registry/runs/RUN").exists())
            self.assertFalse((root / "registry/datasets/DATA").exists())


class TestRunPreflight(unittest.TestCase):
    def test_precreated_slurm_logs_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/output"
            run_dir.mkdir(parents=True)
            (run_dir / "slurm-123.out").write_text("", encoding="utf-8")
            (run_dir / "slurm-123.err").write_text("", encoding="utf-8")
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                result = canonical_main._preflight_output(
                    "all",
                    "RUN",
                    Path("runs/output"),
                    [("datasets", "DATA"), ("checkpoints", "CKPT")],
                    False,
                )
            self.assertEqual(result, run_dir)

    def test_pipeline_artifacts_require_overwrite_and_unregistered_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/output"
            (run_dir / "training").mkdir(parents=True)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                with self.assertRaises(FileExistsError):
                    canonical_main._preflight_output(
                        "train", "RUN", Path("runs/output"), [("checkpoints", "CKPT")], False
                    )
                self.assertEqual(
                    canonical_main._preflight_output(
                        "train", "RUN", Path("runs/output"), [("checkpoints", "CKPT")], True
                    ),
                    run_dir,
                )

    def test_existing_run_registration_blocks_retry_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/output"
            run_dir.mkdir(parents=True)
            entry = root / "registry/runs/OLD"
            entry.parent.mkdir(parents=True)
            entry.symlink_to("../../runs/output", target_is_directory=True)
            with mock.patch.object(canonical_main, "_project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "already registered"):
                    canonical_main._preflight_output(
                        "generate", "NEW", Path("runs/output"), [("datasets", "DATA")], True
                    )


if __name__ == "__main__":
    unittest.main()
