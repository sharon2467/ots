"""Targeted acceptance tests for canonical-pipeline T6 legacy cleanup only."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Install the accepted local dependency fixtures before importing the canonical
# entry modules.  T6 tests cleanup and retained-callability, not dependency
# packaging; the available Torch runtime lacks PyArrow and Matplotlib.
from tests import test_main_t5 as t5_fixture  # noqa: E402,F401

import data  # noqa: E402
import evaluation  # noqa: E402
import main as canonical_main  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


DELETED_ROOT_FILES = {
    "train_v2.py",
    "evaluateModel.py",
    "quick_test.py",
    "genDataSet.py",
    "genDataSet_full_fourier.py",
    "readData.py",
}
DELETED_MODULES = {
    "train_v2",
    "evaluateModel",
    "quick_test",
    "genDataSet",
    "genDataSet_full_fourier",
    "readData",
}
CANONICAL_ROOT_FILES = {"data.py", "evaluation.py", "main.py", "model.py", "train.py"}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


class TestExplicitDeletionList(unittest.TestCase):
    def test_exact_legacy_root_files_are_absent_and_canonical_root_is_unambiguous(self):
        for filename in sorted(DELETED_ROOT_FILES):
            with self.subTest(filename=filename):
                self.assertFalse((PROJECT_ROOT / filename).exists())

        root_python = {path.name for path in PROJECT_ROOT.glob("*.py")}
        self.assertEqual(root_python, CANONICAL_ROOT_FILES)

    def test_deleted_modules_are_not_resolvable_from_project_root(self):
        for module_name in sorted(DELETED_MODULES):
            with self.subTest(module=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

    def test_git_deletion_scope_contains_only_the_six_specified_root_files(self):
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=D", "--", str(PROJECT_ROOT.relative_to(REPOSITORY_ROOT))],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        deleted = {
            Path(line.strip()).name
            for line in result.stdout.splitlines()
            if line.strip()
        }
        self.assertEqual(deleted, DELETED_ROOT_FILES)


class TestNoRetainedLegacyCaller(unittest.TestCase):
    def test_retained_python_sources_do_not_import_deleted_modules(self):
        offenders = {}
        for path in PROJECT_ROOT.rglob("*.py"):
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts[0] in {"tests", "slurm", "__pycache__"}:
                continue
            legacy_imports = _imports(path) & DELETED_MODULES
            if legacy_imports:
                offenders[relative.as_posix()] = sorted(legacy_imports)
        self.assertEqual(offenders, {})

    def test_canonical_dependency_direction_and_no_legacy_runtime_symbols(self):
        imports = {
            name: _imports(PROJECT_ROOT / (name + ".py"))
            for name in ("data", "model", "train", "evaluation", "main")
        }
        self.assertIn("phyEqns", imports["data"])
        self.assertNotIn("evaluation", imports["train"])
        self.assertEqual(imports["train"] & {"data", "model"}, {"data", "model"})
        self.assertEqual(imports["evaluation"] & {"data", "model", "train"}, {"data", "model"})
        self.assertEqual(imports["main"] & {"data", "train", "evaluation"}, {"data", "train", "evaluation"})

        self.assertFalse(hasattr(model, "ModelOTS"))
        self.assertFalse(hasattr(model, "save_model"))
        self.assertFalse(hasattr(model, "load_model"))
        self.assertFalse(hasattr(train, "Loss_func"))
        self.assertFalse(hasattr(train, "train"))
        self.assertTrue(hasattr(model, "OTSRegressor"))
        self.assertTrue(hasattr(train, "train_model"))
        self.assertTrue(hasattr(evaluation, "evaluate_checkpoint"))

    def test_active_core_contains_no_pickle_or_legacy_entrypoint_reference(self):
        forbidden = {
            "pickle",
            "ots_te_model.pickle",
            "train_v2",
            "evaluatemodel",
            "quick_test",
            "gendataset",
            "readdata",
            "modelots",
            "loss_func",
        }
        offenders = {}
        for filename in sorted(CANONICAL_ROOT_FILES):
            text = (PROJECT_ROOT / filename).read_text(encoding="utf-8").lower()
            found = sorted(token for token in forbidden if token in text)
            # model.py must explicitly reject the old ``pet1`` schema, but it
            # should not retain pickle APIs or the old ModelOTS implementation.
            if found:
                offenders[filename] = found
        self.assertEqual(offenders, {})


class TestRetainedBoundary(unittest.TestCase):
    def test_required_physics_paper_reproduction_and_reports_are_preserved(self):
        required = [
            PROJECT_ROOT / "phyEqns" / "ots_fourier.py",
            PROJECT_ROOT / "phyEqns" / "ots_full_fourier.py",
            PROJECT_ROOT / "paper_reproduction" / "generate_paper_dataset.py",
            PROJECT_ROOT / "paper_reproduction" / "train_paper_fig3.py",
            PROJECT_ROOT / "paper_reproduction" / "reproduce_paper_fig3.py",
            PROJECT_ROOT / "paper_reproduction" / "diagnose_paper_fig3_supergaussian.py",
            PROJECT_ROOT / "paper_reproduction" / "PAPER_FIG3_REPRODUCTION_REPORT.md",
            PROJECT_ROOT / "REMOTE_ARTIFACT_LAYOUT.md",
            PROJECT_ROOT / "TRAINING_RECOVERY_REPORT.md",
            REPOSITORY_ROOT / ".git",
        ]
        missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in required if not path.exists()]
        self.assertEqual(missing, [])

    def test_slurm_worktree_is_untouched_and_known_scripts_remain(self):
        slurm_dir = PROJECT_ROOT / "slurm"
        required_scripts = {
            "train_legacy.slurm",
            "train_v2.slurm",
            "train_v2_capacity.slurm",
            "train_v2_rawloss.slurm",
            "train_v2_bscc.slurm",
            "submit_v2.py",
            "monitor_v2.py",
        }
        self.assertTrue(required_scripts.issubset({path.name for path in slurm_dir.iterdir()}))
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(slurm_dir.relative_to(REPOSITORY_ROOT))],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")


class TestCoreEntryAfterCleanup(unittest.TestCase):
    def test_all_retained_non_test_python_sources_compile(self):
        failures = {}
        for path in PROJECT_ROOT.rglob("*.py"):
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts[0] in {"tests", "slurm", "__pycache__"}:
                continue
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as error:
                failures[relative.as_posix()] = str(error)
        self.assertEqual(failures, {})

    def test_canonical_main_help_and_dispatch_still_run(self):
        parser = canonical_main.build_parser()
        help_text = parser.format_help()
        for command in ("generate", "train", "evaluate", "all"):
            self.assertIn(command, help_text)

        handler = mock.Mock(return_value={})
        fake_parser = mock.Mock()
        fake_args = mock.Mock(command="generate", handler=handler)
        fake_parser.parse_args.return_value = fake_args
        output = io.StringIO()
        with mock.patch.object(canonical_main, "build_parser", return_value=fake_parser):
            with contextlib.redirect_stdout(output):
                canonical_main.main()
        handler.assert_called_once_with(fake_args)
        self.assertEqual(output.getvalue().strip(), "generate completed successfully.")


if __name__ == "__main__":
    unittest.main()
