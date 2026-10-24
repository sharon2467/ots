#!/usr/bin/env python3
"""Tiny synthetic continuation tests; no official training or test-data reads."""

import copy
import inspect
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

import adapter
import resume_adapter as resume
from data import OTSDataset


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), "Tensor mismatch"
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)


def must_fail(action):
    try:
        action()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Expected rejection did not occur")


class SyntheticPause(adapter.EpochSnapshots):
    def __init__(self, stop_at=None):
        super().__init__()
        self.stop_at = stop_at

    def boundary(self, frame, final=False):
        if self.stop_at is not None and len(frame.f_locals.get("history", [])) == self.stop_at:
            self.pause_requested = "synthetic_validation_pause"
        super().boundary(frame, final=final)


def main():
    torch.set_num_threads(1)
    adapter.install_adapter()
    original_load = resume.TRAIN.load_dataset
    original_epoch = resume.TRAIN.run_epoch
    protected = [adapter.SOURCE / name for name in ("train.py", "model.py", "main.py", "evaluation.py")]
    protected += [adapter.EXPERIMENT_ROOT / name for name in ("adapter.py", "run_trial.py", "analyze_depth.py")]
    protected_before = {str(path): resume.digest(path) for path in protected}
    official = []
    for label, widths in (("d2", "2048x1024"), ("d4", "2048x1024x1024x1024")):
        path = resume.ARCHIVE / ("runs/depth_fig_" + label + "_s1/training/depth_runtime/" + widths + "/snapshot.pt")
        resume.verify_archived_snapshot(path)
        controller = resume.ResumeController(path)
        official.append({"run_id": controller.snapshot["checkpoint_metadata"]["run_id"],
                         "snapshot_sha256": controller.snapshot_sha256,
                         "completed_epochs": controller.snapshot["completed_epochs"],
                         "next_epoch": controller.snapshot["completed_epochs"] + 1,
                         "best_epoch": controller.snapshot["best_epoch"]})
    layout_meta = controller.snapshot["checkpoint_metadata"]
    generator = torch.Generator().manual_seed(8403)
    dataset = OTSDataset(
        torch.randn(18, 319, generator=generator),
        adapter.canonical_model.hermitian_project(torch.randn(18, 480, generator=generator)),
        layout_meta["wavelengths_nm"], layout_meta["k_grid"], layout_meta["layout"],
    )
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="resume_synthetic_", dir=adapter.EXPERIMENT_ROOT) as temp:
            temp = Path(temp)
            train_path = temp / "synthetic_train_only.parquet"
            train_path.write_bytes(b"Synthetic in-memory loader fixture; no physical data or test file.")
            manifest_path = temp / "synthetic_manifest.json"
            manifest_path.write_text(json.dumps({"run_id": "synthetic_source", "dataset_id": "synthetic",
                "splits": {"train": {"file": train_path.name}}}))

            def synthetic_loader(path, physical=None):
                assert Path(path) == train_path
                return dataset

            resume.TRAIN.load_dataset = synthetic_loader
            for label, sizes in (("d2", [8, 6]), ("d4", [8, 6, 5, 4])):
                for plateau in (False, True):
                    case = label + ("_plateau_early_stop" if plateau else "_actual_validation")
                    base = temp / case
                    config = {"seed": 20261032, "validation_fraction": 0.15,
                              "candidates": [{"hidden_sizes": sizes, "dropout": 0.3}],
                              "batch_size": 4, "max_epochs": 8, "patience": 3 if plateau else 8,
                              "learning_rate": 0.003, "raw_loss_weight": 0.4, "device": "cpu",
                              "scheduler": {"parameters": {"patience": 0}}}
                    actual_validation_losses = []
                    auxiliary_rng_draws = []

                    def recorded_epoch(*args, **kwargs):
                        if kwargs["training"]:
                            auxiliary_rng_draws.append((random.random(), float(np.random.random())))
                        value = original_epoch(*args, **kwargs)
                        if not kwargs["training"]:
                            actual_validation_losses.append(value)
                            if plateau:
                                epoch = inspect.currentframe().f_back.f_locals["epoch"]
                                return 1.0 if epoch == 1 else 1.0 + epoch * 0.01
                        return value

                    resume.TRAIN.run_epoch = recorded_epoch
                    resume.TRAIN.train_candidate = resume.ORIGINAL_CANDIDATE

                    def execute(output, stop=None, snapshot=None):
                        if snapshot is None:
                            resume.TRAIN.train_candidate = resume.ORIGINAL_CANDIDATE
                        else:
                            control = resume.install_resume(snapshot)
                            assert control.audit["epoch_body_ast_unchanged"]
                            assert control.audit["whole_ast_reverses_with_exactly_two_edits"]
                        try:
                            with SyntheticPause(stop):
                                resume.TRAIN.train_model(case, "synthetic", case + "_model", train_path,
                                                        manifest_path, output, config)
                        except SystemExit as error:
                            assert stop is not None and error.code == 75
                        else:
                            assert stop is None

                    continuous = base / "continuous/training"
                    interrupted = base / "interrupted/training"
                    execute(continuous)
                    continuous_losses = list(actual_validation_losses)
                    continuous_auxiliary_rng = list(auxiliary_rng_draws)
                    actual_validation_losses.clear()
                    auxiliary_rng_draws.clear()
                    execute(interrupted, stop=2)
                    widths = "x".join(map(str, sizes))
                    snapshot_path = interrupted / "depth_runtime" / widths / "snapshot.pt"
                    first = torch.load(snapshot_path, weights_only=True)
                    assert first["completed_epochs"] == 2
                    history_prefix = copy.deepcopy(first["history"])
                    # Deliberately disturb all global RNGs and reinitialize setup.
                    torch.rand(19)
                    np.random.random(19)
                    random.random()
                    execute(interrupted, stop=3, snapshot=snapshot_path)
                    second = torch.load(snapshot_path, weights_only=True)
                    equal(second["history"][:2], history_prefix)
                    assert second["completed_epochs"] == 3
                    execute(interrupted, snapshot=snapshot_path)
                    equal(actual_validation_losses, continuous_losses)
                    equal(auxiliary_rng_draws, continuous_auxiliary_rng)
                    full = torch.load(continuous / "depth_runtime" / widths / "snapshot.pt", weights_only=True)
                    restored = torch.load(snapshot_path, weights_only=True)
                    checks = ("current_state", "best_state", "optimizer_state", "scheduler_state", "history",
                              "best_epoch", "best_validation_loss", "epochs_without_improvement", "torch_rng",
                              "python_rng", "numpy_rng", "loader_generator_rng", "cuda_rng", "checkpoint_metadata")
                    for key in checks:
                        equal(full[key], restored[key])
                    full_checkpoint = torch.load(continuous / "model.pt", weights_only=True)
                    resumed_checkpoint = torch.load(interrupted / "model.pt", weights_only=True)
                    equal(full_checkpoint, resumed_checkpoint)
                    if plateau:
                        assert full["completed_epochs"] == 4
                        assert full["epochs_without_improvement"] == 3
                        assert full["scheduler_state"]["_last_lr"][0] < config["learning_rate"]
                    # Completed max-epoch and exhausted-patience snapshots must never
                    # execute an extra epoch merely because resume was requested.
                    must_fail(lambda: resume.ResumeController(snapshot_path))
                    changed = copy.deepcopy(first)
                    changed["history"][0]["epoch"] = 2
                    invalid = base / "invalid.pt"
                    torch.save(changed, invalid)
                    must_fail(lambda: resume.ResumeController(invalid))
                    for key in ("configuration", "statistics"):
                        changed = copy.deepcopy(first)
                        if key == "configuration":
                            changed["checkpoint_metadata"]["training_config"]["raw_loss_weight"] = 0.7
                        else:
                            changed["checkpoint_metadata"]["x_mean"] = changed["checkpoint_metadata"]["x_mean"] + 1
                        torch.save(changed, invalid)
                        mismatch_output = base / ("mismatched_" + key) / "training"
                        must_fail(lambda: execute(mismatch_output, snapshot=invalid))
                        assert not (mismatch_output / "depth_runtime/resume_control.json").exists()
                    cases.append({"case": case, "hidden_sizes": sizes, "dropout": 0.3,
                                  "completed_epochs": full["completed_epochs"],
                                  "interruptions_after_epochs": [2, 3],
                                  "all_checkpoint_fields_bitwise_equal": True,
                                  "all_actual_validation_losses_bitwise_equal": True,
                                  "current_best_optimizer_scheduler_rng_history_bitwise_equal": True,
                                  "nontrivial_python_numpy_rng_draws_bitwise_equal": True,
                                  "changed_configuration_and_statistics_rejected_before_output": True,
                                  "completed_snapshot_and_invalid_history_rejected": True,
                                  "ast_audit": json.loads((interrupted / "depth_runtime/resume_control.json").read_text())["ast_audit"]})
    finally:
        resume.TRAIN.load_dataset = original_load
        resume.TRAIN.run_epoch = original_epoch
        resume.TRAIN.train_candidate = resume.ORIGINAL_CANDIDATE
    assert protected_before == {str(path): resume.digest(path) for path in protected}
    report = {
        "status": "passed", "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "script_sha256": {name: resume.digest(adapter.EXPERIMENT_ROOT / name)
                          for name in ("resume_adapter.py", "validate_resume.py")},
        "cases": cases, "official_snapshot_read_only_checks": official,
        "protected_files_unchanged": True, "official_training_started": False,
        "test_data_read": False,
        "validation_scope": "tiny synthetic train/validation arrays; original train_model and epoch body; two pauses and resumes per case",
    }
    destination = adapter.EXPERIMENT_ROOT / "resume_validation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
