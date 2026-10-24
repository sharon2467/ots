#!/usr/bin/env python3
"""Resume an archived complete-epoch snapshot using the canonical epoch body."""

import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import textwrap

import adapter
import numpy as np
import torch


TRAIN = adapter.canonical_train
ORIGINAL_CANDIDATE = TRAIN.train_candidate
ARCHIVE = adapter.EXPERIMENT_ROOT / "resume_archive_20261021"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_digest(value):
    return hashlib.sha256(ast.dump(value, include_attributes=False).encode()).hexdigest()


def transform_candidate(restore):
    """Insert state loading and change range start; preserve the entire epoch AST."""
    source, start = inspect.getsourcelines(ORIGINAL_CANDIDATE)
    tree = ast.parse(textwrap.dedent("".join(source)))
    function = tree.body[0]
    loops = [node for node in function.body if isinstance(node, ast.For)]
    if len(loops) != 1:
        raise RuntimeError("Canonical candidate must have exactly one top-level loop.")
    loop = loops[0]
    expected_range = ast.parse('range(1, config["max_epochs"] + 1)', mode="eval").body
    if ast.dump(loop.iter) != ast.dump(expected_range):
        raise RuntimeError("Canonical epoch range changed; explicit review required.")
    original_loop_body = ast_digest(ast.Module(body=loop.body, type_ignores=[]))
    unchanged = copy.deepcopy(tree)
    injection = ast.parse(
        "history, best_epoch, best_validation_loss, best_state, "
        "epochs_without_improvement, _resume_start = _depth_resume_restore(locals())"
    ).body[0]
    # Place on the original blank line before the loop so the unchanged snapshot
    # trace sees only restored values at its original for-line boundary.
    for node in ast.walk(injection):
        if hasattr(node, "lineno"):
            node.lineno = loop.lineno - 1
            node.end_lineno = loop.lineno - 1
    function.body.insert(function.body.index(loop), injection)
    loop.iter.args[0] = ast.copy_location(ast.Name(id="_resume_start", ctx=ast.Load()), loop.iter.args[0])
    assert ast_digest(ast.Module(body=loop.body, type_ignores=[])) == original_loop_body
    # Reversing exactly these two authorized edits must reproduce the whole AST.
    reversed_tree = copy.deepcopy(tree)
    reversed_function = reversed_tree.body[0]
    reversed_loop = next(node for node in reversed_function.body if isinstance(node, ast.For))
    reversed_loop.iter.args[0] = ast.Constant(value=1)
    reversed_function.body.pop(reversed_function.body.index(reversed_loop) - 1)
    if ast.dump(reversed_tree) != ast.dump(unchanged):
        raise AssertionError("Unexpected AST change outside state loading/range start.")
    ast.increment_lineno(tree, start - 1)
    ast.fix_missing_locations(tree)
    namespace = dict(ORIGINAL_CANDIDATE.__globals__)
    namespace["_depth_resume_restore"] = restore
    exec(compile(tree, ORIGINAL_CANDIDATE.__code__.co_filename, "exec"), namespace)
    audit = {
        "canonical_train_source_sha256": digest(adapter.SOURCE / "train.py"),
        "original_candidate_ast_sha256": ast_digest(unchanged),
        "transformed_candidate_ast_sha256": ast_digest(tree),
        "epoch_body_ast_sha256": original_loop_body,
        "epoch_body_ast_unchanged": True,
        "whole_ast_reverses_with_exactly_two_edits": True,
        "edits": ["insert pre-loop state restore tuple assignment", "range starts at completed_epochs + 1"],
    }
    return namespace[ORIGINAL_CANDIDATE.__name__], audit


class ResumeController:
    def __init__(self, snapshot_path):
        self.path = Path(snapshot_path).resolve(strict=True)
        self.snapshot_sha256 = digest(self.path)
        self.snapshot = torch.load(self.path, map_location="cpu", weights_only=True)
        state = self.snapshot
        if state.get("format") != "ots_depth_runtime_snapshot_v1":
            raise ValueError("Unknown runtime snapshot format.")
        required = {"completed_epochs", "checkpoint_metadata", "current_state", "best_state", "best_epoch",
                    "best_validation_loss", "history", "epochs_without_improvement", "optimizer_state",
                    "scheduler_state", "torch_rng", "python_rng", "numpy_rng", "loader_generator_rng",
                    "cuda_rng", "train_data_path", "manifest_path"}
        if required - set(state):
            raise ValueError("Snapshot lacks required recovery state: {}".format(sorted(required - set(state))))
        completed = state["completed_epochs"]
        if not isinstance(completed, int) or isinstance(completed, bool) or completed < 1:
            raise ValueError("Resume requires at least one complete epoch.")
        if [row["epoch"] for row in state["history"]] != list(range(1, completed + 1)):
            raise ValueError("Snapshot history does not cover its completed epochs exactly.")
        meta = state["checkpoint_metadata"]
        checkpoint = dict(meta, model_state=state["best_state"], best_epoch=state["best_epoch"],
                          best_validation_loss=state["best_validation_loss"])
        adapter.canonical_model.validate_checkpoint(checkpoint)
        if len(meta["training_config"]["candidates"]) != 1:
            raise ValueError("Resume supports exactly one configured candidate.")
        if completed >= meta["training_config"]["max_epochs"]:
            raise ValueError("Snapshot already reached its original maximum epoch.")
        if state["epochs_without_improvement"] >= meta["training_config"]["patience"]:
            raise ValueError("Snapshot already exhausted early-stopping patience.")
        self.restored = False
        self.audit = None

    def restore(self, current):
        if self.restored:
            raise RuntimeError("Snapshot cannot initialize multiple candidates.")
        state, meta = self.snapshot, self.snapshot["checkpoint_metadata"]
        frame = inspect.currentframe().f_back
        parent = frame.f_back
        while parent is not None and parent.f_code is not TRAIN.train_model.__code__:
            parent = parent.f_back
        if parent is None:
            raise RuntimeError("Resume must be invoked through canonical train_model.")
        training = parent.f_locals
        for key in ("run_id", "dataset_id", "checkpoint_id", "input_dim", "output_dim", "fourier_modes"):
            if training[key] != meta[key]:
                raise ValueError("Resume request differs from snapshot: " + key)
        if current["config"] != meta["training_config"]:
            raise ValueError("Resume must use the exact original resolved configuration.")
        if list(current["hidden_sizes"]) != meta["hidden_sizes"] or current["dropout"] != meta["dropout"]:
            raise ValueError("Resume architecture differs from snapshot.")
        if str(current["device"]) != meta["device"] or str(training["x_train"].dtype).replace("torch.", "") != meta["dtype"]:
            raise ValueError("Resume device/dtype differs from snapshot.")
        for key in ("train_data_path", "manifest_path"):
            if Path(training[key]).resolve() != Path(state[key]).resolve():
                raise ValueError("Resume input path differs from snapshot: " + key)
        if training["dataset"].layout != meta.get("layout"):
            raise ValueError("Resume dataset layout differs from snapshot.")
        for key in ("k_grid", "wavelengths_nm"):
            if not torch.equal(getattr(training["dataset"], key), meta[key]):
                raise ValueError("Resume coordinate differs from snapshot: " + key)
        for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            if not torch.equal(current["statistics"][key].cpu(), meta[key]):
                raise ValueError("Resume standardization differs from snapshot: " + key)
        output = Path(training["output_dir"]).resolve()
        if output == ARCHIVE or ARCHIVE in output.parents:
            raise ValueError("Recovery must never write into the immutable archive.")
        current["model"].load_state_dict(state["current_state"], strict=True)
        current["optimizer"].load_state_dict(state["optimizer_state"])
        current["scheduler"].load_state_dict(state["scheduler_state"])
        if current["train_loader"].generator is None:
            raise ValueError("Canonical training DataLoader must retain its independent generator.")
        # Use snapshot values themselves after verifying the independently rebuilt
        # train-only statistics, preserving dtype and all original metadata.
        for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            current["statistics"][key] = meta[key].clone()
        result = (copy.deepcopy(state["history"]), state["best_epoch"], state["best_validation_loss"],
                  copy.deepcopy(state["best_state"]), state["epochs_without_improvement"], state["completed_epochs"] + 1)
        record = {
            "snapshot": str(self.path), "snapshot_sha256": self.snapshot_sha256,
            "completed_epochs_restored": state["completed_epochs"],
            "next_epoch": state["completed_epochs"] + 1, "torch_version": str(torch.__version__),
            "split_control": adapter.split_control(training), "ast_audit": self.audit,
            "elapsed_seconds_scope": "resumed candidate process only; runner must add pre-pause observed wall time",
        }
        destination = output / "depth_runtime" / "resume_control.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        TRAIN._write_json(destination, record)
        self.restored = True
        # Last stateful work before returning to the original loop. In particular,
        # do not reseed, create a loader iterator or run extra validation afterward.
        random.setstate(state["python_rng"])
        nr = state["numpy_rng"]
        np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        current["train_loader"].generator.set_state(state["loader_generator_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        torch.set_rng_state(state["torch_rng"])
        return result


def install_resume(snapshot_path):
    adapter.install_adapter()
    controller = ResumeController(snapshot_path)
    function, audit = transform_candidate(controller.restore)
    controller.audit = audit
    TRAIN.train_candidate = function
    return controller


def verify_archived_snapshot(path):
    path = Path(path).resolve(strict=True)
    relative = path.relative_to(ARCHIVE.resolve(strict=True)).as_posix()
    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text())
    if manifest.get(relative) != digest(path):
        raise ValueError("Snapshot is absent from immutable archive manifest or its hash changed.")
    return path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", required=True)
    options, remaining = parser.parse_known_args()
    if not remaining or remaining[0] != "train":
        raise ValueError("Resume accepts only the original train CLI command.")
    if adapter.PAUSE_FILE.exists():
        raise SystemExit("PAUSE exists; refusing to resume.")
    snapshot = verify_archived_snapshot(options.snapshot)
    controller = install_resume(snapshot)
    os.chdir(adapter.EXPERIMENT_ROOT)
    import main as canonical_main
    sys.argv = [str(Path(__file__).resolve()), *remaining]
    with adapter.EpochSnapshots():
        canonical_main.main()
    if not controller.restored:
        raise RuntimeError("Canonical command completed without restoring the snapshot.")


if __name__ == "__main__":
    main()
