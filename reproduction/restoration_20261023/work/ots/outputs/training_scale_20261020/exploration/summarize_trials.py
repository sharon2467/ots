#!/usr/bin/env python3
"""Collect timings, training histories and common development scores, never tests."""
import argparse
import hashlib
import json
import re
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SOURCE = ROOT / "OTS_code_share/NN_hotE_large_lmRange"


def summarize(run_id):
    run = SOURCE / "runs" / run_id
    config_path = run / "pipeline_config.json"
    history_path = run / "training/training_history.json"
    is_tail = not config_path.exists()
    parent = None
    if is_tail:
        history = json.loads(history_path.read_text())
        parent_id = history["dataset_source_run_id"]
        parent = summarize(parent_id)
        if parent["status"] != "completed":
            raise ValueError("A cached-data tail requires a completed parent all run.")
        config_path = SOURCE / "runs" / parent_id / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        if history["dataset_id"] != config["dataset_id"]:
            raise ValueError("Tail and parent dataset IDs differ.")
        config["training"] = history["training_config"]
    else:
        config = json.loads(config_path.read_text())
    result = {
        "run_id": run_id,
        "dataset_kind": config["generation"]["dataset"],
        "train_pool": config["generation"]["train_samples"],
        "test_count": config["generation"]["test_samples"],
        "training_options": config["training"],
        "status": "running_or_incomplete",
        "test_metrics_read": False,
    }
    log = STUDY / "logs" / (run_id + ".log")
    if log.exists():
        # The all entry prints only completion; candidate metrics are in history.
        text = log.read_text()
        match = re.search(r"^real\s+(\d+)m([\d.]+)s\s*$", text, re.MULTILINE)
        if match:
            wall = int(match[1]) * 60 + float(match[2])
            if is_tail:
                result["tail_train_evaluate_wall_seconds"] = wall
                result["charged_parent_all_wall_seconds"] = parent["full_wall_seconds"]
                result["full_wall_seconds"] = parent["full_wall_seconds"] + wall
                result["timing_definition"] = (
                    "Conservative charge: measured parent generation/train/evaluation "
                    "wall plus this measured fresh train/evaluation wall; includes "
                    "the parent's original model training."
                )
            else:
                result["full_wall_seconds"] = wall
        result["completion_message"] = (
            "train completed successfully." in text
            and "evaluate completed successfully." in text
        ) if is_tail else "all completed successfully." in text
    if is_tail:
        result["parent_all_run"] = parent["run_id"]
        result["canonical_evaluation_run"] = run_id + "_eval"
        result["canonical_evaluation_artifact_exists"] = (
            SOURCE / "runs" / (run_id + "_eval") / "evaluation/metrics.json"
        ).is_file()
    manifest = run / "data/dataset_manifest.json"
    if manifest.exists():
        result["generation_plus_initialization_io_seconds_approx"] = (
            manifest.stat().st_mtime - config_path.stat().st_mtime
        )
        result["phase_timing_note"] = (
            "Approximate from artifact modification times; authoritative total is Bash real."
        )
    if history_path.exists():
        history = json.loads(history_path.read_text())
        result["sample_counts"] = history["sample_counts"]
        result["selected_candidate"] = history["selected_candidate"]
        candidates = []
        for candidate in history["candidates"]:
            best = candidate["history"][candidate["best_epoch"] - 1]
            candidates.append({
                key: candidate[key] for key in (
                    "hidden_sizes", "dropout", "best_epoch",
                    "best_validation_loss", "elapsed_seconds", "stop_reason",
                    "parameter_count"
                )
            })
            candidates[-1].update(
                epochs_run=len(candidate["history"]),
                training_loss_at_best_epoch=best["train_loss"],
                learning_rate_at_best_epoch=best["learning_rate"],
            )
        result["candidates"] = candidates
        result["sum_candidate_training_seconds"] = sum(
            c["elapsed_seconds"] for c in candidates
        )
        checkpoint = history_path.parent / history["checkpoint"]
        result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit_path = STUDY / "exploration" / (run_id + "_common_validation_metrics.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        result["common_development"] = {
            "count": audit["counts"]["common_validation_count"],
            "selection_score": audit["selection_score"],
            "zero_overlap_with_optimizer": audit["overlap_check"]["zero_overlap"],
            "species": {
                s: {
                    "metrics": values["model"]["metrics"],
                }
                for s, values in audit["species"].items()
            },
            "caution": audit["development_caution"],
        }
    if (result.get("completion_message") and history_path.exists()
            and (not is_tail or result["canonical_evaluation_artifact_exists"])):
        result["status"] = "completed"
        result["under_two_hours"] = result.get("full_wall_seconds", float("inf")) < 7200
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = [summarize(run_id) for run_id in args.run_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trials, indent=2, allow_nan=False) + "\n")
    for trial in trials:
        print(json.dumps({
            key: trial[key]
            for key in ("run_id", "status", "full_wall_seconds", "under_two_hours")
            if key in trial
        }))
