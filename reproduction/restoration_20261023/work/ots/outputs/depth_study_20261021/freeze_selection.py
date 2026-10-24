#!/usr/bin/env python3
"""Freeze validation-based repeats or final checkpoints without opening test."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validated_run(run_id, protocol):
    record = json.loads((STUDY / "execution" / (run_id + ".json")).read_text())
    if (record["status"] != "completed_test_sealed" or record["new_test_metrics_read"]
            or not record["under_two_hours"]):
        raise ValueError("Run is not successfully complete with test sealed: " + run_id)
    validation_path = STUDY / "validation" / run_id / "summary.json"
    validation = json.loads(validation_path.read_text())
    checkpoint = STUDY / "runs" / run_id / "training/model.pt"
    if validation["test_data_accessed"] or validation["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("Validation provenance failed: " + run_id)
    control = json.loads((checkpoint.parent / "depth_runtime/split_control.json").read_text())
    if (control["statistics_sha256"] != protocol["normalization_sha256"]
            or control["train_index_sha256"] != protocol["split_index_sha256"]["optimization"]
            or control["validation_index_sha256"] != protocol["split_index_sha256"]["validation"]):
        raise ValueError("Frozen data membership/normalization mismatch")
    return {"run_id": run_id, "label": record["architecture"].upper(),
            "training_seed": record["training_seed"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "validation_sha256": sha(validation_path),
            "validation_macro_relative_l2": validation["macro_relative_l2"]["mean"],
            "parameter_count": record["parameter_count"],
            "evaluation_dir": str(STUDY / "runs" / (run_id + "_eval") / "evaluation")}


def save_once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("repeats", "final"))
    args = parser.parse_args()
    if (STUDY / "PAUSE").exists():
        raise SystemExit("Study paused")
    protocol = json.loads((STUDY / "protocol.json").read_text())
    primary = [load_validated_run(r, protocol) for r in protocol["primary_run_ids"]]
    if args.stage == "repeats":
        deep = min((r for r in primary if r["label"] in ("D3", "D4")),
                   key=lambda r: (r["validation_macro_relative_l2"], r["parameter_count"]))
        labels = ("d2", deep["label"].lower(), "w" + deep["label"][1:])
        repeats = [f"depth_fig_{label}_s{seed}" for seed in (1, 2) for label in labels]
        result = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "selected_depth": deep["label"], "selected_labels": list(labels),
                  "run_ids": repeats, "primary_validation": primary,
                  "rule": protocol["repeat_rule"], "new_test_results_read": False}
        save_once(STUDY / "repeat_selection.json", result)
    else:
        repeats = json.loads((STUDY / "repeat_selection.json").read_text())
        runs = primary + [load_validated_run(r, protocol) for r in repeats["run_ids"]]
        chosen = min(runs, key=lambda r: (r["validation_macro_relative_l2"],
                                          r["parameter_count"], r["run_id"]))
        result = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "runs": runs, "selected_checkpoint": chosen,
                  "selection_rule": "Minimum validation macro relative-L2 over all completed checkpoints; "
                                    "tie: fewer parameters then lexical run ID. Architecture effects "
                                    "are assessed using paired seeds separately.",
                  "new_test_results_read": False, "protocol_sha256": sha(STUDY / "protocol.json"),
                  "repeat_selection_sha256": sha(STUDY / "repeat_selection.json")}
        save_once(STUDY / "final_selection.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
