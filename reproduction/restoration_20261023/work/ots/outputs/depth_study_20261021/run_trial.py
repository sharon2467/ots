#!/usr/bin/env python3
"""Run exactly one preregistered isolated depth-study pipeline."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
SOURCE = ROOT / "OTS_code_share/NN_hotE_large_lmRange"
PYTHON = ROOT / ".venv/bin/python"
PAUSE = STUDY / "PAUSE"
stop_requested = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def verify_invariants(protocol):
    hashes = json.loads((STUDY / "source_hashes_before.json").read_text())
    if any(digest(ROOT / p) != value for p, value in hashes.items()):
        raise RuntimeError("Protected source changed")
    if digest(protocol["baseline_checkpoint"]) != protocol["baseline_sha256"]:
        raise RuntimeError("Original baseline checkpoint changed")
    for name, value in protocol["data_sha256"].items():
        if digest(Path(protocol["data_directory"]) / name) != value:
            raise RuntimeError("Original data changed: " + name)
    before = json.loads((STUDY / "canonical_registry_before.json").read_text())
    after = {str(p.relative_to(SOURCE / "registry")): str(p.resolve(strict=True))
             for p in (SOURCE / "registry").glob("*/*") if p.is_symlink()}
    if before != after:
        raise RuntimeError("Canonical registry changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if PAUSE.exists():
        raise SystemExit("Study paused; refusing to start.")
    prefix, arch, replicate = args.run_id.rsplit("_", 2)
    if prefix != "depth_fig" or arch not in protocol["architectures"]:
        raise ValueError("Unregistered architecture")
    seeds = [protocol["primary_training_seed"]] + protocol["repeat_training_seeds"]
    if replicate not in ("s0", "s1", "s2"):
        raise ValueError("Unregistered replicate")
    seed = seeds[int(replicate[1:])]
    if replicate != "s0":
        selected = json.loads((STUDY / "repeat_selection.json").read_text())
        if args.run_id not in selected["run_ids"]:
            raise ValueError("Repeat not selected by frozen validation rule")
    command = f"python -B {STUDY / 'run_trial.py'} --run-id {args.run_id}"
    if command not in (STUDY / "README.md").read_text():
        raise RuntimeError("Exact launch command must appear in README before training")
    execution_path = STUDY / "execution" / (args.run_id + ".json")
    run_dir = STUDY / "runs" / args.run_id
    if execution_path.exists() or run_dir.exists():
        raise FileExistsError("Existing trial is immutable; use recovery instructions")
    verify_invariants(protocol)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(STUDY / ".mplconfig"),
               XDG_CACHE_HOME=str(STUDY / ".cache"), PYTHONDONTWRITEBYTECODE="1")
    for name in (".mplconfig", ".cache"):
        (STUDY / name).mkdir(exist_ok=True)
    script_names = ("adapter.py", "run_trial.py", "analyze_depth.py")
    script_hashes = {name: digest(STUDY / name) for name in script_names}
    checkpoint_id = args.run_id + "_model"
    hidden = ",".join(map(str, protocol["architectures"][arch]["hidden_sizes"]))
    adapter = [str(PYTHON), "-B", "-u", str(STUDY / "adapter.py")]
    train_cmd = adapter + [
        "train", "--run-id", args.run_id, "--run-dir", "runs/" + args.run_id,
        "--dataset-id", protocol["dataset_id"], "--checkpoint-id", checkpoint_id,
        "--candidates", hidden, "--max-epochs", "250", "--patience", "35",
        "--batch-size", "128", "--learning-rate", "0.001", "--weight-decay", "0.00001",
        "--raw-loss-weight", "1.0", "--validation-fraction", "0.15",
        "--device", "cpu", "--seed", str(seed),
    ]
    validation_cmd = [
        str(PYTHON), "-B", str(STUDY / "analyze_depth.py"), "validation",
        "--checkpoint", str(run_dir / "training/model.pt"),
        "--output-dir", str(STUDY / "validation" / args.run_id),
    ]
    evaluation_cmd = adapter + [
        "evaluate", "--run-id", args.run_id + "_eval",
        "--run-dir", "runs/" + args.run_id + "_eval",
        "--dataset-id", protocol["dataset_id"], "--checkpoint-id", checkpoint_id,
        "--device", "cpu",
    ]
    started = time.monotonic()
    deadline = started + protocol["compute_budget"]["tail_timeout_seconds"]
    record = {
        "run_id": args.run_id, "architecture": arch, "training_seed": seed,
        "split_seed": protocol["split_seed"], "status": "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parent_pid": os.getpid(), "script_sha256": script_hashes,
        "protocol_sha256": digest(STUDY / "protocol.json"),
        "parameter_count": protocol["architectures"][arch]["parameter_count"],
        "stages": [], "new_test_metrics_read": False,
    }
    write_json(execution_path, record)
    try:
        for stage, cmd in (("train", train_cmd), ("validation", validation_cmd),
                           ("sealed_test_evaluation", evaluation_cmd)):
            if stop_requested or PAUSE.exists() or time.monotonic() >= deadline:
                record["status"] = "paused_before_" + stage
                break
            log_path = STUDY / "logs" / f"{args.run_id}_{stage}.log"
            stage_start = time.monotonic()
            with log_path.open("x") as log:
                child = subprocess.Popen(cmd, cwd=STUDY, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                stage_info = {"name": stage, "command": cmd, "pid": child.pid,
                              "log": str(log_path), "status": "running"}
                record["stages"].append(stage_info)
                write_json(execution_path, record)
                stop_at = None
                while child.poll() is None:
                    exceeded = time.monotonic() >= deadline
                    if (stop_requested or PAUSE.exists() or exceeded) and stop_at is None:
                        stage_info["stop_reason"] = "budget" if exceeded else "pause"
                        os.killpg(child.pid, signal.SIGTERM)
                        stop_at = time.monotonic()
                    if stop_at is not None and time.monotonic() - stop_at > 30:
                        os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(0.5)
                stage_info.update(exit_code=child.returncode,
                                  wall_seconds=time.monotonic() - stage_start,
                                  status="completed" if child.returncode == 0 else "interrupted")
                write_json(execution_path, record)
            if child.returncode != 0:
                record["status"] = "interrupted_" + stage
                break
        else:
            record["status"] = "completed_test_sealed"
            record["checkpoint_sha256"] = digest(run_dir / "training/model.pt")
        verify_invariants(protocol)
        if any(digest(STUDY / name) != value for name, value in script_hashes.items()):
            raise RuntimeError("Experiment code changed while trial was running")
        record["protected_sources_data_model_registry_unchanged"] = True
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["tail_wall_seconds"] = time.monotonic() - started
        record["charged_full_wall_seconds"] = (
            protocol["compute_budget"]["charged_previous_all_seconds"]
            + record["tail_wall_seconds"])
        record["under_two_hours"] = record["charged_full_wall_seconds"] < 7200
        write_json(execution_path, record)
        print(json.dumps({k: record[k] for k in
                          ("run_id", "status", "tail_wall_seconds",
                           "charged_full_wall_seconds", "under_two_hours")}), flush=True)
    return 0 if record["status"] == "completed_test_sealed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
