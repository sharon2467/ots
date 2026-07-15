# -*- coding: utf-8 -*-
"""Read-only status summary for the guarded OTS v2 Slurm job."""

from __future__ import print_function

import os
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
JOB_ID = os.environ.get("OTS_V2_JOB_ID", "59648208")


def run_status(command):
    try:
        return subprocess.check_output(
            command,
            cwd=PROJECT_ROOT,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        return "COMMAND_FAILED: {}".format(error.output.strip())


def tail(path, line_count=40):
    if not os.path.exists(path):
        return "NOT_CREATED"
    with open(path, "r") as handle:
        lines = handle.readlines()
    return "".join(lines[-line_count:]).rstrip() or "EMPTY"


def main():
    queue_output = run_status(
        [
            "squeue",
            "-j",
            JOB_ID,
            "-o",
            "%.18i %.9P %.20j %.8T %.10M %.9l %.6D %R",
        ]
    )
    accounting_output = run_status(
        [
            "sacct",
            "-j",
            JOB_ID,
            "--format=JobID,JobName,Partition,State,Elapsed,ExitCode,AllocTRES%40",
            "-n",
            "-P",
        ]
    )

    print("JOB_ID={}".format(JOB_ID))
    print("SQUEUE_BEGIN")
    print(queue_output)
    print("SQUEUE_END")
    print("SACCT_BEGIN")
    print(accounting_output)
    print("SACCT_END")

    for suffix in ("out", "err"):
        path = os.path.join(PROJECT_ROOT, "train_v2_{}.{}".format(JOB_ID, suffix))
        print("{}_TAIL_BEGIN".format(suffix.upper()))
        print(tail(path))
        print("{}_TAIL_END".format(suffix.upper()))

    print("SUMMARY_SQUEUE={}".format(" | ".join(queue_output.splitlines())))
    print("SUMMARY_SACCT={}".format(" | ".join(accounting_output.splitlines())))


if __name__ == "__main__":
    main()
