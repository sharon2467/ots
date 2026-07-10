# -*- coding: utf-8 -*-
"""Self-test the OTS v2 pipeline and submit its Slurm job.

The self-test is deliberately executed before ``sbatch``.  A missing package,
syntax/runtime incompatibility, or failed numerical invariant therefore stops
the launcher without consuming compute allocation.
"""

from __future__ import print_function

import os
import re
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    training_script = os.path.join(HERE, "train_v2.py")
    slurm_script = os.path.join(HERE, "train_v2.slurm")

    print("Running v2 self-test with {}".format(sys.executable), flush=True)
    subprocess.check_call(
        [sys.executable, training_script, "--self-test"], cwd=HERE
    )

    print("Self-test passed; submitting {}".format(slurm_script), flush=True)
    output = subprocess.check_output(
        ["sbatch", slurm_script], cwd=HERE, universal_newlines=True
    ).strip()
    print(output, flush=True)

    match = re.search(r"(?:Submitted batch job\s+)?(\d+)$", output)
    if not match:
        raise RuntimeError("Could not parse a Slurm job ID from: {}".format(output))
    print("SUBMITTED_JOB_ID={}".format(match.group(1)), flush=True)


if __name__ == "__main__":
    main()
