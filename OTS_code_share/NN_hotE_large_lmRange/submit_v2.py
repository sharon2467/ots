# -*- coding: utf-8 -*-
"""Submit the guarded OTS v2 Slurm job.

The login-node Python does not expose the scientific modules used by the
training image.  ``train_v2.slurm`` therefore runs the numerical self-test
after loading the job environment and exits before training if it fails.
"""

from __future__ import print_function

import os
import re
import subprocess


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    slurm_script = os.path.join(HERE, "train_v2.slurm")

    print("Submitting guarded job {}".format(slurm_script), flush=True)
    try:
        output = subprocess.check_output(
            ["sbatch", slurm_script],
            cwd=HERE,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("SBATCH_FAILED: {}".format(error.output.strip()))
    print(output, flush=True)

    match = re.search(r"(?:Submitted batch job\s+)?(\d+)$", output)
    if not match:
        raise RuntimeError("Could not parse a Slurm job ID from: {}".format(output))
    print("SUBMITTED_JOB_ID={}".format(match.group(1)), flush=True)


if __name__ == "__main__":
    main()
