# BSCC dataset and model layout

The BSCC working directory is `/home/bingxing2/home/scx6090/yhh/ots` on the
GPU route.  Generated Parquet files and checkpoints remain there and are not
tracked by Git or downloaded to the workstation.

## Dataset families

```text
datasets/
├── hot_e_large_range_job1346661/
│   ├── dataset_train.parquet -> ../../dataset_train.parquet
│   ├── dataset_test.parquet  -> ../../dataset_test.parquet
│   └── dataset_manifest.json
└── paper_10gaussian_v1_seed20260715/
    ├── dataset_train.parquet
    ├── dataset_test.parquet
    └── dataset_manifest.json
```

The root-level Parquet files are retained for backward compatibility.  They
belong to the hot-electron/large-wavelength-range model from Job 1346661.  The
paper dataset uses ten Gaussians per EVDF, fixed seeds, and the paper's three
wavelength bands; it is stored physically in its version directory.

Every manifest records the dataset family, generation parameters, file sizes,
and SHA-256 hashes.  A model must name the dataset ID and hashes in its own
training metrics/checkpoint.

## Models and runs

```text
runs/
├── bscc_v2_1346661/                # hot-electron model and metrics
├── paper_fig3_<job-id>/            # paper model, histories, metrics, Fig. 3
└── paper_fig3_diagnostic_<job-id>/ # lightweight frozen-model diagnostics

models/
├── hot_e_large_range_job1346661/job_1346661/
└── paper_10gaussian_v1_seed20260715/
    ├── LATEST.txt
    └── job_<job-id>/
```

The `models/` tree contains lightweight links into immutable job-specific run
directories.  It does not duplicate large checkpoints.  `LATEST.txt` is only
updated after training, validation, Figure-3 generation, and SHA-256 recording
all succeed.

## Paper reproduction command

From the BSCC OTS directory:

```bash
sbatch paper_fig3_bscc.slurm
```

The default run performs the paper's complete 119,000-step full-batch Adam
schedule.  `paper_training_resume.pt` is updated periodically and removed only
after the full schedule succeeds.  The result directory contains a combined
2x2 Figure 3 and four individual panels.

## Verified paper run (2026-07-15)

Formal Job `1347041` completed with exit code `0:0` after all 119,000 steps.
`full_schedule_completed` is true, all ten entries in the run's
`SHA256SUMS.txt` pass, and
`models/paper_10gaussian_v1_seed20260715/LATEST.txt` contains
`job_1347041`.

The immutable paper dataset has these hashes:

```text
dataset_train.parquet  d2c5c3e7828b7e3df045e7cac55a0dc92a6abb4c563e662ba91bc48869d8ae02
dataset_test.parquet   a8f39fcd8f50078277a92961c5e2aa58dabc1aace3bc8cd82bfe3331a8a9cdf3
```

The formal run is at `runs/paper_fig3_1347041/`.  Its test EVDF R2 is
`0.9782841`; Figure-3 panel R2 values are `0.9660`, `0.9940`, `0.8892`, and
`0.8033` for panels (a)-(d).  Diagnostic Job `1347309` is stored at
`runs/paper_fig3_diagnostic_1347309/`; it confirmed that analytic and Fourier
super-Gaussian spectra differ by less than 0.9% in relative L2 and that this
small forward-model difference does not explain the weaker out-of-distribution
panels.

Only logs, JSON/history, the training curve, the combined figure, and four
individual panels were copied to the workstation under the ignored directory
`bscc_results/paper_fig3_1347041/`.  The paper Parquet files and checkpoints
remain on BSCC.
