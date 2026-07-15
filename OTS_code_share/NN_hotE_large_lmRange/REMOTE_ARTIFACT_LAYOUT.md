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
└── paper_fig3_<job-id>/            # paper model, histories, metrics, Fig. 3

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
