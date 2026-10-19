# -*- coding: utf-8 -*-
"""Canonical command-line pipeline for OTS EVDF data, training, and evaluation."""

import argparse
import json
import os
import re
from pathlib import Path

from data import generate_dataset_pair
from evaluation import DEFAULT_EVALUATION_CONFIG, evaluate_checkpoint
from train import DEFAULT_CONFIG, train_model


REGISTRY_KINDS = ("runs", "datasets", "checkpoints")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SLURM_LOG_PATTERN = re.compile(r"slurm-\d+\.(?:out|err)\Z")


def _project_root():
    return Path(__file__).resolve().parent


def _runs_root():
    return _project_root() / "runs"


def _registry_root():
    return _project_root() / "registry"


def _entry_exists(path):
    return os.path.lexists(str(path))


def _validate_registry_id(value, name):
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "{} must be one safe path segment using letters, digits, '.', '_' or '-'.".format(
                name
            )
        )
    return value


def _registry_entry(kind, identifier):
    if kind not in REGISTRY_KINDS:
        raise ValueError("Unknown registry kind: {}".format(kind))
    _validate_registry_id(identifier, kind[:-1] + "_id")
    return _registry_root() / kind / identifier


def _relative_to_runs(path):
    runs_root = _runs_root().resolve(strict=True)
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(runs_root)
    except ValueError as error:
        raise ValueError("Registry target escapes the runs directory: {}".format(path)) from error
    if not relative.parts:
        raise ValueError("Registry target must be below the runs directory.")
    return resolved, relative


def _validate_run_dir(value):
    run_dir = Path(value)
    if run_dir.is_absolute() or len(run_dir.parts) != 2 or run_dir.parts[0] != "runs":
        raise ValueError("run-dir must be exactly runs/RUN_DIR.")
    _validate_registry_id(run_dir.parts[1], "run-dir name")
    absolute = _project_root() / run_dir
    if _entry_exists(absolute) and absolute.is_symlink():
        raise ValueError("run-dir must not be a symbolic link.")
    return absolute


def _run_dir_registration(run_dir):
    registry = _registry_root() / "runs"
    if not registry.exists():
        return None
    for entry in registry.iterdir():
        if not entry.is_symlink():
            raise ValueError("Run registry contains a non-link entry: {}".format(entry))
        try:
            target = entry.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError("Run registry contains a broken link: {}".format(entry)) from error
        if target == run_dir.resolve(strict=False):
            return entry.name
    return None


def _expected_output_names(command):
    if command == "generate":
        return {"data"}
    if command == "train":
        return {"training"}
    if command == "evaluate":
        return {"evaluation"}
    if command == "all":
        return {
            "data",
            "training",
            "evaluation",
            "pipeline_config.json",
            "pipeline_summary.json",
        }
    raise ValueError("Unknown command: {}".format(command))


def _preflight_output(command, run_id, run_dir, output_ids, overwrite):
    _validate_registry_id(run_id, "run_id")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean.")
    absolute_run_dir = _validate_run_dir(run_dir)

    identifiers = [("runs", run_id)] + list(output_ids)
    for kind, identifier in identifiers:
        path = _registry_entry(kind, identifier)
        if _entry_exists(path):
            raise FileExistsError("Refusing to rebind existing registry ID: {}".format(path))

    registered_as = _run_dir_registration(absolute_run_dir)
    if registered_as is not None:
        raise ValueError(
            "run-dir is already registered by run_id {!r}.".format(registered_as)
        )

    if absolute_run_dir.exists():
        allowed_outputs = _expected_output_names(command)
        disallowed = []
        for child in absolute_run_dir.iterdir():
            if child.is_file() and SLURM_LOG_PATTERN.fullmatch(child.name):
                continue
            if overwrite and child.name in allowed_outputs:
                continue
            disallowed.append(str(child))
        if disallowed:
            if overwrite:
                raise FileExistsError(
                    "Retry run-dir contains unrelated entries: {}".format(disallowed)
                )
            raise FileExistsError(
                "Refusing to use run-dir with existing pipeline entries: {}".format(
                    disallowed
                )
            )
    return absolute_run_dir


def _resolve_dataset(dataset_id):
    entry = _registry_entry("datasets", dataset_id)
    if not _entry_exists(entry):
        raise FileNotFoundError("Unknown dataset_id: {}".format(dataset_id))
    if not entry.is_symlink():
        raise ValueError("Dataset registry entry must be a symbolic link: {}".format(entry))
    try:
        target, relative = _relative_to_runs(entry)
    except FileNotFoundError as error:
        raise ValueError("Dataset registry entry is broken: {}".format(entry)) from error
    if not target.is_dir() or target.name != "data":
        raise ValueError("Dataset registry entry must target a run data directory.")
    paths = {
        "registry_entry": entry,
        "directory": target,
        "train_path": target / "dataset_train.parquet",
        "test_path": target / "dataset_test.parquet",
        "manifest_path": target / "dataset_manifest.json",
        "source_run_dir": relative.parts[0],
    }
    for name in ("train_path", "test_path", "manifest_path"):
        if not paths[name].is_file():
            raise FileNotFoundError("Dataset registry is missing {}: {}".format(name, paths[name]))
    return paths


def _resolve_checkpoint(checkpoint_id):
    entry = _registry_entry("checkpoints", checkpoint_id)
    if not _entry_exists(entry):
        raise FileNotFoundError("Unknown checkpoint_id: {}".format(checkpoint_id))
    if entry.is_symlink():
        try:
            directory, relative = _relative_to_runs(entry)
        except FileNotFoundError as error:
            raise ValueError("Checkpoint registry entry is broken: {}".format(entry)) from error
        if not directory.is_dir() or directory.name != "training":
            raise ValueError(
                "Canonical checkpoint registry entry must target a run training directory."
            )
        checkpoint_path = directory / "model.pt"
        source_run_dir = relative.parts[0]
    elif entry.is_dir():
        checkpoint_path = entry / "model.pt"
        if not checkpoint_path.is_symlink():
            raise ValueError(
                "Historical checkpoint registration must provide a model.pt symlink."
            )
        try:
            resolved_model, relative = _relative_to_runs(checkpoint_path)
        except FileNotFoundError as error:
            raise ValueError("Historical checkpoint model link is broken.") from error
        checkpoint_path = resolved_model
        source_run_dir = relative.parts[0]
    else:
        raise ValueError("Checkpoint registry entry must be a directory or directory link.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint model is missing: {}".format(checkpoint_path))
    _relative_to_runs(checkpoint_path)
    return {
        "registry_entry": entry,
        "checkpoint_path": checkpoint_path,
        "source_run_dir": source_run_dir,
    }


def _publish_registry_entries(entries):
    """Publish a batch of relative links without replacing any existing ID."""

    prepared = []
    for kind, identifier, target in entries:
        entry = _registry_entry(kind, identifier)
        if _entry_exists(entry):
            raise FileExistsError("Refusing to rebind existing registry ID: {}".format(entry))
        target, _ = _relative_to_runs(target)
        if not target.is_dir():
            raise ValueError("Registry target must be an existing directory: {}".format(target))
        prepared.append((entry, target))

    created = []
    try:
        for entry, target in prepared:
            entry.parent.mkdir(parents=True, exist_ok=True)
            relative_target = os.path.relpath(str(target), str(entry.parent))
            entry.symlink_to(relative_target, target_is_directory=True)
            created.append(entry)
            if entry.resolve(strict=True) != target:
                raise RuntimeError("Published registry link resolved to the wrong target.")
    except Exception:
        for entry in reversed(created):
            if entry.is_symlink():
                entry.unlink()
        raise
    return created


def _registry_summary(run_id, dataset_id=None, checkpoint_id=None):
    payload = {"run": "registry/runs/{}".format(run_id)}
    if dataset_id is not None:
        payload["dataset"] = "registry/datasets/{}".format(dataset_id)
    if checkpoint_id is not None:
        payload["checkpoint"] = "registry/checkpoints/{}".format(checkpoint_id)
    return payload


def _add_ids(parser, include_checkpoint):
    parser.add_argument("--run-id", required=True, help="ID to publish for this output run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="New runs/RUN_DIR output.")
    parser.add_argument("--dataset-id", required=True, help="Dataset input or output ID.")
    if include_checkpoint:
        parser.add_argument("--checkpoint-id", required=True, help="Checkpoint input or output ID.")


def _add_generation_options(parser):
    parser.add_argument("--dataset-kind", choices=("fixed_ion", "fig1_fig11"), default="fixed_ion")
    parser.add_argument("--generation-config", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])


def _add_training_options(parser, include_seed=True):
    if include_seed:
        parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_CONFIG["validation_fraction"]
    )
    parser.add_argument("--candidates", default=DEFAULT_CONFIG["candidates"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"]
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument(
        "--raw-loss-weight", type=float, default=DEFAULT_CONFIG["raw_loss_weight"]
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=DEFAULT_CONFIG["gradient_clip_norm"]
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_CONFIG["device"]
    )


def _add_evaluation_options(parser, all_command=False):
    option = "--evaluation-batch-size" if all_command else "--batch-size"
    destination = "evaluation_batch_size" if all_command else "batch_size"
    parser.add_argument(
        option,
        dest=destination,
        type=int,
        default=DEFAULT_EVALUATION_CONFIG["batch_size"],
    )
    parser.add_argument(
        "--plot-samples", type=int, default=DEFAULT_EVALUATION_CONFIG["plot_samples"]
    )
    parser.add_argument(
        "--detail-index", type=int, default=DEFAULT_EVALUATION_CONFIG["detail_index"]
    )
    if not all_command:
        parser.add_argument(
            "--device",
            choices=("auto", "cpu", "cuda"),
            default=DEFAULT_EVALUATION_CONFIG["device"],
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one registered dataset.")
    _add_ids(generate, include_checkpoint=False)
    _add_generation_options(generate)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=run_generate)

    train = subparsers.add_parser("train", help="Train from a registered dataset.")
    _add_ids(train, include_checkpoint=True)
    _add_training_options(train)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=run_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a registered checkpoint on a registered dataset."
    )
    _add_ids(evaluate, include_checkpoint=True)
    _add_evaluation_options(evaluate)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=run_evaluate)

    all_parser = subparsers.add_parser("all", help="Run generate, train and evaluate.")
    _add_ids(all_parser, include_checkpoint=True)
    _add_generation_options(all_parser)
    _add_training_options(all_parser, include_seed=False)
    _add_evaluation_options(all_parser, all_command=True)
    all_parser.add_argument("--overwrite", action="store_true")
    all_parser.set_defaults(handler=run_all)
    return parser


def write_json(path, payload, overwrite=False):
    path = Path(path)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be a dictionary.")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean.")
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing JSON: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _training_config(args):
    return {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "candidates": args.candidates,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "raw_loss_weight": args.raw_loss_weight,
        "gradient_clip_norm": args.gradient_clip_norm,
        "device": args.device,
    }


def _evaluation_config(args, all_command=False):
    batch_size = args.evaluation_batch_size if all_command else args.batch_size
    return {
        "batch_size": batch_size,
        "device": args.device,
        "plot_samples": args.plot_samples,
        "detail_index": args.detail_index,
    }


def _relative_path(path, base):
    return Path(os.path.relpath(str(Path(path).resolve()), str(Path(base).resolve()))).as_posix()


def run_generate(args):
    generation_config = None
    if args.generation_config is not None:
        with args.generation_config.open(encoding="utf-8") as stream:
            generation_config = json.load(stream)
        if not isinstance(generation_config, dict):
            raise ValueError("generation-config must contain a JSON object.")
    run_dir = _preflight_output(
        "generate",
        args.run_id,
        args.run_dir,
        [("datasets", args.dataset_id)],
        args.overwrite,
    )
    data_dir = run_dir / "data"
    result = generate_dataset_pair(
        args.run_id,
        args.dataset_id,
        data_dir,
        args.train_samples,
        args.test_samples,
        args.seed,
        overwrite=args.overwrite,
        dataset=args.dataset_kind, generation_config=generation_config,
    )
    _publish_registry_entries(
        [("runs", args.run_id, run_dir), ("datasets", args.dataset_id, data_dir)]
    )
    return result


def run_train(args):
    run_dir = _preflight_output(
        "train",
        args.run_id,
        args.run_dir,
        [("checkpoints", args.checkpoint_id)],
        args.overwrite,
    )
    dataset = _resolve_dataset(args.dataset_id)
    result = train_model(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        dataset["train_path"],
        dataset["manifest_path"],
        run_dir / "training",
        _training_config(args),
        overwrite=args.overwrite,
    )
    _publish_registry_entries(
        [
            ("runs", args.run_id, run_dir),
            ("checkpoints", args.checkpoint_id, run_dir / "training"),
        ]
    )
    return result


def run_evaluate(args):
    run_dir = _preflight_output(
        "evaluate", args.run_id, args.run_dir, [], args.overwrite
    )
    dataset = _resolve_dataset(args.dataset_id)
    checkpoint = _resolve_checkpoint(args.checkpoint_id)
    result = evaluate_checkpoint(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        checkpoint["checkpoint_path"],
        dataset["test_path"],
        dataset["manifest_path"],
        run_dir / "evaluation",
        _evaluation_config(args),
        overwrite=args.overwrite,
    )
    _publish_registry_entries([("runs", args.run_id, run_dir)])
    return result


def _all_artifact_paths(run_dir):
    data_dir = run_dir / "data"
    training_dir = run_dir / "training"
    evaluation_dir = run_dir / "evaluation"
    return {
        "pipeline_config": run_dir / "pipeline_config.json",
        "pipeline_summary": run_dir / "pipeline_summary.json",
        "train_data": data_dir / "dataset_train.parquet",
        "test_data": data_dir / "dataset_test.parquet",
        "dataset_manifest": data_dir / "dataset_manifest.json",
        "checkpoint": training_dir / "model.pt",
        "training_history": training_dir / "training_history.json",
        "learning_curves": training_dir / "learning_curves.png",
        "metrics": evaluation_dir / "metrics.json",
        "predictions": evaluation_dir / "predictions.npz",
        "comparison": evaluation_dir / "comparison.png",
        "sample_detail": evaluation_dir / "sample_detail.png",
    }


def run_all(args):
    generation_config = None
    if args.generation_config is not None:
        with args.generation_config.open(encoding="utf-8") as stream:
            generation_config = json.load(stream)
        if not isinstance(generation_config, dict):
            raise ValueError("generation-config must contain a JSON object.")
    run_dir = _preflight_output(
        "all",
        args.run_id,
        args.run_dir,
        [("datasets", args.dataset_id), ("checkpoints", args.checkpoint_id)],
        args.overwrite,
    )
    paths = _all_artifact_paths(run_dir)
    training_config = _training_config(args)
    evaluation_config = _evaluation_config(args, all_command=True)
    pipeline_config = {
        "run_id": args.run_id,
        "dataset_id": args.dataset_id,
        "checkpoint_id": args.checkpoint_id,
        "directories": {
            "run_dir": ".",
            "data": "data",
            "training": "training",
            "evaluation": "evaluation",
        },
        "registry": _registry_summary(args.run_id, args.dataset_id, args.checkpoint_id),
        "generation": {
            "dataset": args.dataset_kind,
            "generation_config": generation_config,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "seed": args.seed,
        },
        "training": training_config,
        "evaluation": evaluation_config,
        "overwrite": args.overwrite,
    }
    write_json(paths["pipeline_config"], pipeline_config, overwrite=args.overwrite)

    generated = generate_dataset_pair(
        args.run_id,
        args.dataset_id,
        paths["train_data"].parent,
        args.train_samples,
        args.test_samples,
        args.seed,
        overwrite=args.overwrite,
        dataset=args.dataset_kind, generation_config=generation_config,
    )
    trained = train_model(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        generated["train_path"],
        generated["manifest_path"],
        paths["checkpoint"].parent,
        training_config,
        overwrite=args.overwrite,
    )
    evaluated = evaluate_checkpoint(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        trained["checkpoint_path"],
        generated["test_path"],
        generated["manifest_path"],
        paths["metrics"].parent,
        evaluation_config,
        overwrite=args.overwrite,
    )

    summary = {
        "run_id": args.run_id,
        "dataset_id": args.dataset_id,
        "checkpoint_id": args.checkpoint_id,
        "registry": _registry_summary(args.run_id, args.dataset_id, args.checkpoint_id),
        "config": _relative_path(paths["pipeline_config"], run_dir),
        "dataset_manifest": _relative_path(generated["manifest_path"], run_dir),
        "train_data": _relative_path(generated["train_path"], run_dir),
        "test_data": _relative_path(generated["test_path"], run_dir),
        "training_history": _relative_path(trained["history_path"], run_dir),
        "checkpoint": _relative_path(trained["checkpoint_path"], run_dir),
        "learning_curves": _relative_path(trained["learning_curves_path"], run_dir),
        "metrics": _relative_path(evaluated["metrics_path"], run_dir),
        "predictions": ({species: _relative_path(path, run_dir) for species, path in evaluated["predictions_path"].items()}
                        if isinstance(evaluated["predictions_path"], dict) else _relative_path(evaluated["predictions_path"], run_dir)),
        "comparison": ({species: _relative_path(path, run_dir) for species, path in evaluated["comparison_path"].items()}
                        if isinstance(evaluated["comparison_path"], dict) else _relative_path(evaluated["comparison_path"], run_dir)),
        "sample_detail": ({species: _relative_path(path, run_dir) for species, path in evaluated["sample_detail_path"].items()}
                        if isinstance(evaluated["sample_detail_path"], dict) else _relative_path(evaluated["sample_detail_path"], run_dir)),
        "selected_candidate": trained["selected_candidate"],
        "best_validation_loss": trained["selected_candidate"]["best_validation_loss"],
    }
    write_json(paths["pipeline_summary"], summary, overwrite=args.overwrite)
    _publish_registry_entries(
        [
            ("runs", args.run_id, run_dir),
            ("datasets", args.dataset_id, run_dir / "data"),
            ("checkpoints", args.checkpoint_id, run_dir / "training"),
        ]
    )
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    print("{} completed successfully.".format(args.command), flush=True)


if __name__ == "__main__":
    main()
