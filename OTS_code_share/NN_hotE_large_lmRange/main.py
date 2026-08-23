# -*- coding: utf-8 -*-
"""Canonical command-line pipeline for OTS EVDF data, training, and evaluation."""

import argparse
import json
import os
from pathlib import Path

from data import generate_dataset_pair
from evaluation import DEFAULT_EVALUATION_CONFIG, evaluate_checkpoint
from train import DEFAULT_CONFIG, train_model


def _add_ids(parser, include_checkpoint):
    parser.add_argument("--run-id", required=True, help="Explicit pipeline run ID.")
    parser.add_argument(
        "--dataset-id", required=True, help="Explicit generated dataset ID."
    )
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint-id", required=True, help="Explicit model checkpoint ID."
        )


def _add_generation_options(parser):
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of physical train samples.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
        help="Number of physical independent test samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Dataset and training random seed.",
    )


def _add_training_options(parser, include_seed=True):
    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_CONFIG["seed"],
            help="Training random seed.",
        )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_CONFIG["validation_fraction"],
        help="Fraction of train Parquet rows reserved for validation.",
    )
    parser.add_argument(
        "--candidates",
        default=DEFAULT_CONFIG["candidates"],
        help="Semicolon-separated two-layer candidates, e.g. 512,256;1024,512.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_CONFIG["max_epochs"],
        help="Maximum epochs for each candidate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help="Validation early-stopping patience.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--raw-loss-weight",
        type=float,
        default=DEFAULT_CONFIG["raw_loss_weight"],
        help="Weight of normalized raw coefficient MSE.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_CONFIG["gradient_clip_norm"],
        help="Maximum gradient norm.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_CONFIG["device"],
        help="Training device selection.",
    )


def _add_evaluation_options(parser, all_command=False):
    option = "--evaluation-batch-size" if all_command else "--batch-size"
    destination = "evaluation_batch_size" if all_command else "batch_size"
    parser.add_argument(
        option,
        dest=destination,
        type=int,
        default=DEFAULT_EVALUATION_CONFIG["batch_size"],
        help="Frozen-checkpoint inference batch size.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=DEFAULT_EVALUATION_CONFIG["plot_samples"],
        help="Number of samples in the comparison figure.",
    )
    parser.add_argument(
        "--detail-index",
        type=int,
        default=DEFAULT_EVALUATION_CONFIG["detail_index"],
        help="Zero-based sample index for the detail figure.",
    )
    if not all_command:
        parser.add_argument(
            "--device",
            choices=("auto", "cpu", "cuda"),
            default=DEFAULT_EVALUATION_CONFIG["device"],
            help="Evaluation device selection.",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the generate/train/evaluate/all command-line interface."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate train/test Parquet and dataset manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_ids(generate, include_checkpoint=False)
    generate.add_argument("--output-dir", type=Path, required=True)
    _add_generation_options(generate)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=run_generate)

    train = subparsers.add_parser(
        "train",
        help="Train and select models using train/validation only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_ids(train, include_checkpoint=True)
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--dataset-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    _add_training_options(train)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=run_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate one frozen checkpoint on independent test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_ids(evaluate, include_checkpoint=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--test-data", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    _add_evaluation_options(evaluate)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=run_evaluate)

    all_parser = subparsers.add_parser(
        "all",
        help="Run generation, training, and frozen evaluation in sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_ids(all_parser, include_checkpoint=True)
    all_parser.add_argument("--run-dir", type=Path, required=True)
    _add_generation_options(all_parser)
    _add_training_options(all_parser, include_seed=False)
    _add_evaluation_options(all_parser, all_command=True)
    all_parser.add_argument("--overwrite", action="store_true")
    all_parser.set_defaults(handler=run_all)
    return parser


def write_json(path: Path, payload: dict, overwrite: bool = False) -> None:
    """Write sorted UTF-8 JSON while refusing implicit replacement."""

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
    return Path(
        os.path.relpath(str(Path(path).resolve()), str(Path(base).resolve()))
    ).as_posix()


def run_generate(args: argparse.Namespace) -> dict:
    """Execute the generate subcommand."""

    print(
        "generate run_id={!r} dataset_id={!r} output_dir={} train_samples={} "
        "test_samples={} seed={}".format(
            args.run_id,
            args.dataset_id,
            args.output_dir,
            args.train_samples,
            args.test_samples,
            args.seed,
        ),
        flush=True,
    )
    result = generate_dataset_pair(
        args.run_id,
        args.dataset_id,
        args.output_dir,
        args.train_samples,
        args.test_samples,
        args.seed,
        overwrite=args.overwrite,
    )
    print(
        "generated train_data={} test_data={} manifest={}".format(
            result["train_path"], result["test_path"], result["manifest_path"]
        ),
        flush=True,
    )
    return result


def run_train(args: argparse.Namespace) -> dict:
    """Execute the train subcommand without any test input."""

    config = _training_config(args)
    print(
        "train run_id={!r} dataset_id={!r} checkpoint_id={!r} train_data={} "
        "manifest={} output_dir={} config={}".format(
            args.run_id,
            args.dataset_id,
            args.checkpoint_id,
            args.train_data,
            args.dataset_manifest,
            args.output_dir,
            json.dumps(config, sort_keys=True),
        ),
        flush=True,
    )
    result = train_model(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        args.train_data,
        args.dataset_manifest,
        args.output_dir,
        config,
        overwrite=args.overwrite,
    )
    print(
        "trained train_samples={} validation_samples={} checkpoint={} history={} "
        "learning_curves={} selected={}".format(
            result["train_samples"],
            result["validation_samples"],
            result["checkpoint_path"],
            result["history_path"],
            result["learning_curves_path"],
            json.dumps(result["selected_candidate"], sort_keys=True),
        ),
        flush=True,
    )
    return result


def run_evaluate(args: argparse.Namespace) -> dict:
    """Execute the frozen-checkpoint evaluate subcommand."""

    config = _evaluation_config(args)
    print(
        "evaluate run_id={!r} dataset_id={!r} checkpoint_id={!r} checkpoint={} "
        "test_data={} manifest={} output_dir={} config={}".format(
            args.run_id,
            args.dataset_id,
            args.checkpoint_id,
            args.checkpoint,
            args.test_data,
            args.dataset_manifest,
            args.output_dir,
            json.dumps(config, sort_keys=True),
        ),
        flush=True,
    )
    result = evaluate_checkpoint(
        args.run_id,
        args.dataset_id,
        args.checkpoint_id,
        args.checkpoint,
        args.test_data,
        args.dataset_manifest,
        args.output_dir,
        config,
        overwrite=args.overwrite,
    )
    print(
        "evaluated metrics={} predictions={} comparison={} sample_detail={} "
        "test_metrics={}".format(
            result["metrics_path"],
            result["predictions_path"],
            result["comparison_path"],
            result["sample_detail_path"],
            json.dumps(result["metrics"], sort_keys=True),
        ),
        flush=True,
    )
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


def run_all(args: argparse.Namespace) -> dict:
    """Execute the complete canonical pipeline with preflighted outputs."""

    run_dir = Path(args.run_dir)
    paths = _all_artifact_paths(run_dir)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing pipeline artifacts: {}".format(existing)
        )

    training_config = _training_config(args)
    evaluation_config = _evaluation_config(args, all_command=True)
    pipeline_config = {
        "run_id": args.run_id,
        "dataset_id": args.dataset_id,
        "checkpoint_id": args.checkpoint_id,
        "directories": {
            "run_dir": ".",
            "data": _relative_path(paths["train_data"].parent, run_dir),
            "training": _relative_path(paths["checkpoint"].parent, run_dir),
            "evaluation": _relative_path(paths["metrics"].parent, run_dir),
        },
        "generation": {
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "seed": args.seed,
        },
        "training": training_config,
        "evaluation": evaluation_config,
        "overwrite": args.overwrite,
    }
    write_json(paths["pipeline_config"], pipeline_config, overwrite=args.overwrite)
    print(
        "all run_id={!r} dataset_id={!r} checkpoint_id={!r} run_dir={}".format(
            args.run_id, args.dataset_id, args.checkpoint_id, run_dir
        ),
        flush=True,
    )

    generated = generate_dataset_pair(
        args.run_id,
        args.dataset_id,
        paths["train_data"].parent,
        args.train_samples,
        args.test_samples,
        args.seed,
        overwrite=args.overwrite,
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
        "config": _relative_path(paths["pipeline_config"], run_dir),
        "dataset_manifest": _relative_path(generated["manifest_path"], run_dir),
        "train_data": _relative_path(generated["train_path"], run_dir),
        "test_data": _relative_path(generated["test_path"], run_dir),
        "training_history": _relative_path(trained["history_path"], run_dir),
        "checkpoint": _relative_path(trained["checkpoint_path"], run_dir),
        "learning_curves": _relative_path(trained["learning_curves_path"], run_dir),
        "metrics": _relative_path(evaluated["metrics_path"], run_dir),
        "predictions": _relative_path(evaluated["predictions_path"], run_dir),
        "comparison": _relative_path(evaluated["comparison_path"], run_dir),
        "sample_detail": _relative_path(evaluated["sample_detail_path"], run_dir),
        "selected_candidate": trained["selected_candidate"],
        "best_validation_loss": trained["selected_candidate"]["best_validation_loss"],
    }
    write_json(paths["pipeline_summary"], summary, overwrite=args.overwrite)
    print("all summary={}".format(paths["pipeline_summary"]), flush=True)
    return summary


def main() -> None:
    """Parse and dispatch exactly one subcommand."""

    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    print("{} completed successfully.".format(args.command), flush=True)


if __name__ == "__main__":
    main()
