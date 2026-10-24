#!/usr/bin/env python3
"""No-gradient validation: never starts a training run or reads test spectra."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace

import torch

import adapter
from data import OTSDataset


def must_fail(action):
    try:
        action()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid structure/checkpoint was accepted")


def main():
    torch.set_num_threads(4)
    train = adapter.canonical_train
    model = adapter.canonical_model
    unchanged = {name: getattr(train, name) for name in
                 ("coefficient_loss", "run_epoch", "train_candidate", "train_model")}
    original_validator = model.validate_checkpoint
    original_loader = model.load_model_checkpoint
    baseline_path = adapter.SOURCE / "runs/scale_fig_26k_wide/training/model.pt"
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    results = {}
    with torch.no_grad():
        for sizes in ([32, 16], [2048, 1024]):
            torch.manual_seed(20261030)
            original = adapter.CanonicalRegressor(sizes, 0.02, input_dim=319, output_dim=480)
            original_rng = torch.get_rng_state().clone()
            torch.manual_seed(20261030)
            adapted = adapter.DepthRegressor(sizes, 0.02, input_dim=319, output_dim=480)
            assert torch.equal(original_rng, torch.get_rng_state())
            assert list(original.state_dict()) == list(adapted.state_dict())
            assert all(torch.equal(value, adapted.state_dict()[key]) for key, value in original.state_dict().items())
            spectra = torch.randn(5, 319)
            target = torch.randn(5, 480)
            mean, scale = torch.randn(480), torch.rand(480) + 0.01
            for training in (False, True):
                original.train(training)
                adapted.train(training)
                torch.manual_seed(123)
                before = original(spectra)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(123)
                after = adapted(spectra)
                assert torch.equal(before, after) and torch.equal(rng, torch.get_rng_state())
                for weight in (0.0, 0.1, 1.0):
                    assert torch.equal(train.coefficient_loss(before, target, mean, scale, weight),
                                       train.coefficient_loss(after, target, mean, scale, weight))
        results["depth_two_initialization_forward_dropout_rng_loss_bitwise_equal"] = True
        for value in ("512,256;2048,1024", [[32, 16]], [{"hidden_sizes": [32, 16], "dropout": 0.3}]):
            assert adapter.parse_candidates(value) == adapter.canonical_parse_candidates(value)
        for value in ("1,0,3", "", "1,2,3;1,2,3", [{"hidden_sizes": [1, True, 2], "dropout": 0.0}]):
            must_fail(lambda: adapter.parse_candidates(value))
        results["parser_depth_two_compatibility_and_invalid_candidates"] = True
        deep = adapter.DepthRegressor([32, 16, 8], 0.02, input_dim=319, output_dim=480)
        assert len([layer for layer in deep.modules() if isinstance(layer, torch.nn.Linear)]) == 4
        payload = copy.deepcopy(baseline)
        payload.update(model_state=deep.state_dict(), hidden_sizes=[32, 16, 8], dropout=0.02,
                       selected_candidate={"hidden_sizes": [32, 16, 8], "dropout": 0.02})
        payload["training_config"]["candidates"] = [payload["selected_candidate"]]
        must_fail(lambda: original_validator(payload))
        adapter.install_adapter()
        import main as canonical_main
        assert canonical_main._project_root() == adapter.EXPERIMENT_ROOT
        assert all(getattr(train, key) is value for key, value in unchanged.items())
        assert model.validate_checkpoint is original_validator and model.load_model_checkpoint is original_loader
        model.validate_checkpoint(payload)
        for mutate in (
            lambda p: p.update(unexpected=True),
            lambda p: p.update(input_dim=318),
            lambda p: p.update(x_scale=torch.zeros(319)),
            lambda p: p["selected_candidate"].update(hidden_sizes=[32, 16]),
            lambda p: p["training_config"]["optimizer"].update(name="SGD"),
            lambda p: p["training_config"].update(unexpected=True),
        ):
            bad = copy.deepcopy(payload)
            mutate(bad)
            must_fail(lambda: model.validate_checkpoint(bad))
        with tempfile.TemporaryDirectory(prefix="adapter_validation_", dir=adapter.EXPERIMENT_ROOT) as temporary:
            temporary = Path(temporary)
            path = temporary / "deep.pt"
            torch.save(payload, path)
            loaded, actual_payload = model.load_model_checkpoint(path, "cpu", payload["checkpoint_id"])
            assert actual_payload["hidden_sizes"] == [32, 16, 8]
            deep.eval()
            assert torch.equal(loaded(spectra), deep(spectra))
            must_fail(lambda: model.load_model_checkpoint(path, "cpu", "wrong_id"))
            bad = copy.deepcopy(payload)
            bad["model_state"]["network.0.weight"] = torch.zeros(1)
            torch.save(bad, path)
            must_fail(lambda: model.load_model_checkpoint(path, "cpu", payload["checkpoint_id"]))
            dataset = OTSDataset(torch.randn(20, 319), torch.randn(20, 480),
                                 baseline["wavelengths_nm"], baseline["k_grid"], baseline["layout"])
            reference = adapter.canonical_split_train_validation(dataset, 0.15, adapter.SPLIT_SEED)
            assert all(torch.equal(a, b) for seed in (20261030, 20261032, 20261033)
                       for a, b in zip(reference, train.split_train_validation(dataset, 0.15, seed)))
            statistics = dict(zip(("x_mean", "x_scale", "y_mean", "y_scale"),
                                  train.normalization_statistics(reference[0], reference[1])))
            config = copy.deepcopy(payload["training_config"])
            training = dict(statistics, dataset=dataset, resolved_config=config,
                            input_dim=319, output_dim=480, fourier_modes=80,
                            run_id="validation_only", dataset_id=payload["dataset_id"],
                            checkpoint_id=payload["checkpoint_id"], x_train=reference[0],
                            loader_generator=torch.Generator().manual_seed(123),
                            train_data_path=temporary / "unused_train.parquet",
                            manifest_path=temporary / "unused_manifest.json", output_dir=temporary)
            candidate = dict(hidden_sizes=[32, 16, 8], dropout=0.02, statistics=statistics,
                             device=torch.device("cpu"), model=deep, config=config, best_state=deep.state_dict(),
                             best_epoch=1, best_validation_loss=0.1, epoch=1,
                             epochs_without_improvement=0,
                             optimizer=torch.optim.AdamW(deep.parameters()), history=[])
            candidate["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(candidate["optimizer"])
            parent = SimpleNamespace(f_code=train.train_model.__code__, f_locals=training, f_back=None)
            frame = SimpleNamespace(f_locals=candidate, f_back=parent)
            snapshots = adapter.EpochSnapshots()
            for epoch in range(1, 11):
                candidate["epoch"] = epoch
                candidate["history"].append(dict(epoch=epoch, train_loss=0.2, validation_loss=0.1, learning_rate=0.001))
                rng = torch.get_rng_state().clone()
                snapshots.boundary(frame)
                assert torch.equal(rng, torch.get_rng_state())
                assert (temporary / "depth_runtime/32x16x8/progress.json").is_file()
            snapshot_path = temporary / "depth_runtime/32x16x8/snapshot.pt"
            snapshot = torch.load(snapshot_path, weights_only=True)
            assert snapshot["completed_epochs"] == 10
            exported = temporary / "exported.pt"
            adapter.export_snapshot_best(snapshot_path, exported)
            model.load_model_checkpoint(exported, "cpu", payload["checkpoint_id"])
            snapshots.active = frame
            snapshots.stop(signal.SIGTERM, None)
            try:
                snapshots.boundary(frame)
            except SystemExit as error:
                assert error.code == 75
            else:
                raise AssertionError("Pause did not stop")
            assert torch.load(snapshot_path, weights_only=True)["reason"] == "signal_15"
            split_record = json.loads((temporary / "depth_runtime/split_control.json").read_text())
            assert split_record["split_seed"] == adapter.SPLIT_SEED
            results["synthetic_split_control"] = split_record
        results["unchanged_canonical_functions_strict_schema_state_id_load_checks"] = True
        results["fixed_split_across_training_seeds"] = True
        results["snapshot_progress_periodic_export_signal_no_rng_consumption"] = True
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_hash
    results.update(baseline_sha256=baseline_hash, no_gradients_or_training_runs=True)
    report = adapter.EXPERIMENT_ROOT / "adapter_validation.json"
    report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
