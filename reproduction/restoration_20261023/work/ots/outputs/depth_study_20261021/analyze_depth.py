#!/usr/bin/env python3
"""Independent fixed-split validation and explicitly invoked test aggregation.

Validation never loads a test path. Test aggregation is available only through
the separate summarize subcommand after the experiment protocol is frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent
SOURCE = STUDY.parents[1] / 'OTS_code_share' / 'NN_hotE_large_lmRange'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(STUDY))
import adapter
import data
import model
import train
import evaluation

SPECIES = ('e', 'i1', 'i2')
CONDITIONAL_CI = ('Paired bootstrap over physical sample rows, keeping species '
                  'together. Conditional on these trained checkpoints; excludes '
                  'training-seed and model-selection uncertainty. Existing '
                  'seed20261031 test was previously exposed.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def describe(a):
    a = np.asarray(a)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite sample metrics.')
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def sample_metrics(target, predicted, k_grid, velocity):
    """Use canonical float32/float64 arithmetic and its denominator floor."""
    velocity, truth = evaluation.reconstruct_real_evdf(target, k_grid, velocity=velocity)
    _, inferred = evaluation.reconstruct_real_evdf(predicted, k_grid, velocity=velocity)
    relative = torch.sqrt((inferred - truth).square().sum(1) /
                          truth.square().sum(1).clamp(min=1e-30))
    negative = torch.trapz(torch.relu(-inferred), velocity, dim=1) / torch.trapz(
        inferred.abs(), velocity, dim=1).clamp(min=1e-30)
    return relative.numpy(), negative.numpy()


def prepare_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ('summary.json', 'samples.npz')):
        raise FileExistsError('Refusing to replace analysis outputs: ' + str(path))
    return path


def validation(args):
    install = getattr(adapter, 'install_adapter', None) or adapter.install
    install()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    # Deliberately access only the train entry; never resolve/open test data.
    train_path = manifest_path.parent / manifest['splits']['train']['file']
    dataset = data.load_dataset(train_path, manifest['physical_configuration'])
    if len(dataset.spectra) != 26000:
        raise ValueError('This protocol requires the original 26000 training candidates.')
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    regressor, cp = model.load_model_checkpoint(checkpoint_path, torch.device('cpu'), payload['checkpoint_id'])
    if cp['dataset_id'] != manifest['dataset_id'] or cp.get('layout') != dataset.layout:
        raise ValueError('Checkpoint training dataset/layout does not match fixed data.')
    for name, actual in (('k_grid', dataset.k_grid), ('wavelengths_nm', dataset.wavelengths_nm)):
        if not torch.equal(cp[name].cpu().to(actual.dtype), actual):
            raise ValueError('Checkpoint ' + name + ' differs from dataset.')
    x, y, vx, vy = train.split_train_validation(dataset, .15, 20261030)
    statistics = train.normalization_statistics(x, y)
    stats_hash = {}
    for name, actual in zip(('x_mean', 'x_scale', 'y_mean', 'y_scale'), statistics):
        if not torch.equal(cp[name].cpu(), actual):
            raise ValueError('Checkpoint statistics differ from the frozen split: ' + name)
        stats_hash[name] = array_sha(actual.numpy())
    generator = torch.Generator().manual_seed(20261030)
    permutation = torch.randperm(26000, generator=generator)
    indices = permutation[:3900].numpy()
    predicted = evaluation.predict_coefficients(regressor, vx, cp, 128, torch.device('cpu'))
    modes = cp['fourier_modes']
    selected = [(s, key) for s, key in data.SPECIES_BLOCKS if cp['layout'][key]['used_in_model']]
    if tuple(s for s, _ in selected) != SPECIES:
        raise ValueError('Fig protocol requires all e/H/C outputs.')
    metrics, relative, negative = {}, [], []
    for index, (species, key) in enumerate(selected):
        section = slice(index * 2 * modes, (index + 1) * 2 * modes)
        k = dataset.k_grid[index * modes:(index + 1) * modes]
        velocity = cp['layout'][key]['velocity_grid_m_s']
        target, inferred = vy[:, section], predicted[:, section]
        rel, neg = sample_metrics(target, inferred, k, velocity)
        relative.append(rel)
        negative.append(neg)
        metrics[species] = evaluation.prediction_metrics(target, inferred, k, velocity=velocity)
        metrics[species]['relative_l2_distribution'] = describe(rel)
        metrics[species]['negative_mass_distribution'] = describe(neg)
    relative, negative = np.stack(relative, 1), np.stack(negative, 1)
    macro = relative.mean(1)
    out = prepare_output(args.output_dir)
    np.savez_compressed(out / 'samples.npz', source_sample_index=indices,
                        relative_l2=relative, negative_mass_fraction=negative,
                        macro_relative_l2=macro, species=np.asarray(SPECIES),
                        target_coefficients=vy.numpy(), predicted_coefficients=predicted.numpy())
    summary = dict(stage='validation', checkpoint=str(checkpoint_path),
                   checkpoint_sha256=sha(checkpoint_path), manifest=str(manifest_path),
                   train_sha256=sha(train_path), split_seed=20261030,
                   training_samples=22100, validation_samples=3900,
                   validation_indices_sha256=array_sha(indices), statistics_sha256=stats_hash,
                   training_seed=cp['training_config']['seed'], hidden_sizes=cp['hidden_sizes'],
                   best_epoch=cp['best_epoch'], metrics=metrics,
                   macro_relative_l2=describe(macro), test_data_accessed=False)
    write_json(out / 'summary.json', summary)
    print(json.dumps({'output': str(out), 'macro_relative_l2': describe(macro)}))


def bootstrap_difference(candidate, reference, repetitions=10000, seed=20261021):
    if np.shape(candidate) != np.shape(reference):
        raise ValueError('Paired metric vectors must have identical shapes.')
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError('Bootstrap repetitions must be a positive integer.')
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if difference.ndim != 1 or not np.isfinite(difference).all() or not len(difference):
        raise ValueError('Paired metric vectors must be finite and nonempty.')
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        indices = rng.integers(0, len(difference), (count, len(difference)))
        means[start:start + count] = difference[indices].mean(1)
    return dict(difference_candidate_minus_reference=float(difference.mean()),
                ci95_percentile=np.quantile(means, [.025, .975]).tolist(),
                repetitions=repetitions, bootstrap_seed=seed,
                interpretation='Negative favors candidate. ' + CONDITIONAL_CI)


def summarize(args):
    """Explicit test-stage operation. Never called during validation."""
    runs = json.loads(args.runs_json.read_text())
    if isinstance(runs, dict):
        runs = runs['runs']
    identities = [(r['label'], r['training_seed']) for r in runs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate label/training_seed.')
    collected, reference_hashes, arrays, comparisons = [], None, {}, []
    for run in runs:
        directory = Path(run['evaluation_dir'])
        metric_payload = json.loads((directory / 'metrics.json').read_text())
        checkpoint_path = Path(run['checkpoint']).resolve()
        if (directory / metric_payload['checkpoint']).resolve() != checkpoint_path:
            raise ValueError('Evaluation provenance differs from the declared checkpoint.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['training_config']['seed'] != run['training_seed']:
            raise ValueError('Declared training seed differs from checkpoint.')
        canonical = metric_payload['metrics']
        rels, negs, hashes, details = [], [], {}, {}
        for species in SPECIES:
            with np.load(directory / species / 'predictions.npz', allow_pickle=False) as payload:
                target = torch.from_numpy(payload['target_coefficients'].copy())
                predicted = torch.from_numpy(payload['predicted_coefficients'].copy())
                k = torch.from_numpy(payload['k_grid'].copy())
                velocity = torch.from_numpy(payload['velocity_m_s'].copy())
                hashes[species] = {key: array_sha(payload[key]) for key in
                                   ('target_coefficients', 'k_grid', 'velocity_m_s')}
            metrics = evaluation.prediction_metrics(target, predicted, k, velocity=velocity)
            for key, value in metrics.items():
                if not np.isclose(value, canonical[species][key], rtol=1e-6, atol=1e-12):
                    raise ValueError(f'Canonical metric mismatch: {run["run_id"]}/{species}/{key}')
            rel, neg = sample_metrics(target, predicted, k, velocity)
            rels.append(rel)
            negs.append(neg)
            details[species] = dict(canonical=metrics, relative_l2=describe(rel), negative_mass=describe(neg))
        if reference_hashes is None:
            reference_hashes = hashes
        if hashes != reference_hashes:
            raise ValueError('Test targets/grids/order differ; cannot pair runs.')
        rels, negs = np.stack(rels, 1), np.stack(negs, 1)
        identity = run['run_id']
        if identity in arrays:
            raise ValueError('Duplicate run_id.')
        arrays[identity] = dict(relative_l2=rels, negative_mass_fraction=negs, macro_relative_l2=rels.mean(1))
        collected.append(dict(**run, checkpoint_sha256=sha(checkpoint_path),
                              metrics=details, macro_relative_l2=describe(rels.mean(1))))
    for candidate in runs:
        label = candidate['label']
        references = ['D2'] + (['W' + label[1:]] if label in ('D3', 'D4') else [])
        for reference_label in references:
            if label == reference_label:
                continue
            reference = next((r for r in runs if r['label'] == reference_label and
                              r['training_seed'] == candidate['training_seed']), None)
            if reference is None:
                continue
            c, r = arrays[candidate['run_id']], arrays[reference['run_id']]
            result = dict(candidate=candidate['run_id'], reference=reference['run_id'],
                          label=label, reference_label=reference_label,
                          training_seed=candidate['training_seed'])
            result['macro'] = bootstrap_difference(c['macro_relative_l2'], r['macro_relative_l2'])
            result['species'] = {s: bootstrap_difference(c['relative_l2'][:, i], r['relative_l2'][:, i])
                                 for i, s in enumerate(SPECIES)}
            comparisons.append(result)
    seed_summary = []
    for pair in sorted(set((c['label'], c['reference_label']) for c in comparisons)):
        rows = [c for c in comparisons if (c['label'], c['reference_label']) == pair]
        deltas = [c['macro']['difference_candidate_minus_reference'] for c in rows]
        seed_summary.append(dict(candidate_label=pair[0], reference_label=pair[1],
                                 seeds=[c['training_seed'] for c in rows], deltas=deltas,
                                 mean_delta=float(np.mean(deltas)),
                                 sample_std_delta=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                                 seed_count=len(deltas), negative_delta_count=sum(d < 0 for d in deltas),
                                 note='Descriptive training-seed replication; sample CI is not a seed CI.'))
    out = prepare_output(args.output_dir)
    np.savez_compressed(out / 'samples.npz', **{run + '__' + key: value
                        for run, metrics in arrays.items() for key, value in metrics.items()})
    write_json(out / 'summary.json', dict(stage='test', runs=collected, comparisons=comparisons,
               seed_summary=seed_summary, target_hashes=reference_hashes, species=list(SPECIES),
               uncertainty_scope=CONDITIONAL_CI, selection_bias='Architecture selected using validation. '
               'This previously exposed test is not a new unbiased confirmation set.'))
    print(json.dumps({'output': str(out), 'runs': len(runs), 'comparisons': len(comparisons)}))


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validation')
    v.add_argument('--checkpoint', type=Path, required=True)
    v.add_argument('--output-dir', type=Path, required=True)
    v.add_argument('--manifest', type=Path, default=SOURCE / 'runs/scale_fig_26k/data/dataset_manifest.json')
    v.set_defaults(function=validation)
    s = sub.add_parser('summarize')
    s.add_argument('--runs-json', type=Path, required=True,
                   help='JSON list of run_id,label,training_seed,checkpoint,evaluation_dir records.')
    s.add_argument('--output-dir', type=Path, required=True)
    s.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
