#!/usr/bin/env python3
"""Read frozen Fig checkpoints and one existing test; never train or generate data."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'OTS_code_share/NN_hotE_large_lmRange'
DEPTH = ROOT / 'outputs/depth_study_20261021'
sys.path.insert(0, str(DEPTH))
import adapter
adapter.install_adapter()
import data
import model
import evaluation
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def describe(x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite metric')
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()))


def verify_files(plan):
    for name, expected in plan['protected_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Protected artifact changed: ' + name)
    for directory, expected in plan['registry_snapshots'].items():
        root = Path(directory)
        actual = {str(p.relative_to(root)): str(p.resolve(strict=True))
                  for p in root.glob('*/*') if p.is_symlink()}
        if actual != expected:
            raise ValueError('Registry changed: ' + directory)


def metrics_for(predicted, dataset, modes):
    relative, negative, by_species = [], [], {}
    for i, (species, key) in enumerate(data.SPECIES_BLOCKS):
        section = slice(i * 2 * modes, (i + 1) * 2 * modes)
        target, prediction = dataset.coefficients[:, section], predicted[:, section]
        k = dataset.k_grid[i * modes:(i + 1) * modes]
        velocity = dataset.layout[key]['velocity_grid_m_s']
        axis, truth = evaluation.reconstruct_real_evdf(target, k, velocity=velocity)
        _, inferred = evaluation.reconstruct_real_evdf(prediction, k, velocity=velocity)
        rel = torch.sqrt((inferred - truth).square().sum(1) /
                         truth.square().sum(1).clamp(min=1e-30)).numpy()
        neg = (torch.trapz(torch.relu(-inferred), axis, dim=1) /
               torch.trapz(inferred.abs(), axis, dim=1).clamp(min=1e-30)).numpy()
        canonical = evaluation.prediction_metrics(target, prediction, k, velocity=velocity)
        # A fresh canonical call must agree with the per-sample calculation.
        for metric, value in [('evdf_relative_l2_mean', rel.mean()),
                              ('evdf_relative_l2_median', np.median(rel)),
                              ('negative_mass_fraction_mean', neg.mean())]:
            if not np.isclose(canonical[metric], value, rtol=1e-6, atol=1e-12):
                raise ValueError('Fresh canonical/per-sample mismatch: ' + species + '/' + metric)
        relative.append(rel)
        negative.append(neg)
        by_species[species] = dict(relative_l2=describe(rel), negative_mass=describe(neg),
                                   canonical=canonical)
    relative, negative = np.stack(relative, axis=1), np.stack(negative, axis=1)
    return relative, negative, by_species


def evaluate(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if (HERE / 'PAUSE').exists():
        raise RuntimeError('Paused; refusing evaluation')
    if (HERE / 'results.json').exists():
        raise FileExistsError('Existing results are immutable; use a new output directory')
    verify_files(plan)
    # Composite is an existing saved choice, never optimized on this test.
    singles = {r['model_id']: r for r in plan['models'] if r['kind'] == 'single'}
    for item in plan['models']:
        if item['kind'] != 'ensemble':
            continue
        frozen = json.loads(Path(item['recipe_path']).read_text())
        original = frozen['comparisons']['fig1_fig11']['selected_ensemble']['components']
        if not frozen['frozen'] or len(original) != len(item['components']):
            raise ValueError('Missing original frozen ensemble recipe')
        for old, component in zip(original, item['components']):
            member = singles[component['model_id']]
            if (Path(member['checkpoint']).resolve() !=
                    (Path(old['training_run']) / 'training/model.pt').resolve() or
                    component['weight'] != old['weight'] or
                    component['checkpoint_sha256'] != old['checkpoint_sha256'] or
                    member['checkpoint_sha256'] != old['checkpoint_sha256']):
                raise ValueError('Composite differs from frozen recipe')
        if item['parameter_count'] != sum(singles[c['model_id']]['parameter_count']
                                          for c in item['components']):
            raise ValueError('Composite parameter count is not member sum')
    torch.set_num_threads(2)
    start = time.perf_counter()
    spec = plan['dataset']
    manifest = json.loads(Path(spec['manifest']).read_text())
    test_path = Path(spec['test_data'])
    if (Path(spec['manifest']).parent / manifest['splits']['test']['file']).resolve() != test_path.resolve():
        raise ValueError('Test path disagrees with manifest')
    dataset = data.load_dataset(test_path, manifest['physical_configuration'])
    if len(dataset.spectra) != spec['physical_samples']:
        raise ValueError('Wrong number of physical test rows')
    expected_species = ['e', 'i1', 'i2']
    actual_species = [s for s, key in data.SPECIES_BLOCKS if dataset.layout[key]['used_in_model']]
    if actual_species != expected_species:
        raise ValueError('Expected all three Fig output species')
    folders = HERE / 'predictions'
    folders.mkdir(exist_ok=False)
    np.savez_compressed(HERE / 'common_test_targets.npz',
                        source_sample_index=np.arange(len(dataset.spectra)),
                        spectra=dataset.spectra.numpy(),
                        target_coefficients=dataset.coefficients.numpy(),
                        k_grid=dataset.k_grid.numpy(), wavelengths_nm=dataset.wavelengths_nm.numpy())
    predicted_by_id, completed = {}, []
    for item in plan['models']:
        if (HERE / 'PAUSE').exists():
            write_json(HERE / 'progress.json', dict(status='paused', completed=completed))
            return 75
        began = time.perf_counter()
        if item['kind'] == 'single':
            network, cp = model.load_model_checkpoint(item['checkpoint'], 'cpu', item['checkpoint_id'])
            if cp.get('layout') != dataset.layout:
                raise ValueError('Layout meaning differs: ' + item['model_id'])
            if not torch.equal(cp['wavelengths_nm'].to(dataset.wavelengths_nm), dataset.wavelengths_nm):
                raise ValueError('Wavelength axis differs: ' + item['model_id'])
            if not torch.equal(cp['k_grid'].to(dataset.k_grid), dataset.k_grid):
                raise ValueError('Fourier K grid differs: ' + item['model_id'])
            if (cp['input_dim'], cp['output_dim']) != (dataset.spectra.shape[1], dataset.coefficients.shape[1]):
                raise ValueError('Input/output dimensions differ')
            if sum(p.numel() for p in network.parameters()) != item['parameter_count']:
                raise ValueError('Parameter count differs')
            if cp['hidden_sizes'] != item['hidden_sizes']:
                raise ValueError('Hidden sizes differ')
            predicted = evaluation.predict_coefficients(network, dataset.spectra, cp, 128, 'cpu')
            modes = cp['fourier_modes']
            del network, cp
        elif item['kind'] == 'ensemble':
            # Preserve the already frozen coefficient-average recipe, order and dtype.
            components = item['components']
            if abs(sum(c['weight'] for c in components) - 1) > 1e-12:
                raise ValueError('Frozen ensemble weights must sum to one')
            predicted = sum(c['weight'] * predicted_by_id[c['model_id']] for c in components)
            modes = dataset.layout['fourier_modes']
        else:
            raise ValueError('Unknown model kind')
        rel, neg, details = metrics_for(predicted, dataset, modes)
        macro = rel.mean(axis=1)
        predicted_by_id[item['model_id']] = predicted
        path = folders / (item['model_id'] + '.npz')
        np.savez_compressed(path, predicted_coefficients=predicted.numpy(),
                            relative_l2=rel, negative_mass_fraction=neg,
                            macro_relative_l2=macro)
        row = dict(item, macro_relative_l2=describe(macro), species_metrics=details,
                   predictions_path=str(path), predicted_coefficients_sha256=array_sha(predicted.numpy()),
                   seconds=time.perf_counter() - began,
                   compatibility='exact_layout_axes_and_dimensions',
                   fresh_canonical_metric_check='passed')
        completed.append(row)
        write_json(HERE / 'progress.json', dict(status='running', completed=completed))
        print(json.dumps({'completed':len(completed), 'total':len(plan['models']),
                          'model':item['model_id'], 'test_macro_percent':float(macro.mean()) * 100},
                         ensure_ascii=False), flush=True)
    verify_files(plan)
    ranked = sorted(completed, key=lambda r: (-r['macro_relative_l2']['mean'], r['model_id']))
    result = dict(status='completed', evaluated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  plan=str(plan_path), plan_sha256=sha(plan_path), dataset=spec,
                  metric='Equal-species mean of per-row VDF relative-L2, then mean across all test rows',
                  order='descending unified test macro relative L2; lower is better',
                  species=expected_species, target_coefficients_sha256=array_sha(dataset.coefficients.numpy()),
                  total_seconds=time.perf_counter() - start, models=ranked,
                  protected_artifacts_unchanged=True, no_generation_or_training=True)
    write_json(HERE / 'results.json', result)
    fields = ['rank', 'model_id', 'kind', 'parameter_count', 'hidden_sizes', 'training_pool_samples',
              'optimization_samples', 'validation_samples', 'seed', 'learning_rate', 'raw_loss_weight',
              'dropout', 'best_epoch', 'test_macro_mean', 'test_macro_p95', 'test_macro_maximum',
              'test_e_mean', 'test_H_mean', 'test_C_mean', 'checkpoint', 'checkpoint_sha256']
    with (HERE / 'ranking.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(ranked, 1):
            flat = {key: row.get(key) for key in fields if key in row}
            for key in ('hidden_sizes', 'seed', 'best_epoch'):
                flat[key] = json.dumps(flat[key], ensure_ascii=False)
            flat.update(rank=index, test_macro_mean=row['macro_relative_l2']['mean'],
                        test_macro_p95=row['macro_relative_l2']['p95'],
                        test_macro_maximum=row['macro_relative_l2']['maximum'],
                        test_e_mean=row['species_metrics']['e']['relative_l2']['mean'],
                        test_H_mean=row['species_metrics']['i1']['relative_l2']['mean'],
                        test_C_mean=row['species_metrics']['i2']['relative_l2']['mean'])
            writer.writerow(flat)
    write_json(HERE / 'progress.json', dict(status='completed', model_count=len(completed),
                                          seconds=result['total_seconds']))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=HERE/'evaluation_plan.json')
    args = parser.parse_args()
    raise SystemExit(evaluate(args.plan))
