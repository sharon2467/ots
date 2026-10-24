#!/usr/bin/env python3
"""Summarize frozen diagnostics; no inference, training, or weight selection.

For the added paired comparison only: selected=ensemble, baseline=frozen single.
The old-study model remains the separate 'baseline' role in the metric tables.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from final_comparison import paired_comparison


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compact(model):
    metrics = model['fourier_target']['metrics']
    raw = model['raw_signed_temperature_vs_original']
    taylor = model['fixed_taylor3_temperature_vs_original']
    original = model['original_vdf_relative_l2']
    return {
        'fourier_target': {
            'mean': metrics['evdf_relative_l2_mean'],
            'p50': metrics['evdf_relative_l2_p50'],
            'p95': metrics['evdf_relative_l2_p95'],
            'r2_global_flattened': metrics['evdf_r2'],
            'negative_mass_mean': metrics['negative_mass_fraction_mean'],
        },
        'original_vdf': {key: original[key] for key in ('mean', 'p95', 'max')},
        'raw_temperature': {
            'mae_ev_all_finite': raw['all_finite']['mae_ev'],
            'finite_count': raw['all_finite']['count'],
            'negative_count': raw['negative_temperature_count'],
            'nonfinite_count': raw['nonfinite_count'],
        },
        'taylor3_temperature': {
            'mae_ev_all_finite': taylor['all_finite']['mae_ev'],
            'finite_count': taylor['all_finite']['count'],
            'negative_count': taylor['negative_temperature_count'],
            'nonfinite_count': taylor['nonfinite_count'],
        },
    }


def main():
    json_path = ROOT / 'final_results_summary.json'
    csv_path = ROOT / 'final_results_metrics.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Refusing to overwrite final summary or CSV')
    frozen = {role: ROOT / ('frozen_' + role + '_selection.json')
              for role in ('single', 'ensemble')}
    trials_path = ROOT / 'trials.json'
    trials = json.loads(trials_path.read_text())
    if not isinstance(trials, list) or len(trials) != 7:
        raise ValueError('Expected seven recorded trials in trials.json')
    hashes = {role: {'path': str(path), 'sha256': sha256(path)}
              for role, path in frozen.items()}
    summary = {
        'schema_version': 1,
        'purpose': 'Compact frozen final metrics plus conditional ensemble versus frozen-single paired comparison; no tuning',
        'metric_table_roles': {
            'baseline': 'Historical frozen study C checkpoint, evaluated on the same new 1024 test states',
            'single': 'Frozen selected single checkpoint',
            'ensemble': 'Frozen physical-Fourier convex combination; unchanged weights',
        },
        'paired_comparison_roles': {'selected': 'ensemble', 'baseline': 'single'},
        'units': {'relative_l2': 'fraction, not percent', 'negative_mass': 'fraction', 'temperature_mae': 'eV'},
        'temperature_convention': 'All finite estimates, including finite negative temperatures; nonfinite and negative counts are separate',
        'uncertainty': 'Conditional paired percentile bootstrap, 2000 resamples; same physical indices across roles and species; excludes training and adaptive-selection uncertainty',
        'frozen_selections': hashes,
        'trials': {'path': str(trials_path), 'sha256': sha256(trials_path), 'count': len(trials)},
        'paired_function': {'path': str(HERE / 'final_comparison.py'), 'sha256': sha256(HERE / 'final_comparison.py'), 'name': 'paired_comparison'},
        'datasets': {},
    }
    rows = []
    for kind in ('fixed_ion', 'fig1_fig11'):
        paths = {role: ROOT / 'diagnostics' / ('final_' + role) / kind / 'final_comparison.json'
                 for role in ('single', 'ensemble')}
        docs = {role: json.loads(path.read_text()) for role, path in paths.items()}
        single, ensemble = docs['single'], docs['ensemble']
        identity_keys = ('dataset_kind', 'test_dataset_id', 'test_seed', 'test_sample_count',
                         'test_manifest_sha256', 'test_parquet_sha256', 'species_order',
                         'frozen_auxiliary_temperature_method')
        for key in identity_keys:
            if single[key] != ensemble[key]:
                raise ValueError(f'{kind}: test identity mismatch: {key}')
        if single['test_sample_count'] != 1024:
            raise ValueError('Expected exactly 1024 test states')
        for role, doc in docs.items():
            if Path(doc['selection_file']).resolve() != frozen[role].resolve() or doc['selection_file_sha256'] != hashes[role]['sha256']:
                raise ValueError(f'{kind}: frozen {role} hash/path mismatch')
        names = single['species_order']
        result = {
            'test_identity': {key: single[key] for key in identity_keys},
            'sources': {role: {'path': str(path), 'sha256': sha256(path)} for role, path in paths.items()},
            'provenance_checks': {'same_test_hashes': True, 'baseline_models_exact_equal': single['models']['baseline'] == ensemble['models']['baseline']},
            'species_order': names, 'species': {},
        }
        if not result['provenance_checks']['baseline_models_exact_equal']:
            raise ValueError(f'{kind}: historical baseline checkpoint provenance differs')
        errors = {'single': [], 'ensemble': []}
        baseline_payload_equal = {}
        expected_indices = list(range(1024))
        for species in names:
            old = single['species'][species]['models']['baseline']
            if old != ensemble['species'][species]['models']['baseline']:
                raise ValueError(f'{kind}/{species}: baseline predictions/metrics differ')
            baseline_payload_equal[species] = True
            models = {'baseline': old,
                      'single': single['species'][species]['models']['selected'],
                      'ensemble': ensemble['species'][species]['models']['selected']}
            result['species'][species] = {}
            for role, model in models.items():
                samples = model['fourier_target']['per_sample']
                indices = [row['test_sample_index_zero_based'] for row in samples]
                parquet_rows = [row['parquet_row_including_metadata_zero_based'] for row in samples]
                if indices != expected_indices or parquet_rows != list(range(1, 1025)):
                    raise ValueError(f'{kind}/{species}/{role}: test index ordering differs')
                metric = compact(model)
                result['species'][species][role] = metric
                row = {'dataset_kind': kind, 'species': species, 'role': role, 'test_count': 1024}
                for section, values in metric.items():
                    row.update({section + '_' + key: value for key, value in values.items()})
                rows.append(row)
                if role in errors:
                    vector = np.array([entry['evdf_relative_l2'] for entry in samples], dtype=np.float64)
                    if not np.isfinite(vector).all() or not np.isclose(vector.mean(), metric['fourier_target']['mean'], rtol=1e-12, atol=1e-14):
                        raise ValueError('Per-sample error vector does not match reported mean')
                    errors[role].append(vector)
        result['provenance_checks'].update({
            'same_index_order_all_species_roles': True,
            'ordered_test_indices_sha256': hashlib.sha256(np.arange(1024, dtype='<i8').tobytes()).hexdigest(),
            'historical_baseline_prediction_metric_payload_exact_equal': baseline_payload_equal,
        })
        result['macro_fourier_target_relative_l2_mean'] = {
            role: float(np.mean([result['species'][species][role]['fourier_target']['mean'] for species in names]))
            for role in ('baseline', 'single', 'ensemble')}
        result['ensemble_vs_frozen_single'] = {
            'selected': 'ensemble', 'baseline': 'single', 'metric': 'per_sample_fourier_target_relative_l2',
            'species_order': names,
            **paired_comparison(np.stack(errors['ensemble']), np.stack(errors['single'])),
        }
        summary['datasets'][kind] = result
    # Exclusive creation: neither existing deliverable may be overwritten.
    with json_path.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for kind, data in summary['datasets'].items():
        print(json.dumps({'dataset_kind': kind,
                          'comparison_roles': {'selected': 'ensemble', 'baseline': 'single'},
                          'macro': data['ensemble_vs_frozen_single']['macro'],
                          'checks_passed': True}))
    print(json.dumps({'summary': str(json_path), 'metrics_csv': str(csv_path), 'metric_rows': len(rows)}))


if __name__ == '__main__':
    main()
