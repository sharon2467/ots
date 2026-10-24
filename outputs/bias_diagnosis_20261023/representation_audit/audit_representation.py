"""Existing-row, no-NN representation diagnostics; does not generate datasets.

Uses the saved smoke checkpoint ONLY for coordinate/layout metadata. No model is
instantiated or evaluated. Original f and per-row ne are absent in this snapshot.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import pyarrow.parquet as pq

ROOT = Path('/workspace/ots')
SRC = ROOT / 'OTS_code_share/NN_hotE_large_lmRange'
OUT = ROOT / 'outputs/bias_diagnosis_20261023/representation_audit'
RUN = SRC / 'runs/cloud_fig_e2198eeab80d'
sys.path.insert(0, str(SRC))
from evaluation import reconstruct_real_evdf, prediction_metrics
from data import distribution_moments, normalize_distribution, SPECIES_BLOCKS

torch.set_num_threads(2)
QE = 1.602176634e-19

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def signed_diagnostics(v, f, mass):
    # Diagnostic signed moments, explicitly NOT the positive-only public API.
    # Full support, trapz and signed normalization; no abs/clipping/subsampling.
    norm = torch.trapz(f, v)
    mean = torch.trapz(v * f, v) / norm
    m2 = torch.trapz(v.square() * f, v)
    second = m2 / norm - mean.square()
    neg = torch.trapz(torch.relu(-f), v)
    absolute = torch.trapz(f.abs(), v)
    values = dict(norm=float(norm), mean_m_s=float(mean),
                  temperature_ev=float(mass * second / QE),
                  raw_second_integral_m2_s2=float(m2),
                  negative_mass=float(neg), negative_mass_fraction=float(neg / absolute),
                  min_f=float(f.min()), max_f=float(f.max()))
    for fraction in (.5, .8):
        # Zero-masked integrands preserve full-grid quadrature, including boundaries.
        mask = v.abs() > fraction * v.abs().max()
        tail_m2 = torch.trapz(torch.where(mask, v.square() * f, 0.), v)
        tail_abs = torch.trapz(torch.where(mask, f.abs(), 0.), v)
        values[f'tail_abs_mass_gt_{fraction}V'] = float(tail_abs)
        values[f'tail_signed_T_contribution_gt_{fraction}V_ev'] = float(mass * tail_m2 / norm / QE)
    return values

def main():
    checkpoint_path = RUN / 'training/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    layout = checkpoint['layout']
    modes = checkpoint['fourier_modes']
    source_paths = [SRC / p for p in ['data.py', 'evaluation.py', 'model.py', 'phyEqns/ots_full_fourier.py']]
    input_paths = [checkpoint_path, *(RUN / f'data/dataset_{split}.parquet' for split in ['test', 'train'])]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths + input_paths}
    result = {
        'scope': 'No NN calls, no dataset generation, no original-f replay. Existing smoke rows only.',
        'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__},
        'checkpoint_metadata_only': {k: checkpoint[k] for k in ['run_id', 'dataset_id', 'checkpoint_id', 'dtype', 'input_dim', 'output_dim', 'fourier_modes']},
        'sha256_before': before,
        'missing': ['dataset_manifest.json', 'original physical VDFs', 'per-row original ne', 'per-row original actual T/u'],
        'unmeasured': ['original-f VDF relative-L2', 'temperature bias relative to manifest actual', 'PV at original per-row ne versus saved spectrum', 'D4/best-model performance'],
        'selection': 'All 4 existing test rows (indices 0..3), then first 4 train rows (indices 0..3). Zero-based physical-row indices; parquet row is index+1.',
        'grid': {}, 'rows': [],
    }
    arrays = {}
    for split in ['test', 'train']:
        table = pq.read_table(RUN / f'data/dataset_{split}.parquet')
        values = table.to_pandas().to_numpy(dtype=np.float64, copy=True)
        result.setdefault('parquet', {})[split] = {'total_rows': table.num_rows, 'columns': table.num_columns, 'schema_metadata': str(table.schema.metadata)}
        for species_index, (species, key) in enumerate(SPECIES_BLOCKS):
            block = layout[key]
            if not block['used_in_model']:
                continue
            offset = checkpoint['input_dim'] + species_index * 2 * modes
            # Preserve saved label and metadata values, promoted from their storage precision.
            k = torch.tensor(values[0, offset:offset + modes], dtype=torch.float64)
            y = torch.tensor(values[1:5, offset:offset + 2 * modes], dtype=torch.float64)
            v = torch.tensor(block['velocity_grid_m_s'], dtype=torch.float64)
            _, f = reconstruct_real_evdf(y, k, velocity=v)
            legacy_v, legacy_f = reconstruct_real_evdf(y, k)
            dk = k[2] - k[1]
            alias_period = 2 * math.pi / float(dk)
            # Evaluate the identical public inverse at an offset axis: even M gives
            # f(v+2pi/dK)=-f(v) for a perfectly uniform half-integer K grid.
            _, shifted_f = reconstruct_real_evdf(y, k, velocity=v + alias_period)
            alias_residual = torch.max(torch.abs(shifted_f + f)) / torch.max(torch.abs(f))
            result['grid'][species] = {
                'Nv': len(v), 'M': modes, 'velocity_limit_m_s': float(v[-1]),
                'velocity_support_width_m_s': float(v[-1] - v[0]),
                'k_min': float(k[0]), 'k_max': float(k[-1]), 'minimum_abs_K': float(k.abs().min()),
                'contains_K0': bool(torch.any(k == 0)), 'actual_dK': float(dk),
                'alias_antiperiod_m_s': alias_period,
                'support_width_over_alias_antiperiod': float(v[-1] - v[0]) / alias_period,
                'antiperiodicity_relative_max_residual_saved_K': float(alias_residual),
            }
            arrays[f'{split}_{species}_velocity'] = v.numpy()
            arrays[f'{split}_{species}_target_reconstruction'] = f.numpy()
            arrays[f'{split}_{species}_legacy_velocity'] = legacy_v.numpy()
            arrays[f'{split}_{species}_legacy_target_reconstruction'] = legacy_f.numpy()
            exact_label_metrics = prediction_metrics(y, y, k, velocity=v)
            result.setdefault('self_comparison_metrics', {}).setdefault(split, {})[species] = exact_label_metrics
            for index in range(len(y)):
                raw = signed_diagnostics(v, f[index], block['mass_kg'])
                clipped = normalize_distribution(v, torch.relu(f[index]), species)
                clipped_moments = distribution_moments(v, clipped, block['mass_kg'])
                legacy = signed_diagnostics(legacy_v, legacy_f[index], block['mass_kg'])
                row = {'split': split, 'physical_index': index, 'parquet_row': index + 1, 'species': species,
                       'raw_signed': raw, 'clipped_normalized': clipped_moments, 'legacy_M_grid_signed': legacy}
                result['rows'].append(row)
    after = {str(p.relative_to(ROOT)): sha(p) for p in source_paths + input_paths}
    result['sha256_after'] = after
    result['inputs_and_sources_unchanged'] = before == after
    assert before == after
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'target_reconstructions.npz', **arrays)
    flat = []
    for row in result['rows']:
        flat.append({**{k: row[k] for k in ['split', 'physical_index', 'parquet_row', 'species']},
                     **{'raw_' + k: v for k, v in row['raw_signed'].items()},
                     **{'clipnorm_' + k: v for k, v in row['clipped_normalized'].items()},
                     **{'legacy_' + k: v for k, v in row['legacy_M_grid_signed'].items()}})
    with (OUT / 'moments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    for species in ['e', 'i1', 'i2']:
        rows = [r for r in result['rows'] if r['species'] == species]
        print(species, 'raw T:', [round(r['raw_signed']['temperature_ev'], 3) for r in rows])
        print(species, 'clipnorm T:', [round(r['clipped_normalized']['actual_temperature_ev'], 3) for r in rows])
        print(species, 'norm range', min(r['raw_signed']['norm'] for r in rows), max(r['raw_signed']['norm'] for r in rows))
        print(species, 'negative fraction range', min(r['raw_signed']['negative_mass_fraction'] for r in rows), max(r['raw_signed']['negative_mass_fraction'] for r in rows))
    print('saved', OUT / 'results.json')

if __name__ == '__main__':
    main()
