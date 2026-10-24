"""Descriptive validation-only error stratification; no fitting or test access."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[2] / 'OTS_code_share/NN_hotE_large_lmRange'
for name in sys.argv[1:]:
    audit = json.loads((OUT / (name + '_validation_metrics.json')).read_text())
    manifest = json.loads(Path(audit['inputs']['training_manifest']).read_text())
    indices = np.array(audit['split']['validation_indices'])
    records = [manifest['splits']['train']['samples'][int(i)] for i in indices]
    band = manifest['physical_configuration']['wavelength_bands_nm'][0]
    table = pq.read_table(audit['inputs']['train_parquet'])
    x = table.select(list(range(len(band)))).to_pandas().values[indices + 1]
    features = {'density_m3': np.array([r['electron_density_m3'] for r in records]),
                'IAW_points_above_half_peak': (x > .5).sum(1),
                'IAW_equivalent_width_nm': np.trapezoid(x, np.array(band), axis=1)}
    for s in ('e', 'i1', 'i2'):
        features[s + '_temperature_ev'] = np.array([r['species'][s]['actual_temperature_ev'] for r in records])
    results = {}
    for s, entry in audit['species'].items():
        error = np.array([r['evdf_relative_l2'] for r in entry['model']['per_sample']])
        rows = []
        for label, values in features.items():
            if np.ptp(values) <= abs(np.mean(values))*1e-12:
                continue
            lo, hi = np.quantile(values, [.25, .75])
            low, high = values <= lo, values >= hi
            rows.append({'feature': label, 'spearman_r': float(spearmanr(values, error).statistic),
                         'low_quartile_max': float(lo), 'high_quartile_min': float(hi),
                         'low_count': int(low.sum()), 'high_count': int(high.sum()),
                         'low_mean_relative_l2': float(error[low].mean()),
                         'high_mean_relative_l2': float(error[high].mean())})
        results[s] = sorted(rows, key=lambda row: abs(row['spearman_r']), reverse=True)
    payload = {'run_id': name, 'test_read': False,
               'scope': 'Descriptive associations on the fixed validation set, not causal effects. Quartile counts can differ due to ties.',
               'species': results}
    (OUT / (name + '_validation_strata.json')).write_text(json.dumps(payload, indent=2))
    print(json.dumps({'run': name, 'strongest_associations': {s: rows[:2] for s, rows in results.items()}}, indent=2))
