#!/usr/bin/env python3
"""Read-only study aggregation. Test summaries require prior frozen selection."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'OTS_code_share/NN_hotE_large_lmRange'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(include_test):
    if include_test:
        selection = read(STUDY / 'frozen_selection.json')
        if selection.get('frozen') is not True:
            raise ValueError('Test summaries require frozen=true')
    runs, candidates = [], []
    for prefix, kind in [('fixed', 'fixed_ion'), ('fig', 'fig1_fig11')]:
        for stage in 'abc':
            name = f'study_{prefix}_{stage}'
            run = SOURCE / 'runs' / name
            history_path = run / 'training/training_history.json'
            validation_path = STUDY / 'exploration' / (name + '_validation_metrics.json')
            if not history_path.exists() or not validation_path.exists():
                continue
            history, validation = read(history_path), read(validation_path)
            config = history['training_config']
            assert validation['test_data_read'] is False
            log = (STUDY / 'logs' / (name + '.log')).read_text()
            elapsed = re.search(r'^real\s+(\d+)m([\d.]+)s$', log, re.M)
            assert 'all completed successfully.' in log
            entry = {
                'run_id': name, 'dataset_kind': kind, 'stage': stage.upper(),
                'raw_loss_weight': config['raw_loss_weight'],
                'selected_candidate': history['selected_candidate'],
                'training_config': config,
                'total_wall_seconds': int(elapsed[1])*60 + float(elapsed[2]) if elapsed else None,
                'candidate_training_seconds': sum(c['elapsed_seconds'] for c in history['candidates']),
                'validation_macro_relative_l2_mean': validation['selection_score'],
                'validation_species': {s: v['model']['metrics'] for s, v in validation['species'].items()},
                'train_parquet_sha256': sha(run/'data/dataset_train.parquet'),
                'test_parquet_sha256': sha(run/'data/dataset_test.parquet'),
                'checkpoint_sha256': sha(run/'training/model.pt'),
                'validation_split': {k: v for k, v in validation['split'].items() if not isinstance(v, list)},
                'paths': {'run': str(run), 'checkpoint': str(run/'training/model.pt'), 'history': str(history_path), 'validation': str(validation_path)}
            }
            if include_test:
                payload = read(run/'evaluation/metrics.json')
                assert payload['run_id'] == name
                metrics = payload['metrics']
                entry['test_species'] = {'e': metrics} if 'evdf_r2' in metrics else metrics
                entry['test_macro_relative_l2_mean'] = sum(v['evdf_relative_l2_mean'] for v in entry['test_species'].values())/len(entry['test_species'])
            for c in history['candidates']:
                record = {k: v for k, v in c.items() if k != 'history'}
                record.update(run_id=name, dataset_kind=kind, raw_loss_weight=config['raw_loss_weight'], epochs_completed=len(c['history']), selected=c['hidden_sizes']==history['selected_candidate']['hidden_sizes'])
                record['final_epoch'] = c['history'][-1]
                record['best_epoch_history'] = next(row for row in c['history'] if row['epoch']==c['best_epoch'])
                candidates.append(record)
            runs.append(entry)
    data_checks = {}
    for kind in ['fixed_ion', 'fig1_fig11']:
        group = [r for r in runs if r['dataset_kind']==kind]
        data_checks[kind] = {key: len({r[key] for r in group}) == 1 for key in ['train_parquet_sha256', 'test_parquet_sha256']}
        assert all(data_checks[kind].values())
    payload = {'test_metrics_read': include_test, 'runs': runs, 'network_candidates': candidates, 'data_hash_identity_checks': data_checks}
    stem = 'results_summary' if include_test else 'validation_summary'
    (STUDY/(stem+'.json')).write_text(json.dumps(payload, indent=2)+'\n')
    fields = ['run_id','dataset_kind','raw_loss_weight','validation_macro_relative_l2_mean','total_wall_seconds','candidate_training_seconds']
    if include_test: fields.append('test_macro_relative_l2_mean')
    with (STUDY/(stem+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in runs)
    print(json.dumps({'output':str(STUDY/(stem+'.json')),'runs':len(runs),'network_candidates':len(candidates),'test_metrics_read':include_test}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--include-test',action='store_true');a=p.parse_args();summarize(a.include_test)
