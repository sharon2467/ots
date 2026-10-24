#!/usr/bin/env python3
"""Continue one archived interrupted run, charging both command segments."""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_trial import STUDY, PYTHON, PAUSE, digest, verify_invariants, write_json

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def process_start():
    fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    return (time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
            + int(fields[19]) / os.sysconf('SC_CLK_TCK'))


def main():
    process_epoch = process_start()
    start_mono = time.monotonic() - (time.time() - process_epoch)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True,
                        choices=['depth_fig_d2_s1', 'depth_fig_d4_s1'])
    args = parser.parse_args()
    if PAUSE.exists():
        raise SystemExit('Study paused; refusing to resume.')
    command = f"python -B {STUDY / 'resume_trial.py'} --run-id {args.run_id}"
    if command not in (STUDY / 'README.md').read_text():
        raise RuntimeError('Write the exact resume command in README first')
    archive = STUDY / 'resume_archive_20261021'
    manifest = json.loads((archive / 'archive_manifest.json').read_text())
    old_path = archive / 'execution' / (args.run_id + '.json')
    execution = STUDY / 'execution' / (args.run_id + '.json')
    old = json.loads(old_path.read_text())
    if old['status'] != 'interrupted_train' or digest(execution) != digest(old_path):
        raise RuntimeError('Only the original untouched interrupted execution may resume once')
    if not (STUDY / 'resume_authorization.json').exists():
        raise RuntimeError('Missing explicit resume authorization record')
    validation = json.loads((STUDY / 'resume_validation.json').read_text())
    if validation.get('status') != 'passed':
        raise RuntimeError('Resume equivalence validation must pass first')
    for name in ('resume_adapter.py', 'validate_resume.py'):
        if validation.get('script_sha256', {}).get(name) != digest(STUDY / name):
            raise RuntimeError('Resume validation does not match current script: ' + name)
    protocol = json.loads((STUDY / 'protocol.json').read_text())
    if digest(STUDY / 'protocol.json') != old['protocol_sha256']:
        raise RuntimeError('Frozen protocol changed since interrupted training')
    if any(digest(STUDY / name) != value for name, value in old['script_sha256'].items()):
        raise RuntimeError('Original study scripts changed since interrupted training')
    verify_invariants(protocol)
    run_dir = STUDY / 'runs' / args.run_id
    if any((run_dir / 'training' / f).exists()
           for f in ('model.pt', 'training_history.json', 'learning_curves.png')):
        raise RuntimeError('Refusing to overwrite completed training products')
    snapshot = next((archive / 'runs' / args.run_id / 'training/depth_runtime').glob('*/snapshot.pt'))
    if digest(snapshot) != manifest[str(snapshot.relative_to(archive))]:
        raise RuntimeError('Archived recovery snapshot changed')
    starts = json.loads((archive / 'observed_process_starts.json').read_text())
    original_start = starts[args.run_id]['process_started_at_utc']
    prior_seconds = (dt.datetime.fromisoformat(old['finished_at_utc'])
                     - dt.datetime.fromisoformat(original_start)).total_seconds()
    deadline = start_mono + protocol['compute_budget']['tail_timeout_seconds'] - prior_seconds
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(STUDY / '.mplconfig'),
               XDG_CACHE_HOME=str(STUDY / '.cache'), PYTHONDONTWRITEBYTECODE='1')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    scripts = ('adapter.py', 'run_trial.py', 'analyze_depth.py',
               'resume_adapter.py', 'resume_trial.py', 'validate_resume.py')
    hashes = {name: digest(STUDY / name) for name in scripts}
    train_cmd = [str(PYTHON), '-B', '-u', str(STUDY / 'resume_adapter.py'),
                 '--snapshot', str(snapshot)] + old['stages'][0]['command'][4:] + ['--overwrite']
    val_cmd = [str(PYTHON), '-B', str(STUDY / 'analyze_depth.py'), 'validation',
               '--checkpoint', str(run_dir / 'training/model.pt'),
               '--output-dir', str(STUDY / 'validation' / args.run_id)]
    eval_cmd = [str(PYTHON), '-B', '-u', str(STUDY / 'adapter.py'), 'evaluate',
                '--run-id', args.run_id + '_eval', '--run-dir', 'runs/' + args.run_id + '_eval',
                '--dataset-id', protocol['dataset_id'], '--checkpoint-id', args.run_id + '_model',
                '--device', 'cpu']
    record = {k: copy.deepcopy(old[k]) for k in
              ('run_id', 'architecture', 'training_seed', 'split_seed',
               'protocol_sha256', 'parameter_count', 'stages', 'new_test_metrics_read')}
    record.update(status='running', parent_pid=os.getpid(), script_sha256=hashes,
                  started_at_utc=dt.datetime.fromtimestamp(process_epoch, dt.timezone.utc).isoformat(),
                  prior_command_seconds=prior_seconds,
                  resume_snapshot=str(snapshot), resume_snapshot_sha256=digest(snapshot),
                  prior_execution=str(old_path), prior_execution_sha256=digest(old_path),
                  execution_segments=[{'started_at_utc': original_start,
                                       'finished_at_utc': old['finished_at_utc'],
                                       'command_seconds': prior_seconds}],
                  timing_definition='Sum of active command segments; excludes quota-pause interval')
    write_json(execution, record)
    try:
        for stage, cmd in [('train', train_cmd), ('validation', val_cmd),
                           ('sealed_test_evaluation', eval_cmd)]:
            if stop_requested or PAUSE.exists() or time.monotonic() >= deadline:
                record['status'] = 'paused_before_' + stage
                break
            log_path = STUDY / 'logs' / f'{args.run_id}_resume1_{stage}.log'
            stage_start = time.monotonic()
            with log_path.open('x') as log:
                child = subprocess.Popen(cmd, cwd=STUDY, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                info = dict(name=stage, segment='resume1', command=cmd, pid=child.pid,
                            log=str(log_path), status='running')
                record['stages'].append(info)
                write_json(execution, record)
                stop_at = None
                while child.poll() is None:
                    exceeded = time.monotonic() >= deadline
                    if (stop_requested or PAUSE.exists() or exceeded) and stop_at is None:
                        info['stop_reason'] = 'budget' if exceeded else 'pause'
                        os.killpg(child.pid, signal.SIGTERM)
                        stop_at = time.monotonic()
                    if stop_at is not None and time.monotonic() - stop_at > 30:
                        os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(0.5)
                info.update(exit_code=child.returncode, wall_seconds=time.monotonic()-stage_start,
                            status='completed' if child.returncode == 0 else 'interrupted')
                write_json(execution, record)
            if child.returncode != 0:
                record['status'] = 'interrupted_' + stage
                break
        else:
            record['status'] = 'completed_test_sealed'
            record['checkpoint_sha256'] = digest(run_dir / 'training/model.pt')
        verify_invariants(protocol)
        if any(digest(STUDY / name) != value for name, value in hashes.items()):
            raise RuntimeError('Experiment code changed while resuming')
        record['protected_sources_data_model_registry_unchanged'] = True
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        raise
    finally:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        segment_seconds = time.monotonic()-start_mono
        record.update(finished_at_utc=now, resumed_command_seconds=segment_seconds,
                      tail_wall_seconds=prior_seconds+segment_seconds)
        record['execution_segments'].append({'started_at_utc': record['started_at_utc'],
                                            'finished_at_utc': now,
                                            'command_seconds': segment_seconds})
        record['charged_full_wall_seconds'] = (protocol['compute_budget']['charged_previous_all_seconds']
                                                + record['tail_wall_seconds'])
        record['under_two_hours'] = record['charged_full_wall_seconds'] < 7200
        write_json(execution, record)
        print(json.dumps({k: record[k] for k in ('run_id', 'status', 'tail_wall_seconds',
                         'charged_full_wall_seconds', 'under_two_hours')}), flush=True)
    return 0 if record['status'] == 'completed_test_sealed' else 75


if __name__ == '__main__':
    raise SystemExit(main())
