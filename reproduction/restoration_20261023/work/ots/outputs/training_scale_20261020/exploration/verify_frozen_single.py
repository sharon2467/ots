#!/usr/bin/env python3
"""After freeze only, cross-check independent best-single and real CLI evaluation.

Never runs inference, training or canonical evaluation. Uses existing artifacts.
Exit codes: 0 passed, 2 incomplete/not frozen, 3 mismatch, 4 verifier error.
Example: --selection-file frozen_selection.json --selection-sha256 HEX
 --kind fixed_ion --final-dir FINAL/fixed_ion --canonical-run REAL_CLI_RUN
 --output NEW_verification.json
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'OTS_code_share/NN_hotE_large_lmRange'
BASE=Path(__file__).resolve().parent.parent


class Incomplete(Exception):pass
class Mismatch(Exception):pass


def require(value,message):
    if not value:raise Mismatch(message)


def exists(path):
    if not path.is_file():raise Incomplete('Required artifact missing: '+str(path))
    return path


def read(path):
    return json.loads(exists(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with exists(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def same_array(a,b,label,report,*,exact=False,rtol=3e-5,atol=2e-6):
    require(a.shape==b.shape,label+': shape differs')
    require(np.isfinite(a).all() and np.isfinite(b).all(),label+': nonfinite values')
    error=np.abs(a.astype('float64')-b.astype('float64'))
    report[label]={'max_abs_difference':float(error.max(initial=0)),
                   'relative_l2_difference':float(np.linalg.norm(error)/max(np.linalg.norm(a.astype('float64')),1e-30)),
                   'exact':exact,'rtol':0 if exact else rtol,'atol':0 if exact else atol}
    require(np.array_equal(a,b) if exact else np.allclose(a,b,rtol=rtol,atol=atol),label+': numerical values differ beyond tolerance')


def scalar(a,b,label,report):
    # SI VDF MAE/RMSE must not inherit dimensionless absolute tolerances.
    atol=1e-13 if label.endswith(('evdf_mae','evdf_rmse')) else 2e-7
    require(np.isfinite(a) and np.isfinite(b),label+': nonfinite metric')
    report[label]={'independent':float(a),'canonical':float(b),'absolute_difference':float(abs(a-b)),
                   'rtol':5e-5,'atol':atol}
    require(np.isclose(a,b,rtol=5e-5,atol=atol),label+': metric differs beyond tolerance')


def source_and_registry(args,report):
    hashes=read(args.source_hashes)
    require(len(hashes)==21,'Protected source baseline must contain 21 files')
    checked=[]
    for name,expected in hashes.items():
        path=ROOT/name
        require(path.resolve().is_relative_to(SOURCE.resolve()),'Protected path escapes source tree')
        require(digest(path)==expected,'Protected source changed: '+name)
        checked.append(name)
    report['protected_source']={'count':len(checked),'baseline_sha256':digest(args.source_hashes),'all_match':True}
    registry=SOURCE/'registry';runs=(SOURCE/'runs').resolve();links=[]
    for group in ('runs','datasets','checkpoints'):
        folder=registry/group
        if not folder.is_dir():raise Incomplete('Registry directory missing: '+str(folder))
        for path in sorted(folder.iterdir()):
            require(path.is_symlink(),'Registry entry is not a symlink: '+str(path))
            require(path.exists(),'Broken registry link: '+str(path))
            target=path.resolve()
            require(target.is_relative_to(runs),'Registry link escapes runs: '+str(path))
            links.append({'registry':str(path),'target':str(target)})
    expected={}
    # All scale_* runs are from this study; evaluate runs contain no history.
    for run in sorted(runs.glob('scale_*')):
        if not run.is_dir():continue
        expected[registry/'runs'/run.name]=run
        history_path=run/'training/training_history.json'
        if history_path.is_file():
            history=read(history_path)
            require(history['run_id']==run.name,'Training run directory/identity differs')
            expected[registry/'checkpoints'/history['checkpoint_id']]=run/'training'
        manifest_path=run/'data/dataset_manifest.json'
        if manifest_path.is_file():
            manifest=read(manifest_path)
            require(manifest['run_id']==run.name,'Dataset run directory/identity differs')
            expected[registry/'datasets'/manifest['dataset_id']]=run/'data'
    for link,target in expected.items():
        if not link.exists():raise Incomplete('This-round registry link missing: '+str(link))
        require(link.is_symlink() and link.resolve()==target.resolve(),'This-round registry link points elsewhere: '+str(link))
    require(bool(expected),'No scale_* registry expectations discovered')
    report['registry']={'all_existing_links_checked':len(links),'this_round_expected_links':len(expected),
                        'scope':'All existing registry links plus required scale_* run/dataset/checkpoint links',
                        'links':links,'passed':True}


def verify(args,report):
    # Strictly before reading ANY test manifest, test hash, metrics or predictions.
    selection=read(args.selection_file)
    if selection.get('frozen') is not True:raise Incomplete('Explicit frozen=true required before test-artifact access')
    selection_hash=digest(args.selection_file)
    require(selection_hash==args.selection_sha256,'Frozen-selection SHA256 differs from caller-approved hash')
    entry=selection.get('comparisons',{}).get(args.kind)
    if not isinstance(entry,dict):raise Incomplete('Frozen kind entry missing')
    require('selected_ensemble' not in entry,'Best-single verifier cannot compare an ensemble as a canonical single model')
    report['frozen_selection']={'path':str(args.selection_file.resolve()),'sha256':selection_hash,'frozen':True}
    source_and_registry(args,report)
    training=Path(entry['selected_training_run']).resolve()
    history_path=training/'training/training_history.json';history=read(history_path)
    checkpoint=(history_path.parent/history['checkpoint']).resolve()
    require(digest(checkpoint)==entry['selected_checkpoint_sha256'],'Best-single checkpoint differs from frozen hash')
    manifest_path=Path(entry['test_dataset_run'])/'data/dataset_manifest.json'
    require(digest(manifest_path)==entry['test_manifest_sha256'],'Test manifest differs from frozen hash')
    manifest=read(manifest_path)
    require(manifest['physical_configuration']['dataset']==args.kind,'Frozen manifest kind differs')
    require(manifest['splits']['test']['seed']==entry['test_seed'],'Test seed differs from freeze')
    test_path=(manifest_path.parent/manifest['splits']['test']['file']).resolve()
    # Hash bytes only; no direct loading of raw test arrays or model inference.
    require(digest(test_path)==entry['test_parquet_sha256'],'Test Parquet differs from frozen hash')
    final_path=args.final_dir/'final_comparison.json';final=read(final_path)
    require(final['selection_file_sha256']==selection_hash,'Independent final output uses another frozen selection')
    require(final['dataset_kind']==args.kind and final['test_dataset_id']==manifest['dataset_id'],'Final dataset identity differs')
    for key in ('test_manifest_sha256','test_parquet_sha256','test_seed'):
        require(final[key]==entry[key],'Independent final provenance differs: '+key)
    selected=final['models']['selected']
    require(selected.get('type')!='convex_ensemble','Independent final selected is ensemble')
    for key,value in (('run_id',history['run_id']),('checkpoint_id',history['checkpoint_id']),
                      ('checkpoint_sha256',entry['selected_checkpoint_sha256']),('training_dataset_id',history['dataset_id'])):
        require(selected[key]==value,'Independent selected identity differs: '+key)
    require(Path(selected['checkpoint']).resolve()==checkpoint,'Independent checkpoint path differs')
    require(selected['history_sha256']==digest(history_path),'Independent training history hash differs')
    evaluation=args.canonical_run.resolve()/'evaluation'
    canonical_path=evaluation/'metrics.json';canonical=read(canonical_path)
    require(canonical['run_id']==args.canonical_run.name,'Canonical run identity differs')
    for key,value in (('checkpoint_id',history['checkpoint_id']),('checkpoint_source_run_id',history['run_id']),
                      ('checkpoint_training_dataset_id',history['dataset_id']),('dataset_id',manifest['dataset_id']),
                      ('dataset_source_run_id',manifest['run_id'])):
        require(canonical[key]==value,'Canonical provenance differs: '+key)
    require((evaluation/canonical['checkpoint']).resolve()==checkpoint,'Canonical checkpoint path differs')
    require((evaluation/canonical['test_data']).resolve()==test_path,'Canonical test path differs')
    report['provenance']={'checkpoint_sha256':digest(checkpoint),'final_json_sha256':digest(final_path),
                          'canonical_metrics_sha256':digest(canonical_path),'test_parquet_sha256':entry['test_parquet_sha256'],
                          'canonical_run':str(args.canonical_run.resolve()),'no_evaluation_fabricated':True}
    names=final['species_order']
    expected=[s for s,k in (('e','electron_dis'),('i1','ion1_dis'),('i2','ion2_dis'))
              if manifest['physical_configuration'][k]['used_in_model']]
    require(names==expected,'Independent species order differs from manifest')
    n=manifest['splits']['test']['physical_samples']
    require(n==final['test_sample_count']==1024,'Final sample count differs from protocol')
    report['species']={}
    for s in names:
        result=report['species'][s]={'arrays':{},'metrics':{}}
        independent_path=args.final_dir/(s+'_comparison_predictions.npz')
        canonical_predictions=evaluation/('predictions.npz' if names==['e'] else s+'/predictions.npz')
        with np.load(exists(independent_path),allow_pickle=False) as independent, np.load(exists(canonical_predictions),allow_pickle=False) as real:
            require(independent['selected_coefficients'].shape[0]==n,'Independent prediction count differs')
            for key in ('target_coefficients','k_grid'):
                same_array(independent[key],real[key],key,result['arrays'],exact=True)
            same_array(independent['selected_coefficients'],real['predicted_coefficients'],'predicted_coefficients',result['arrays'])
            records=final['species'][s]['models']['selected']['fourier_target']['per_sample']
            require([r['test_sample_index_zero_based'] for r in records]==list(range(n)),'Final per-sample order differs')
            error=real['predicted_evdf'].astype('float64')-real['target_evdf'].astype('float64')
            relative=np.sqrt((error**2).sum(1)/np.maximum((real['target_evdf'].astype('float64')**2).sum(1),1e-30))
            same_array(np.array([r['evdf_relative_l2'] for r in records]),relative,'per_sample_vdf_relative_l2',result['arrays'],rtol=5e-5,atol=2e-7)
            same_array(independent['selected_fourier_target_relative_l2'],relative,'saved_vdf_relative_l2',result['arrays'],rtol=5e-5,atol=2e-7)
        real_metrics=canonical['metrics'] if names==['e'] else canonical['metrics'][s]
        independent_metrics=final['species'][s]['models']['selected']['fourier_target']['metrics']
        require(all(k in independent_metrics for k in real_metrics),'Independent metric keys missing')
        for key,value in real_metrics.items():scalar(independent_metrics[key],value,key,result['metrics'])
        result['artifact_hashes']={'independent_predictions':digest(independent_path),'canonical_predictions':digest(canonical_predictions)}
        result['passed']=True
    require(digest(args.selection_file)==selection_hash,'Frozen selection changed during verification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-file',type=Path,required=True)
    p.add_argument('--selection-sha256',required=True)
    p.add_argument('--kind',choices=('fixed_ion','fig1_fig11'),required=True)
    p.add_argument('--final-dir',type=Path,required=True)
    p.add_argument('--canonical-run',type=Path,required=True)
    p.add_argument('--source-hashes',type=Path,default=BASE/'source_hashes_before.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite '+str(args.output))
    report={'status':'incomplete','kind':args.kind,'training_started':False,'inference_run':False,
            'canonical_evaluation_run':False,'verifier_sha256':digest(Path(__file__))}
    try:
        verify(args,report);report['status']='passed';code=0
    except Incomplete as error:report['status']='incomplete';report['reason']=str(error);code=2
    except Mismatch as error:report['status']='mismatch';report['reason']=str(error);code=3
    except Exception as error:report['status']='verifier_error';report['reason']=type(error).__name__+': '+str(error);code=4
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'output':str(args.output),'reason':report.get('reason')}))
    sys.exit(code)
