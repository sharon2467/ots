#!/usr/bin/env python3
"""Frozen two-model convex probe on the previous study's 750 development rows.

No training or independent test reads. Example (after both common audits exist):
  .venv/bin/python outputs/training_scale_20261020/exploration/ensemble_development.py \
    RUN_A RUN_B --parent-all-wall-seconds 4500 \
    --additional-train-evaluate-wall-seconds 1200 --output NEW.json
Budget JSON alternatively requires parent_all_wall_seconds and
additional_train_evaluate_wall_seconds (actual completed stage walls, not estimates).
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
import common_validation_audit as common

ALPHAS=(0.,.25,.5,.75,1.)
BUDGET_LIMIT=7200.


def require(condition,message):
    if not condition:
        raise ValueError(message)


def mix(a,b,alpha):
    require(alpha in ALPHAS,'Alpha must belong to the fixed five-point grid')
    require(a.shape==b.shape,'Coefficient shapes differ')
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Nonfinite coefficients')
    # Preserve endpoint predictions exactly, including float32 rounding.
    return a if alpha==1. else b if alpha==0. else alpha*a+(1.-alpha)*b


def budget(parent,additional):
    require(all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in (parent,additional)),
            'Both completed-stage actual wall seconds must be finite and positive')
    total=float(parent+additional)
    return {'parent_all_wall_seconds':float(parent),'additional_train_evaluate_wall_seconds':float(additional),
            'total_wall_seconds':total,'limit_seconds':BUDGET_LIMIT,'eligible':total<BUDGET_LIMIT,
            'accounting':'Parent complete generate/train/evaluate wall plus additional model train/evaluate wall; includes original single-model training',
            'reason':'Within strict two-hour budget' if total<BUDGET_LIMIT else 'Total is >=7200 seconds; ineligible'}


def compare_tree(actual,expected,label):
    """Compare all saved metrics and every per-sample error, not just score."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),label+': keys differ')
        for key in expected:compare_tree(actual[key],expected[key],label+'.'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': lengths differ')
        for i,(a,b) in enumerate(zip(actual,expected)):compare_tree(a,b,label+f'[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if isinstance(expected,int):require(actual==expected,label+': index differs')
        else:require(bool(np.isclose(actual,expected,rtol=2.e-5,atol=2.e-10)),label+': endpoint metric differs')
    else:require(actual==expected,label+': value differs')


def common_identity(a,b):
    for key in ('dataset_id','dataset_kind'):
        require(a[key]==b[key],'Two audits differ in '+key)
    require(a['training_config']['seed']==b['training_config']['seed'],'Internal split seeds differ')
    require(a['training_config']['validation_fraction']==b['training_config']['validation_fraction'],'Internal validation fractions differ')
    for key in ('common_reference_physical_indices_zero_based','common_indices_sha256','optimizer_indices_sha256','internal_validation_indices_sha256'):
        require(a['indices'][key]==b['indices'][key],'Audit indices differ: '+key)
    for key in ('reference_run','reference_history_sha256','reference_train_manifest','reference_train_manifest_sha256','reference_train_parquet','reference_train_parquet_sha256','train_manifest_sha256','train_parquet_sha256'):
        require(a['inputs'][key]==b['inputs'][key],'Audit provenance differs: '+key)
    for doc in (a,b):
        require(doc['counts']['common_validation_count']==750,'Common holdout must contain 750 rows')
        require(doc['layout_K_wavelengths_match'] is True,'Audit layout gate absent')
        for key in ('test_data_read','test_metrics_read','test_predictions_read'):
            require(doc[key] is False,'Audit read independent test')
        overlap=doc['overlap_check']
        require(overlap['zero_overlap'] is True and overlap['overlapping_unique_row_count']==0 and overlap['overlapping_common_row_count']==0 and overlap['overlapping_common_rows']==[],
                'Audit optimization/common-development rows overlap')
    require(a['overlap_check']['common_row_sha256']==b['overlap_check']['common_row_sha256'],'Common row sequence differs')


def verified_model(run,audit_path):
    doc=json.loads(audit_path.read_text())
    current=common.context(run)
    history=current['history']
    checkpoint_path=(current['history_path'].parent/history['checkpoint']).resolve()
    inputs=doc['inputs']
    for key,path in (('checkpoint',checkpoint_path),('history',current['history_path']),('train_manifest',current['manifest_path']),('train_parquet',current['train_path'])):
        require(Path(inputs[key]).resolve()==path,key+' path differs from audit')
        require(common.file_hash(path)==inputs[key+'_sha256'],key+' hash differs from audit')
    model,checkpoint=common.load_model_checkpoint(checkpoint_path,'cpu',history['checkpoint_id'])
    for key in ('run_id','dataset_id','checkpoint_id'):
        require(history[key]==checkpoint[key]==doc[key],'Identity differs: '+key)
    require(checkpoint['training_config']==doc['training_config'],'Training config differs from audit')
    require(current['manifest']['physical_configuration']['dataset']==doc['dataset_kind'],'Dataset kind differs')
    expected_reference=common.SOURCE/('runs/study_fixed_c' if doc['dataset_kind']=='fixed_ion' else 'runs/study_fig_c')
    require(doc['dataset_kind'] in ('fixed_ion','fig1_fig11'),'Unsupported kind')
    require(Path(inputs['reference_run']).resolve()==expected_reference.resolve(),'Reference must be the canonical old C run')
    reference=common.context(expected_reference)
    for key,path in (('reference_history',reference['history_path']),('reference_train_manifest',reference['manifest_path']),('reference_train_parquet',reference['train_path'])):
        require(common.file_hash(path)==inputs[key+'_sha256'],'Reference hash differs: '+key)
    common.compatible(checkpoint,current['dataset'])
    common.compatible(checkpoint,reference['dataset'])
    require(current['dataset'].layout==reference['dataset'].layout,'Training/reference physical layouts differ')
    for key in ('k_grid','wavelengths_nm'):
        require(torch.equal(getattr(current['dataset'],key),getattr(reference['dataset'],key)),'Training/reference axes differ')
    _,_,_,optimizer,internal=common.split_indices(current['dataset'],checkpoint['training_config'])
    _,x,y,_,indices=common.split_indices(reference['dataset'],reference['history']['training_config'])
    require(len(indices)==750,'Actual holdout must contain 750 rows')
    require(indices.tolist()==doc['indices']['common_reference_physical_indices_zero_based'],'Actual common indices differ')
    for key,index in (('common_indices_sha256',indices),('optimizer_indices_sha256',optimizer),('internal_validation_indices_sha256',internal)):
        require(common.index_hash(index.numpy())==doc['indices'][key],'Actual index hash differs: '+key)
    hashes=common.row_hashes(reference['dataset'].spectra,reference['dataset'].coefficients,indices)
    opt_hashes=common.row_hashes(current['dataset'].spectra,current['dataset'].coefficients,optimizer)
    require(hashes==doc['overlap_check']['common_row_sha256'],'Actual common row hashes differ')
    require(not set(hashes).intersection(opt_hashes),'Actual optimization/common rows overlap')
    for key,rows in (('common',hashes),('optimizer',opt_hashes)):
        require(common.digest_sequence_hash(rows)==doc['overlap_check'][key+'_ordered_row_digests_sha256'],'Actual row digest differs: '+key)
    return doc,current,checkpoint,model,reference,x,y,indices


def metrics(predicted,target,checkpoint,dataset,indices,chunk):
    layout=checkpoint.get('layout');m=checkpoint['fourier_modes']
    selected=[(s,k) for s,k in common.SPECIES_BLOCKS if layout[k]['used_in_model']] if layout else [('e','electron_dis')]
    species={}
    for j,(s,key) in enumerate(selected):
        block=slice(j*2*m,(j+1)*2*m)
        species[s]=common.summarize(target[:,block],predicted[:,block],dataset.k_grid[j*m:(j+1)*m],
                                    layout[key]['velocity_grid_m_s'] if layout else None,indices,chunk)
    score=float(np.mean([v['metrics']['evdf_relative_l2_mean'] for v in species.values()]))
    return {'species':species,'macro_score':score}


def probe(args):
    require(not args.output.exists(),'Refusing to overwrite '+str(args.output))
    timing=budget(args.parent_all_wall_seconds,args.additional_train_evaluate_wall_seconds)
    paths=[args.audit_a or Path(__file__).parent/(args.run_a.name+'_common_validation_metrics.json'),
           args.audit_b or Path(__file__).parent/(args.run_b.name+'_common_validation_metrics.json')]
    a=verified_model(args.run_a,paths[0]);b=verified_model(args.run_b,paths[1])
    common_identity(a[0],b[0])
    require(a[2]['checkpoint_id']!=b[2]['checkpoint_id'],'Two distinct checkpoints required')
    require(a[2].get('layout')==b[2].get('layout'),'Checkpoint layouts differ')
    require(a[2]['fourier_modes']==b[2]['fourier_modes'],'Fourier modes differ')
    for key in ('k_grid','wavelengths_nm'):
        require(torch.equal(a[2][key],b[2][key]),'Checkpoint axes differ: '+key)
    require(torch.equal(a[5],b[5]) and torch.equal(a[6],b[6]) and torch.equal(a[7],b[7]),'Actual common tensors differ')
    # Each call uses that model's own physical input/output normalization statistics.
    predictions=[common.predict_coefficients(v[3],v[5],v[2],args.chunk_size,'cpu') for v in (a,b)]
    results=[]
    for alpha in ALPHAS:
        row=metrics(mix(*predictions,alpha),a[6],a[2],a[4]['dataset'],a[7],args.chunk_size)
        results.append({'alpha':alpha,**row})
        if alpha in (0.,1.):
            original=a[0] if alpha==1. else b[0]
            for s,value in row['species'].items():compare_tree(value,original['species'][s]['model'],'endpoint '+str(alpha)+' '+s)
            compare_tree(row['macro_score'],original['selection_score'],'endpoint macro')
    winner=min(results,key=lambda r:r['macro_score'])
    payload={'schema_version':1,'diagnostic':'Adaptive common old 750 development ensemble probe; not independent test',
             'formula':'alpha * model_A + (1-alpha) * model_B in physical Fourier coefficients',
             'alpha_grid':list(ALPHAS),'shared_alpha_all_species':True,'selection_rule':'Equal-species macro mean per-sample VDF relative-L2',
             'test_data_read':False,'test_metrics_read':False,'test_predictions_read':False,'cpu_threads':torch.get_num_threads(),
             'chunk_size':args.chunk_size,'endpoint_checks_passed':True,'endpoint_tolerance':{'rtol':2e-5,'atol':2e-10},
             'budget':timing,'eligible':timing['eligible'],'selected_alpha':winner['alpha'],'selected_macro_score':winner['macro_score'],
             'provenance':{'probe_sha256':common.file_hash(Path(__file__)),'common_audit_implementation_sha256':common.file_hash(Path(common.__file__)),
                           'metric_implementation_sha256':common.file_hash(common.OLD_AUDIT),
                           'audits':[{'path':str(p.resolve()),'sha256':common.file_hash(p),'inputs':v[0]['inputs'],'run_id':v[0]['run_id'],
                                      'dataset_id':v[0]['dataset_id'],'checkpoint_id':v[0]['checkpoint_id']} for p,v in zip(paths,(a,b))],
                           'indices':a[0]['indices'],'actual_optimizer_zero_overlap_verified_for_both':True,
                           'common_ordered_row_digests_sha256':a[0]['overlap_check']['common_ordered_row_digests_sha256']},
             'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(args.output),'selected_alpha':winner['alpha'],'macro_score':winner['macro_score'],'eligible':timing['eligible']}))


def self_check():
    a=torch.tensor([[1.,2.],[3.,4.]])
    b=torch.tensor([[5.,6.],[7.,8.]])
    require(torch.equal(mix(a,b,1.),a) and torch.equal(mix(a,b,0.),b),'Endpoint combination failed')
    require(torch.equal(mix(a,b,.5),(a+b)/2),'Interior combination failed')
    require(budget(5000,2199)['eligible'] and not budget(5000,2200)['eligible'] and not budget(5000,2201)['eligible'],'Budget boundary failed')
    rejected=0
    for action in (lambda:mix(a,b,.1),lambda:budget(float('nan'),1),lambda:budget(-1,2),
                   lambda:compare_tree({'p95':.2},{'p95':.3},'synthetic endpoint')):
        try:action()
        except ValueError:rejected+=1
    require(rejected==4,'Synthetic rejection gate failed')
    print(json.dumps({'self_check':'passed','alpha_grid':list(ALPHAS),'synthetic_only':True,'training_started':False,'test_data_read':False,'budget_boundary_seconds':7200}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_a',nargs='?',type=Path);p.add_argument('run_b',nargs='?',type=Path)
    p.add_argument('--audit-a',type=Path);p.add_argument('--audit-b',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--parent-all-wall-seconds',type=float)
    p.add_argument('--additional-train-evaluate-wall-seconds',type=float)
    p.add_argument('--budget-json',type=Path)
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.self_check:self_check()
    else:
        if not args.run_a or not args.run_b or not args.output:p.error('Two runs and --output required')
        if args.chunk_size<1:p.error('--chunk-size must be positive')
        if args.budget_json:
            if args.parent_all_wall_seconds is not None or args.additional_train_evaluate_wall_seconds is not None:p.error('Choose budget JSON or explicit actual wall seconds')
            doc=json.loads(args.budget_json.read_text())
            args.parent_all_wall_seconds=doc['parent_all_wall_seconds']
            args.additional_train_evaluate_wall_seconds=doc['additional_train_evaluate_wall_seconds']
        probe(args)
