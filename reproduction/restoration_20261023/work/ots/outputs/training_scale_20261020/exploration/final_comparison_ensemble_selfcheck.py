#!/usr/bin/env python3
"""Synthetic-only gates/combination checks; no real runs, data, or test reads."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,copy,json,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import torch
import final_comparison as fc

def check():
    entry={'selected_training_run':'/synthetic/component_a','selected_checkpoint_sha256':'a'*64,'baseline_training_run':'/synthetic/old_c','baseline_checkpoint_sha256':'c'*64,'test_dataset_run':'/synthetic/data_run','test_manifest_sha256':'d'*64,'test_parquet_sha256':'e'*64,'test_seed':20261031}
    assert fc.ensemble_entry(entry) is None
    ensemble={'method':'convex_fourier_average','components':[{'training_run':entry['selected_training_run'],'checkpoint_sha256':'a'*64,'weight':.25},{'training_run':'/synthetic/component_b','checkpoint_sha256':'b'*64,'weight':.75}],'parent_all_wall_seconds':5000.,'additional_train_evaluate_wall_seconds':2199.}
    entry['selected_ensemble']=ensemble
    assert fc.ensemble_entry(entry)['total_wall_seconds']==7199.
    mutation_cases=[('selected_ensemble.method','wrong'),('selected_ensemble.components',[]),('selected_ensemble.components.0.weight',0.),('selected_ensemble.components.0.weight',1.),('selected_ensemble.components.0.weight',.3),('selected_ensemble.components.0.weight',True),('selected_ensemble.components.0.weight',float('nan')),('selected_ensemble.components.1.weight',.5),('selected_ensemble.components.1.training_run','/synthetic/component_a'),('selected_ensemble.components.1.checkpoint_sha256','a'*64),('selected_ensemble.components.0.training_run','/synthetic/not_a'),('selected_ensemble.components.0.checkpoint_sha256','f'*64),('selected_ensemble.parent_all_wall_seconds',0.),('selected_ensemble.parent_all_wall_seconds',float('inf')),('selected_ensemble.additional_train_evaluate_wall_seconds',2200.),('selected_ensemble.additional_train_evaluate_wall_seconds',2201.)]
    for path,value in mutation_cases:
        changed=copy.deepcopy(entry);keys=path.split('.');parent=changed
        for key in keys[:-1]:parent=parent[int(key)] if isinstance(parent,list) else parent[key]
        if isinstance(parent,list):parent[int(keys[-1])]=value
        else:parent[keys[-1]]=value
        try:fc.ensemble_entry(changed)
        except ValueError:pass
        else:raise AssertionError('Ensemble gate accepted '+path+'='+str(value))
    for weight in (.25,.5,.75):
        changed=copy.deepcopy(entry);changed['selected_ensemble']['components'][0]['weight']=weight;changed['selected_ensemble']['components'][1]['weight']=1-weight
        assert fc.ensemble_entry(changed)['eligible']
    identity={key:'same' for key in ('dataset_id','train_parquet_sha256','train_pool_count','generation_seed','split_seed','validation_fraction','optimizer_count','internal_validation_count','optimizer_indices_sha256','internal_validation_indices_sha256','layout_sha256','physical_configuration_sha256')}
    assert fc.shared_training_identity([identity,copy.deepcopy(identity)])==identity
    for key in identity:
        changed=copy.deepcopy(identity);changed[key]='different'
        try:fc.shared_training_identity([identity,changed])
        except ValueError:pass
        else:raise AssertionError('Training identity gate accepted '+key)
    a={'run_id':'a','checkpoint_id':'ca','dataset_id':'shared','fourier_modes':4,'input_dim':2,'output_dim':8,'layout':{'species':'e'},'k_grid':torch.arange(4.),'wavelengths_nm':torch.tensor([1.,2.])}
    b=copy.deepcopy(a);b.update(run_id='b',checkpoint_id='cb')
    fc.component_checkpoint_identity([a,b])
    for key in ('dataset_id','fourier_modes','input_dim','output_dim','layout','k_grid','wavelengths_nm','run_id','checkpoint_id'):
        changed=copy.deepcopy(b)
        if key in ('k_grid','wavelengths_nm'):changed[key][0]+=1
        elif key in ('run_id','checkpoint_id'):changed[key]=a[key]
        else:changed[key]='different'
        try:fc.component_checkpoint_identity([a,changed])
        except ValueError:pass
        else:raise AssertionError('Checkpoint identity gate accepted '+key)
    calls=[];original_predict=fc.predict_coefficients
    def fake_predict(model,spectra,checkpoint,chunk,device):
        calls.append(checkpoint['own_normalization_marker']);return checkpoint['prediction']
    fa=torch.tensor([[1.,2.],[3.,4.]]);fb=torch.tensor([[5.,6.],[7.,8.]])
    components=[{'model':object(),'checkpoint':{'own_normalization_marker':'A','prediction':fa},'weight':.25},{'model':object(),'checkpoint':{'own_normalization_marker':'B','prediction':fb},'weight':.75}]
    fc.predict_coefficients=fake_predict
    try:
        single,raw=fc.component_predictions(components[:1],None,32)
        assert single is fa and raw[0] is fa and calls==['A']
        calls.clear();combined,raw=fc.component_predictions(components,None,32)
        assert calls==['A','B'] and torch.equal(combined,.25*fa+.75*fb)
        assert raw[0] is fa and raw[1] is fb
    finally:fc.predict_coefficients=original_predict
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.ensemble_gate_') as temporary:
        path=Path(temporary)/'selection.json'
        selection={'frozen':True,'auxiliary_temperature':{'method':fc.METHOD},'comparisons':{k:copy.deepcopy(entry) for k in fc.KINDS}}
        path.write_text(json.dumps(selection));assert fc.frozen_selection(path)[0]['frozen']
        selection['frozen']=False;path.write_text(json.dumps(selection))
        try:fc.frozen_selection(path)
        except ValueError:pass
        else:raise AssertionError('Unfrozen ensemble accepted')
    result={'self_check':'passed','synthetic_only':True,'real_test_read':False,'training_performed':False,'schema_rejection_cases':len(mutation_cases),'training_identity_rejection_cases':len(identity),'checkpoint_identity_rejection_cases':9,'valid_weight_pairs':[[.25,.75],[.5,.5],[.75,.25]],'single_tensor_identity_preserved':True,'each_component_own_checkpoint_used':True,'strict_budget_boundary_seconds':7200}
    output=Path(__file__).with_suffix('.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);check()
