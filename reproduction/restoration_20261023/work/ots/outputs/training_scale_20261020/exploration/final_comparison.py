#!/usr/bin/env python3
"""Independent old/new comparison on a hash-bound, frozen common new test set."""
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import sys,json,argparse,hashlib,warnings,math
sys.dont_write_bytecode=True
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'OTS_code_share/NN_hotE_large_lmRange'
OLD_EXPLORATION=ROOT/'outputs/training_study_20261020/exploration'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(OLD_EXPLORATION))
from data import load_dataset,SPECIES_BLOCKS
from model import load_model_checkpoint
from evaluation import predict_coefficients,reconstruct_real_evdf
from final_diagnostics import moments,clean,sha256
from moment_probe import taylor,error_stats
from positivity_probe import replay_config
from phyEqns import ots_full_fourier as phy
from common_validation_audit import compatible,summarize
from ensemble_development import mix as convex_mix,budget as ensemble_budget

METHOD='taylor_even4_odd3_n3'
KINDS=('fixed_ion','fig1_fig11')
ROLES=('selected','baseline')
REPETITIONS=2000
BOOTSTRAP_SEED=20261020

def ensemble_entry(entry):
    """Validate frozen scalar weights/provenance/budget without reading data."""
    if 'selected_ensemble' not in entry:return None
    ensemble=entry['selected_ensemble']
    if not isinstance(ensemble,dict) or ensemble.get('method')!='convex_fourier_average':raise ValueError('Unsupported selected ensemble method')
    components=ensemble.get('components')
    if not isinstance(components,list) or len(components)!=2:raise ValueError('Exactly two ensemble components are required')
    for component in components:
        if not isinstance(component,dict):raise ValueError('Ensemble components must be objects')
        run=component.get('training_run');value=component.get('checkpoint_sha256');weight=component.get('weight')
        if not isinstance(run,str) or not Path(run).is_absolute():raise ValueError('Component training_run must be absolute')
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid component checkpoint SHA256')
        if not isinstance(weight,(int,float)) or isinstance(weight,bool) or not math.isfinite(weight) or weight not in (.25,.5,.75):raise ValueError('Ensemble weights must be fixed-grid interior values .25/.5/.75')
    if not math.isclose(sum(c['weight'] for c in components),1.,rel_tol=0.,abs_tol=1e-12):raise ValueError('Ensemble weights must sum to one')
    if Path(components[0]['training_run']).resolve()!=Path(entry['selected_training_run']).resolve() or components[0]['checkpoint_sha256']!=entry['selected_checkpoint_sha256']:raise ValueError('Ensemble component A must equal the original selected run/hash')
    if Path(components[0]['training_run']).resolve()==Path(components[1]['training_run']).resolve() or components[0]['checkpoint_sha256']==components[1]['checkpoint_sha256']:raise ValueError('Two distinct component runs and checkpoint hashes are required')
    timing=ensemble_budget(ensemble.get('parent_all_wall_seconds'),ensemble.get('additional_train_evaluate_wall_seconds'))
    if not timing['eligible']:raise ValueError('Ensemble completed-stage total must be strictly below 7200 seconds')
    return timing

def frozen_selection(path):
    raw=path.read_bytes();selection=json.loads(raw)
    if selection.get('frozen') is not True:raise ValueError('Explicit frozen=true is required before any test access')
    if selection.get('auxiliary_temperature',{}).get('method')!=METHOD:raise ValueError('Auxiliary temperature must be frozen to '+METHOD)
    for kind in KINDS:
        entry=selection.get('comparisons',{}).get(kind)
        if not isinstance(entry,dict):raise ValueError('Missing comparisons.'+kind)
        for field in ('selected_training_run','baseline_training_run','test_dataset_run'):
            if not isinstance(entry.get(field),str) or not Path(entry[field]).is_absolute():raise ValueError(field+' must be an absolute run path')
        for field in ('selected_checkpoint_sha256','baseline_checkpoint_sha256','test_manifest_sha256','test_parquet_sha256'):
            value=entry.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid SHA256: '+field)
        if type(entry.get('test_seed')) is not int or entry['test_seed']!=20261031:raise ValueError('The final common test seed must be 20261031')
        ensemble_entry(entry)
    return selection,hashlib.sha256(raw).hexdigest()

def checked_model(entry,role):
    run=Path(entry[role+'_training_run']).resolve();history_path=run/'training/training_history.json'
    history=json.loads(history_path.read_text())
    checkpoint_path=(history_path.parent/history['checkpoint']).resolve()
    actual_hash=sha256(checkpoint_path)
    if actual_hash!=entry[role+'_checkpoint_sha256']:raise ValueError(role+' checkpoint differs from frozen hash')
    model,checkpoint=load_model_checkpoint(checkpoint_path,'cpu',history['checkpoint_id'])
    if checkpoint['run_id']!=history['run_id'] or checkpoint['dataset_id']!=history['dataset_id']:raise ValueError('Checkpoint training identity differs from its own history')
    provenance={'training_run':str(run),'run_id':history['run_id'],'training_dataset_id':history['dataset_id'],'checkpoint_id':history['checkpoint_id'],'checkpoint':str(checkpoint_path),'checkpoint_sha256':actual_hash,'history_sha256':sha256(history_path),'training_config':checkpoint['training_config']}
    return model,checkpoint,provenance

def component_checkpoint_identity(checkpoints):
    a,b=checkpoints
    if a['run_id']==b['run_id'] or a['checkpoint_id']==b['checkpoint_id']:raise ValueError('Ensemble component identities must be distinct')
    for key in ('dataset_id','fourier_modes','input_dim','output_dim','layout'):
        if a.get(key)!=b.get(key):raise ValueError('Ensemble checkpoints differ in '+key)
    for key in ('k_grid','wavelengths_nm'):
        if not torch.equal(a[key],b[key]):raise ValueError('Ensemble checkpoint coordinates differ: '+key)

def component_training_identity(run,checkpoint,hash_cache):
    """Check the real shared train artifact and deterministic optimizer split."""
    history_path=Path(run)/'training/training_history.json';history=json.loads(history_path.read_text())
    if checkpoint['training_config']!=history['training_config']:raise ValueError('Component checkpoint/history training config differs')
    manifest_path=(history_path.parent/history['dataset_manifest']).resolve();manifest=json.loads(manifest_path.read_text())
    train_path=(manifest_path.parent/manifest['splits']['train']['file']).resolve()
    if train_path!=(history_path.parent/history['train_data']).resolve() or manifest['dataset_id']!=checkpoint['dataset_id']:raise ValueError('Component training dataset identity/path differs')
    if train_path not in hash_cache:hash_cache[train_path]=sha256(train_path)
    n=manifest['splits']['train']['physical_samples'];config=checkpoint['training_config']
    if type(n) is not int or n<2:raise ValueError('Invalid component training pool size')
    nval=min(n-1,max(1,int(round(n*config['validation_fraction']))))
    order=torch.randperm(n,generator=torch.Generator().manual_seed(config['seed']))
    ih=lambda x:hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()
    jh=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    return {'dataset_id':checkpoint['dataset_id'],'train_parquet':str(train_path),'train_parquet_sha256':hash_cache[train_path],'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path),'train_pool_count':n,'generation_seed':manifest['splits']['train']['seed'],'split_seed':config['seed'],'validation_fraction':config['validation_fraction'],'optimizer_count':n-nval,'internal_validation_count':nval,'optimizer_indices_sha256':ih(order[nval:].numpy()),'internal_validation_indices_sha256':ih(order[:nval].numpy()),'layout_sha256':jh(checkpoint.get('layout')),'physical_configuration_sha256':jh(manifest['physical_configuration'])}

def shared_training_identity(identities):
    a,b=identities
    keys=('dataset_id','train_parquet_sha256','train_pool_count','generation_seed','split_seed','validation_fraction','optimizer_count','internal_validation_count','optimizer_indices_sha256','internal_validation_indices_sha256','layout_sha256','physical_configuration_sha256')
    for key in keys:
        if a[key]!=b[key]:raise ValueError('Ensemble components do not share training dataset/split: '+key)
    return {key:a[key] for key in keys}

def inference_models(entry):
    """Single path stays unchanged; ensemble has no fabricated checkpoint."""
    timing=ensemble_entry(entry);inference={};provenance={}
    for role in ROLES:
        if role!='selected' or timing is None:
            model,checkpoint,record=checked_model(entry,role)
            inference[role]=[{'model':model,'checkpoint':checkpoint,'weight':1.}];provenance[role]=record
            continue
        inference[role]=[];records=[];identities=[];hash_cache={}
        for component in entry['selected_ensemble']['components']:
            metadata={'component_training_run':component['training_run'],'component_checkpoint_sha256':component['checkpoint_sha256']}
            model,checkpoint,record=checked_model(metadata,'component')
            inference[role].append({'model':model,'checkpoint':checkpoint,'weight':component['weight']})
            records.append({**record,'weight':component['weight']})
            identities.append(component_training_identity(record['training_run'],checkpoint,hash_cache))
        component_checkpoint_identity([c['checkpoint'] for c in inference[role]])
        shared=shared_training_identity(identities)
        provenance[role]={'type':'convex_ensemble','method':'convex_fourier_average','formula':'weight_A * physical_Fourier_A + weight_B * physical_Fourier_B; each component uses its own saved normalization and canonical Hermitian projection','components':records,'component_training_artifacts':identities,'shared_training_data_and_split':shared,'budget':timing}
    return inference,provenance

def component_predictions(components,spectra,chunk_size):
    predictions=[predict_coefficients(c['model'],spectra,c['checkpoint'],chunk_size,'cpu') for c in components]
    if len(predictions)==1:return predictions[0],predictions
    return convex_mix(predictions[0],predictions[1],components[0]['weight']),predictions

def vector_summary(a):
    a=np.asarray(a);finite=np.isfinite(a);valid=a[finite]
    return {'finite_count':int(finite.sum()),'nonfinite_count':int((~finite).sum()),'mean':float(valid.mean()) if len(valid) else None,'median':float(np.median(valid)) if len(valid) else None,'p95':float(np.quantile(valid,.95)) if len(valid) else None,'max':float(valid.max()) if len(valid) else None}

def paired_comparison(selected,baseline):
    """S,N errors; same physical-state indices across roles and species."""
    selected=np.asarray(selected);baseline=np.asarray(baseline)
    if selected.ndim!=2 or selected.shape!=baseline.shape:raise ValueError('Paired metric shapes differ')
    valid=np.isfinite(selected)&np.isfinite(baseline)
    a=np.where(valid,selected,np.nan);b=np.where(valid,baseline,np.nan)
    n=selected.shape[1];rng=np.random.default_rng(BOOTSTRAP_SEED)
    adraw=np.empty((REPETITIONS,len(selected)));bdraw=np.empty_like(adraw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for start in range(0,REPETITIONS,100):
            stop=min(start+100,REPETITIONS);indices=rng.integers(0,n,size=(stop-start,n))
            adraw[start:stop]=np.nanmean(a[:,indices],axis=2).T
            bdraw[start:stop]=np.nanmean(b[:,indices],axis=2).T
        def metrics(av,bv):
            ratio=av/np.maximum(bv,1e-30)
            return {'selected_mean':av,'baseline_mean':bv,'baseline_minus_selected_absolute':bv-av,'selected_over_baseline_ratio':ratio,'relative_error_reduction':1-ratio}
        point=metrics(np.nanmean(a,axis=1),np.nanmean(b,axis=1));draw=metrics(adraw,bdraw)
        species=[]
        for j in range(len(selected)):
            species.append({'common_finite_count':int(valid[j].sum()),'excluded_pair_count':int((~valid[j]).sum()),'point':{k:float(v[j]) for k,v in point.items()},'ci95':{k:np.nanquantile(v[:,j],[.025,.975]).tolist() for k,v in draw.items()}})
        macro_point=metrics(np.nanmean(a,axis=1).mean(),np.nanmean(b,axis=1).mean())
        macro_draw=metrics(adraw.mean(1),bdraw.mean(1))
        macro={'point':{k:float(v) for k,v in macro_point.items()},'ci95':{k:np.nanquantile(v,[.025,.975]).tolist() for k,v in macro_draw.items()}}
    return {'repetitions':REPETITIONS,'seed':BOOTSTRAP_SEED,'method':'Paired percentile bootstrap; identical sampled physical indices across both roles and every species; common finite pairs, retaining finite negative-temperature failures in absolute-error vectors','interpretation':'Conditional sampling uncertainty for the frozen estimators; excludes retraining/selection uncertainty. Macro means weight species equally. Ratios compare means, not mean per-sample ratios.','species':species,'macro':macro}

def decode(coefficients,K,velocity,chunk_size,dtype):
    pieces=[]
    for start in range(0,len(coefficients),chunk_size):
        v,f=reconstruct_real_evdf(coefficients[start:start+chunk_size].to(dtype),K.to(dtype),velocity=velocity)
        pieces.append(f.numpy())
    return v.numpy(),np.concatenate(pieces)

def original_truth(manifest,dataset,names):
    physical=manifest['physical_configuration'];n=len(dataset.spectra)
    if manifest['splits']['test']['seed']!=manifest['splits']['train']['seed']+1:raise ValueError('Recorded test seed must equal train seed+1')
    config,factory=replay_config(physical);sampler=factory(np.random.default_rng(manifest['splits']['test']['seed']),config)
    original={s:np.empty((n,len(physical[key]['velocity_grid_m_s']))) for s,key in names}
    checks=[];chosen=set(np.linspace(0,n-1,min(12,n),dtype=int).tolist());m=physical['fourier_modes']
    for row in range(n):
        state=sampler.sample()
        for j,(s,key) in enumerate(names):
            v,f=state['distributions'][s];original[s][row]=f.numpy()
            if row in chosen:
                K,F=phy.ft_x_n(v,f,m);label=torch.cat((F.real,F.imag)).float()
                target=dataset.coefficients[row,j*2*m:(j+1)*2*m]
                if not torch.equal(K.float(),dataset.k_grid[j*m:(j+1)*m]) or not torch.equal(label,target):raise ValueError('Original sampler replay differs from target labels/K')
                checks.append({'species':s,'test_index_zero_based':row,'max_abs_float32_coefficient_error':float((label-target).abs().max())})
    return original,checks

def canonical_summary(target,predicted,K,velocity,chunk_size):
    result=summarize(target,predicted,K,velocity,torch.arange(len(target)),chunk_size)
    for row in result['per_sample']:
        row['test_sample_index_zero_based']=row.pop('train_parquet_physical_index_zero_based')
    return result

def plots(outdir,species,velocity,target,curves,original,relative,selected_label='Selected model'):
    fig,axes=plt.subplots(2,2,figsize=(12,7),squeeze=False)
    representatives={}
    for row,role in enumerate(ROLES):
        order=np.argsort(relative[role]);chosen=[int(order[len(order)//2]),int(order[-1])]
        representatives[role]={'median':chosen[0],'worst':chosen[1]}
        for ax,index,label in zip(axes[row],chosen,('median','worst')):
            ax.plot(velocity,original[index],color='.65',ls=':',label='Original sampled VDF')
            ax.plot(velocity,target[index],'k-',label='Fourier target')
            ax.plot(velocity,curves['baseline'][index],label='Old C baseline')
            ax.plot(velocity,curves['selected'][index],label=selected_label)
            ax.set(xlabel='Velocity (m/s)',ylabel='Signed VDF',title=f'{species}: {role} {label}, sample {index}; L2={relative[role][index]:.3g}')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(outdir/(species+'_representatives.png'),dpi=150);plt.close(fig)
    return representatives

def compare(kind,entry,selection_path,selection_hash,output_root,chunk_size):
    current_selection,current_hash=frozen_selection(selection_path)
    if current_hash!=selection_hash or current_selection['comparisons'][kind]!=entry:raise ValueError('Frozen selection changed or caller supplied an unfrozen comparison')
    outdir=output_root/kind
    if outdir.exists():raise FileExistsError('Refusing to overwrite final comparison '+str(outdir))
    inference,provenance=inference_models(entry)
    is_ensemble=provenance['selected'].get('type')=='convex_ensemble'
    selected_label='Selected ensemble' if is_ensemble else 'Selected model'
    test_run=Path(entry['test_dataset_run']).resolve();manifest_path=test_run/'data/dataset_manifest.json'
    if sha256(manifest_path)!=entry['test_manifest_sha256']:raise ValueError('Test manifest differs from frozen hash')
    manifest=json.loads(manifest_path.read_text());physical=manifest['physical_configuration']
    if physical['dataset']!=kind or manifest['splits']['test']['seed']!=entry['test_seed']:raise ValueError('Frozen test kind/seed mismatch')
    n=manifest['splits']['test']['physical_samples']
    if n!=1024:raise ValueError('This final protocol requires 1024 common test states')
    test_path=(manifest_path.parent/manifest['splits']['test']['file']).resolve()
    if sha256(test_path)!=entry['test_parquet_sha256']:raise ValueError('Test Parquet differs from frozen hash')
    dataset=load_dataset(test_path,physical)
    if len(dataset.spectra)!=n:raise ValueError('Test sample count differs from manifest')
    rows=manifest['splits']['test']['samples']
    if len(rows)!=n or any(row['row']!=i+1 for i,row in enumerate(rows)):raise ValueError('Manifest test row order mismatch')
    for components in inference.values():
        for component in components:compatible(component['checkpoint'],dataset)
    names=[(s,k) for s,k in SPECIES_BLOCKS if physical[k]['used_in_model']]
    original,replay_checks=original_truth(manifest,dataset,names)
    predicted={};component_coefficients={}
    for role in ROLES:predicted[role],component_coefficients[role]=component_predictions(inference[role],dataset.spectra,chunk_size)
    outdir.mkdir(parents=True)
    species_results={};filtered_errors={role:[] for role in ROLES};original_errors={role:[] for role in ROLES}
    raw_temperature_errors={role:[] for role in ROLES};taylor_temperature_errors={role:[] for role in ROLES}
    cdf,cdf_axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False)
    m=physical['fourier_modes']
    for j,(s,key) in enumerate(names):
        block=slice(j*2*m,(j+1)*2*m);K=dataset.k_grid[j*m:(j+1)*m];velocity=np.asarray(physical[key]['velocity_grid_m_s'],dtype=np.float64);mass=physical[key]['mass_kg']
        true_temperature=np.array([row['species'][s]['actual_temperature_ev'] for row in rows])
        if not np.isfinite(true_temperature).all() or (true_temperature<=0).any():raise ValueError('Original temperatures must be finite positive')
        target_coefficients=dataset.coefficients[:,block]
        canonical_velocity,target_curve=decode(target_coefficients,K,velocity,chunk_size,torch.float32)
        _,target_curve64=decode(target_coefficients,K,velocity,chunk_size,torch.float64)
        target_moments=moments(target_curve.astype(np.float64),canonical_velocity.astype(np.float64),mass)
        true_taylor,true_taylor_detail=taylor(target_coefficients.double().numpy(),K.double().numpy(),3,mass)
        denominator=np.maximum((original[s]**2).sum(1),1e-30)
        representation=np.sqrt(((target_curve64-original[s])**2).sum(1)/denominator)
        replayed_temperature=moments(original[s],velocity,mass)['temperature_ev']
        if not np.allclose(replayed_temperature,true_temperature,rtol=1e-9,atol=1e-6):raise ValueError('Replayed temperatures differ from manifest actual moments')
        result={'target_fourier_vs_original_vdf_relative_l2':vector_summary(representation),'true_fourier_direct_temperature_bias':error_stats(target_moments['temperature_ev'],true_temperature),'true_fourier_taylor3_temperature_bias':error_stats(true_taylor,true_temperature),'true_fourier_taylor3_diagnostics':true_taylor_detail,'replayed_temperature_max_abs_difference_ev':float(np.max(np.abs(replayed_temperature-true_temperature))),'models':{}}
        curves={};relative={};saved={'target_coefficients':target_coefficients.numpy(),'k_grid':K.numpy(),'original_velocity_m_s':velocity,'original_actual_temperature_ev':true_temperature,'target_fourier_vs_original_relative_l2':representation,'target_direct_temperature_ev':target_moments['temperature_ev'],'target_taylor3_temperature_ev':true_taylor}
        for role in ROLES:
            coefficients=predicted[role][:,block]
            metrics=canonical_summary(target_coefficients,coefficients,K,velocity,chunk_size)
            relative[role]=np.array([row['evdf_relative_l2'] for row in metrics['per_sample']]);filtered_errors[role].append(relative[role])
            v,curve=decode(coefficients,K,velocity,chunk_size,torch.float32);curves[role]=curve
            _,curve64=decode(coefficients,K,velocity,chunk_size,torch.float64)
            original_relative=np.sqrt(((curve64-original[s])**2).sum(1)/denominator);original_errors[role].append(original_relative)
            raw=moments(curve.astype(np.float64),v.astype(np.float64),mass)
            estimated_t,detail=taylor(coefficients.double().numpy(),K.double().numpy(),3,mass)
            raw_temperature_errors[role].append(np.abs(raw['temperature_ev']-true_temperature));taylor_temperature_errors[role].append(np.abs(estimated_t-true_temperature))
            result['models'][role]={'fourier_target':metrics,'original_vdf_relative_l2':vector_summary(original_relative),'raw_signed_temperature_vs_original':error_stats(raw['temperature_ev'],true_temperature),'fixed_taylor3_temperature_vs_original':error_stats(estimated_t,true_temperature),'raw_nonpositive_mass_count':raw['nonpositive_mass_count'],'raw_integral_mass_absolute_error_mean':raw['mass_absolute_error_from_one_mean'],'taylor3_diagnostics':detail}
            saved.update({role+'_coefficients':coefficients.numpy(),role+'_fourier_target_relative_l2':relative[role],role+'_original_relative_l2':original_relative,role+'_raw_temperature_ev':raw['temperature_ev'],role+'_taylor3_temperature_ev':estimated_t})
            values=np.sort(relative[role]);cdf_axes[0,j].step(values,np.arange(1,n+1)/n,where='post',label='selected ensemble' if role=='selected' and is_ensemble else role)
        cdf_axes[0,j].set(xlabel='Fourier-target VDF relative L2',ylabel='Empirical CDF',title=s);cdf_axes[0,j].grid(alpha=.25);cdf_axes[0,j].legend()
        result['representative_indices']=plots(outdir,s,canonical_velocity,target_curve,curves,original[s],relative,selected_label)
        if is_ensemble:
            for number,coefficients in enumerate(component_coefficients['selected']):saved['selected_component_'+str(number)+'_coefficients']=coefficients[:,block].numpy()
            saved['selected_component_weights']=np.array([c['weight'] for c in inference['selected']])
        np.savez_compressed(outdir/(s+'_comparison_predictions.npz'),**saved)
        species_results[s]=result
    cdf.tight_layout();cdf.savefig(outdir/'error_cdf.png',dpi=150);plt.close(cdf)
    comparisons={'fourier_target_relative_l2':paired_comparison(filtered_errors['selected'],filtered_errors['baseline']),'original_vdf_relative_l2':paired_comparison(original_errors['selected'],original_errors['baseline']),'raw_temperature_absolute_error_ev':paired_comparison(raw_temperature_errors['selected'],raw_temperature_errors['baseline']),'taylor3_temperature_absolute_error_ev':paired_comparison(taylor_temperature_errors['selected'],taylor_temperature_errors['baseline'])}
    for role in ROLES:
        comparison=paired_comparison(taylor_temperature_errors[role],raw_temperature_errors[role])
        comparison['error_vector_labels']={'selected':'Fixed Taylor3 absolute temperature error of '+role+' model','baseline':'Direct signed-VDF absolute temperature error of '+role+' model'}
        comparisons[role+'_raw_minus_taylor3_temperature_error']=comparison
    payload={'dataset_kind':kind,'selection_file':str(selection_path.resolve()),'selection_file_sha256':selection_hash,'purpose':'Independent frozen old/new comparison on one untouched common new test set; no fake canonical run or dataset provenance','test_dataset_run':str(test_run),'test_dataset_id':manifest['dataset_id'],'test_seed':entry['test_seed'],'test_sample_count':n,'test_manifest_sha256':sha256(manifest_path),'test_parquet_sha256':sha256(test_path),'models':provenance,'frozen_auxiliary_temperature_method':METHOD,'species_order':[s for s,k in names],'species':species_results,'paired_comparisons':comparisons,'fourier_replay_checks':replay_checks,'cpu_threads':torch.get_num_threads(),'chunk_size':chunk_size,'sampler_source_sha256':sha256(SOURCE/'data.py'),'notes':['Model input/output statistics come from each own real training checkpoint.','Shape metrics use canonical float32 inverse Fourier targets; global R2 is flattened against one grand mean.','Original-VDF L2 uses float64 inverse reconstruction of unchanged float32 coefficients on original float64 axes.','Temperature truth is manifest actual temperature. all_finite errors include finite negative temperatures; nonfinite/negative counts are separate.','True-Fourier direct integral bias and fixed Taylor3 estimator bias are reported separately.','Paired CIs use common finite pairs. Ratio is selected mean error divided by baseline mean error; positive absolute improvement means baseline minus selected.','Representative examples independently choose median and worst Fourier-target error for each role.','No training, PV spectrum calculation, postprocessing, hyperparameter change, or method selection occurred.']}
    if is_ensemble:
        payload['notes'][-1]='No training, PV, clipping, renormalization, post-test weight adjustment, or method selection occurred. The frozen physical-Fourier average is the declared selected ensemble.'
        payload['notes'].append('Selected is a separately reported convex ensemble, not component A or a fabricated canonical checkpoint. Component weights and completed-stage budget were frozen before test access.')
    path=outdir/'final_comparison.json';path.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(path),'kind':kind,'replay_checks':len(replay_checks)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection-file',required=True,type=Path);p.add_argument('--output-root',type=Path,default=Path(__file__).parent/'final_comparison');p.add_argument('--kind',choices=KINDS);p.add_argument('--chunk-size',type=int,default=32)
    a=p.parse_args()
    if a.chunk_size<1:p.error('--chunk-size must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    selection,selection_hash=frozen_selection(a.selection_file)
    for kind in ([a.kind] if a.kind else KINDS):compare(kind,selection['comparisons'][kind],a.selection_file,selection_hash,a.output_root,a.chunk_size)
