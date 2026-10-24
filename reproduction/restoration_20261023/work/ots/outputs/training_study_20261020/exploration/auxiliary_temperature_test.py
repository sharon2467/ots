#!/usr/bin/env python3
"""Frozen-selection-only final auxiliary temperature diagnostics. No tuning."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse,warnings
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import numpy as np
import torch
from moment_probe import taylor,error_stats
from final_diagnostics import moments,clean,sha256

METHOD='taylor_even4_odd3_n3'
SPECIES_KEYS={'e':'electron_dis','i1':'ion1_dis','i2':'ion2_dis'}

def frozen_run_context(run_dir,selection_file):
    """This gate must precede all reads of test manifests or predictions."""
    selection=json.loads(selection_file.read_text())
    if selection.get('frozen') is not True:
        raise ValueError('Selection file must explicitly contain frozen=true')
    if selection.get('auxiliary_temperature',{}).get('method')!=METHOD:
        raise ValueError('Auxiliary temperature method must be frozen to '+METHOD)
    best=selection.get('best',{})
    allowed={}
    for kind in ('fixed_ion','fig1_fig11'):
        entry=best.get(kind)
        if not isinstance(entry,dict) or not isinstance(entry.get('run_id'),str) or not entry['run_id']:
            raise ValueError('Selection must name best.'+kind+'.run_id')
        allowed[entry['run_id']]=kind
    if len(allowed)!=2:raise ValueError('The two selected datasets must identify distinct runs')
    config=json.loads((run_dir/'pipeline_config.json').read_text())
    if config['run_id'] not in allowed:
        raise ValueError('Run is not an authorized frozen best run')
    return selection,config,allowed[config['run_id']]

def frozen_test_manifest(run_dir,selection_file):
    selection,config,kind=frozen_run_context(run_dir,selection_file)
    manifest_path=run_dir/'data/dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['dataset_id']!=config['dataset_id'] or manifest['physical_configuration']['dataset']!=kind:
        raise ValueError('Frozen run dataset identity differs from its manifest')
    names=[s for s,key in SPECIES_KEYS.items() if manifest['physical_configuration'][key]['used_in_model']]
    return selection,config,manifest,manifest_path,names

def prediction_path(run_dir,names,species):
    return run_dir/'evaluation'/('predictions.npz' if names==['e'] else species+'/predictions.npz')

def validate_prediction_arrays(archive,manifest,species):
    physical=manifest['physical_configuration'];m=physical['fourier_modes'];n=manifest['splits']['test']['physical_samples']
    target=archive['target_coefficients'].astype(np.float64)
    predicted=archive['predicted_coefficients'].astype(np.float64)
    k=archive['k_grid'].astype(np.float64);velocity=archive['velocity_m_s'].astype(np.float64)
    expected_v=np.asarray(physical[SPECIES_KEYS[species]]['velocity_grid_m_s'])
    if target.shape!=(n,2*m) or predicted.shape!=target.shape or k.shape!=(m,):raise ValueError('Prediction coefficient layout mismatch')
    if velocity.shape!=expected_v.shape or not np.allclose(velocity,expected_v,rtol=2e-6,atol=1e-8):raise ValueError('Prediction species velocity mismatch')
    if not np.all(np.diff(k)>0) or not np.allclose(k,-k[::-1],rtol=1e-6,atol=0):raise ValueError('K must be increasing and conjugate symmetric')
    if (k>0).sum()<3:raise ValueError('Frozen Taylor3 requires at least three positive K values')
    if not all(np.isfinite(a).all() for a in (target,predicted,k,velocity)):raise ValueError('Nonfinite prediction arrays')
    if not np.all(np.diff(velocity)>0):raise ValueError('Velocity must increase')
    return target,predicted,k,velocity

def paired_intervals(temperatures,original,repetitions=2000,seed=20261020):
    """S,M,N arrays; one state-index draw shared by all methods and species."""
    finite=np.isfinite(temperatures)&np.isfinite(original[:,None,:])
    common=finite.all(1)&(original>0)
    errors=np.where(common[:,None,:],np.abs(temperatures-original[:,None,:]),np.nan)
    relative=errors/original[:,None,:]
    s,m,n=temperatures.shape;mae=np.full((repetitions,s,m),np.nan);median=np.full_like(mae,np.nan)
    rng=np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for start in range(0,repetitions,100):
            stop=min(start+100,repetitions);indices=rng.integers(0,n,size=(stop-start,n))
            mae[start:stop]=np.nanmean(errors[:,:,indices],axis=-1).transpose(2,0,1)
            median[start:stop]=np.nanmedian(relative[:,:,indices],axis=-1).transpose(2,0,1)
        ci=lambda x:np.nanquantile(x,[.025,.975],axis=0).T.tolist()
        results=[]
        for j in range(s):
            results.append({'common_finite_sample_count':int(common[j].sum()),'excluded_sample_count':int(n-common[j].sum()),'method_mae_ev_ci95':ci(mae[:,j,:]),'method_median_absolute_relative_error_ci95':ci(median[:,j,:]),'raw_minus_taylor_mae_ev_ci95':ci(mae[:,j,0]-mae[:,j,1]),'raw_minus_taylor_median_absolute_relative_error_ci95':ci(median[:,j,0]-median[:,j,1]),'bootstrap_replicates_with_finite_statistics':np.isfinite(mae[:,j,:]).sum(0).tolist()})
    return {'repetitions':repetitions,'seed':seed,'method':'Paired percentile bootstrap; identical physical sample indices for every species and estimator; finite negative temperatures are included. CIs use the common finite subset for each species.','interpretation':'Conditional uncertainty for frozen estimators/model on iid sampled synthetic states; excludes selection and retraining uncertainty. Method point statistics outside bootstrap retain each estimator own finite subset.','species_results':results}

def diagnose(run_dir,selection_file,output_root):
    run_dir=run_dir.resolve()
    _,config,manifest,manifest_path,names=frozen_test_manifest(run_dir,selection_file)
    outdir=output_root/config['run_id']
    if outdir.exists():raise FileExistsError('Refusing to overwrite final auxiliary diagnostics')
    physical=manifest['physical_configuration'];n=manifest['splits']['test']['physical_samples']
    rows=manifest['splits']['test']['samples']
    if len(rows)!=n or any(r['row']!=i+1 for i,r in enumerate(rows)):raise ValueError('Manifest test sample order mismatch')
    labels=['raw_signed_vdf_model','fixed_taylor3_model','fixed_taylor3_true_fourier_bias']
    results={};temps=[];originals=[];inputs=[]
    for s in names:
        path=prediction_path(run_dir,names,s);inputs.append({'path':str(path),'sha256':sha256(path)})
        with np.load(path,allow_pickle=False) as archive:
            target,predicted,k,velocity=validate_prediction_arrays(archive,manifest,s)
            vdf=archive['predicted_evdf'].astype(np.float64)
        if vdf.shape!=(n,len(velocity)) or not np.isfinite(vdf).all():raise ValueError('Invalid canonical predicted VDF')
        mass=physical[SPECIES_KEYS[s]]['mass_kg']
        original=np.array([r['species'][s]['actual_temperature_ev'] for r in rows],dtype=np.float64)
        if not np.isfinite(original).all() or (original<=0).any():raise ValueError('Original actual temperatures must be finite positive')
        raw=moments(vdf,velocity,mass)
        model_t,model_detail=taylor(predicted,k,3,mass);true_t,true_detail=taylor(target,k,3,mass)
        estimates=[raw['temperature_ev'],model_t,true_t]
        results[s]={'estimators':{label:error_stats(t,original) for label,t in zip(labels,estimates)},'raw_nonpositive_mass_count':raw['nonpositive_mass_count'],'raw_integral_mass_absolute_error_mean':raw['mass_absolute_error_from_one_mean'],'model_taylor_diagnostics':model_detail,'true_fourier_taylor_diagnostics':true_detail}
        temps.append(estimates);originals.append(original)
    temperatures=np.asarray(temps);original=np.asarray(originals)
    bootstrap=paired_intervals(temperatures,original)
    for s,b in zip(names,bootstrap.pop('species_results')):results[s]['paired_bootstrap']=b
    payload={'run_id':config['run_id'],'dataset_id':config['dataset_id'],'checkpoint_id':config['checkpoint_id'],'purpose':'Final frozen fixed-Taylor3 auxiliary temperature evaluation; not VDF repair or a tuning criterion','selection_file':str(selection_file.resolve()),'selection_file_sha256':sha256(selection_file),'frozen_method':METHOD,'estimator_order':labels,'species_order':names,'test_sample_count':n,'cpu_threads':torch.get_num_threads(),'inputs':inputs+[{'path':str(manifest_path),'sha256':sha256(manifest_path)}],'reporting':'all_finite MAE includes finite negative temperatures. Nonnegative-only errors, negative and nonfinite counts are separate. Original truth is manifest actual_temperature_ev, not configured targets. True-Fourier Taylor3 quantifies fixed-window estimator bias.','bootstrap':bootstrap,'species':results,'training_performed':False,'pv_called':False,'postprocessing':'No VDF or Fourier coefficients changed'}
    outdir.mkdir(parents=True)
    path=outdir/'auxiliary_temperature_test.json';path.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(outdir/'temperature_estimates.npz',temperatures_ev=temperatures,original_temperatures_ev=original)
    print(json.dumps({'output':str(path),'frozen_method':METHOD}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dirs',nargs='+',type=Path);p.add_argument('--selection-file',required=True,type=Path);p.add_argument('--output-root',type=Path,default=Path(__file__).parent/'auxiliary_temperature_final')
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for run in a.run_dirs:diagnose(run,a.selection_file,a.output_root)
