#!/usr/bin/env python3
"""Fixed-window validation Fourier moment estimators; never train/read test/PV."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[3]/'OTS_code_share/NN_hotE_large_lmRange'
sys.path.insert(0,str(SOURCE))
import data
from model import load_model_checkpoint
from train import split_train_validation
from evaluation import predict_coefficients
from final_diagnostics import moments,clean,sha256

E_CHARGE=data.ELEMENTARY_CHARGE

def taylor(coefficients,k,n,mass):
    m=len(k);positive=np.flatnonzero(k>0)[:n]
    keep=np.sort(np.r_[positive,m-1-positive]);scale=np.max(np.abs(k[keep]))
    x=k[keep]/scale
    real_design=np.stack([np.ones_like(x),x*x,x**4],axis=1)
    imag_design=np.stack([x,x**3],axis=1)
    real=coefficients[:,:m][:,keep];imag=coefficients[:,m:][:,keep]
    a=real@np.linalg.pinv(real_design).T;b=imag@np.linalg.pinv(imag_design).T
    f0=a[:,0];valid=f0>0
    mean=np.full(len(f0),np.nan);second=np.full(len(f0),np.nan)
    mean[valid]=-b[valid,0]/(f0[valid]*scale)
    second[valid]=-2*a[valid,1]/(f0[valid]*scale**2)
    temperature=mass/E_CHARGE*(second-mean*mean)
    return temperature,{'f0':f0,'mean_velocity_m_s':mean,'invalid_nonpositive_f0_count':int((~valid).sum()),'positive_k_count':n,'max_abs_K_s_per_m':scale,'real_design_condition':float(np.linalg.cond(real_design)),'imag_design_condition':float(np.linalg.cond(imag_design))}

def log_cumulant(coefficients,k,n,mass):
    m=len(k);positive=np.flatnonzero(k>0)[:n]
    keep=np.sort(np.r_[positive,m-1-positive]);scale=np.max(np.abs(k[keep]));x=k[keep]/scale
    amplitude=np.hypot(coefficients[:,:m][:,keep],coefficients[:,m:][:,keep])
    valid=(amplitude>0).all(1)&np.isfinite(amplitude).all(1)
    design=np.stack([np.ones_like(x),x*x,x**4],axis=1)
    a=np.full((len(coefficients),3),np.nan)
    a[valid]=np.log(amplitude[valid])@np.linalg.pinv(design).T
    temperature=mass/E_CHARGE*(-2*a[:,1]/scale**2)
    return temperature,{'f0':np.exp(a[:,0]),'invalid_zero_or_nonfinite_amplitude_count':int((~valid).sum()),'positive_k_count':n,'max_abs_K_s_per_m':scale,'design_condition':float(np.linalg.cond(design))}

METHODS=[('taylor_even4_odd3_n3',lambda c,k,m:taylor(c,k,3,m)),('taylor_even4_odd3_n5',lambda c,k,m:taylor(c,k,5,m)),('taylor_even4_odd3_n8',lambda c,k,m:taylor(c,k,8,m)),('log_amplitude_cumulant_even4_n5',lambda c,k,m:log_cumulant(c,k,5,m))]

def error_stats(estimate,true):
    finite=np.isfinite(estimate)&np.isfinite(true);valid=finite&(estimate>=0)&(true>0)
    def errors(mask):
        if not mask.any():return {'count':0,'mae_ev':None,'median_absolute_relative_error':None,'bias_ev':None}
        delta=estimate[mask]-true[mask]
        return {'count':int(mask.sum()),'mae_ev':float(np.abs(delta).mean()),'median_absolute_relative_error':float(np.median(np.abs(delta)/true[mask])),'bias_ev':float(delta.mean())}
    return {'nonfinite_count':int((~finite).sum()),'negative_temperature_count':int((finite&(estimate<0)).sum()),'all_finite':errors(finite),'nonnegative_finite':errors(valid)}

def probe(run_dir,output_root,positivity_root,chunk_size):
    run_dir=run_dir.resolve();hpath=run_dir/'training/training_history.json';h=json.loads(hpath.read_text())
    ckpath=(hpath.parent/h['checkpoint']).resolve();model,checkpoint=load_model_checkpoint(ckpath,'cpu',h['checkpoint_id'])
    manifest=json.loads((hpath.parent/h['dataset_manifest']).resolve().read_text())
    physical=manifest['physical_configuration'];train_path=(hpath.parent/h['train_data']).resolve()
    pp_path=positivity_root/(h['run_id']+'_positivity_probe.json');pp=json.loads(pp_path.read_text())
    if pp['checkpoint_sha256']!=sha256(ckpath) or pp['train_parquet_sha256']!=sha256(train_path): raise ValueError('Positivity evidence does not match this checkpoint/train dataset')
    dataset=data.load_dataset(train_path,physical);cfg=checkpoint['training_config']
    _,_,xval,yval=split_train_validation(dataset,cfg['validation_fraction'],cfg['seed'])
    indices=torch.randperm(len(dataset.spectra),generator=torch.Generator().manual_seed(cfg['seed']))[:len(xval)]
    with np.load(pp['truth_cache'],allow_pickle=False) as cache:
        if not np.array_equal(cache['validation_indices'],indices.numpy()):raise ValueError('Validation truth order mismatch')
        original={s:cache[s+'_vdf'].copy() for s,k in data.SPECIES_BLOCKS if physical[k]['used_in_model']}
    with np.load(pp_path.with_suffix('.npz'),allow_pickle=False) as previous:
        if not np.array_equal(previous['validation_indices'],indices.numpy()):raise ValueError('Positivity prediction order mismatch')
        raw_temperature={s:previous[s+'_raw_prediction_temperature_ev'].copy() for s in original}
    guess=predict_coefficients(model,xval,checkpoint,chunk_size,'cpu').double().numpy();truth=yval.double().numpy()
    modes=checkpoint['fourier_modes'];results={};arrays={'validation_indices':indices.numpy()}
    for j,(s,kname) in enumerate([(s,k) for s,k in data.SPECIES_BLOCKS if physical[k]['used_in_model']]):
        mass=physical[kname]['mass_kg'];velocity=np.asarray(physical[kname]['velocity_grid_m_s'])
        original_t=moments(original[s],velocity,mass)['temperature_ev'];K=dataset.k_grid[j*modes:(j+1)*modes].double().numpy();block=slice(j*2*modes,(j+1)*2*modes)
        methods={};all_model_t={'direct_signed_vdf_moment':raw_temperature[s]}
        for name,fun in METHODS:
            pred_t,detail=fun(guess[:,block],K,mass);true_t,true_detail=fun(truth[:,block],K,mass)
            all_model_t[name]=pred_t
            methods[name]={'model_vs_original':error_stats(pred_t,original_t),'true_fourier_vs_original_estimator_bias':error_stats(true_t,original_t),'model_estimator_diagnostics':detail,'target_estimator_diagnostics':true_detail}
            arrays[s+'_'+name+'_model_temperature_ev']=pred_t;arrays[s+'_'+name+'_target_temperature_ev']=true_t
        common=np.isfinite(original_t)&(original_t>0)
        for t in all_model_t.values():common &= np.isfinite(t)&(t>=0)
        common_results={name:error_stats(np.where(common,t,np.nan),original_t) for name,t in all_model_t.items()}
        results[s]={'original_temperature_ev':{'min':float(original_t.min()),'max':float(original_t.max()),'mean':float(original_t.mean())},'direct_signed_vdf_moment_vs_original':error_stats(raw_temperature[s],original_t),'methods':methods,'common_nonnegative_all_model_estimators_subset':{'count':int(common.sum()),'comparisons':common_results}}
        arrays[s+'_original_temperature_ev']=original_t
    output_root.mkdir(parents=True,exist_ok=True);out=output_root/(h['run_id']+'_moment_probe.json')
    payload={'run_id':h['run_id'],'checkpoint_id':h['checkpoint_id'],'scope':'Validation-only, fixed model and fixed global windows; no training, PV, test read, network-output change, or per-sample truth-based window selection','test_data_read':False,'pv_called':False,'validation_count':len(xval),'cpu_threads':torch.get_num_threads(),'inputs':{'checkpoint_sha256':sha256(ckpath),'train_parquet_sha256':sha256(train_path),'positivity_evidence':str(pp_path),'truth_cache':pp['truth_cache']},'formulas':{'taylor':'Fit Re F=a0+a2*x^2+a4*x^4 and Im F=b1*x+b3*x^3 on conjugate symmetric fixed windows x=K/Kmax. u=-b1/(a0*Kmax); E[v^2]=-2*a2/(a0*Kmax^2); T=m/e*(E[v^2]-u^2). Nonpositive a0 is invalid.','log_cumulant':'Fit log|F|=a0+a2*x^2+a4*x^4; T=-(m/e)*2*a2/Kmax^2. This is an auxiliary characteristic-function estimator, not a repaired VDF.','sign_convention':'F(K)=integral f(v) exp(-i*K*v)dv; K in s/m, velocity m/s, temperature eV.'},'reporting':'All-finite MAE includes finite negative temperatures to prevent improved scores through exclusion. Separate nonnegative-only errors and invalid counts are provided. True Fourier estimates expose finite-window fit bias. No per-sample estimator or window is selected.','species':results}
    out.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps({'output':str(out),'validation_count':len(xval)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dirs',nargs='+',type=Path)
    p.add_argument('--output-root',type=Path,default=Path(__file__).parent/'moment_estimation');p.add_argument('--positivity-root',type=Path,default=Path(__file__).parent/'positivity');p.add_argument('--chunk-size',type=int,default=32)
    a=p.parse_args()
    if a.chunk_size<1:p.error('--chunk-size must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for run in a.run_dirs:probe(run,a.output_root,a.positivity_root,a.chunk_size)
