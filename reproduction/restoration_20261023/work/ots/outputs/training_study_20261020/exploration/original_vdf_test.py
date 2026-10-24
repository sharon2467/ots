#!/usr/bin/env python3
"""Replay original test VDFs only after explicit frozen-selection authorization."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from auxiliary_temperature_test import frozen_test_manifest,prediction_path,validate_prediction_arrays,SPECIES_KEYS
from positivity_probe import replay_config
from final_diagnostics import clean,sha256
from evaluation import reconstruct_real_evdf
from phyEqns import ots_full_fourier as phy

def summarize(a):
    return {'mean':float(np.mean(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(np.max(a))}

def diagnose(run_dir,selection_file,output_root,chunk_size):
    run_dir=run_dir.resolve()
    _,config,manifest,manifest_path,names=frozen_test_manifest(run_dir,selection_file)
    outdir=output_root/config['run_id']
    if outdir.exists():raise FileExistsError('Refusing to overwrite final original-VDF diagnostics')
    physical=manifest['physical_configuration'];n=manifest['splits']['test']['physical_samples']
    seed=manifest['splits']['test']['seed']
    if seed!=manifest['splits']['train']['seed']+1:raise ValueError('Test seed differs from train seed+1')
    sampler_config,factory=replay_config(physical);sampler=factory(np.random.default_rng(seed),sampler_config)
    targets={};predictions={};ks={};inputs=[]
    original={s:np.empty((n,len(physical[SPECIES_KEYS[s]]['velocity_grid_m_s']))) for s in names}
    for s in names:
        path=prediction_path(run_dir,names,s);inputs.append({'path':str(path),'sha256':sha256(path)})
        with np.load(path,allow_pickle=False) as archive:
            targets[s],predictions[s],ks[s],_=validate_prediction_arrays(archive,manifest,s)
    check_indices=set(np.linspace(0,n-1,min(12,n),dtype=int).tolist());checks=[]
    for row in range(n):
        state=sampler.sample()
        for s in names:
            v,f=state['distributions'][s];original[s][row]=f.numpy()
            if row in check_indices:
                k,F=phy.ft_x_n(v,f,physical['fourier_modes'])
                label=torch.cat((F.real,F.imag)).float().numpy()
                if not np.array_equal(k.float().numpy(),ks[s].astype(np.float32)):raise ValueError('Replayed K differs from canonical prediction metadata')
                error=float(np.max(np.abs(label-targets[s][row])))
                if not np.array_equal(label,targets[s][row].astype(np.float32)):raise ValueError(f'Replayed target label differs for {s}, row {row}: {error}')
                checks.append({'species':s,'test_sample_index_zero_based':row,'max_abs_float32_coefficient_error':error})
    results={};arrays={}
    for s in names:
        v=np.asarray(physical[SPECIES_KEYS[s]]['velocity_grid_m_s'],dtype=np.float64)
        relative={label:[] for label in ('model_vs_original','target_fourier_vs_original')}
        for start in range(0,n,chunk_size):
            truth=original[s][start:start+chunk_size]
            denominator=np.maximum((truth*truth).sum(1),1e-30)
            for label,coefficients in [('model_vs_original',predictions[s]),('target_fourier_vs_original',targets[s])]:
                _,reconstructed=reconstruct_real_evdf(torch.from_numpy(coefficients[start:start+chunk_size]),torch.from_numpy(ks[s]),velocity=v)
                error=np.sqrt(((reconstructed.numpy()-truth)**2).sum(1)/denominator)
                if not np.isfinite(error).all():raise ValueError('Nonfinite original VDF error')
                relative[label].append(error)
        relative={label:np.concatenate(values) for label,values in relative.items()}
        results[s]={label:{'raw_vdf_relative_l2':summarize(values)} for label,values in relative.items()}
        for label,values in relative.items():arrays[s+'_'+label+'_relative_l2']=values
    payload={'run_id':config['run_id'],'checkpoint_id':config['checkpoint_id'],'dataset_id':config['dataset_id'],'purpose':'Final frozen-model comparison to original sampled VDF; no model or estimator selection','selection_file':str(selection_file.resolve()),'selection_file_sha256':sha256(selection_file),'test_sample_count':n,'replay_seed':seed,'cpu_threads':torch.get_num_threads(),'fourier_replay_checks':checks,'inputs':inputs+[{'path':str(manifest_path),'sha256':sha256(manifest_path)}],'numerics':'Replay original VDF on original float64 velocity axes. Reconstruct canonical float32 target/model coefficients in float64 on these axes; relative L2 uses unweighted sums, consistent with validation positivity diagnostic. This avoids counting lowpass label targets as original VDF truth.','species':results,'training_performed':False,'pv_called':False,'postprocessing':'None; original model Fourier coefficients are unchanged'}
    outdir.mkdir(parents=True);path=outdir/'original_vdf_test.json'
    path.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n');np.savez_compressed(outdir/'original_vdf_errors.npz',**arrays)
    print(json.dumps({'output':str(path),'replay_checks':len(checks)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dirs',nargs='+',type=Path);p.add_argument('--selection-file',required=True,type=Path);p.add_argument('--output-root',type=Path,default=Path(__file__).parent/'original_vdf_final');p.add_argument('--chunk-size',type=int,default=32)
    a=p.parse_args()
    if a.chunk_size<1:p.error('--chunk-size must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for run in a.run_dirs:diagnose(run,a.selection_file,a.output_root,a.chunk_size)
