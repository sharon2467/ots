#!/usr/bin/env python3
"""Validation-only positivity exploration; replay samplers, never call PV/test."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.mplconfig'))
import sys,json,argparse,copy,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[3]/'OTS_code_share/NN_hotE_large_lmRange'
sys.path.insert(0,str(SOURCE))
import data
from train import split_train_validation
from model import load_model_checkpoint
from evaluation import predict_coefficients,reconstruct_real_evdf
from phyEqns import ots_full_fourier as phy
from final_diagnostics import moments,temperature_error,metric_vectors,clean,sha256,trapz

def stats(a):
    a=np.asarray(a);valid=np.isfinite(a)
    return {'valid_count':int(valid.sum()),'invalid_count':int((~valid).sum()),'mean':float(a[valid].mean()) if valid.any() else None,'p95':float(np.quantile(a[valid],.95)) if valid.any() else None,'min':float(a[valid].min()) if valid.any() else None,'max':float(a[valid].max()) if valid.any() else None}

def integration_weights(v):
    w=np.empty_like(v)
    w[0]=(v[1]-v[0])/2;w[-1]=(v[-1]-v[-2])/2
    w[1:-1]=(v[2:]-v[:-2])/2
    return w

def positive_projection(values,velocity):
    """Exact argmin sum w*(g-f)^2, g>=0, sum w*g=1."""
    w=integration_weights(velocity)
    out=np.empty_like(values)
    for i,f in enumerate(values):
        order=np.argsort(f)[::-1];ordered=f[order];weights=w[order]
        thresholds=(np.cumsum(weights*ordered)-1)/np.cumsum(weights)
        rho=np.flatnonzero(ordered>thresholds)[-1]
        out[i]=np.maximum(f-thresholds[rho],0)
    assert np.all(out>=0)
    assert np.max(np.abs(out@w-1))<1e-11
    return out

def replay_config(physical):
    axes={'wavelength_bands_nm':physical['wavelength_bands_nm'],'velocity_grids_m_s':{s:physical[k]['velocity_grid_m_s'] for s,k in data.SPECIES_BLOCKS},'used_in_model':{s:physical[k]['used_in_model'] for s,k in data.SPECIES_BLOCKS},'fourier_modes':physical['fourier_modes']}
    if physical['dataset']=='fixed_ion':
        config=data._physical_config(**axes)
        config['n_e']=physical['electron_density_m3']
        factory=data.Fixed_ion_sampler
    elif physical['dataset']=='fig1_fig11':
        config=data._physical_config_fig1_fig11(**axes)
        config['sampling']=copy.deepcopy(physical['sampling'])
        config['points_csv']=physical['points_csv']
        factory=data.Fig1_Fig11_sampler
    else: raise ValueError('Unsupported replay sampler')
    for s,k in data.SPECIES_BLOCKS:
        for field in ['mass_kg','charge']:
            if config[k][field]!=physical[k][field]:
                raise ValueError('Replay physical identity differs from generation configuration')
    return config,factory

def recover_original(manifest,dataset,indices,cache_root,train_hash):
    physical=manifest['physical_configuration'];modes=physical['fourier_modes']
    selected=[(s,k) for s,k in data.SPECIES_BLOCKS if physical[k]['used_in_model']]
    index_hash=hashlib.sha256(indices.numpy().astype('<i8').tobytes()).hexdigest()
    cache_key=hashlib.sha256((train_hash+index_hash+sha256(SOURCE/'data.py')).encode()).hexdigest()[:20]
    cache_path=cache_root/('validation_original_'+cache_key+'.npz')
    meta_path=cache_path.with_suffix('.json')
    if cache_path.exists() and meta_path.exists():
        metadata=json.loads(meta_path.read_text())
        if metadata['train_parquet_sha256']!=train_hash or metadata['validation_indices_sha256']!=index_hash: raise ValueError('Invalid truth cache')
        with np.load(cache_path,allow_pickle=False) as archive:
            original={s:archive[s+'_vdf'].copy() for s,k in selected}
        return original,metadata,cache_path
    config,factory=replay_config(physical)
    sampler=factory(np.random.default_rng(manifest['splits']['train']['seed']),config)
    original={s:np.empty((len(indices),len(physical[k]['velocity_grid_m_s']))) for s,k in selected}
    lookup={int(row):i for i,row in enumerate(indices)}
    checks=set(np.linspace(0,len(indices)-1,min(12,len(indices)),dtype=int).tolist())
    check_records=[]
    for row in range(manifest['splits']['train']['physical_samples']):
        state=sampler.sample()
        if row not in lookup: continue
        val_index=lookup[row]
        for j,(s,k) in enumerate(selected):
            v,f=state['distributions'][s];original[s][val_index]=f.numpy()
            if val_index in checks:
                K,F=phy.ft_x_n(v,f,modes)
                coef=torch.cat((F.real,F.imag)).float()
                stored=dataset.coefficients[row,j*2*modes:(j+1)*2*modes]
                error=float((coef-stored).abs().max())
                if not torch.equal(coef,stored): raise ValueError(f'Replay Fourier label differs at row {row}: {error}')
                check_records.append({'species':s,'validation_index':val_index,'physical_index':row,'max_abs_float32_coefficient_error':error})
    metadata={'train_parquet_sha256':train_hash,'validation_indices_sha256':index_hash,'sampler_source_sha256':sha256(SOURCE/'data.py'),'generation_seed':manifest['splits']['train']['seed'],'validation_count':len(indices),'replayed_physical_samples':manifest['splits']['train']['physical_samples'],'fourier_checks':check_records,'pv_called':False,'test_data_read':False}
    cache_root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,validation_indices=indices.numpy(),**{s+'_vdf':f for s,f in original.items()})
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    return original,metadata,cache_path

def examine(run_dir,output_root,chunk_size):
    run_dir=run_dir.resolve();history_path=run_dir/'training/training_history.json'
    history=json.loads(history_path.read_text());checkpoint_path=(history_path.parent/history['checkpoint']).resolve()
    model,checkpoint=load_model_checkpoint(checkpoint_path,'cpu',history['checkpoint_id'])
    manifest_path=(history_path.parent/history['dataset_manifest']).resolve();manifest=json.loads(manifest_path.read_text())
    train_path=(manifest_path.parent/manifest['splits']['train']['file']).resolve()
    if train_path!=(history_path.parent/history['train_data']).resolve(): raise ValueError('Training history path mismatch')
    if manifest['dataset_id']!=checkpoint['dataset_id']: raise ValueError('Dataset identity mismatch')
    dataset=data.load_dataset(train_path,manifest['physical_configuration'])
    config=checkpoint['training_config']
    _,_,x_val,y_val=split_train_validation(dataset,config['validation_fraction'],config['seed'])
    indices=torch.randperm(len(dataset.spectra),generator=torch.Generator().manual_seed(config['seed']))[:len(x_val)]
    assert torch.equal(dataset.spectra[indices],x_val) and torch.equal(dataset.coefficients[indices],y_val)
    original,replay,cache_path=recover_original(manifest,dataset,indices,output_root/'truth_cache',sha256(train_path))
    predicted=predict_coefficients(model,x_val,checkpoint,chunk_size,'cpu')
    physical=manifest['physical_configuration'];modes=checkpoint['fourier_modes']
    selected=[(s,k) for s,k in data.SPECIES_BLOCKS if physical[k]['used_in_model']]
    species_results={};arrays={}
    for j,(s,k) in enumerate(selected):
        velocity=np.asarray(physical[k]['velocity_grid_m_s'],dtype=np.float64);mass=physical[k]['mass_kg']
        K=dataset.k_grid[j*modes:(j+1)*modes].double();block=slice(j*2*modes,(j+1)*2*modes)
        targets=[];guesses=[]
        for start in range(0,len(y_val),chunk_size):
            _,f=reconstruct_real_evdf(y_val[start:start+chunk_size,block].double(),K,velocity=velocity);targets.append(f.numpy())
            _,f=reconstruct_real_evdf(predicted[start:start+chunk_size,block].double(),K,velocity=velocity);guesses.append(f.numpy())
        target=np.concatenate(targets);raw=np.concatenate(guesses)
        clipped=np.maximum(raw,0);clipped_mass=trapz(clipped,velocity)
        clip_norm=np.full_like(clipped,np.nan);valid_clip=clipped_mass>0
        clip_norm[valid_clip]=clipped[valid_clip]/clipped_mass[valid_clip,None]
        methods={'raw_prediction':raw,'clip_normalize':clip_norm,'positive_unit_mass_l2_projection':positive_projection(raw,velocity)}
        truths={'fourier_target':target,'original_sampled_vdf':original[s]}
        truth_moments={label:moments(f,velocity,mass) for label,f in truths.items()}
        results={};method_temperatures={}
        for label,f in methods.items():
            mm=moments(f,velocity,mass);comparisons={};method_temperatures[label]=mm['temperature_ev']
            for truth_label,truth in truths.items():
                relative,negative=metric_vectors(truth,f,velocity)
                reference,_=metric_vectors(truth,raw,velocity)
                comparisons[truth_label]={'relative_l2':stats(relative),'negative_mass_fraction':stats(negative),'temperature_error_ev':temperature_error(mm['temperature_ev'],truth_moments[truth_label]['temperature_ev']),'paired_relative_l2_change_from_raw':stats(relative-reference),'improved_sample_count':int(np.sum(relative<reference)),'worsened_sample_count':int(np.sum(relative>reference))}
                arrays[s+'_'+label+'_'+truth_label+'_relative_l2']=relative
            results[label]={'comparisons':comparisons,'integral_mass':stats(mm['integral_mass']),'absolute_integral_mass_error':stats(np.abs(mm['integral_mass']-1)),'nonpositive_mass_count':mm['nonpositive_mass_count'],'negative_temperature_count':mm['negative_temperature_count'],'valid_nonnegative_temperature_count':mm['valid_nonnegative_temperature_count'],'temperature_ev':stats(mm['temperature_ev'])}
            arrays[s+'_'+label+'_temperature_ev']=mm['temperature_ev']
            arrays[s+'_'+label+'_integral_mass']=mm['integral_mass']
        for truth_label,truth_mm in truth_moments.items():
            truth_temperature=truth_mm['temperature_ev']
            common_valid=np.isfinite(truth_temperature)&(truth_temperature>=0)
            for temperature in method_temperatures.values():
                common_valid &= np.isfinite(temperature)&(temperature>=0)
            for label,temperature in method_temperatures.items():
                results[label]['comparisons'][truth_label]['common_all_methods_valid_temperature_error_ev']=temperature_error(np.where(common_valid,temperature,np.nan),np.where(common_valid,truth_temperature,np.nan))
        representation,_=metric_vectors(original[s],target,velocity)
        species_results[s]={'methods':results,'representation':{'relative_l2':stats(representation),'temperature_fourier_minus_original_ev':temperature_error(truth_moments['fourier_target']['temperature_ev'],truth_moments['original_sampled_vdf']['temperature_ev'])},'clip_zero_mass_count':int((~valid_clip).sum()),'mass_kg':mass}
    payload={'run_id':history['run_id'],'dataset_id':history['dataset_id'],'checkpoint_id':history['checkpoint_id'],'scope':'Validation-only exploration; no training, PV, test read, or canonical edits','test_data_read':False,'pv_called':False,'cpu_threads':torch.get_num_threads(),'validation_count':len(x_val),'checkpoint_sha256':sha256(checkpoint_path),'train_parquet_sha256':sha256(train_path),'replay':replay,'truth_cache':str(cache_path),'projection_definition':'minimize trapezoid-weighted sum w_i*(g_i-f_i)^2 subject to g_i>=0 and sum w_i*g_i=1; exact sorted active-set threshold','numerics':'Original sampler VDF float64. Network coefficients and training labels loaded as canonical float32; both inverse transformed in float64 on original float64 velocity axis for this diagnostic. Relative L2 uses canonical unweighted sample sums.','temperature_caution':'Signed moments without clipping unless method explicitly clips/projects. Temperature comparisons require finite nonnegative values on both sides and report exclusions. Fourier target may itself contain negative lobes. Projection improves weighted L2 to any feasible original VDF; canonical unweighted relative L2 is also reported.','species':species_results}
    output_root.mkdir(parents=True,exist_ok=True)
    result_path=output_root/(history['run_id']+'_positivity_probe.json')
    # This file belongs only to this isolated, replayable exploration.
    result_path.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(result_path.with_suffix('.npz'),validation_indices=indices.numpy(),**arrays)
    print(json.dumps({'output':str(result_path),'species':species_results},default=clean))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path)
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'positivity')
    p.add_argument('--chunk-size',type=int,default=32)
    a=p.parse_args()
    if a.chunk_size<1:p.error('--chunk-size must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    examine(a.run_dir,a.output_root,a.chunk_size)
