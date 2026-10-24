#!/usr/bin/env python3
"""Compare frozen models on the previous study's common development holdout.

Only training Parquets, manifests, histories, and checkpoints are read. No test
Parquet, test metrics, or test predictions are opened. This reused development
holdout is not an unbiased final test.
"""
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import sys,json,argparse,hashlib,importlib.util
sys.dont_write_bytecode=True
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'OTS_code_share/NN_hotE_large_lmRange'
OLD_AUDIT=ROOT/'outputs/training_study_20261020/exploration/validation_audit.py'
sys.path.insert(0,str(SOURCE))
from data import load_dataset,SPECIES_BLOCKS,dataset_layout,_physical_config
from train import split_train_validation
from model import load_model_checkpoint
from evaluation import predict_coefficients

spec=importlib.util.spec_from_file_location('previous_study_validation_audit',OLD_AUDIT)
old_audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_audit)
summarize,file_hash=old_audit.summarize,old_audit.file_hash

def index_hash(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()

def digest_sequence_hash(digests):
    return hashlib.sha256(b''.join(bytes.fromhex(d) for d in digests)).hexdigest()

def row_hashes(x,y,indices):
    """Hash actual float32 model input followed by float32 coefficient labels."""
    result=[]
    for i in indices.tolist():
        row=np.concatenate((x[i].numpy(),y[i].numpy())).astype('<f4',copy=False)
        result.append(hashlib.sha256(row.tobytes(order='C')).hexdigest())
    return result

def context(run_dir):
    run_dir=run_dir.resolve();history_path=run_dir/'training/training_history.json'
    history=json.loads(history_path.read_text())
    manifest_path=(history_path.parent/history['dataset_manifest']).resolve()
    manifest=json.loads(manifest_path.read_text())
    train_path=(manifest_path.parent/manifest['splits']['train']['file']).resolve()
    if train_path!=(history_path.parent/history['train_data']).resolve():raise ValueError('History and training manifest paths differ')
    if manifest['dataset_id']!=history['dataset_id']:raise ValueError('History and manifest dataset IDs differ')
    dataset=load_dataset(train_path,manifest['physical_configuration'])
    if len(dataset.spectra)!=manifest['splits']['train']['physical_samples']:raise ValueError('Training sample count differs from manifest')
    return {'run_dir':run_dir,'history_path':history_path,'history':history,'manifest_path':manifest_path,'manifest':manifest,'train_path':train_path,'dataset':dataset}

def split_indices(dataset,config):
    xtrain,ytrain,xval,yval=split_train_validation(dataset,config['validation_fraction'],config['seed'])
    order=torch.randperm(len(dataset.spectra),generator=torch.Generator().manual_seed(config['seed']))
    val_indices=order[:len(xval)];optimizer_indices=order[len(xval):]
    if not torch.equal(dataset.spectra[val_indices],xval) or not torch.equal(dataset.coefficients[val_indices],yval):raise ValueError('Validation split reconstruction differs from canonical')
    if not torch.equal(dataset.coefficients[optimizer_indices],ytrain):raise ValueError('Optimization split reconstruction differs from canonical')
    return ytrain,xval,yval,optimizer_indices,val_indices

def compatible(checkpoint,dataset):
    layout=checkpoint.get('layout')
    if layout is None:
        if dataset.layout is not None and dataset.layout!=dataset_layout(_physical_config()):raise ValueError('Legacy checkpoint requires original fixed layout')
    elif dataset.layout!=layout:raise ValueError('Checkpoint and dataset layouts differ')
    k=checkpoint['k_grid'].detach().cpu().to(dataset.k_grid.dtype)
    lm=checkpoint['wavelengths_nm'].detach().cpu().to(dataset.wavelengths_nm.dtype)
    if k.shape!=dataset.k_grid.shape or not torch.allclose(k,dataset.k_grid,rtol=1e-5,atol=1e-12):raise ValueError('Checkpoint and dataset K differ')
    if lm.shape!=dataset.wavelengths_nm.shape or not torch.allclose(lm,dataset.wavelengths_nm,rtol=1e-5,atol=1e-5):raise ValueError('Checkpoint and dataset wavelengths differ')

def audit(run_dir,references,output_dir,chunk_size,overwrite):
    history_path=run_dir.resolve()/'training/training_history.json'
    history_header=json.loads(history_path.read_text())
    output_path=output_dir/(history_header['run_id']+'_common_validation_metrics.json')
    if output_path.exists() and not overwrite:raise FileExistsError('Refusing to overwrite '+str(output_path))
    current=context(run_dir);history=current['history']
    checkpoint_path=(current['history_path'].parent/history['checkpoint']).resolve()
    model,checkpoint=load_model_checkpoint(checkpoint_path,'cpu',history['checkpoint_id'])
    if checkpoint['dataset_id']!=history['dataset_id'] or checkpoint['run_id']!=history['run_id']:raise ValueError('Checkpoint training identity differs from history')
    kind=current['manifest']['physical_configuration']['dataset']
    if kind not in references:raise ValueError('Unsupported dataset kind')
    reference=current if references[kind].resolve()==current['run_dir'] else context(references[kind])
    if reference['manifest']['physical_configuration']['dataset']!=kind:raise ValueError('Common holdout reference dataset kind mismatch')
    compatible(checkpoint,current['dataset']);compatible(checkpoint,reference['dataset'])
    if current['dataset'].layout!=reference['dataset'].layout or not torch.equal(current['dataset'].k_grid,reference['dataset'].k_grid) or not torch.equal(current['dataset'].wavelengths_nm,reference['dataset'].wavelengths_nm):raise ValueError('Common holdout and training coordinates/layout differ')
    ytrain,_,_,optimizer_indices,internal_validation_indices=split_indices(current['dataset'],checkpoint['training_config'])
    ref_config=reference['history']['training_config']
    _,xcommon,ycommon,_,common_indices=split_indices(reference['dataset'],ref_config)
    if len(xcommon)!=750:raise ValueError('This protocol requires exactly 750 previous-study heldout states')
    common_hashes=row_hashes(reference['dataset'].spectra,reference['dataset'].coefficients,common_indices)
    optimizer_hashes=row_hashes(current['dataset'].spectra,current['dataset'].coefficients,optimizer_indices)
    overlap=set(common_hashes).intersection(optimizer_hashes)
    overlap_rows=[{'common_validation_position':i,'reference_physical_index_zero_based':int(common_indices[i]),'row_sha256':digest} for i,digest in enumerate(common_hashes) if digest in overlap]
    if overlap:raise ValueError('Common development rows occur in actual optimization data: '+json.dumps(overlap_rows[:5]))
    predicted=predict_coefficients(model,xcommon,checkpoint,chunk_size,'cpu')
    baseline=ytrain.mean(0,keepdim=True).expand(len(ycommon),-1)
    layout=checkpoint.get('layout');m=checkpoint['fourier_modes']
    selected=[(s,k) for s,k in SPECIES_BLOCKS if layout[k]['used_in_model']] if layout else [('e','electron_dis')]
    species={}
    for j,(s,key) in enumerate(selected):
        block=slice(j*2*m,(j+1)*2*m);K=reference['dataset'].k_grid[j*m:(j+1)*m]
        velocity=layout[key]['velocity_grid_m_s'] if layout else None
        species[s]={'train_mean_coefficients':ytrain[:,block].mean(0).tolist(),'model':summarize(ycommon[:,block],predicted[:,block],K,velocity,common_indices,chunk_size),'optimizer_mean_curve_baseline':summarize(ycommon[:,block],baseline[:,block],K,velocity,common_indices,chunk_size)}
    score=float(np.mean([v['model']['metrics']['evdf_relative_l2_mean'] for v in species.values()]))
    baseline_score=float(np.mean([v['optimizer_mean_curve_baseline']['metrics']['evdf_relative_l2_mean'] for v in species.values()]))
    same_training_file=current['train_path']==reference['train_path']
    payload={'diagnostic':'Common previous-study development holdout; never an unbiased final test','run_id':history['run_id'],'dataset_id':history['dataset_id'],'checkpoint_id':history['checkpoint_id'],'dataset_kind':kind,'test_data_read':False,'test_metrics_read':False,'test_predictions_read':False,'cpu_threads':torch.get_num_threads(),'chunk_size':chunk_size,'selection_score':score,'selection_rule':'Equal-species macro mean of common-development per-sample VDF relative-L2; electron only for fixed_ion','baseline_selection_score':baseline_score,'worst_species_relative_l2_mean':max(v['model']['metrics']['evdf_relative_l2_mean'] for v in species.values()),'training_config':checkpoint['training_config'],'inputs':{'checkpoint':str(checkpoint_path),'checkpoint_sha256':file_hash(checkpoint_path),'history':str(current['history_path']),'history_sha256':file_hash(current['history_path']),'train_manifest':str(current['manifest_path']),'train_manifest_sha256':file_hash(current['manifest_path']),'train_parquet':str(current['train_path']),'train_parquet_sha256':file_hash(current['train_path']),'reference_run':str(reference['run_dir']),'reference_history_sha256':file_hash(reference['history_path']),'reference_train_manifest':str(reference['manifest_path']),'reference_train_manifest_sha256':file_hash(reference['manifest_path']),'reference_train_parquet':str(reference['train_path']),'reference_train_parquet_sha256':file_hash(reference['train_path']),'reused_metric_implementation':str(OLD_AUDIT),'reused_metric_implementation_sha256':file_hash(OLD_AUDIT)},'counts':{'train_pool_count':len(current['dataset'].spectra),'optimizer_count':len(optimizer_indices),'internal_validation_count':len(internal_validation_indices),'common_validation_count':len(common_indices)},'indices':{'common_reference_physical_indices_zero_based':common_indices.tolist(),'common_indices_sha256':index_hash(common_indices.numpy()),'optimizer_indices_sha256':index_hash(optimizer_indices.numpy()),'internal_validation_indices_sha256':index_hash(internal_validation_indices.numpy()),'index_hash_format':'Ordered physical indices, little-endian int64'},'overlap_check':{'row_encoding':'SHA256 of [spectrum float32, coefficient labels float32] concatenated as little-endian float32 C-order bytes','comparison':'Common validation rows versus checkpoint actual optimization rows, not entire original training pool','same_training_file_as_reference':same_training_file,'common_unique_row_count':len(set(common_hashes)),'optimizer_unique_row_count':len(set(optimizer_hashes)),'common_ordered_row_digests_sha256':digest_sequence_hash(common_hashes),'optimizer_ordered_row_digests_sha256':digest_sequence_hash(optimizer_hashes),'common_sorted_unique_row_digests_sha256':digest_sequence_hash(sorted(set(common_hashes))),'optimizer_sorted_unique_row_digests_sha256':digest_sequence_hash(sorted(set(optimizer_hashes))),'common_row_sha256':common_hashes,'overlapping_unique_row_count':len(overlap),'overlapping_common_row_count':len(overlap_rows),'overlapping_common_rows':overlap_rows,'zero_overlap':True},'layout_K_wavelengths_match':True,'r2_definition':'Canonical global flattened R2 against one grand mean; float64 accumulation','vdf_truth_definition':'Inverse transform of heldout target Fourier coefficients, not original sampler VDF','per_sample_index_context':'Every per_sample index refers to the reference training Parquet, not the new run training Parquet','development_caution':'These 750 states were already used for previous tuning and are reused adaptively; their estimates/intervals are development diagnostics. Final generalization requires the untouched new test set.','species':species}
    output_dir.mkdir(parents=True,exist_ok=True)
    temporary=output_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(output_path)
    print(json.dumps({'output':str(output_path),'selection_score':score,'baseline_selection_score':baseline_score,'optimizer_count':len(optimizer_indices),'common_validation_count':len(common_indices),'overlap':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dirs',nargs='+',type=Path)
    p.add_argument('--fixed-reference-run',type=Path,default=SOURCE/'runs/study_fixed_c')
    p.add_argument('--fig-reference-run',type=Path,default=SOURCE/'runs/study_fig_c')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    p.add_argument('--chunk-size',type=int,default=32);p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.chunk_size<1:p.error('--chunk-size must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    references={'fixed_ion':a.fixed_reference_run,'fig1_fig11':a.fig_reference_run}
    for run in a.run_dirs:audit(run,references,a.output_dir,a.chunk_size,a.overwrite)
