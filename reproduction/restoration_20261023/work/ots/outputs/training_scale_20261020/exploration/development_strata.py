#!/usr/bin/env python3
"""Descriptive paired strata on reused old development rows; never read test."""
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import sys,json,argparse,hashlib,math
sys.dont_write_bytecode=True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DENSITY_EDGES=np.array([2.8,3.35,3.9,4.45,5.0])*1e26
KINDS=('fixed_ion','fig1_fig11')
SPECIES_NAMES={'e':'e','i1':'H','i2':'C'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):return json.loads(path.read_text())

def error_vector(document,species,indices):
    rows=document['species'][species]['model']['per_sample']
    actual=[row['train_parquet_physical_index_zero_based'] for row in rows]
    if actual!=indices:raise ValueError('Per-sample reference ordering differs from common indices')
    a=np.array([row['evdf_relative_l2'] for row in rows])
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid L2 vector')
    return a

def paired_summary(old,new,mask,old_worst,new_worst):
    n=int(mask.sum());a=old[mask];b=new[mask]
    return {'count':n,'old_mean_relative_l2':float(a.mean()) if n else None,'new_mean_relative_l2':float(b.mean()) if n else None,'old_p95_relative_l2':float(np.quantile(a,.95)) if n else None,'new_p95_relative_l2':float(np.quantile(b,.95)) if n else None,'absolute_mean_improvement':float((a-b).mean()) if n else None,'relative_mean_error_reduction':float((a.mean()-b.mean())/a.mean()) if n and a.mean()>0 else None,'improved_count':int(np.sum(b<a)),'improved_fraction':float(np.mean(b<a)) if n else None,'worsened_count':int(np.sum(b>a)),'equal_count':int(np.sum(b==a)),'old_global_worst_count':int((old_worst&mask).sum()),'new_global_worst_count':int((new_worst&mask).sum()),'retained_global_worst_count':int((old_worst&new_worst&mask).sum()),'escaped_global_worst_count':int((old_worst&~new_worst&mask).sum()),'newly_global_worst_count':int((~old_worst&new_worst&mask).sum())}

def strata(values,edges,old,new,old_worst,new_worst):
    if np.any(np.diff(edges)<=0):raise ValueError('Stratum boundaries must be strictly increasing')
    if values.min()<edges[0]-max(abs(edges[0]),1)*1e-12 or values.max()>edges[-1]+max(abs(edges[-1]),1)*1e-12:raise ValueError('Values outside fixed grouping boundaries')
    membership=np.searchsorted(edges[1:-1],values,side='right')
    result=[]
    for j in range(len(edges)-1):
        row=paired_summary(old,new,membership==j,old_worst,new_worst)
        row.update({'group':j+1,'lower_bound':float(edges[j]),'upper_bound':float(edges[j+1]),'upper_bound_inclusive':j==len(edges)-2})
        result.append(row)
    return {'edges':edges.tolist(),'interval_rule':'[lower, upper), last interval includes upper bound','groups':result},membership

def analyze(new_path,baseline_path):
    new=read(new_path);old=read(baseline_path);kind=new['dataset_kind']
    if old['dataset_kind']!=kind:raise ValueError('Baseline dataset kind differs')
    if new['indices']['common_indices_sha256']!=old['indices']['common_indices_sha256']:raise ValueError('Common index hashes differ')
    if new['overlap_check']['common_ordered_row_digests_sha256']!=old['overlap_check']['common_ordered_row_digests_sha256']:raise ValueError('Common float32 row hash sequences differ')
    indices=old['indices']['common_reference_physical_indices_zero_based']
    if len(indices)!=750 or len(set(indices))!=750:raise ValueError('Expected 750 distinct common development states')
    if new['indices']['common_reference_physical_indices_zero_based']!=indices:raise ValueError('Common row order differs')
    if not new['overlap_check']['zero_overlap'] or not old['overlap_check']['zero_overlap']:raise ValueError('Optimization overlap audit failed')
    manifest_path=Path(old['inputs']['reference_train_manifest'])
    if digest(manifest_path)!=old['inputs']['reference_train_manifest_sha256'] or digest(manifest_path)!=new['inputs']['reference_train_manifest_sha256']:raise ValueError('Reference manifest content differs from audited metadata')
    manifest=read(manifest_path)
    train=manifest['splits']['train']['samples']
    selected=[]
    for index in indices:
        row=train[index]
        if row['row']!=index+1:raise ValueError('Manifest train row indexing mismatch')
        selected.append(row)
    density=np.array([r['electron_density_m3'] for r in selected])
    result={'dataset_kind':kind,'new_run_id':new['run_id'],'baseline_run_id':old['run_id'],'common_sample_count':750,'common_indices_sha256':old['indices']['common_indices_sha256'],'inputs':{'new_common_validation':str(new_path.resolve()),'new_common_validation_sha256':digest(new_path),'old_common_validation':str(baseline_path.resolve()),'old_common_validation_sha256':digest(baseline_path),'old_train_manifest':str(manifest_path),'old_train_manifest_sha256':digest(manifest_path)},'species':{}}
    for s in old['species']:
        a=error_vector(old,s,indices);b=error_vector(new,s,indices)
        count=math.ceil(.05*len(a));old_worst=np.zeros(len(a),dtype=bool);new_worst=np.zeros(len(a),dtype=bool)
        old_worst[np.argsort(a,kind='stable')[-count:]]=True;new_worst[np.argsort(b,kind='stable')[-count:]]=True
        temperature=np.array([r['species'][s]['actual_temperature_ev'] for r in selected])
        edges=np.quantile(temperature,[0,.25,.5,.75,1])
        ts,temperature_groups=strata(temperature,edges,a,b,old_worst,new_worst)
        overlap=old_worst&new_worst;union=old_worst|new_worst
        species={'overall':paired_summary(a,b,np.ones(len(a),dtype=bool),old_worst,new_worst),'actual_temperature_ev_quartiles':ts,'worst_definition':'Largest ceil(0.05*750)=38 relative-L2 errors separately for each model and species; descriptive worst tail, not a physical failure threshold','worst_overlap':{'count_each':count,'intersection_count':int(overlap.sum()),'retained_fraction_of_old_worst':float(overlap.sum()/count),'jaccard':float(overlap.sum()/union.sum()),'old_worst_reference_physical_indices':[indices[i] for i in np.flatnonzero(old_worst)],'new_worst_reference_physical_indices':[indices[i] for i in np.flatnonzero(new_worst)],'overlap_reference_physical_indices':[indices[i] for i in np.flatnonzero(overlap)]}}
        density_groups=None
        if kind=='fig1_fig11':
            species['electron_density_m3_strata'],density_groups=strata(density,DENSITY_EDGES,a,b,old_worst,new_worst)
        species['per_sample']=[{'reference_physical_index_zero_based':index,'old_relative_l2':a[i],'new_relative_l2':b[i],'actual_temperature_ev':temperature[i],'electron_density_m3':density[i],'temperature_quartile':int(temperature_groups[i])+1,'density_stratum':int(density_groups[i])+1 if density_groups is not None else None,'old_global_worst':bool(old_worst[i]),'new_global_worst':bool(new_worst[i])} for i,index in enumerate(indices)]
        result['species'][s]=species
    return result

def panel(ax,grouping,title,density=False):
    groups=grouping['groups'];x=np.arange(len(groups));width=.36
    a=np.array([r['old_mean_relative_l2'] for r in groups])*100;b=np.array([r['new_mean_relative_l2'] for r in groups])*100
    ax.bar(x-width/2,a,width,color='C0');ax.bar(x+width/2,b,width,color='C1')
    ax.scatter(x-width/2,[r['old_p95_relative_l2']*100 for r in groups],marker='D',color='C0',s=24)
    ax.scatter(x+width/2,[r['new_p95_relative_l2']*100 for r in groups],marker='D',color='C1',s=24)
    if density:labels=[f"{r['lower_bound']/1e26:.2f}–{r['upper_bound']/1e26:.2f}\nn={r['count']}" for r in groups]
    else:labels=[f"Q{j+1}\nn={r['count']}" for j,r in enumerate(groups)]
    ax.set_xticks(x,labels,fontsize=8);ax.set(ylabel='Relative L2 (%)',title=title)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_xlabel('Electron density (10^26 m^-3)' if density else 'Actual-temperature quartiles: Q1 cold → Q4 hot')

def figure(results,path):
    fixed=results.get('fixed_ion');fig=results.get('fig1_fig11')
    rows=(1 if fixed else 0)+(2 if fig else 0)
    canvas=plt.figure(figsize=(13,3.1*rows+1.0),layout='constrained');grid=canvas.add_gridspec(rows,3)
    row=0
    if fixed:
        panel(canvas.add_subplot(grid[0,:]),fixed['species']['e']['actual_temperature_ev_quartiles'],'Fixed e: temperature strata');row+=1
    if fig:
        for j,s in enumerate(('e','i1','i2')):
            panel(canvas.add_subplot(grid[row,j]),fig['species'][s]['electron_density_m3_strata'],'Fig '+SPECIES_NAMES[s]+': density strata',True)
            panel(canvas.add_subplot(grid[row+1,j]),fig['species'][s]['actual_temperature_ev_quartiles'],'Fig '+SPECIES_NAMES[s]+': temperature strata')
    legend=[Patch(color='C0',label='Old C baseline: mean'),Patch(color='C1',label='Expanded-data model: mean'),Line2D([],[],color='.35',marker='D',ls='',label='Diamonds: p95 (same color)')]
    canvas.legend(handles=legend,loc='outside upper center',ncol=3,frameon=False)
    canvas.supxlabel('Same reused 750 development states per dataset; descriptive associations only',fontsize=10)
    canvas.savefig(path,dpi=170);plt.close(canvas)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('new_metric_paths',nargs='+',type=Path)
    p.add_argument('--name',required=True);p.add_argument('--baseline-fixed',type=Path,default=Path(__file__).parent/'study_fixed_c_common_validation_metrics.json');p.add_argument('--baseline-fig',type=Path,default=Path(__file__).parent/'study_fig_c_common_validation_metrics.json');p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'development_strata');p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):p.error('--name must be a simple identifier')
    output=args.output_dir/(args.name+'.json');plot=args.output_dir/(args.name+'.png')
    if not args.overwrite and (output.exists() or plot.exists()):raise FileExistsError('Refusing to overwrite existing strata outputs')
    results={}
    for path in args.new_metric_paths:
        kind=read(path)['dataset_kind']
        if kind in results:p.error('Use one expanded model per kind per report; rerun with another --name for later stages')
        if kind not in KINDS:p.error('Unsupported dataset kind')
        results[kind]=analyze(path,args.baseline_fixed if kind=='fixed_ion' else args.baseline_fig)
    payload={'scope':'Descriptive paired common-development analysis only; old holdout reused for adaptive tuning, not an unbiased final test','test_data_read':False,'density_edges_m3':DENSITY_EDGES.tolist(),'temperature_boundary_source':'Actual species temperatures of the same original 750 heldout states; fixed across all model comparisons','notes':['No training, PV generation, model changes, or new test/evaluation reads.','All L2 truths are the same lowpass Fourier targets used by common_validation_audit.','Groups may differ jointly in temperature, density, shape and flow. Associations do not identify causes or prove non-identifiability without density input.','Worst-tail memberships are defined globally per species/model, not recomputed inside each stratum.','No confidence intervals or causal claims are inferred from these descriptive group summaries.'],'comparisons':results}
    args.output_dir.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');figure(results,plot)
    print(json.dumps({'json':str(output),'figure':str(plot)}))
