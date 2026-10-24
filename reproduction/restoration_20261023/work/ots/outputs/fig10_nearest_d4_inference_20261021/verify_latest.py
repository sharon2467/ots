"""Independent read-only verification of native Fig.10 -> frozen D4 artifacts.

Writes only this directory's verification.json and numerical_comparison.json.
Repeat after plot_summary.json exists to include the plot artifact audit.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MPLCONFIGDIR']='/tmp/fig10_latest_d4_matplotlib'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SOURCE=ROOT/'OTS_code_share/NN_hotE_large_lmRange'
DEPTH=ROOT/'outputs/depth_study_20261021'
sys.path[:0]=[str(SOURCE),str(DEPTH)]
import adapter
adapter.install_adapter()
from model import load_model_checkpoint
from evaluation import predict_coefficients,reconstruct_real_evdf
torch.set_num_threads(2)
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition: raise AssertionError(name)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def exact(name,a,b): check(name,bits(a,b))
def registry():
    return {str(r):{str(p.relative_to(r)):os.readlink(p) for p in sorted(r.rglob('*')) if p.is_symlink()} for r in (SOURCE/'registry',DEPTH/'registry')}
def diag(v,f,t,rows):
    v,f=v.astype(np.float64),f.astype(np.float64)
    area=np.trapezoid(f,v,axis=1)
    frac=np.trapezoid(np.maximum(-f,0),v,axis=1)/np.maximum(np.trapezoid(np.abs(f),v,axis=1),1e-30)
    imax=int(np.argmax(frac))
    return dict(shape=list(f.shape),finite=bool(np.isfinite(f).all()),velocity_range_m_s=[float(v[0]),float(v[-1])],vdf_range_s_per_m=[float(f.min()),float(f.max())],signed_integral_range=[float(area.min()),float(area.max())],signed_integral_mean=float(area.mean()),rows_with_nonpositive_integral=int((area<=0).sum()),negative_mass_fraction_definition='integral(max(-f,0))/integral(abs(f))',negative_mass_fraction_mean=float(frac.mean()),negative_mass_fraction_p95=float(np.quantile(frac,.95)),negative_mass_fraction_max=float(frac.max()),negative_mass_fraction_max_array_row=imax,negative_mass_fraction_max_source_row_index=int(rows[imax]),negative_mass_fraction_max_time_ns=float(t[imax]),rows_with_any_negative_vdf=int((f.min(axis=1)<0).sum()),ground_truth_available=False)
def profile(v,f):
    x=v.astype(np.float64); y=f.astype(np.float64)
    pos=np.maximum(y,0); area=np.trapezoid(pos,x)
    mean=np.trapezoid(x*pos,x)/area
    return {'min_s_m':float(y.min()),'max_s_m':float(y.max()),'positive_mass_rms_width_m_s':float(np.sqrt(np.trapezoid((x-mean)**2*pos,x)/area)),'signed_area':float(np.trapezoid(y,x)),'negative_mass_fraction':float(np.trapezoid(np.maximum(-y,0),x)/np.trapezoid(np.abs(y),x))}
def rel(a,b): return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-30))
def main():
    plan=read(HERE/'inference_plan.json'); summary=read(HERE/'results/summary.json')
    check('plan frozen',plan['status']=='frozen_before_inference')
    check('inference script frozen',sha(HERE/'infer_fig10_latest.py')==plan['inference_script_sha256'])
    for p,h in plan['protected_sha256'].items(): check('protected SHA '+p,sha(p)==h)
    check('registry48+35 unchanged before',registry()==plan['registry_snapshots'])
    ck=Path(plan['checkpoint'])
    check('D4 SHA',sha(ck)=='b822cac9b18878dd45ee324c2bbf2e480f4ece56742e10e1126a571bcc737019')
    model,cp=load_model_checkpoint(ck,'cpu','depth_fig_d4_s0_model')
    check('four hidden layers',cp['hidden_sizes']==[2048,1024,1024,1024])
    check('5354976 parameters',sum(p.numel() for p in model.parameters())==5354976)
    check('model eval',not model.training)
    axis=np.asarray(cp['layout']['wavelength_bands_nm'][0],dtype=np.float64)
    cached={}; comparisons={}; full_changes={}
    for name,nrow,ncol in [('fig10a',556,527),('fig10b',480,319),('fig10c',480,320)]:
        inp=ROOT/'outputs/fig10_nearest_row_fill_20261021/datasets'/f'{name}.npz'
        out=HERE/'results'/name/'predictions.npz'
        with np.load(inp,allow_pickle=False) as z: native={k:z[k] for k in z.files}
        with np.load(out,allow_pickle=False) as z: saved={k:z[k] for k in z.files}
        cached[name]=saved
        rec=summary['datasets'][name]
        check(name+' input SHA',sha(inp)==rec['input_sha256'])
        check(name+' output SHA',sha(out)==rec['output_sha256'])
        check(name+' diagnostics same as summary',read(out.parent/'diagnostics.json')==rec)
        check(name+' native shape',native['intensity'].shape==(nrow,ncol))
        for k,x in native.items(): exact(name+' native '+k,x,saved['native_'+k])
        for k in ['time_paper_ns','source_row_index']: exact(name+' '+k,native[k],saved[k])
        wave=native['wavelength_nm']; raw=native['intensity'].astype(np.float64)
        if name=='fig10b':
            check(name+' native319',raw.shape[1]==319)
            exact(name+' unchanged wavelength',wave,saved['wavelength_nm'])
            exact(name+' stored checkpoint axis',wave.astype(cp['wavelengths_nm'].numpy().dtype),cp['wavelengths_nm'].numpy())
        else:
            check(name+' no extrapolation',wave[0]<=axis[0] and wave[-1]>=axis[-1])
            raw=np.stack([np.interp(axis,wave,row) for row in raw])
            exact(name+' aligned wavelength',axis,saved['wavelength_nm'])
        exact(name+' aligned spectra',raw,saved['spectra_before_peak_normalization'])
        scale=np.max(np.abs(raw),axis=1); norm64=raw/scale[:,None]; norm=norm64.astype(np.float32)
        exact(name+' float64 scale',scale,saved['input_peak_scale'])
        exact(name+' float64 normalization',norm64,saved['spectra_peak_normalized_float64'])
        exact(name+' float32 normalization',norm,saved['spectra_peak_normalized'])
        check(name+' finite normalized',np.isfinite(norm).all())
        with torch.inference_mode(): coeff=predict_coefficients(model,torch.from_numpy(norm),cp,batch_size=128,device='cpu')
        exact(name+' canonical forward',coeff.numpy(),saved['predicted_coefficients'])
        for i,(s,key) in enumerate([('e','electron_dis'),('H','ion1_dis'),('C','ion2_dis')]):
            k=cp['k_grid'][i*80:(i+1)*80]
            v,f=reconstruct_real_evdf(coeff[:,i*160:(i+1)*160],k,velocity=cp['layout'][key]['velocity_grid_m_s'])
            exact(name+'/'+s+' canonical axis',v.numpy(),saved[s+'_velocity_m_s'])
            exact(name+'/'+s+' canonical VDF',f.numpy(),saved[s+'_vdf'])
            exact(name+'/'+s+' canonical k',k.numpy(),saved[s+'_k_grid'])
            check(name+'/'+s+' finite',bool(torch.isfinite(f).all()))
            check(name+'/'+s+' all diagnostics',diag(v.numpy(),f.numpy(),saved['time_paper_ns'],saved['source_row_index'])==rec['species'][s])
        check(name+' no truth claim',rec['species']['e']['ground_truth_available'] is False)
        with np.load(ROOT/'outputs/fig10_best_d4_inference_20261021/results'/name/'predictions.npz',allow_pickle=False) as z: old={k:z[k] for k in z.files}
        for k in ['time_paper_ns','source_row_index','e_velocity_m_s','H_velocity_m_s','C_velocity_m_s']: exact(name+' old/new coordinate '+k,saved[k],old[k])
        full_changes[name]={k:{'bitwise_identical':bits(saved[k],old[k]),'rows_with_any_changed_value':int(np.any(saved[k]!=old[k],axis=1).sum()),'relative_l2_change':rel(saved[k],old[k])} for k in ['spectra_peak_normalized','predicted_coefficients','e_vdf','H_vdf','C_vdf']}
        rep=[]
        for target in [2.,3.,4.]:
            row=int(np.argmin(abs(saved['time_paper_ns']-target))); neighborhood=[]
            for j in range(max(0,row-2),min(nrow,row+3)):
                item={'array_row':j,'source_row_index':int(saved['source_row_index'][j]),'actual_time_ns':float(saved['time_paper_ns'][j]),'native_replaced_pixel_count':int(native['replacement_mask'][j].sum()),'new_peak_scale':float(scale[j]),'old_peak_scale':float(old['input_peak_scale'][j]),'normalized_input_relative_l2_change':rel(norm[j],old['spectra_peak_normalized'][j]),'species':{}}
                for s in ['e','H','C']:
                    f=saved[s+'_vdf']; fo=old[s+'_vdf']; v=saved[s+'_velocity_m_s']
                    item['species'][s]={'new':profile(v,f[j]),'old':profile(v,fo[j]),'new_vs_old_relative_l2':rel(f[j],fo[j]),'new_vs_previous_row_relative_l2':rel(f[j],f[j-1]) if j else None,'old_vs_previous_row_relative_l2':rel(fo[j],fo[j-1]) if j else None}
                neighborhood.append(item)
            rep.append({'target_ns':target,'representative_row':row,'neighborhood':neighborhood})
        comparisons[name]=rep
    check('total all rows',summary['total_rows']==1516)
    check('experimental accuracy not inferred',summary['ground_truth_available'] is False and 'synthetic held-out accuracy does not measure accuracy' in summary['interpretation'])
    plot_status='pending'
    if (HERE/'plot_summary.json').exists():
        ps=read(HERE/'plot_summary.json')
        check('plot script SHA',sha(HERE/'plot_fig10_latest.py')==ps['script_sha256'])
        for name,saved in cached.items():
            r=ps['inputs'][name]
            check(name+' plot NPZ SHA',sha(r['file'])==r['sha256']==summary['datasets'][name]['output_sha256'])
            check(name+' plot corrected SHA',sha(r['corrected_dataset']['file'])==r['corrected_dataset']['sha256'])
            for target,rep in zip([2.,3.,4.],r['representative_rows']):
                j=int(np.argmin(abs(saved['time_paper_ns']-target)))
                check(name+f' plot representative {target}',rep['target_time_ns']==target and rep['saved_row_index']==j and rep['actual_time_ns']==float(saved['time_paper_ns'][j]) and rep['source_row_index']==int(saved['source_row_index'][j]) and rep['corrected_dataset_row_index']==j)
        for s in ['e','H','C']:
            lim=max(float(np.abs(x[s+'_vdf']).max()) for x in cached.values())
            check(s+' shared symmetric full extrema colors',ps['shared_species_color_limits_s_m'][s]==[-lim,lim])
            arr=[x[s+'_vdf'][[np.argmin(abs(x['time_paper_ns']-t)) for t in [2.,3.,4.]]] for x in cached.values()]
            lo=min(float(x.min()) for x in arr); hi=max(float(x.max()) for x in arr); margin=.075*(hi-lo or 1)
            check(s+' shared curve y limits',ps['shared_species_representative_y_limits_s_m'][s]==[min(lo,0)-margin,max(hi,0)+margin])
        check('10 plots/21 output artifacts',len(ps['page_order'])==10 and len(ps['outputs'])==21)
        for p,r in ps['outputs'].items():
            check('plot SHA '+p,sha(HERE/p)==r['sha256'])
            check('plot bytes '+p,(HERE/p).stat().st_size==r['bytes'])
        import subprocess,re
        pdfinfo=subprocess.run(['pdfinfo',str(HERE/'combined_figures.pdf')],check=True,capture_output=True,text=True).stdout
        check('PDF 10 pages',int(re.search(r'^Pages:\s+(\d+)',pdfinfo,re.M).group(1))==10)
        plot_status='passed'
    for p,h in plan['protected_sha256'].items(): check('final protected SHA '+p,sha(p)==h)
    check('registry48+35 unchanged after',registry()==plan['registry_snapshots'])
    (HERE/'numerical_comparison.json').write_text(json.dumps({'interpretation':'Descriptive change and neighbor comparisons only. Positive-part RMS width is a profile diagnostic, not temperature or physical truth. Relative L2 denominators use the named comparator.','full_array_changes_from_old_D4':full_changes,'datasets':comparisons},indent=2,allow_nan=False)+'\n')
    report={'status':'passed','check_count':len(checks),'checks':checks,'protected_file_count':len(plan['protected_sha256']),'registry_counts':{k:len(v) for k,v in plan['registry_snapshots'].items()},'bitwise_canonical_reproduction':True,'plot_verification':plot_status,'no_experimental_ground_truth':True,'comparison_file':'numerical_comparison.json','script_sha256':sha(__file__)}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
if __name__=='__main__': main()
