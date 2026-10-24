"""Authorized copies -> unchanged repository PV function, on native axes."""
from pathlib import Path
import sys, json, time, hashlib, argparse, csv
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / 'OTS_code_share/NN_hotE_large_lmRange'
sys.path.insert(0, str(SOURCE))
from data import normalize_distribution, _physical_config_fig1_fig11
from phyEqns.ots_full_fourier import OTS_spectrum_PV

SPECIES = [('e','e'),('H','i1'),('C','i2')]
MASSES = {'e':9.1093837015e-31,'H':1.67262192369e-27,'C':12*1.67262192369e-27}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def moments(v,f,mass):
    integral=np.trapezoid(f,v,axis=-1)
    u=np.trapezoid(v*f,v,axis=-1)/integral
    temperature=mass*np.trapezoid((v-u[...,None])**2*f,v,axis=-1)/integral/1.602176634e-19
    return integral,u,temperature

def load_panel(name):
    predpath=REPO/f'outputs/fig10_nearest_d4_inference_20261021/results/{name}/predictions.npz'
    datapath=REPO/f'outputs/fig10_nearest_row_fill_20261021/datasets/{name}.npz'
    return np.load(predpath),np.load(datapath),predpath,datapath

def prepare(pred,rows=None):
    result={};stats=[]
    for sp,key in SPECIES:
        oldv=np.asarray(pred[sp+'_velocity_m_s'],dtype=np.float64)
        oldf=np.asarray(pred[sp+'_vdf'],dtype=np.float64)
        if rows is not None:oldf=oldf[rows]
        v=np.linspace(oldv[0],oldv[-1],len(oldv),dtype=np.float64)
        assert v[0]==oldv[0] and v[-1]==oldv[-1]
        interp=np.stack([np.interp(v,oldv,f) for f in oldf])
        clipped=np.maximum(interp,0.)
        f=np.stack([normalize_distribution(torch.from_numpy(v),torch.from_numpy(x),sp).numpy() for x in clipped])
        original_i,original_u,original_t=moments(oldv,oldf,MASSES[sp])
        new_i,new_u,new_t=moments(v,f,MASSES[sp])
        neg_mass=np.trapezoid(np.maximum(-oldf,0.),oldv,axis=1)
        interp_neg=np.trapezoid(np.maximum(-interp,0.),v,axis=1)
        result.update({sp+'_velocity_m_s':v,sp+'_vdf':f,sp+'_original_velocity_m_s':oldv,
                       sp+'_original_integral':original_i,sp+'_original_u_m_s':original_u,sp+'_original_T_eV':original_t,
                       sp+'_processed_integral':new_i,sp+'_processed_u_m_s':new_u,sp+'_processed_T_eV':new_t,
                       sp+'_original_negative_mass':neg_mass,sp+'_interpolated_negative_mass':interp_neg,
                       sp+'_interpolated_integral':np.trapezoid(interp,v,axis=1),
                       sp+'_pre_normalization_positive_integral':np.trapezoid(clipped,v,axis=1),
                       sp+'_clipped_node_count':np.count_nonzero(interp<0,axis=1),
                       sp+'_interpolation_relativeL2':np.linalg.norm(interp-oldf,axis=1)/np.linalg.norm(oldf,axis=1),
                       sp+'_total_preparation_relativeL2':np.linalg.norm(f-oldf,axis=1)/np.linalg.norm(oldf,axis=1)})
        dv=np.diff(v)
        stats.append({'species':sp,'maximum_grid_displacement_m_s':float(np.max(np.abs(v-oldv))),
                      'new_grid_relative_spacing_error':float(np.max(np.abs(dv-dv.mean()))/dv.mean()),
                      'maximum_original_negative_mass':float(neg_mass.max()),
                      'maximum_preparation_relativeL2':float(result[sp+'_total_preparation_relativeL2'].max()),
                      'original_T_min_eV':float(original_t.min()),'original_T_max_eV':float(original_t.max()),
                      'processed_T_min_eV':float(new_t.min()),'processed_T_max_eV':float(new_t.max())})
    return result,stats

def make_config():
    cfg=_physical_config_fig1_fig11()
    cfg['n_e']=np.float64(3.5e26)
    for k in ['lm_0','n_e','Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V']:cfg[k]=np.float64(cfg[k])
    for k in ['ss','k_sh','E_i0','u_e','u_i1','u_i2']:cfg[k]=torch.as_tensor(cfg[k],dtype=torch.float64)
    return cfg

def forward(wave,prepared,row,cfg):
    args=[torch.from_numpy(np.asarray(wave,dtype=np.float64)*1e-9)]
    args.extend(cfg[k] for k in ['ss','k_sh','E_i0','lm_0','n_e','Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V'])
    kwargs={'fourier_modes':80}
    for sp,key in SPECIES:
        kwargs['v_'+key]=torch.from_numpy(prepared[sp+'_velocity_m_s'])
        kwargs['f_'+key]=torch.from_numpy(prepared[sp+'_vdf'][row])
        kwargs['u_'+key]=cfg['u_'+key]
    with torch.no_grad(): y=OTS_spectrum_PV(*args,**kwargs).numpy()
    assert np.isfinite(y).all() and np.isclose(np.max(np.abs(y)),1.,rtol=0,atol=1e-12)
    return y

def simple_config(cfg):
    keys=['lm_0','n_e','Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V','ss','k_sh','E_i0','u_e','u_i1','u_i2','fourier_modes']
    return {k:cfg[k].tolist() if isinstance(cfg[k],torch.Tensor) else float(cfg[k]) for k in keys}

def run(smoke=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    HERE.mkdir(parents=True,exist_ok=True)
    plan=HERE.parent/'analysis_plan.json'
    cfg=make_config();start=time.perf_counter()
    if smoke:
        pred,data,pp,dp=load_panel('fig10a'); prepared,stats=prepare(pred,[0])
        t=time.perf_counter();y=forward(data['wavelength_nm'],prepared,0,cfg);elapsed=time.perf_counter()-t
        report={'panel':'fig10a','row':0,'shape':list(y.shape),'finite':bool(np.isfinite(y).all()),
                'range':[float(y.min()),float(y.max())],'forward_seconds':elapsed,'torch_threads':torch.get_num_threads(),
                'predictions_sha256':sha(pp),'dataset_sha256':sha(dp),'config':simple_config(cfg),'preparation':stats}
        (HERE/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(HERE/'smoke_result.npz',predicted_PV=y,wavelength_nm=data['wavelength_nm'])
        print(json.dumps(report,indent=2),flush=True);return
    if (HERE/'summary.json').exists(): raise RuntimeError('Refusing to overwrite completed forward results')
    summary={'analysis_plan_sha256':sha(plan),'config':simple_config(cfg),'torch_threads':torch.get_num_threads(),
             'forward_api':str(SOURCE/'phyEqns/ots_full_fourier.py')+':OTS_spectrum_PV',
             'api_sha256':sha(SOURCE/'phyEqns/ots_full_fourier.py'),'panels':{}}
    for name in ['fig10a','fig10b','fig10c']:
        panelstart=time.perf_counter();pred,data,pp,dp=load_panel(name)
        t=np.asarray(data['time_paper_ns']);wave=np.asarray(data['wavelength_nm']);obs=np.asarray(data['intensity'],dtype=np.float64)
        rows=np.asarray(data['source_row_index']);assert np.array_equal(t,pred['time_paper_ns']) and np.array_equal(rows,pred['source_row_index'])
        assert np.array_equal(wave,pred['native_wavelength_nm']) and np.array_equal(obs,pred['native_intensity'])
        dest=HERE/name;dest.mkdir(exist_ok=True)
        prepared,stats=prepare(pred)
        prepared.update(time_paper_ns=t,source_row_index=rows)
        np.savez_compressed(dest/'processed_vdfs.npz',**prepared)
        scale=np.max(np.abs(obs),axis=1);norm=np.full_like(obs,np.nan);validobs=scale>0;norm[validobs]=obs[validobs]/scale[validobs,None]
        predicted=np.full_like(obs,np.nan);elapsed=np.zeros(len(t));success=np.zeros(len(t),dtype=bool);errors=[]
        for row in range(len(t)):
            rstart=time.perf_counter()
            try:predicted[row]=forward(wave,prepared,row,cfg);success[row]=True;errors.append('')
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
            elapsed[row]=time.perf_counter()-rstart
            if row%100==0:print(name,row+1,'/',len(t),'elapsed',time.perf_counter()-panelstart,'success',int(success.sum()),flush=True)
        residual=predicted-norm;denom=np.linalg.norm(norm,axis=1)
        relative=np.linalg.norm(residual,axis=1)/denom
        rmse=np.sqrt(np.mean(residual**2,axis=1))
        np.savez_compressed(dest/'forward_results.npz',time_paper_ns=t,wavelength_nm=wave,source_row_index=rows,
                            observed=obs,observed_normalized=norm,observed_peak_scale=scale,predicted_PV=predicted,
                            residual=residual,relativeL2=relative,rmse=rmse,success=success,error=np.asarray(errors),
                            forward_seconds=elapsed,fixed_density_m3=np.float64(3.5e26))
        with (dest/'row_metrics.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['row','source_row_index','time_paper_ns','success','relativeL2','rmse','forward_seconds','error'])
            writer.writerows(zip(range(len(t)),rows,t,success,relative,rmse,elapsed,errors))
        good=success&validobs&np.isfinite(relative)
        panel={'rows':len(t),'wavelength_points':len(wave),'success_count':int(success.sum()),'failed_count':int((~success).sum()),
               'comparison_valid_count':int(good.sum()),'relativeL2_mean':float(relative[good].mean()),
               'relativeL2_median':float(np.median(relative[good])),'relativeL2_min':float(relative[good].min()),
               'relativeL2_max':float(relative[good].max()),'rmse_mean':float(rmse[good].mean()),
               'forward_seconds_sum':float(elapsed.sum()),'panel_wall_seconds':time.perf_counter()-panelstart,
               'preparation':stats,'sources':{str(pp):sha(pp),str(dp):sha(dp)},
               'outputs':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
        summary['panels'][name]=panel
        (dest/'summary.json').write_text(json.dumps(panel,indent=2)+'\n')
        print(name,'complete',json.dumps({k:panel[k] for k in ['success_count','failed_count','relativeL2_mean','forward_seconds_sum']}),flush=True)
    frozen=json.loads(plan.read_text())
    changed={p:{'expected':s,'actual':sha(p)} for p,s in frozen['protected_sha256'].items() if sha(p)!=s}
    summary['protected_files_checked']=len(frozen['protected_sha256']);summary['protected_changed']=changed
    summary['total_wall_seconds']=time.perf_counter()-start
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if changed:raise AssertionError('Protected SHA changed: '+str(changed))
    print('COMPLETE',summary['total_wall_seconds'],'seconds; protected files unchanged',summary['protected_files_checked'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(args.smoke)
