"""Independent recomputation; writes only this verification directory."""
import ast,csv,hashlib,json,os,sys,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
REPO=BASE.parents[1]
SP=('e','H','C')
MASS={'e':9.1093837015e-31,'H':1.67262192369e-27,'C':12*1.67262192369e-27}
Q=1.602176634e-19
T=np.trapezoid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,rtol=2e-12,atol=1e-10):
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)
def readcsv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def moment(v,f,m):
    a=T(f,v,axis=-1);u=T(f*v,v,axis=-1)/a
    t=m/Q*(T(f*v*v,v,axis=-1)/a-u*u)
    return a,u,t

def main(forward=False):
    report={'scope':'Independent all-row array recomputation; original files read only','forward_checked':forward}
    plan=json.loads((BASE/'analysis_plan.json').read_text())
    changed=[p for p,s in plan['protected_sha256'].items() if sha(p)!=s]
    assert not changed,changed
    registries={}
    for root,saved in plan['registry_snapshots'].items():
        root=Path(root);actual={str(p.relative_to(root)):os.readlink(p) for p in root.rglob('*') if p.is_symlink()}
        assert actual==saved
        registries[str(root)]=len(actual)
    report['protected']={'files_checked':len(plan['protected_sha256']),'changed':changed,'registry_links':registries}
    constants={}
    for node in ast.parse((REPO/'OTS_code_share/NN_hotE_large_lmRange/data.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for name in node.targets:
                if isinstance(name,ast.Name) and name.id in ('ELEMENTARY_CHARGE','ELECTRON_MASS','PROTON_MASS'):constants[name.id]=ast.literal_eval(node.value)
    assert constants=={'ELEMENTARY_CHARGE':Q,'ELECTRON_MASS':MASS['e'],'PROTON_MASS':MASS['H']}
    report['constants']=constants;report['moments']={};total=0
    for panel in ('fig10a','fig10b','fig10c'):
        raw=np.load(REPO/f'outputs/fig10_nearest_d4_inference_20261021/results/{panel}/predictions.npz')
        saved=np.load(BASE/f'moments/{panel}_signed_moments.npz');result={}
        assert np.array_equal(raw['time_paper_ns'],saved['time_paper_ns'])
        assert np.array_equal(raw['source_row_index'],saved['source_row_index'])
        for sp in SP:
            v=raw[sp+'_velocity_m_s'].astype('float64');f=raw[sp+'_vdf'].astype('float64')
            assert v.shape==(4097,) and f.shape==(len(raw['time_paper_ns']),4097)
            a,u,t=moment(v,f,MASS[sp]);close(a,saved[sp+'_mass_integral']);close(u,saved[sp+'_u_m_s']);close(t,saved[sp+'_T_eV'])
            close(-T(np.minimum(f,0),v,axis=-1)/T(abs(f),v,axis=-1),saved[sp+'_negative_mass_fraction'])
            result[sp]={'rows':len(t),'temperature_range_eV':[float(t.min()),float(t.max())],'negative_temperature_rows':int((t<0).sum()),'max_recomputed_temperature_error_eV':float(abs(t-saved[sp+'_T_eV']).max())}
            total+=len(t)
        report['moments'][panel]=result
    report['total_species_rows']=total
    refs=readcsv(BASE/'references/fig11_experimental_points.csv')
    source=readcsv(REPO/'outputs/training_study_20261020/paper_experimental_points.csv')
    expected=[r for r in source if r['figure']=='fig11' and r['shot_id']=='20170811020']
    assert len(refs)==len(expected)
    for a,b in zip(refs,expected):
        assert set(a)==set(b)
        for k in a:
            try: close(float(a[k]),float(b[k]),atol=1e-10)
            except ValueError: assert a[k]==b[k]
    counts={p:sum(r['parameter']==p for r in refs) for p in ('T_e_ev','T_i_ev')};assert counts=={'T_e_ev':27,'T_i_ev':27}
    metrics=json.loads((BASE/'moments_comparison.json').read_text())['experimental_errors']
    rows=readcsv(BASE/'moments/signed_moments_all_rows.csv');errors={}
    for sp in SP:
        rs=[r for r in rows if r['panel']=='fig10a' and r['species']==sp]
        rr=[r for r in refs if r['parameter']==('T_e_ev' if sp=='e' else 'T_i_ev')]
        time=np.array([float(r['time_paper_ns']) for r in rs]);val=np.array([float(r['T_eV']) for r in rs])
        rt=np.array([float(r['time_ns']) for r in rr]);truth=np.array([float(r['value']) for r in rr]);assert rt.min()>=time.min() and rt.max()<=time.max()
        d=np.interp(rt,time,val)-truth
        calculated={'n':len(rr),'MAE_eV':abs(d).mean(),'RMSE_eV':np.sqrt((d*d).mean()),'bias_eV':d.mean(),'mean_absolute_relative_difference':(abs(d)/truth).mean()}
        for k,x in calculated.items():close(x,metrics[sp][k])
        errors[sp]={k:float(x) for k,x in calculated.items()}
    report['experimental']={'counts':counts,'metrics':errors,'ion_reference':'Same published Ti compared to H and C separately; no separate species truth'}
    audit=json.loads((BASE/'references/reference_audit.json').read_text());assert audit['simulation_raw_numeric_data_found'] is False
    from PIL import Image
    image=np.array(Image.open(BASE/'references/page10_image-001.png').convert('RGB'))
    envelope=readcsv(BASE/'references/fig11_simulation_ink_envelopes.csv')
    expected_envelope={}
    for parameter,x0,x1,maximum in [('T_e_ev',114,595,2500),('T_i_ev',775,1256,2000)]:
        for condition,channel in [('without_probe',2),('with_probe',0)]:
            other=[c for c in range(3) if c!=channel]
            mask=(image[:,:,channel]>220)&(image[:,:,other[0]]<90)&(image[:,:,other[1]]<90)
            mask[:13]=False;mask[474:]=False;mask[35:181,x0+28:x0+294]=False
            for x in range(x0+1,x1):
                y=np.flatnonzero(mask[:,x])
                if len(y):expected_envelope[(parameter,condition,x)]=(int(y.min()),int(y.max()),1.5+(x-x0)*3/481,(474-y.max())*maximum/462,(474-y.min())*maximum/462)
    assert len(envelope)==len(expected_envelope)
    for row in envelope:
        key=(row['parameter'],row['condition'],int(row['pixel_x']))
        low,high,time,evlow,evhigh=expected_envelope.pop(key)
        close([low,high,time,evlow,evhigh],[float(row[k]) for k in ['pixel_y_ink_top','pixel_y_ink_bottom','time_ns','ink_envelope_low_ev','ink_envelope_high_ev']])
        assert row['both_physical_bounds_certified']=='False'
    assert not expected_envelope
    report['simulation_reference']={'numeric_truth_available':False,'independent_native_ink_rows_checked':len(envelope),'limitations':audit['limitations']}
    if forward:verify_forward(report)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

def verify_forward(report):
    import torch
    sys.path.insert(0,str(REPO/'OTS_code_share/NN_hotE_large_lmRange'))
    from data import _physical_config_fig1_fig11
    from phyEqns.ots_full_fourier import OTS_spectrum_PV
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=_physical_config_fig1_fig11();cfg['n_e']=np.float64(3.5e26)
    for k in ['lm_0','n_e','Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V']:cfg[k]=np.float64(cfg[k])
    for k in ['ss','k_sh','E_i0','u_e','u_i1','u_i2']:cfg[k]=torch.as_tensor(cfg[k],dtype=torch.float64)
    assert all(torch.equal(cfg[k],torch.zeros_like(cfg[k])) for k in ['u_e','u_i1','u_i2'])
    summary=json.loads((BASE/'forward/summary.json').read_text());assert summary['config']['n_e']==3.5e26
    report['forward']={}
    for panel in ('fig10a','fig10b','fig10c'):
        raw=np.load(REPO/f'outputs/fig10_nearest_d4_inference_20261021/results/{panel}/predictions.npz')
        native=np.load(REPO/f'outputs/fig10_nearest_row_fill_20261021/datasets/{panel}.npz')
        proc=np.load(BASE/f'forward/{panel}/processed_vdfs.npz');saved=np.load(BASE/f'forward/{panel}/forward_results.npz')
        for key in ('time_paper_ns','source_row_index','wavelength_nm'):assert np.array_equal(saved[key],native[key])
        assert np.array_equal(saved['observed'],native['intensity'])
        assert np.array_equal(saved['wavelength_nm'],raw['native_wavelength_nm']) and np.array_equal(saved['observed'],raw['native_intensity'])
        preparation={}
        for sp in SP:
            oldv=raw[sp+'_velocity_m_s'].astype('float64');oldf=raw[sp+'_vdf'].astype('float64')
            v=np.linspace(oldv[0],oldv[-1],4097,dtype='float64');assert np.array_equal(v,proc[sp+'_velocity_m_s'])
            interp=np.stack([np.interp(v,oldv,f) for f in oldf]);clip=np.maximum(interp,0);norm=T(clip,v,axis=-1);f=clip/norm[:,None]
            close(f,proc[sp+'_vdf'],atol=1e-25);assert np.isfinite(f).all() and (f>=0).all();close(T(f,v,axis=-1),1)
            original=moment(oldv,oldf,MASS[sp]);processed=moment(v,f,MASS[sp])
            for name,x,y in zip(('integral','u_m_s','T_eV'),original,processed):
                close(x,proc[sp+'_original_'+name],atol=1e-9);close(y,proc[sp+'_processed_'+name],atol=1e-9)
            relative=np.linalg.norm(f-oldf,axis=-1)/np.linalg.norm(oldf,axis=-1);close(relative,proc[sp+'_total_preparation_relativeL2'])
            preparation[sp]={'temperature_delta_range_eV':[float((processed[2]-original[2]).min()),float((processed[2]-original[2]).max())],'max_relativeL2':float(relative.max()),'max_grid_displacement_m_s':float(abs(v-oldv).max())}
        obs=native['intensity'].astype('float64');scale=np.max(abs(obs),axis=1);assert (scale>0).all();norm=obs/scale[:,None]
        close(norm,saved['observed_normalized']);close(scale,saved['observed_peak_scale']);res=saved['predicted_PV']-norm;relative=np.linalg.norm(res,axis=1)/np.linalg.norm(norm,axis=1)
        close(res,saved['residual']);close(relative,saved['relativeL2']);close(np.sqrt((res*res).mean(axis=1)),saved['rmse'])
        assert saved['success'].all() and np.isfinite(saved['predicted_PV']).all();close(np.max(abs(saved['predicted_PV']),axis=1),1.)
        reps=[]
        for target in (2.,3.,4.):
            row=int(np.argmin(abs(saved['time_paper_ns']-target)));args=[torch.tensor(saved['wavelength_nm'].astype('float64')*1e-9)]
            args.extend(cfg[k] for k in ['ss','k_sh','E_i0','lm_0','n_e','Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V'])
            kwargs={'fourier_modes':80}
            for sp,key in zip(SP,('e','i1','i2')):
                kwargs['v_'+key]=torch.from_numpy(proc[sp+'_velocity_m_s']);kwargs['f_'+key]=torch.from_numpy(proc[sp+'_vdf'][row]);kwargs['u_'+key]=cfg['u_'+key]
            with torch.no_grad():y=OTS_spectrum_PV(*args,**kwargs).numpy()
            close(y,saved['predicted_PV'][row],atol=1e-12)
            reps.append({'row':row,'time_ns':float(saved['time_paper_ns'][row]),'max_PV_absolute_difference':float(abs(y-saved['predicted_PV'][row]).max())})
        report['forward'][panel]={'rows':len(obs),'success_count':int(saved['success'].sum()),'all_spectra_finite':True,'mean_relativeL2':float(relative.mean()),'preparation':preparation,'independent_existing_public_PV_recomputations':reps}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forward',action='store_true');main(p.parse_args().forward)
