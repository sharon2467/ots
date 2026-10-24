"""Small deterministic forward-model probes; no training or production edits."""
import sys, os, json, math, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
sys.path.insert(0, '/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange')
import data
from evaluation import reconstruct_real_evdf
from phyEqns import ots_full_fourier as phy
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cfg = data._physical_config_fig1_fig11()
forward_times=[]
curves={}
species = ['e', 'i1', 'i2']
blocks = dict(data.SPECIES_BLOCKS)
def vdf(s, t, u=0., p=2.):
    b=cfg[blocks[s]]
    v=b['velocity_grid_m_s']
    sigma=math.sqrt(data.ELEMENTARY_CHARGE*t/b['mass_kg'])
    std=math.sqrt(math.gamma(3/p)/math.gamma(1/p))
    f=torch.exp(-torch.abs((v-u)/sigma*std).pow(p))
    return v, data.normalize_distribution(v,f,s)
def spectrum(ts, ps=(2.,2.,2.), us=(0.,0.,0.), ne=4e26):
    ds=[vdf(s,t,u,p) for s,t,u,p in zip(species,ts,us,ps)]
    kwargs={}
    for s,(v,f) in zip(species,ds):
        kwargs['v_'+s]=v
        kwargs['f_'+s]=f
    tick=time.perf_counter()
    y=phy.OTS_spectrum_PV(cfg['wavelength_bands_nm'][0]*1e-9,
        *[cfg[k] for k in ('ss','k_sh','E_i0','lm_0')],ne,
        *[cfg[k] for k in ('Z_i1','N_i1','m_i1','Z_i2','N_i2','m_i2','V')],
        **kwargs, u_e=cfg['u_e'],u_i1=cfg['u_i1'],u_i2=cfg['u_i2'],fourier_modes=80)
    forward_times.append(time.perf_counter()-tick)
    return y,ds
def rel(a,b): return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))
def coef(v,f):
    k,F=phy.ft_x_n(v,f,80)
    return k,torch.cat((F.real,F.imag))
result={'scope':'Deterministic, clean single-cell PV spectra. Sensitivity is numerical local evidence, not global uniqueness proof.', 'torch_threads':torch.get_num_threads(), 'same_temperature_shape':[], 'equal_static_screening':[], 'fourier_roundtrip':[]}
for name,ts,ne in [('fig1a',[209.,78.,78.],3.5e26),('fig1b',[1392.,912.,912.],4e26)]:
    y0,ds0=spectrum(ts,ne=ne)
    curves[name+'_iaw_gaussian']=y0.numpy()
    curves[name+'_electron_velocity']=ds0[0][0].numpy()
    curves[name+'_electron_gaussian']=ds0[0][1].numpy()
    for j,s in enumerate(species):
        ps=[2.,2.,2.];ps[j]=4.
        yy,ds=spectrum(ts,ps,ne=ne)
        v,f0=ds0[j];f=ds[j][1]
        _,c0=coef(v,f0);_,c=coef(v,f)
        result['same_temperature_shape'].append({'anchor':name,'changed_species':s,'spectrum_relative_l2':rel(y0,yy),'spectrum_max_abs_change':float((y0-yy).abs().max()),'raw_vdf_relative_l2':rel(f0,f),'coefficient_relative_l2':rel(c0,c)})
        if s=='e':
            curves[name+'_iaw_same_temperature_sg']=yy.numpy()
            curves[name+'_electron_same_temperature_sg']=f.numpy()
    ve,fe0=ds0[0]
    response0=phy._pv_fd_landau_response(ve,fe0,torch.tensor([0.],dtype=torch.float64),name='probe').real.item()
    def screen_diff(t):
        v,f=vdf('e',t,p=4.)
        return phy._pv_fd_landau_response(v,f,torch.tensor([0.],dtype=torch.float64),name='probe').real.item()/response0-1
    tmatch=brentq(screen_diff,.25*ts[0],.8*ts[0],xtol=1e-10)
    yy,ds=spectrum([tmatch,*ts[1:]],ps=(4.,2.,2.),ne=ne)
    _,c0=coef(ve,fe0);_,c=coef(ve,ds[0][1])
    result['equal_static_screening'].append({'anchor':name,'gaussian_Te_ev':ts[0],'supergaussian_p4_Te_ev':tmatch,'spectrum_relative_l2':rel(y0,yy),'spectrum_max_abs_change':float((y0-yy).abs().max()),'raw_e_vdf_relative_l2':rel(fe0,ds[0][1]),'electron_coefficient_relative_l2':rel(c0,c),'static_response_relative_difference':screen_diff(tmatch)})
    curves[name+'_iaw_equal_response_sg']=yy.numpy()
    curves[name+'_electron_equal_response_sg']=ds[0][1].numpy()
for s in species:
    temps=([89.288,209.,1392.,2426.953] if s=='e' else [33.55,78.,912.,1641.6])
    for t in temps:
        for p in [2.,4.]:
            v,f=vdf(s,t,p=p)
            k,c=coef(v,f)
            _,ff=reconstruct_real_evdf(c[None],k,velocity=v)
            ff=ff[0]
            result['fourier_roundtrip'].append({'species':s,'T_ev':t,'p':p,'raw_vdf_relative_l2':rel(f,ff),'reconstruction_mass':float(torch.trapz(ff,v)),'negative_mass_fraction':float(torch.trapz(torch.clamp(-ff,min=0),v)/torch.trapz(ff.abs(),v))})
# Finite-difference sensitivity under explicitly stated physical scales.
base=np.array([1392.,912.,912.,0.,0.,0.,4e26])
scales=np.array([1392.,912.,912.,3e5,5e4,5e4,2.2e26])
def unpack(x): return spectrum(x[:3],us=x[3:6],ne=x[6])[0]
y0=unpack(base)
cols=[]
for j in range(7):
    d=np.zeros(7);d[j]=scales[j]*1e-3
    cols.append(((unpack(base+d)-unpack(base-d))/(2e-3)).numpy())
J=np.stack(cols,axis=1)
u,sv,vh=np.linalg.svd(J,full_matrices=False)
result['local_svd']={'parameter_order':['Te','TH','TC','ue','uH','uC','ne'],'parameter_scales':scales.tolist(),'singular_values':sv.tolist(),'condition_number':float(sv[0]/sv[-1]),'weakest_scaled_direction':vh[-1].tolist(),'one_percent_step_spectral_relative_norm':[float(np.linalg.norm(J[:,j]*.01)/torch.linalg.vector_norm(y0)) for j in range(7)]}
# Exact measured phase-speed window, with refraction as used in PV.
c=data.SPEED_OF_LIGHT; lm=cfg['wavelength_bands_nm'][0]*1e-9
w0=2*math.pi*c/cfg['lm_0'];w=2*math.pi*c/lm
wp2=4e26*data.ELEMENTARY_CHARGE**2/(data.ELECTRON_MASS*8.854187817e-12)
kmag=torch.sqrt((w0*w0-wp2)+(w*w-wp2))/c
vp=(w-w0)/kmag
result['phase_velocity_window_m_s']=[float(vp.min()),float(vp.max())]
result['electron_thermal_speed_range_m_s']=[math.sqrt(data.ELEMENTARY_CHARGE*t/data.ELECTRON_MASS) for t in (89.288,2426.953)]
iaw_axis=cfg['wavelength_bands_nm'][0]
curves['iaw_wavelength_nm']=iaw_axis.numpy()
result['extra_electron_bands']=[]
for name,ts,ne in [('fig1a',[209.,78.,78.],3.5e26),('fig1b',[1392.,912.,912.],4e26)]:
    tmatch=next(r['supergaussian_p4_Te_ev'] for r in result['equal_static_screening'] if r['anchor']==name)
    for band,(lo,hi) in [('blue',(200.,250.)),('red',(280.,350.))]:
        axis=torch.linspace(lo,hi,181,dtype=torch.float64)
        cfg['wavelength_bands_nm']=[axis]
        yy,_=spectrum(ts,ne=ne)
        matched,_=spectrum([tmatch,*ts[1:]],ps=(4.,2.,2.),ne=ne)
        same,_=spectrum(ts,ps=(4.,2.,2.),ne=ne)
        result['extra_electron_bands'].append({'anchor':name,'band':band,'wavelength_range_nm':[lo,hi],'points':181,'independent_peak_normalization':True,'matched_static_response_spectrum_relative_l2':rel(yy,matched),'same_temperature_spectrum_relative_l2':rel(yy,same),'gaussian_peak_wavelength_nm':float(axis[yy.argmax()]),'matched_peak_wavelength_nm':float(axis[matched.argmax()])})
        curves[name+'_'+band+'_gaussian']=yy.numpy()
        curves[name+'_'+band+'_equal_response_sg']=matched.numpy()
        curves[name+'_'+band+'_same_temperature_sg']=same.numpy()
        curves[band+'_wavelength_nm']=axis.numpy()
cfg['wavelength_bands_nm']=[iaw_axis]
cfg['points_csv']='/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020/paper_experimental_points.csv'
rng=np.random.default_rng(20261020)
sampler=data.Fig1_Fig11_sampler(rng,cfg)
sample_times=[]
for _ in range(5):
    tick=time.perf_counter()
    data.generate_sample(rng,cfg,sampler)
    sample_times.append(time.perf_counter()-tick)
result['timing_seconds']={'forward_n':len(forward_times),'forward_median':float(np.median(forward_times)),'forward_min':min(forward_times),'full_generate_sample_times':sample_times,'full_generate_sample_median':float(np.median(sample_times)),'note':'One torch/BLAS thread; measured concurrently with two parent baseline jobs, so includes resource contention. Excludes imports/Parquet writing.'}
out=Path(__file__).with_name('physics_probe.json')
out.write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(Path(__file__).with_name('physics_probe_curves.npz'),**curves)
print(json.dumps(result,indent=2))
