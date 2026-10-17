"""Controlled pilot improvements; keep raw data and the original pilot intact."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import find_peaks
import torch

import spectral_fitting as sf


def load_inputs(repo: Path) -> dict:
    baseline = repo/'outputs/ots_experimental_fits/pilot_20260914'
    config = json.loads((baseline/'config.json').read_text())
    manifest = json.loads((baseline/'manifest.json').read_text())
    hashes = manifest['input_and_code_hashes'].copy()
    for shot in manifest['shots']:
        for out in shot['outputs']:
            hashes[str(baseline/out['path'])] = out['sha256']
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Baseline/input/code changed: {path}')
    data = {}
    for settings in config['shots']:
        sid = settings['shot_id']
        with np.load(Path(config['input_dir'])/sid/'ordered_arrays.npz') as saved:
            arrays = {k:saved[k] for k in saved.files}
        rows = json.loads((baseline/sid/'fit_results.json').read_text())['rows']
        data[sid] = {'arrays':arrays, 'rows':rows, 'settings':settings,
                     'physics':{**config['physics'], 'instrument_fwhm_nm':settings['instrument_fwhm_nm']}}
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {'shots':data, 'config':config, 'hashes':hashes}


def signal_model(x: np.ndarray, nonlinear: np.ndarray, density: float, physics: dict) -> np.ndarray:
    te, th, tc, ui, ue = np.asarray(nonlinear, dtype=np.float64)
    theta = np.deg2rad(physics['scattering_angle_deg'])
    incident = np.array([1., 0., 0.])
    scattered = np.array([np.cos(theta), np.sin(theta), 0.])
    direction = scattered-incident
    direction /= np.linalg.norm(direction)
    fraction = float(physics['ion1_number_fraction'])
    z1, z2 = float(physics['ion1_charge']), float(physics['ion2_charge'])
    ne = float(density)*1e6
    ni = ne/(fraction*z1+(1-fraction)*z2)
    signal = sf.OTS_spectrum_maxwellian_two_ion(
        torch.as_tensor(x*1e-9, dtype=torch.float64), scattered, incident, np.array([0., 0., 1.]),
        float(physics['probe_wavelength_nm'])*1e-9, ne, te,
        z1, fraction*ni, float(physics['ion1_mass_amu'])*sf._ATOMIC_MASS_UNIT_KG, th,
        z2, (1-fraction)*ni, float(physics['ion2_mass_amu'])*sf._ATOMIC_MASS_UNIT_KG, tc, 1.,
        u_e=float(ue)*1e-2*direction, u_i1=float(ui)*1e-2*direction, u_i2=float(ui)*1e-2*direction,
    ).numpy()
    sigma = physics['instrument_fwhm_nm']/(2*np.sqrt(2*np.log(2))*(x[1]-x[0]))
    convolved = gaussian_filter1d(signal, sigma, mode='constant', truncate=8.)
    convolved /= gaussian_filter1d(np.ones_like(signal), sigma, mode='constant', truncate=8.)
    return convolved/np.max(np.abs(convolved))


def solve_linear(signal: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple:
    fraction = (x-x[0])/(x[-1]-x[0])
    design = np.column_stack([signal, 1-fraction, fraction])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if np.any(coefficients < sf._LOWER_BOUNDS[5:]) or np.any(coefficients > sf._UPPER_BOUNDS[5:]):
        coefficients = lsq_linear(design, y, bounds=(sf._LOWER_BOUNDS[5:],sf._UPPER_BOUNDS[5:]), method='bvls', tol=1e-12).x
    fitted = design@coefficients
    return coefficients, fitted, fitted-y


def fit_multistart(x: np.ndarray, y: np.ndarray, density: float, physics: dict,
                   starts: list, shared_ti: bool=False) -> dict:
    selected = np.array([0,1,3,4] if shared_ti else [0,1,2,3,4])
    expand = np.array([0,1,1,2,3] if shared_ti else [0,1,2,3,4])
    units = np.array([1000.,1000.,1000.,1e7,1e7])[selected]
    lower, upper = sf._LOWER_BOUNDS[selected]/units, sf._UPPER_BOUNDS[selected]/units
    candidates = []
    for i, start in enumerate(starts):
        begin = time.perf_counter()
        guess = np.clip(np.asarray(start)[selected]/units, lower+1e-7, upper-1e-7)
        try:
            result = least_squares(
                lambda z: solve_linear(signal_model(x, (z*units)[expand], density, physics), x, y)[2],
                guess, bounds=(lower,upper), jac='3-point', x_scale='jac',
                max_nfev=600, ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            nonlinear = (result.x*units)[expand]
            linear, fitted, residual = solve_linear(signal_model(x, nonlinear, density, physics), x, y)
            p = np.r_[nonlinear,linear]
            candidate = {'parameters':p.tolist(), 'sse':float(residual@residual),
                         'success':bool(result.success), 'nfev':result.nfev, 'optimality':result.optimality,
                         'termination':result.message, 'status':int(result.status)}
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            candidate = {'success':False, 'error':str(exc), 'parameters':None, 'sse':None}
        candidate.update(start=np.asarray(start).tolist(), index=i, elapsed_seconds=time.perf_counter()-begin)
        candidates.append(candidate)
    good = [c for c in candidates if c['parameters'] is not None and np.isfinite(c['sse'])]
    if not good:
        raise RuntimeError('All multistart fits failed')
    best = min(good, key=lambda c:c['sse'])
    p = np.array(best['parameters'])
    _, fitted, residual = solve_linear(signal_model(x,p[:5],density,physics),x,y)
    r = float(np.sqrt(np.mean(residual**2))/p[5]) if p[5]>1e-8 else None
    distance = np.minimum(p-sf._LOWER_BOUNDS,sf._UPPER_BOUNDS-p)/(sf._UPPER_BOUNDS-sf._LOWER_BOUNDS)
    flags = []
    if not best['success']: flags.append('optimizer_not_converged')
    if r is None: flags.append('negligible_amplitude')
    if np.any(distance<.001): flags.append('at_bound')
    if r is not None and r>.2: flags.append('large_residual')
    if p[5]<3*np.std(residual): flags.append('weak_signal')
    if any(c['index']!=best['index'] and c['sse']<=1.01*best['sse']+1e-12
           and np.any(np.abs(np.array(c['parameters'][:3])-p[:3])>.2*p[:3]) for c in good):
        flags.append('nonunique_temperature')
    return {'parameters':dict(zip(sf._PARAMETER_NAMES,p.tolist())), 'sse':best['sse'], 'R':r,
            'status':'review' if flags else 'ok', 'flags':flags, 'selected_candidate':best['index'],
            'bound_parameters':[k for k,d in zip(sf._PARAMETER_NAMES,distance) if d<.001],
            'candidates':candidates, 'fitted':fitted, 'residual':residual,
            'fit_seconds':sum(c['elapsed_seconds'] for c in candidates)}


def make_starts(x: np.ndarray, y: np.ndarray, previous: list, shared_ti: bool=False) -> list:
    initial = [list(p[:5]) for p in previous]
    smooth = gaussian_filter1d(y, .035/(x[1]-x[0]))
    peaks, _ = find_peaks(smooth, distance=max(1,int(.12/(x[1]-x[0]))), prominence=.02)
    peaks = sorted(peaks,key=lambda i:smooth[i],reverse=True)
    center = np.mean(x[peaks[:2]]) if len(peaks)>=2 else x[np.argmax(smooth)]
    ui = -sf.SPEED_OF_LIGHT*100*(center-263.3)/(263.3*np.sqrt(2))
    for te,th,tc in ([200,80,120],[900,350,600],[1500,800,1200],[2000,2400,3300]):
        for drift in [ui, 0.]:
            initial.append([te,th,tc,drift,drift])
    for delta in [-1e8,1e8]:
        initial.append([1400,800,1200,ui,ui+delta])
    # Fixed seed covers asymmetric widths without using the paper answer as a target.
    rng = np.random.default_rng(20260915)
    for _ in range(4):
        initial.append([rng.uniform(200,3000),rng.uniform(60,4000),rng.uniform(60,4000),ui+rng.uniform(-1e7,1e7),ui+rng.uniform(-1e8,1e8)])
    out=[]
    for p in initial:
        p=np.clip(p,sf._LOWER_BOUNDS[:5]+1e-4,sf._UPPER_BOUNDS[:5]-1e-4).tolist()
        if shared_ti: p[2]=p[1]
        if p not in out:out.append(p)
    return out


def compare_case(case: dict, settings: dict) -> dict:
    arrays=settings['arrays']; old=next((r for r in settings['rows'] if r['source_row_index']==case['source_row_index']),None)
    i=int(np.flatnonzero(arrays['source_row_index']==case['source_row_index'])[0])
    axis=arrays['wavelength_nm']+case.get('wavelength_reference_shift_nm',0.)
    columns=np.flatnonzero((axis>=case['wavelength_range_nm'][0])&(axis<=case['wavelength_range_nm'][1]))
    x=axis[columns]; original=arrays['intensity'][i,columns].astype(float)
    if len(x)<8 or not np.all(np.isfinite(original)):
        raise ValueError('A finite spectrum with at least eight points is required')
    scale=float(np.max(np.abs(original)))
    if scale<=0:
        raise ValueError('Cannot normalize an all-zero spectrum')
    y=original/scale
    np.testing.assert_allclose(scale*y,original,rtol=1e-14,atol=1e-12)
    previous=[[c['parameters'][k] for k in sf._PARAMETER_NAMES] for c in old['candidates'] if c['status']=='success'] if old else []
    previous += case.get('additional_starts',[])
    starts=make_starts(x,y,previous,case.get('shared_ti',False))
    if 'max_starts' in case: starts=starts[:case['max_starts']]
    fit=fit_multistart(x,y,case['electron_density_cm3'],settings['physics'],starts,case.get('shared_ti',False))
    fit.update(case=case, time_fig_ns=float(arrays['time_fig_ns'][i]), scale=scale,
               source_row_index=case['source_row_index'], source_col_index=arrays['source_col_index'][columns],
               wavelength_nm=x, source_wavelength_nm=arrays['wavelength_nm'][columns], measured=y, baseline=old)
    return fit


def write_comparison(root: Path, cases: list, provenance: dict) -> None:
    root.mkdir(parents=True,exist_ok=True)
    for fit in cases:
        cid=fit['case']['case_id'];dest=root/cid;dest.mkdir(exist_ok=True)
        excluded=('fitted','residual','source_col_index','wavelength_nm','source_wavelength_nm','measured')
        (dest/'result.json').write_text(json.dumps({k:v for k,v in fit.items() if k not in excluded},indent=2,allow_nan=False)+'\n')
        np.savez_compressed(dest/'spectra.npz',**{k:fit[k] for k in excluded},scale=np.array(fit['scale']),source_row_index=np.array(fit['source_row_index']))
        saved_json=json.loads((dest/'result.json').read_text())
        with np.load(dest/'spectra.npz',allow_pickle=False) as saved:
            for key in excluded:
                np.testing.assert_array_equal(saved[key],fit[key])
            assert saved['scale'].item()==saved_json['scale']==fit['scale']
            assert saved['source_row_index'].item()==saved_json['source_row_index']
            np.testing.assert_array_equal(saved['residual'],saved['fitted']-saved['measured'])
            np.testing.assert_allclose(saved['residual']@saved['residual'],saved_json['sse'],rtol=1e-12,atol=1e-15)
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
        axes[0].plot(fit['wavelength_nm'],fit['measured'],color='.45',lw=.65,label='Measured')
        axes[0].plot(fit['wavelength_nm'],fit['fitted'],color='#cf471d',lw=1.4,label='Improved fit')
        axes[1].plot(fit['wavelength_nm'],fit['residual'],color='#1565c0',lw=.65)
        axes[1].axhline(0,color='.5',lw=.5)
        axes[0].legend();axes[0].set_ylabel('Normalized intensity');axes[1].set_ylabel('Fit - measured')
        axes[1].set_xlabel('Wavelength (nm)')
        r_text='undefined' if fit['R'] is None else f"{fit['R']:.4g}"
        fig.suptitle(f"{cid} | t_FIG={fit['time_fig_ns']:.4f} ns | R={r_text}\n{fit['status']}: {', '.join(fit['flags'])}")
        fig.savefig(dest/'fit.png',dpi=140);plt.close(fig)
        provenance.setdefault('output_hashes',{}).update({str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                                       for path in (dest/'result.json',dest/'spectra.npz',dest/'fit.png')})
    summaries={c['case_id']:c for c in provenance.get('cases',[])}
    summaries.update({f['case']['case_id']:{'case_id':f['case']['case_id'],'sse':f['sse'],'R':f['R'],'status':f['status'],'flags':f['flags'],'fit_seconds':f['fit_seconds']} for f in cases})
    provenance['cases']=list(summaries.values())
    (root/'manifest.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('Output must be new')
    torch.set_num_threads(1)
    inputs=load_inputs(Path(__file__).resolve().parents[2])
    if args.validate:
        errors=[]
        x=np.linspace(261.3,265.3,925);physics=inputs['shots']['20160909153']['physics']
        for p in ([200,80,120,-1e6,2e6,.8,-.06,.04],[1200,600,900,1e6,-1e7,.8,.04,-.06],[2500,2200,3300,3e7,8e7,.8,-.06,.04]):
            p=np.array(p,dtype=float);signal=signal_model(x,p[:5],6e20,physics)
            fraction=(x-x[0])/(x[-1]-x[0]);fast=p[5]*signal+(1-fraction)*p[6]+fraction*p[7]
            old=sf._fit_model(p,x,6e20,**physics)
            eq=float(np.linalg.norm(old-fast)/np.linalg.norm(old));assert eq<1e-10,eq
            synthetic=fit_multistart(x,fast,6e20,physics,[(p[:5]*np.array([1.03,1.03,1.03,1,1])).tolist()])
            error=float(np.linalg.norm(synthetic['fitted']-fast)/np.linalg.norm(fast));assert error<1e-5
            errors.append({'equivalence_relative_l2':eq,'synthetic_relative_l2':error})
            print(errors[-1],flush=True)
        args.output.mkdir(parents=True)
        (args.output/'validation.json').write_text(json.dumps({'status':'passed','checks':errors},indent=2)+'\n')
        return 0
    cases=json.loads(args.cases.read_text()) if args.cases else [
        {'case_id':f"{sid}_{r['source_row_index']}_same_window", 'shot_id':sid,'source_row_index':r['source_row_index'],
         'wavelength_range_nm':[261.3,265.3], 'electron_density_cm3':s['settings']['electron_density_cm3']}
        for sid,s in inputs['shots'].items() for r in s['rows']]
    if not cases or len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Case IDs must be nonempty and unique')
    for case in cases:
        if not case['case_id'] or Path(case['case_id']).name!=case['case_id'] or case['case_id'] in ('.','..'):
            raise ValueError('Each case ID must be a single directory name')
    args.output.mkdir(parents=True)
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    provenance={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'input_hashes':inputs['hashes'],
                'full_run':False,'cases_config':cases,'source_config':inputs['config'],
                'runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                           'scipy':scipy.__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads()}}
    write_comparison(args.output,[],provenance)
    done=[];started=time.perf_counter()
    try:
        for case in cases:
            print('START',case['case_id'],flush=True)
            fit=compare_case(case,inputs['shots'][case['shot_id']]);done.append(fit)
            write_comparison(args.output,[fit],provenance)
            print('DONE',case['case_id'],fit['sse'],fit['R'],fit['flags'],list(fit['parameters'].values())[:5],flush=True)
        for path,digest in inputs['hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Input/code changed during run: {path}')
    except Exception as exc:
        provenance.update(status='failed',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write_comparison(args.output,[],provenance)
        raise
    provenance.update(status='completed',elapsed_seconds=time.perf_counter()-started)
    write_comparison(args.output,[],provenance)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
