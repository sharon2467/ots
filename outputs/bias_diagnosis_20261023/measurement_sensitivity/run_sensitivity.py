"""Bounded fixed-VDF controls: six PV calls and three existing convolutions only."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch
from scipy.signal import find_peaks, peak_widths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / 'OTS_code_share/NN_hotE_large_lmRange'
sys.path.insert(0, str(CODE))
from data import _physical_config_fig1_fig11
from phyEqns.ots_full_fourier import OTS_spectrum_PV
from spectral_fitting import _gaussian_convolve

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shape(x, y):
    # Each half's strongest local peak; half-prominence widths avoid assuming zero background.
    peaks, _ = find_peaks(y)
    chosen = []
    for mask in (x[peaks] < 263.3, x[peaks] > 263.3):
        subset = peaks[mask]
        if len(subset):
            chosen.append(int(subset[np.argmax(y[subset])]))
    if len(chosen) != 2:
        raise ValueError('Expected one dominant local peak on each side')
    widths, _, left, right = peak_widths(y, chosen, rel_height=.5)
    grid = np.arange(len(x))
    return {'blue_peak_nm': float(x[chosen[0]]), 'red_peak_nm': float(x[chosen[1]]),
            'peak_separation_nm': float(x[chosen[1]]-x[chosen[0]]),
            'blue_half_prominence_width_nm': float(np.interp(right[0],grid,x)-np.interp(left[0],grid,x)),
            'red_half_prominence_width_nm': float(np.interp(right[1],grid,x)-np.interp(left[1],grid,x)),
            'center_intensity': float(np.interp(263.3,x,y)),
            'area_nm': float(np.trapezoid(y,x))}
def compare(x, y, baseline):
    a,b = shape(x,y), shape(x,baseline)
    return {'spectral_relative_l2': float(np.linalg.norm(y-baseline)/np.linalg.norm(baseline)),
            'max_absolute_change_peak_units': float(np.max(np.abs(y-baseline))),
            'shape': a, 'shape_delta': {k:a[k]-b[k] for k in a},
            'half_prominence_width_relative_changes': {side: a[side+'_half_prominence_width_nm']/b[side+'_half_prominence_width_nm']-1 for side in ('blue','red')}}
def main():
    start=time.monotonic()
    source=HERE.parent/'representation_audit/controlled_arrays.npz'
    truth_json=HERE.parent/'representation_audit/controlled_results.json'
    protected=[source,truth_json,CODE/'data.py',CODE/'phyEqns/ots_full_fourier.py',CODE/'spectral_fitting.py']
    before={str(p):sha(p) for p in protected}
    data=np.load(source)
    config=_physical_config_fig1_fig11()
    x=data['wavelength_nm']
    assert np.array_equal(x,config['wavelength_bands_nm'][0].numpy())
    results={'scope':'Three standalone known-truth Gaussian states; fixed original positive VDFs, no fitting, NN, training, registry, D4 reevaluation or 26k-row replay.',
             'geometry':{'probe_nm':263.3,'scattering_angle_deg':90,'composition':'fully ionized C5H12; H:C number ratio 12:5'},
             'baseline_density_m3':3.5e26,'density_controls_m3':[2.8e26,5e26],
             'instrument_sensitivity_assumption':{'Gaussian_fwhm_nm':.057,'implementation':'existing spectral_fitting._gaussian_convolve, then divide by max(abs(signal))','limitation':'Sensitivity assumption, not an exact replication of the paper response. Paper simulations include spectral/time response and volume averaging; exact Gaussian response is not disclosed. No time response or volume averaging is evaluated here.'},
             'metric_notes':'Spectral L2 compares peak-normalized spectra on the same 319-point wavelength grid. Widths are dominant local peak half-prominence widths, not assumed isolated-line FWHM. Peak positions are grid sampled. Spectral differences cannot establish inverse-temperature error without a fit/inverse evaluation.',
             'states':[]}
    saved={'wavelength_nm':x}
    fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
    pv_count=convolution_count=0
    for idx,(te,ti) in enumerate([(100,60),(209,78),(1392,912)]):
        baseline=data[f'state{idx}_original_PV']
        kwargs={f'{prefix}_{s}':torch.from_numpy(data[f'state{idx}_{s}_{suffix}']) for s in ('e','i1','i2') for prefix,suffix in [('v','v'),('f','original')]}
        for s in ('e','i1','i2'):
            assert bool(torch.all(kwargs[f'f_{s}'] >= 0))
        rec={'state_index':idx,'Te_ev':te,'Ti_ev':ti,'baseline_shape':shape(x,baseline),'density_controls':[]}
        saved[f'state{idx}_baseline']=baseline
        ax=axs[idx]
        ax.plot(x,baseline,label='baseline 3.5e26',color='black')
        for density,color in [(2.8e26,'tab:blue'),(5e26,'tab:orange')]:
            y=OTS_spectrum_PV(torch.from_numpy(x)*1e-9,config['ss'],config['k_sh'],config['E_i0'],config['lm_0'],density,config['Z_i1'],config['N_i1'],config['m_i1'],config['Z_i2'],config['N_i2'],config['m_i2'],config['V'],**kwargs,u_e=config['u_e'],u_i1=config['u_i1'],u_i2=config['u_i2'],fourier_modes=config['fourier_modes']).numpy()
            pv_count+=1
            rec['density_controls'].append({'density_m3':density,**compare(x,y,baseline)})
            saved[f'state{idx}_density_{density:.1e}']=y
            ax.plot(x,y,label=f'ne={density/1e26:.1f}e26',color=color)
        broadened=_gaussian_convolve(x,baseline,.057)
        convolution_count+=1
        rec['instrument_peak_before_renormalization']=float(np.max(np.abs(broadened)))
        broadened=broadened/np.max(np.abs(broadened))
        rec['instrument_control']=compare(x,broadened,baseline)
        saved[f'state{idx}_gaussian057']=broadened
        ax.plot(x,broadened,'--',label='Gaussian 0.057 nm assumed',color='tab:green')
        ax.set_title(f'Te/Ti = {te}/{ti} eV')
        ax.set_xlim(263.3-.6,263.3+.6)
        ax.set_xlabel('Wavelength (nm)')
        results['states'].append(rec)
        print(te,ti,'density L2',[r['spectral_relative_l2'] for r in rec['density_controls']],'instrument L2',rec['instrument_control']['spectral_relative_l2'],flush=True)
    axs[0].set_ylabel('Peak-normalized intensity')
    axs[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE/'sensitivity.png',dpi=180)
    plt.close(fig)
    results['counts']={'new_PV_calls':pv_count,'existing_gaussian_convolutions':convolution_count,'torch_threads':torch.get_num_threads()}
    assert pv_count==6 and convolution_count==3
    after={str(p):sha(p) for p in protected}
    assert before==after
    results['protected_input_sha256']=before
    results['protected_inputs_unchanged']=True
    results['elapsed_seconds']=time.monotonic()-start
    results['interpretation']='Density controls quantify forward spectral shape changes only; they do not prove or exclude 30% inverse temperature errors. Compare spectral L2 magnitudes to the assumed instrument response, without equating them to temperature error.'
    np.savez_compressed(HERE/'spectra.npz',**saved)
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    artifacts=[HERE/'run_sensitivity.py',HERE/'results.json',HERE/'spectra.npz',HERE/'sensitivity.png']
    (HERE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in artifacts))
if __name__=='__main__':
    main()
