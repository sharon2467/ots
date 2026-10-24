#!/usr/bin/env python3
"""Full-grid algebraic moments of unchanged signed predicted VDF arrays."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / 'OTS_code_share/NN_hotE_large_lmRange/data.py'
INPUT = ROOT / 'outputs/fig10_nearest_d4_inference_20261021/results'
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # Parse literal constants without importing data.py or invoking a physical model.
    constants = {}
    for node in ast.parse(DATA.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ['ELEMENTARY_CHARGE', 'ELECTRON_MASS', 'PROTON_MASS']:
                    constants[target.id] = ast.literal_eval(node.value)
    masses = {'e': constants['ELECTRON_MASS'], 'H': constants['PROTON_MASS'], 'C': 12.0 * constants['PROTON_MASS']}
    charge = constants['ELEMENTARY_CHARGE']
    summary = {'meaning': 'Signed algebraic 1D full-grid moments; not validated physical VDF temperatures.',
               'formula': 'A=trapz(f,v); u=trapz(v*f,v)/A; T=m/e*trapz((v-u)^2*f,v)/A; no factor 1/3',
               'policy': 'Saved float32 arrays converted exactly to float64; all 4097 points; no clipping, smoothing, grid change, Maxwellian fit or renormalization.',
               'diagnostics': {'negative_mass_fraction': 'integral(max(-f,0))/integral(abs(f))',
                               'negative_raw_second_fraction_abs': 'abs(integral(v^2*min(f,0)))/integral(v^2*abs(f))',
                               'negative_region_integration': 'Trapezoids over full original grid with zero outside f<0; no removal of grid points.'},
               'constants': constants, 'data_py_sha256': sha(DATA), 'script_sha256': sha(Path(__file__)), 'panels': {}}
    all_rows = []
    for panel in ['fig10a', 'fig10b', 'fig10c']:
        path = INPUT / panel / 'predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            time = z['time_paper_ns'].copy(); source = z['source_row_index'].copy()
            saved = {'time_paper_ns': time, 'source_row_index': source}
            ps = {'input': str(path.relative_to(ROOT)), 'input_sha256': sha(path), 'rows': len(time), 'species': {}}
            for sp, mass in masses.items():
                v = z[sp + '_velocity_m_s'].astype(np.float64)
                f = z[sp + '_vdf'].astype(np.float64)
                assert v.shape == (4097,) and f.shape == (len(time),4097)
                assert np.all(np.diff(v)>0) and np.isfinite(f).all() and np.isfinite(v).all()
                A = TRAPZ(f,v,axis=-1)
                first = TRAPZ(f*v,v,axis=-1)
                u = first/A
                second = TRAPZ(f*v*v,v,axis=-1)
                centered = TRAPZ(f*(v-u[:,None])**2,v,axis=-1)
                T = mass/charge*centered/A
                Traw = mass/charge*(second/A-u*u)
                # Masking is diagnostic decomposition only: original f remains unchanged.
                fneg = np.minimum(f,0.0); fpos = np.maximum(f,0.0)
                neg = -TRAPZ(fneg,v,axis=-1); pos = TRAPZ(fpos,v,axis=-1)
                negsecond = TRAPZ(fneg*v*v,v,axis=-1)
                possecond = TRAPZ(fpos*v*v,v,axis=-1)
                negTraw = mass/charge*negsecond/A
                frac = neg/(pos+neg)
                frac2 = -negsecond/(possecond-negsecond)
                err = np.abs(T-Traw)
                relerr = err/np.maximum(np.abs(T), np.abs(Traw))
                assert np.isfinite(T).all() and np.isfinite(u).all()
                assert np.allclose(T,Traw,rtol=2e-12,atol=1e-9)
                assert np.allclose(A,pos-neg,rtol=2e-12,atol=1e-12)
                assert np.allclose(second,possecond+negsecond,rtol=2e-12,atol=1e-3)
                arrays = {'mass_integral':A,'u_m_s':u,'u_km_s':u/1000,'T_eV':T,'T_raw_formula_eV':Traw,
                          'negative_mass_integral':neg,'positive_mass_integral':pos,'negative_mass_fraction':frac,
                          'raw_first_moment_m_s':first,'raw_second_moment_m2_s2':second,
                          'negative_raw_second_moment_m2_s2':negsecond,
                          'negative_raw_second_T_contribution_eV':negTraw,
                          'negative_raw_second_fraction_abs':frac2,'centered_raw_abs_difference_eV':err}
                saved.update({sp+'_'+k:val for k,val in arrays.items()})
                reps = []
                for target in [2,3,4]:
                    j = int(np.argmin(abs(time-target)))
                    reps.append({'requested_time_ns':target, 'row_index':j, 'source_row_index':int(source[j]), 'time_paper_ns':float(time[j]), **{k:float(val[j]) for k,val in arrays.items()}})
                ss = {'mass_kg':mass, 'velocity_points':4097, 'velocity_min_m_s':float(v.min()), 'velocity_max_m_s':float(v.max()),
                      'negative_T_count':int(np.sum(T<0)), 'nonfinite_T_count':int(np.sum(~np.isfinite(T))),
                      'centered_raw_max_absolute_difference_eV':float(err.max()), 'centered_raw_max_relative_difference':float(relerr.max()),
                      'audit_pass':True, 'representative_nearest':reps}
                for key,val in arrays.items():
                    jmin=int(np.argmin(val));jmax=int(np.argmax(val))
                    ss[key]={'min':float(val[jmin]),'max':float(val[jmax]),'min_row_index':jmin,'max_row_index':jmax,
                             'min_time_ns':float(time[jmin]),'max_time_ns':float(time[jmax])}
                ps['species'][sp] = ss
                for j in range(len(time)):
                    all_rows.append({'panel':panel,'species':sp,'row_index':j,'source_row_index':int(source[j]),'time_paper_ns':float(time[j]),'mass_kg':mass,
                                     **{k:float(val[j]) for k,val in arrays.items()}})
            np.savez_compressed(OUT/(panel+'_signed_moments.npz'),**saved)
            summary['panels'][panel] = ps
    with (OUT/'signed_moments_all_rows.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    summary['total_species_rows'] = len(all_rows)
    summary['negative_T_count'] = sum(r['T_eV']<0 for r in all_rows)
    summary['centered_raw_max_absolute_difference_eV']=max(r['centered_raw_abs_difference_eV'] for r in all_rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'artifact_sha256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_sha256.json'},indent=2)+'\n')
    print(json.dumps({'rows':len(all_rows),'negative_T_count':summary['negative_T_count'],'max_audit_delta_eV':summary['centered_raw_max_absolute_difference_eV']}))

if __name__ == '__main__':
    main()
