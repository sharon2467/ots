"""Read-only independent array checks and sampled original PV reevaluation."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
from run_forward import make_config,forward,sha

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter();checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition: raise AssertionError(name)
    plan=json.loads((HERE.parent/'analysis_plan.json').read_text())
    for p,s in plan['protected_sha256'].items():check('protected SHA '+p,sha(p)==s)
    cfg=make_config();summary=json.loads((HERE/'summary.json').read_text())
    for panel in ['fig10a','fig10b','fig10c']:
        dest=HERE/panel
        raw=np.load(REPO/f'outputs/fig10_nearest_row_fill_20261021/datasets/{panel}.npz')
        pred=np.load(REPO/f'outputs/fig10_nearest_d4_inference_20261021/results/{panel}/predictions.npz')
        out=np.load(dest/'forward_results.npz');prep=np.load(dest/'processed_vdfs.npz')
        n=len(raw['time_paper_ns'])
        for key in ['time_paper_ns','source_row_index','wavelength_nm']:
            check(panel+' original '+key,np.array_equal(out[key],raw[key]))
        check(panel+' raw observed bitwise',np.array_equal(out['observed'],raw['intensity']))
        scale=np.max(np.abs(raw['intensity']),axis=1)
        normalized=raw['intensity']/scale[:,None]
        check(panel+' observed normalization',np.array_equal(out['observed_normalized'],normalized))
        check(panel+' observed scale',np.array_equal(out['observed_peak_scale'],scale))
        check(panel+' all statuses present',len(out['error'])==n and len(out['success'])==n)
        good=out['success']
        check(panel+' successful rows finite',np.isfinite(out['predicted_PV'][good]).all())
        check(panel+' failed rows NaN',np.isnan(out['predicted_PV'][~good]).all())
        check(panel+' PV per-row maxabs',np.allclose(np.max(np.abs(out['predicted_PV'][good]),axis=1),1.,rtol=0,atol=1e-12))
        check(panel+' residual',np.array_equal(out['residual'],out['predicted_PV']-normalized,equal_nan=True))
        rel=np.linalg.norm(out['residual'],axis=1)/np.linalg.norm(normalized,axis=1)
        check(panel+' metric relativeL2',np.array_equal(rel,out['relativeL2'],equal_nan=True))
        check(panel+' metric rmse',np.array_equal(np.sqrt(np.mean(out['residual']**2,axis=1)),out['rmse'],equal_nan=True))
        check(panel+' mean summary',float(rel[good].mean())==summary['panels'][panel]['relativeL2_mean'])
        for sp in ['e','H','C']:
            oldv=pred[sp+'_velocity_m_s'].astype('float64');oldf=pred[sp+'_vdf'].astype('float64')
            v=np.linspace(oldv[0],oldv[-1],len(oldv));f=prep[sp+'_vdf']
            interpolated=np.array([np.interp(v,oldv,x) for x in oldf])
            positive=np.maximum(interpolated,0.)
            norm=np.trapezoid(positive,v,axis=1)
            independently=positive/norm[:,None]
            check(panel+' '+sp+' v endpoint-equivalent uniform',np.array_equal(v,prep[sp+'_velocity_m_s']))
            check(panel+' '+sp+' uniform tolerance',np.max(np.abs(np.diff(v)-np.diff(v).mean())) <= 1e-4*np.diff(v).mean())
            check(panel+' '+sp+' no extrapolation',v[0]==oldv[0] and v[-1]==oldv[-1])
            check(panel+' '+sp+' preprocessing independently',np.allclose(f,independently,rtol=5e-15,atol=0))
            check(panel+' '+sp+' f float64 finite nonnegative',f.dtype==np.float64 and np.isfinite(f).all() and (f>=0).all())
            check(panel+' '+sp+' f unit integrals',np.allclose(np.trapezoid(f,v,axis=1),1.,rtol=0,atol=1e-14))
            check(panel+' '+sp+' actual clip count',np.array_equal(prep[sp+'_clipped_node_count'],np.count_nonzero(interpolated<0,axis=1)))
            check(panel+' '+sp+' negative mass',np.array_equal(prep[sp+'_original_negative_mass'],np.trapezoid(np.maximum(-oldf,0.),oldv,axis=1)))
        sampled=sorted(set([0]+[int(np.argmin(np.abs(raw['time_paper_ns']-t))) for t in [2,3,4]]))
        for row in sampled:
            y=forward(raw['wavelength_nm'],prep,row,cfg)
            check(panel+' original PV reevaluation row '+str(row),np.array_equal(y,out['predicted_PV'][row]))
    report={'passed':all(c['passed'] for c in checks),'checks_count':len(checks),'checks':checks,
            'sampled_original_PV_rows_per_panel':4,'duration_seconds':time.perf_counter()-start,
            'verification_scope':'All protected SHA and all processed arrays/metrics; sampled original public PV reevaluation. Does not rerun every PV row.'}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
