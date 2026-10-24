#!/usr/bin/env python3
"""Read-only lineage and scaling audit; no inference or data corrections."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fig10_normalization_audit_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots')
OUT=Path(__file__).resolve().parent
GROUPS=('fig10a','fig10b','fig10c')
LABELS=('(a) Experiment','(b) Simulation without probe','(c) Simulation with probe')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edges(x):
    mid=(x[1:]+x[:-1])/2
    return np.r_[2*x[0]-mid[0],mid,2*x[-1]-mid[-1]]

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':220})
    fig,axes=plt.subplots(3,2,figsize=(12.5,12.5),sharex=True,sharey=True,layout='constrained')
    report={'scope':'Read-only latest data and saved PV comparison arrays; plots only, no new correction or inference.',
            'normalization':'I(t,lambda)/max_lambda(abs(I(t,lambda)))','panels':{}}
    for i,(g,label) in enumerate(zip(GROUPS,LABELS)):
        dp=ROOT/f'outputs/fig10_nearest_row_fill_20261021/datasets/{g}.npz'
        fp=ROOT/f'outputs/fig10_moments_pv_20261021/forward/{g}/forward_results.npz'
        with np.load(dp,allow_pickle=False) as d,np.load(fp,allow_pickle=False) as f:
            t=d['time_paper_ns'];wl=d['wavelength_nm'];raw=d['intensity'];norm=f['observed_normalized'];scale=np.max(np.abs(raw),axis=1)
            assert np.array_equal(raw,f['observed'])
            assert np.array_equal(raw/scale[:,None],norm)
            assert np.array_equal(wl,f['wavelength_nm']) and np.array_equal(t,f['time_paper_ns'])
            assert np.array_equal(scale,f['observed_peak_scale'])
            record={'latest_dataset':str(dp),'latest_sha256':sha(dp),'forward_result_sha256':sha(fp),
                    'observed_exactly_equals_latest':True,'normalized_exactly_equals_latest_divided_by_row_peak':True,
                    'shape':list(raw.shape),'examples':[]}
            for target in (1.52,1.6,1.65,1.7,2.,3.,4.):
                k=int(np.argmin(abs(t-target)));x=raw[k]
                record['examples'].append({'target_ns':target,'actual_ns':float(t[k]),'row':k,
                    'source_row':int(d['source_row_index'][k]),'peak':float(scale[k]),'minimum':float(x.min()),
                    'median':float(np.median(x)),'normalized_median':float(np.median(norm[k])),
                    'pixels_equal_row_peak':int(np.count_nonzero(x==scale[k])),
                    'replacement_pixels':int(d['replacement_mask'][k].sum())})
            for j,z in enumerate((raw,norm)):
                ax=axes[i,j]
                vmax=3200 if i==0 and j==0 else 1
                im=ax.pcolormesh(edges(wl-263.3),edges(t),z,vmin=0,vmax=vmax,cmap='viridis',shading='flat',rasterized=True)
                ax.set_xlim(-1,1);ax.set_ylim(4.5,1.5);ax.set_xticks(np.arange(-1,1.01,.5))
                ax.set_yticks(np.arange(1.5,4.51,.5))
                fig.colorbar(im,ax=ax,pad=.015,shrink=.92,label='Counts after subtracting 1691' if i==0 and j==0 else ('Digitized intensity' if j==0 else 'Intensity / row peak'))
                if j==0:ax.set_ylabel(label+'\nTime (ns)')
                if i==2:ax.set_xlabel(r'$\Delta\lambda=\lambda-263.3$ nm')
                if i==0:ax.set_title('Latest saved values\nFixed scale over the entire time series' if j==0 else 'Same data, divided by each row peak\nThe left column of the previous PV figure',fontsize=12,pad=12)
                q=record['examples'][1]
                text=(f'At {q["actual_ns"]:.4f} ns\nMedian = {q["median"]:.5g}; peak = {q["peak"]:.5g}' if j==0 else
                      f'At {q["actual_ns"]:.4f} ns\nNormalized median = {q["normalized_median"]:.3f}')
                ax.text(.035,.035,text,va='bottom',transform=ax.transAxes,color='white',fontsize=10,
                        bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':4})
            report['panels'][g]=record
    fig.suptitle('Why the top becomes bright: the same latest data on two intensity scales',fontsize=16)
    for ext in ('png','pdf'):fig.savefig(OUT/f'latest_values_vs_row_normalized.{ext}')
    plt.close(fig)
    for g,x in report['panels'].items():
        assert sha(Path(x['latest_dataset']))==x['latest_sha256']
    report['outputs_sha256']={p.name:sha(p) for p in OUT.glob('latest_values_vs_row_normalized.*')}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({g:x['examples'][1] for g,x in report['panels'].items()},indent=2))

if __name__=='__main__':main()
