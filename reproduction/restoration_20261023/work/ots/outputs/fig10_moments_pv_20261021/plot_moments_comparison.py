#!/usr/bin/env python3
"""Plot saved signed moments against published Fig.11; no forward physics."""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fig10_moments_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
GROUPS = ('fig10a', 'fig10b', 'fig10c')
SPECIES = ('e', 'H', 'C')
LABELS = ('(a) Experiment', '(b) Simulation without probe', '(c) Simulation with probe')
COLORS = {'e': '#D55E00', 'H': '#009E73', 'C': '#7851A9'}
SLABELS = {'e': 'Electrons', 'H': 'Hydrogen', 'C': 'Carbon'}

def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def series(rows, group, species, field):
    r = [x for x in rows if x['panel'] == group and x['species'] == species]
    return np.array([float(x['time_paper_ns']) for x in r]), np.array([float(x[field]) for x in r])

def style(ax):
    ax.grid(True, color='#d9dfe5', lw=.6, alpha=.75)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(1.5, 4.5)
    ax.set_xticks(np.arange(1.5, 4.51, .5))

def reference(ax, references, envelopes, group, parameter):
    if group == 'fig10a':
        r = [x for x in references if x['parameter'] == parameter]
        ax.scatter([float(x['time_ns']) for x in r], [float(x['value']) for x in r],
                   color='#c74ceb', marker='s', s=26, edgecolors='white', linewidths=.4,
                   zorder=5, label='Fig.11 experimental markers')
    else:
        r = [x for x in envelopes if x['fig10_panel'] == group[-1] and x['parameter'] == parameter]
        t = np.array([float(x['time_ns']) for x in r])
        lo = np.array([float(x['ink_envelope_low_ev']) for x in r])
        hi = np.array([float(x['ink_envelope_high_ev']) for x in r])
        color = '#0058d9' if group == 'fig10b' else '#e02727'
        # Break at missing native image columns; never bridge obscured ink.
        gaps = np.flatnonzero(np.diff(t) > .006237006237006237 * 1.5) + 1
        for n, inds in enumerate(np.split(np.arange(len(t)), gaps)):
            ax.fill_between(t[inds], lo[inds], hi[inds], color=color, alpha=.17)
            ax.plot(t[inds], lo[inds], color=color, lw=1.1,
                    label='Fig.11 simulation ink envelope' if n == 0 else None)
            ax.plot(t[inds], hi[inds], color=color, lw=1.1)

def main():
    out = ROOT / 'plots'; out.mkdir(exist_ok=True)
    rows = read_csv(ROOT / 'moments/signed_moments_all_rows.csv')
    refs = read_csv(ROOT / 'references/fig11_experimental_points.csv')
    env = read_csv(ROOT / 'references/fig11_simulation_ink_envelopes.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.titlesize':12, 'axes.labelsize':11, 'savefig.dpi':220})
    manifest = {'moment_definition':'Full saved signed VDF: u=int(v*f)/int(f), T=m/e*int((v-u)^2*f)/int(f); no clipping.',
                'experimental_comparison':'Unweighted differences at Fig.11 digitized marker times, linear interpolation of calculated moment time series only.',
                'simulation_reference':'Published color-ink extent with gaps retained; not numerical truth or a statistical confidence interval.',
                'experimental_errors':{}, 'representative_rows':[], 'plots':{}}
    with PdfPages(ROOT / 'moments_fig11_comparison.pdf') as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.1), sharex=True, sharey='row')
        for j, group in enumerate(GROUPS):
            for i, species_group in enumerate((('e',), ('H', 'C'))):
                ax = axes[i,j]; style(ax)
                reference(ax, refs, env, group, 'T_e_ev' if i == 0 else 'T_i_ev')
                for s in species_group:
                    t,y = series(rows, group, s, 'T_eV')
                    ax.plot(t,y,color=COLORS[s],lw=1.4,label=f'Inferred {s}: full signed moment')
                ax.set_ylim(0, 2100 if i == 0 else 2800)
                ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=.93, fontsize=8)
                if j == 0: ax.set_ylabel('Electron temperature (eV)' if i == 0 else 'Ion temperature (eV)')
                if i == 0: ax.set_title(LABELS[j], pad=12)
                else: ax.set_xlabel('Time (ns)')
        fig.suptitle('Latest Fig.10 VDF moments compared with Fig.11', fontsize=17, y=.985)
        fig.text(.05,.032,'Original inferred VDFs; 4097 velocity samples per species; no clipping, smoothing or temperature fitting.',fontsize=10)
        fig.text(.05,.012,'Fig.11 gives one ion temperature, not separate H/C truth. Shading reflects spatial inhomogeneity; digitized ink is approximate.',fontsize=10)
        fig.tight_layout(rect=(.01,.065,.995,.95))
        name='temperatures_vs_fig11'; fig.savefig(out/(name+'.png')); fig.savefig(out/(name+'.pdf')); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3,3,figsize=(15.4,10.1),sharex=True,sharey='row')
        for i,s in enumerate(SPECIES):
            for j,group in enumerate(GROUPS):
                ax=axes[i,j];style(ax)
                t,u=series(rows,group,s,'u_km_s')
                ax.plot(t,u,color=COLORS[s],lw=1.4)
                ax.axhline(0,color='#5e6570',ls='--',lw=.8)
                ax.set_ylim((-125,15) if s=='e' else (-3,18))
                if j==0:ax.set_ylabel(f'{SLABELS[s]} flow (km/s)')
                if i==0:ax.set_title(LABELS[j],pad=12)
                if i==2:ax.set_xlabel('Time (ns)')
        fig.suptitle('Projected flow velocity from the latest inferred VDFs',fontsize=17,y=.985)
        fig.text(.05,.025,'Signed first moments on the saved velocity axis. Fig.11 has no flow data; dashed zero is a visual reference, not a measured truth.',fontsize=10)
        fig.tight_layout(rect=(.01,.055,.995,.95))
        name='projected_flow_velocities';fig.savefig(out/(name+'.png'));fig.savefig(out/(name+'.pdf'));pdf.savefig(fig);plt.close(fig)

    comparisons=[]
    for s in SPECIES:
        parameter='T_e_ev' if s=='e' else 'T_i_ev'
        r=[x for x in refs if x['parameter']==parameter]
        tref=np.array([float(x['time_ns']) for x in r]); truth=np.array([float(x['value']) for x in r])
        t,pred=series(rows,'fig10a',s,'T_eV');assert tref.min()>=t.min() and tref.max()<=t.max()
        estimate=np.interp(tref,t,pred);delta=estimate-truth
        manifest['experimental_errors'][s]={'n':len(tref),'MAE_eV':float(np.abs(delta).mean()),
            'RMSE_eV':float(np.sqrt(np.mean(delta**2))),'bias_eV':float(delta.mean()),
            'mean_absolute_relative_difference':float(np.mean(np.abs(delta)/truth))}
        for x,estimate_value,d in zip(r,estimate,delta):
            comparisons.append({'species':s,'time_ns':x['time_ns'],'paper_value_eV':x['value'],
                                'calculated_value_eV':estimate_value,'difference_eV':d,'paper_point_id':x['point_id']})
    with (ROOT/'experimental_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    for group in GROUPS:
        for target in (2.,3.,4.):
            for s in SPECIES:
                r=[x for x in rows if x['panel']==group and x['species']==s]
                pick=min(r,key=lambda x:abs(float(x['time_paper_ns'])-target))
                manifest['representative_rows'].append({'target_time_ns':target,**pick})
    with (ROOT/'representative_moments.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest['representative_rows'][0]));w.writeheader();w.writerows(manifest['representative_rows'])
    for p in sorted(out.glob('*')):
        manifest['plots'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'moments_comparison.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['experimental_errors'],indent=2))

if __name__=='__main__':main()
