#!/usr/bin/env python3
"""Plot existing PV outputs; never evaluate or implement a physical model."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fig10_moments_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT=Path(__file__).resolve().parent
GROUPS=('fig10a','fig10b','fig10c')
LABELS=('(a) Experiment','(b) Simulation without probe','(c) Simulation with probe')
COLORS={'e':'#D55E00','H':'#009E73','C':'#7851A9'}

def edges(x):
    mid=(x[:-1]+x[1:])/2
    return np.r_[2*x[0]-mid[0],mid,2*x[-1]-mid[-1]]

def style(ax):
    ax.grid(True,color='#d9dfe5',lw=.6,alpha=.75)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)

def main():
    summary=json.loads((ROOT/'forward/summary.json').read_text())
    out=ROOT/'plots';out.mkdir(exist_ok=True)
    data={};prep={};records={}
    for group in GROUPS:
        p=ROOT/'forward'/group/'forward_results.npz'
        with np.load(p,allow_pickle=False) as d:data[group]={k:d[k] for k in d.files}
        with np.load(ROOT/'forward'/group/'processed_vdfs.npz',allow_pickle=False) as d:
            prep[group]={k:d[k] for k in d.files if '_T_eV' in k or '_u_m_s' in k}
        d=data[group];assert d['success'].all()
        records[group]={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'representative_rows':[],'preparation_moment_changes':{}}
        for target in (2.,3.,4.):
            row=int(np.argmin(np.abs(d['time_paper_ns']-target)))
            records[group]['representative_rows'].append({'target_time_ns':target,'row':row,
                'actual_time_ns':float(d['time_paper_ns'][row]),'source_row_index':int(d['source_row_index'][row]),
                'relativeL2':float(d['relativeL2'][row]),'rmse':float(d['rmse'][row])})
        for s in COLORS:
            deltaT=prep[group][s+'_processed_T_eV']-prep[group][s+'_original_T_eV']
            deltau=(prep[group][s+'_processed_u_m_s']-prep[group][s+'_original_u_m_s'])/1000
            records[group]['preparation_moment_changes'][s]={'delta_T_min_eV':float(deltaT.min()),
                'delta_T_max_eV':float(deltaT.max()),'delta_T_mean_eV':float(deltaT.mean()),
                'max_abs_delta_u_km_s':float(np.abs(deltau).max())}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':12,'savefig.dpi':220})
    pdf=PdfPages(ROOT/'spectral_comparison.pdf')
    def save(fig,name):
        fig.savefig(out/(name+'.png'));fig.savefig(out/(name+'.pdf'));pdf.savefig(fig);plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(14.8,12.5),sharex=True,sharey=True,layout='constrained')
    for i,group in enumerate(GROUPS):
        d=data[group];x=edges(d['wavelength_nm']-263.3);t=edges(d['time_paper_ns'])
        for j,field in enumerate(('observed_normalized','predicted_PV','residual')):
            ax=axes[i,j]
            im=ax.pcolormesh(x,t,d[field],shading='flat',cmap='RdBu_r' if j==2 else 'viridis',
                             vmin=-1 if j==2 else 0,vmax=1,rasterized=True)
            ax.set_xlim(-1,1);ax.set_ylim(4.5,1.5);ax.set_xticks(np.arange(-1,1.01,.5));ax.set_yticks(np.arange(1.5,4.51,.5))
            for target in (2.,3.,4.):ax.axhline(target,lw=.5,ls=':',color='white' if j<2 else '#555',alpha=.6)
            if i==0:ax.set_title(('Latest data / row peak','Existing PV reconstruction','PV minus latest data')[j],pad=10)
            if j==0:ax.set_ylabel(LABELS[i]+'\nTime (ns)')
            if i==2:ax.set_xlabel(r'$\Delta\lambda=\lambda-263.3$ nm')
        axes[i,1].text(.03,.96,f'Mean relative L2: {100*summary["panels"][group]["relativeL2_mean"]:.1f}%',
                        transform=axes[i,1].transAxes,va='top',color='white',fontsize=10,
                        bbox={'facecolor':'black','alpha':.55,'edgecolor':'none','pad':4})
    fig.colorbar(axes[0,0].collections[0],ax=axes[:,:2],shrink=.85,pad=.015,label='Normalized intensity')
    fig.colorbar(axes[0,2].collections[0],ax=axes[:,2],shrink=.85,pad=.015,label='Normalized intensity difference')
    fig.suptitle('Spectral closure on the latest native wavelength grids\n'
                 r'Existing OTS_spectrum_PV; fixed $n_e=3.5\times10^{20}$ cm$^{-3}$; authorized VDF copies',fontsize=16)
    save(fig,'spectral_heatmaps_comparison')

    fig,axes=plt.subplots(3,3,figsize=(15.4,11),sharex=True,sharey=True)
    for i,group in enumerate(GROUPS):
        d=data[group];x=d['wavelength_nm']-263.3
        for j,rec in enumerate(records[group]['representative_rows']):
            ax=axes[i,j];style(ax);row=rec['row']
            ax.plot(x,d['observed_normalized'][row],color='#303944',lw=1.6,label='Latest data / row peak')
            ax.plot(x,d['predicted_PV'][row],color='#D55E00',lw=1.5,label='Existing PV reconstruction')
            ax.set_xlim(-1,1);ax.set_ylim(-.03,1.07);ax.set_xticks(np.arange(-1,1.01,.5))
            ax.set_title(f'{rec["actual_time_ns"]:.6f} ns | source row {rec["source_row_index"]}',fontsize=11)
            ax.text(.98,.92,f'Rel. L2 = {100*rec["relativeL2"]:.1f}%',ha='right',va='top',transform=ax.transAxes,fontsize=10)
            if j==0:ax.set_ylabel(LABELS[i]+'\nNormalized intensity')
            if i==2:ax.set_xlabel(r'$\Delta\lambda=\lambda-263.3$ nm')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.953),ncol=2,frameon=False)
    fig.suptitle('Representative spectra nearest 2, 3 and 4 ns',fontsize=17,y=.99)
    fig.text(.055,.025,'Native wavelength axes; independent row-peak normalization. Fixed density; no spectral shift, smoothing, fit or instrument convolution.',fontsize=10)
    fig.tight_layout(rect=(.01,.055,.995,.92));save(fig,'representative_spectra_comparison')

    fig,axes=plt.subplots(2,3,figsize=(15.4,8.1),sharex=True,sharey='row')
    for j,group in enumerate(GROUPS):
        d=data[group];t=d['time_paper_ns']
        for i,(field,mult,label) in enumerate((('relativeL2',100,'Relative L2 difference (%)'),('rmse',1,'Normalized-intensity RMSE'))):
            ax=axes[i,j];style(ax);ax.plot(t,d[field]*mult,color='#2166ac',lw=1.4)
            ax.set_xlim(1.5,4.5);ax.set_ylim(bottom=0)
            if j==0:ax.set_ylabel(label)
            if i==0:ax.set_title(LABELS[j])
            else:ax.set_xlabel('Time (ns)')
    for i,(field,mult) in enumerate((('relativeL2',100),('rmse',1))):
        maximum=max(float(np.max(data[g][field]))*mult for g in GROUPS)
        axes[i,0].set_ylim(0, maximum*1.08)
    fig.suptitle('Spectral discrepancy over time',fontsize=17,y=.985)
    fig.text(.055,.025,'Relative L2 = ||PV - normalized data||₂ / ||normalized data||₂. Comparison of spectral shape, not absolute optical power.',fontsize=10)
    fig.tight_layout(rect=(.01,.06,.995,.95));save(fig,'spectral_error_over_time')

    fig,axes=plt.subplots(2,3,figsize=(15.4,8.5),sharex=True,sharey='row')
    for j,group in enumerate(GROUPS):
        t=data[group]['time_paper_ns']
        for i,(suffix,scale,label) in enumerate((('T_eV',1,'Temperature change (eV)'),('u_m_s',1/1000,'Flow change (km/s)'))):
            ax=axes[i,j];style(ax)
            for s,color in COLORS.items():
                delta=(prep[group][s+'_processed_'+suffix]-prep[group][s+'_original_'+suffix])*scale
                ax.plot(t,delta,color=color,lw=1.2,label=s)
            ax.axhline(0,color='#777',ls='--',lw=.8);ax.set_xlim(1.5,4.5)
            if j==0:ax.set_ylabel(label)
            if i==0:ax.set_title(LABELS[j]);ax.legend(ncol=3,loc='upper right',fontsize=9)
            else:ax.set_xlabel('Time (ns)')
    fig.suptitle('Effect of the authorized preparation: PV copy minus original inferred VDF',fontsize=16,y=.985)
    fig.text(.055,.025,'Uniform float64 grid → linear interpolation → negative values set to zero → unit integral. Original inference arrays remain unchanged.',fontsize=10)
    fig.tight_layout(rect=(.01,.06,.995,.95));save(fig,'preparation_moment_changes')
    pdf.close()
    manifest={'data':records,'plots':{}}
    for name in ('spectral_heatmaps_comparison','representative_spectra_comparison','spectral_error_over_time','preparation_moment_changes'):
        for ext in ('png','pdf'):
            p=out/(name+'.'+ext);manifest['plots'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'forward_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({g:{'relativeL2_mean':summary['panels'][g]['relativeL2_mean'],
                          'moment_changes':records[g]['preparation_moment_changes']} for g in GROUPS},indent=2))

if __name__=='__main__':main()
