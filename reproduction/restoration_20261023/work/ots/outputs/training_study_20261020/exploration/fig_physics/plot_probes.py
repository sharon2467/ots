import os,json
os.environ['MPLCONFIGDIR']='/tmp/matplotlib-fig-physics'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
c=np.load(root/'physics_probe_curves.npz')
d=json.loads((root/'physics_probe.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10.4,7.1),constrained_layout=True)
for row,(key,title) in enumerate([('same_temperature_sg','Equal kinetic Te: 1392 eV'),('equal_response_sg','Matched static response: Te = 1392 / 636.09 eV')]):
    v=c['fig1b_electron_velocity']/1e6
    for curve,label in [('gaussian','Gaussian'),(key,'Super-Gaussian p=4')]:
        axs[row,0].plot(v,c['fig1b_electron_'+curve]*1e6,label=label,lw=1.8)
    axs[row,0].set(xlim=(-65,65),xlabel='Electron velocity (10^6 m/s)',ylabel='VDF (10^-6 s/m)',title=title)
    for col,band,label in [(1,'iaw','IAW band')]:
        axis=c['iaw_wavelength_nm' if band=='iaw' else 'red_wavelength_nm']
        y0=c['fig1b_'+band+'_gaussian'];yy=c['fig1b_'+band+'_'+key]
        err=np.linalg.norm(yy-y0)/np.linalg.norm(y0)
        axs[row,col].plot(axis,y0,label='Gaussian',lw=1.8)
        axs[row,col].plot(axis,yy,label='Super-Gaussian',lw=1.4,ls='--')
        axs[row,col].set(xlabel='Wavelength (nm)',ylabel='Peak-normalized intensity',title=f'{label}; relative L2 difference = {100*err:.2f}%')
    axs[row,0].legend(frameon=False)
axs[0,1].legend(frameon=False)
fig.suptitle('Clean PV probes: electron shape and temperature can compensate in IAW',fontsize=14)
fig.savefig(root/'shape_temperature_compensation.png',dpi=170)
plt.close(fig)
r=json.loads((root/'representation_probe.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11.8,4.5),constrained_layout=True)
groups=[('fixed_ion','e','Fixed e'),('fig1_fig11','e','Fig e'),('fig1_fig11','i1','Fig H'),('fig1_fig11','i2','Fig C')]
values=[]
for dataset,s,label in groups:
    values.append([max(1e-11,x['raw_vdf_relative_l2'])*100 for x in r[dataset]['samples'] if x['species']==s])
axs[0].boxplot(values,tick_labels=[x[2] for x in groups],showfliers=True)
axs[0].set(yscale='log',ylabel='Raw-VDF reconstruction relative L2 (%)',title='Exact target coefficients, M=80; 64 prior draws each')
conv=json.loads((root/'mode_convergence.json').read_text())
for s,color,name in [('i1','C1','H'),('i2','C2','C')]:
    rows=[x for x in conv['samples'] if x['species']==s]
    axs[1].plot([x['M'] for x in rows],[x['raw_vdf_relative_l2']*100 for x in rows],color=color,marker='o',label=f'{name}: raw-VDF error')
    axs[1].plot([x['M'] for x in rows],[x['negative_mass_fraction']*100 for x in rows],color=color,ls='--',marker='s',label=f'{name}: negative mass')
axs[1].set(yscale='log',xlabel='Fourier modes M',ylabel='Error or negative mass (%)',title='33.55 eV, p=4: improve M on unchanged Fig axes')
axs[1].set_xticks([80,160,320])
axs[1].legend(ncol=1,fontsize=8,frameon=False)
for ax in axs: ax.grid(True,alpha=.2)
fig.suptitle('Fourier representation error exists before training; canonical EVDF metrics use filtered targets',fontsize=13)
fig.savefig(root/'fourier_representation_error.png',dpi=170)
plt.close(fig)
print('Saved shape_temperature_compensation.png and fourier_representation_error.png')
