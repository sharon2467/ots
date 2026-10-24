import os,sys,json,time
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/matplotlib-fig-physics'
sys.path.insert(0,'/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange')
import numpy as np
import torch
import data
from phyEqns import ots_full_fourier as phy
from evaluation import reconstruct_real_evdf
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
out={}
for name,config,factory,outputs in [('fixed_ion',data._physical_config(),data.Fixed_ion_sampler,['e']),('fig1_fig11',data._physical_config_fig1_fig11(),data.Fig1_Fig11_sampler,['e','i1','i2'])]:
    config['points_csv']='/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020/paper_experimental_points.csv'
    sampler=factory(np.random.default_rng(20261020),config)
    rows=[]
    for i in range(64):
        state=sampler.sample()
        for s in outputs:
            v,f=state['distributions'][s]
            k,F=phy.ft_x_n(v,f,80)
            coeff=torch.cat((F.real,F.imag))
            _,ff=reconstruct_real_evdf(coeff[None],k,velocity=v)
            ff=ff[0]
            rows.append({'sample_index':i,'species':s,**data.distribution_moments(v,f,config[dict(data.SPECIES_BLOCKS)[s]]['mass_kg']), 'raw_vdf_relative_l2':float(torch.linalg.vector_norm(ff-f)/torch.linalg.vector_norm(f)), 'negative_mass_fraction':float(torch.trapz(torch.relu(-ff),v)/torch.trapz(ff.abs(),v)), 'mass':float(torch.trapz(ff,v))})
    stats={}
    for s in outputs:
        r=[x for x in rows if x['species']==s]
        stats[s]={}
        for key in ['raw_vdf_relative_l2','negative_mass_fraction','mass']:
            a=np.array([x[key] for x in r])
            stats[s][key]={'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
    out[name]={'n_samples':64,'seed':20261020,'statistics':stats,'samples':rows}
from pathlib import Path
Path(__file__).with_name('representation_probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v['statistics'] for k,v in out.items()},indent=2))
