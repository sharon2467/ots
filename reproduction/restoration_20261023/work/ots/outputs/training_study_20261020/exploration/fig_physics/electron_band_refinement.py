"""Milder p=2.5 case, to check that added-band contrast survives refinement."""
from pathlib import Path
prefix=Path(__file__).with_name('physics_probe.py').read_text().split("result={'scope'")[0]
exec(compile(prefix, str(Path(__file__).with_name('physics_probe.py')), 'exec'))
ts=[1392.,912.,912.];ne=4e26;p=2.5
v,f=vdf('e',1392.)
z=torch.tensor([0.],dtype=torch.float64)
response=phy._pv_fd_landau_response(v,f,z,name='probe').real.item()
def fun(t):
    vv,ff=vdf('e',t,p=p)
    return phy._pv_fd_landau_response(vv,ff,z,name='probe').real.item()/response-1
tmatch=brentq(fun,.4*1392.,1392.)
out={'p':p,'gaussian_Te_ev':1392.,'matching_Te_ev':tmatch,'bands':[]}
y0,ds=spectrum(ts,ne=ne)
ym,dsm=spectrum([tmatch,912.,912.],ps=(p,2.,2.),ne=ne)
out['iaw_relative_l2']=rel(y0,ym)
out['raw_e_vdf_relative_l2']=rel(ds[0][1],dsm[0][1])
curves={}
for n in (181,721,1441):
    axis=torch.linspace(280.,350.,n,dtype=torch.float64)
    cfg['wavelength_bands_nm']=[axis]
    yg,_=spectrum(ts,ne=ne)
    ys,_=spectrum([tmatch,912.,912.],ps=(p,2.,2.),ne=ne)
    out['bands'].append({'n':n,'range_nm':[280.,350.],'relative_l2':rel(yg,ys),'gaussian_peak_nm':float(axis[yg.argmax()]),'supergaussian_peak_nm':float(axis[ys.argmax()])})
    curves['axis_'+str(n)]=axis.numpy();curves['gaussian_'+str(n)]=yg.numpy();curves['sg_'+str(n)]=ys.numpy()
out['coarse_vs_fine_relative_l2']={s:float(np.linalg.norm(curves[s+'_181']-curves[s+'_721'][::4])/np.linalg.norm(curves[s+'_721'][::4])) for s in ('gaussian','sg')}
out['721_vs_1441_relative_l2']={s:float(np.linalg.norm(curves[s+'_721']-curves[s+'_1441'][::2])/np.linalg.norm(curves[s+'_1441'][::2])) for s in ('gaussian','sg')}
root=Path(__file__).parent
(root/'electron_band_refinement.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(root/'electron_band_refinement.npz',**curves)
print(json.dumps(out,indent=2))
