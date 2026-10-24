from pathlib import Path
prefix=Path(__file__).with_name('physics_probe.py').read_text().split("result={'scope'")[0]
exec(compile(prefix, str(Path(__file__).with_name('physics_probe.py')), 'exec'))
rows=[]
for s in ['i1','i2']:
    v,f=vdf(s,33.55,p=4.)
    for m in [80,160,320]:
        k,F=phy.ft_x_n(v,f,m)
        _,ff=reconstruct_real_evdf(torch.cat((F.real,F.imag))[None],k,velocity=v)
        ff=ff[0]
        rows.append({'species':s,'temperature_ev':33.55,'p':4.,'M':m,'raw_vdf_relative_l2':rel(f,ff),'negative_mass_fraction':float(torch.trapz(torch.relu(-ff),v)/torch.trapz(ff.abs(),v)),'mass':float(torch.trapz(ff,v))})
out={'scope':'Unchanged default Fig velocity axes; exact Fourier labels; no training or PV. M is exposed by generation-config fourier_modes.','samples':rows}
Path(__file__).with_name('mode_convergence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
