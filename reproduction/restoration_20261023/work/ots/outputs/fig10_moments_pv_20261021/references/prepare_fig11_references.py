"""Published Fig11 image readings; never asserts original simulation centerline."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
R=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots')
P=R/'Zhao 等 - 2019 - Progress in optical Thomson scattering diagnostics for ICF gas-filled hohlraums.pdf'
S=R/'outputs/training_study_20261020/paper_experimental_points.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['pdftotext','-f','9','-l','10','-layout',str(P),str(H/'paper_pages9_10.txt')],check=True)
d=pd.read_csv(S);e=d[d.figure.eq('fig11') & d.shot_id.eq(20170811020)].copy()
assert e.groupby('parameter').size().to_dict()=={'T_e_ev':27,'T_i_ev':27}
e.to_csv(H/'fig11_experimental_points.csv',index=False)
a=np.array(Image.open(H/'page10_image-001.png').convert('RGB'))
assert a.shape==(564,1278,3)
panels={'T_e_ev':dict(panel='a',x0=114,x1=595,maximum=2500),'T_i_ev':dict(panel='b',x0=775,x1=1256,maximum=2000)}
rows=[]
for parameter,c in panels.items():
 q=e[e.parameter.eq(parameter)]
 assert np.max(np.abs(1.5+(q.pixel_x-c['x0'])*3/481-q.time_ns))<1e-12
 assert np.max(np.abs((474-q.pixel_y)*c['maximum']/462-q.value))<1e-10
 for condition,color,channel in [('without_probe','blue',2),('with_probe','red',0)]:
  other=[i for i in range(3) if i!=channel]
  mask=(a[:,:,channel]>220)&(a[:,:,other[0]]<90)&(a[:,:,other[1]]<90)
  mask[:13]=False;mask[474:]=False
  mask[35:181,c['x0']+28:c['x0']+294]=False
  for x in range(c['x0']+1,c['x1']):
   y=np.flatnonzero(mask[:,x])
   if not len(y):continue
   lo,hi=int(y.min()),int(y.max())
   groups=np.split(y,np.flatnonzero(np.diff(y)>1)+1)
   rows.append(dict(parameter=parameter,panel=c['panel'],fig10_panel='b' if condition=='without_probe' else 'c',condition=condition,color=color,pixel_x=x,pixel_y_ink_top=lo,pixel_y_ink_bottom=hi,ink_pixel_count=len(y),ink_components=len(groups),time_ns=1.5+(x-c['x0'])*3/481,ink_envelope_low_ev=(474-hi)*c['maximum']/462,ink_envelope_high_ev=(474-lo)*c['maximum']/462,derived_ink_midpoint_ev=(474-(lo+hi)/2)*c['maximum']/462,time_resolution_per_pixel_ns=3/481,value_resolution_per_pixel_ev=c['maximum']/462,conservative_reading_halfwidth_ev=2*c['maximum']/462,reference_kind='published_simulation_color_ink_envelope_not_numeric_truth',both_physical_bounds_certified=False))
r=pd.DataFrame(rows);r.to_csv(H/'fig11_simulation_ink_envelopes.csv',index=False)
fig,ax=plt.subplots(figsize=(15,6.6));ax.imshow(a)
for _,q in r.groupby(['parameter','condition']):
 ax.scatter(q.pixel_x,q.pixel_y_ink_top,s=1,color='lime',alpha=.7)
 ax.scatter(q.pixel_x,q.pixel_y_ink_bottom,s=1,color='cyan',alpha=.7)
ax.set_title('Native Fig11: green/cyan = selected ink envelope; no centerline assumed');ax.axis('off');fig.tight_layout();fig.savefig(H/'fig11_ink_selection_overlay.png',dpi=160);plt.close(fig)
audit=dict(source_pdf=str(P),source_pdf_sha256=sha(P),pdf_page=10,printed_page='055201-9',source_image_object=308,native_image_size=[1278,564],source_image_sha256=sha(H/'page10_image-001.png'),experimental_source=str(S),experimental_source_sha256=sha(S),experimental_points=54,experimental_point_counts={'T_e_ev':27,'T_i_ev':27},simulation_raw_numeric_data_found=False,axis_calibration=panels,axis_y0=12,axis_y1=474,axis_time_limits_ns=[1.5,4.5],method='Strict saturated red/blue ink per native column; min/max selected ink; omit absent columns; no interpolation. Derived midpoint is not simulation centerline.',color_threshold=dict(dominant_channel_min_exclusive=220,other_channels_max_exclusive=90),reading_uncertainty='Conservative +/-2 native pixels: Te +/-10.82 eV, Ti +/-8.66 eV, time +/-0.01247 ns. Not calibrated confidence interval. Sharp rises amplify horizontal uncertainty.',limitations=['No distinct center curve reliably identified; colored outlines bound spatial-inhomogeneity shading.','Envelopes include stroke thickness and may miss edges obscured by experimental markers; both_physical_bounds_certified=false for every row.','Shading is plasma inhomogeneity, not measurement statistical error.','Experimental square values inferred by spectral fitting are not simulation truth.','Common Ti does not give separate H/C temperature truth.','No quantitative flow reference; text only says central flow and Doppler shift are small.'],correspondence={'fig10_a':'Fig11 experimental purple squares','fig10_b':'blue without probe','fig10_c':'red with probe'},geometry=dict(location='hohlraum center; simulation Z=R=0',scattering_angle_deg=90,scattering_volume_um=[100,100,100],fill='C5H12 0.35atm',fully_ionized_fill_baseline_ne_cm3=3.5e20,density_caveat='Fill baseline only, not per-time local density',instrument_fwhm_nm=.057,probe_energy_J=90,probe_duration_ns=3,probe_delay_ns=.3,main_pulse_start_ns=1.5),rows_per_curve={f'{p}/{c}':len(q) for (p,c),q in r.groupby(['parameter','condition'])})
(H/'reference_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(H/'README.txt').write_text('Fig11 references\n\nfig11_experimental_points.csv: 54 existing square readings, 27 Te/27 Ti; all original fields retained.\n\nfig11_simulation_ink_envelopes.csv: native-image saturated color-ink envelopes only. derived_ink_midpoint_ev is an image midpoint, NOT original simulation centerline or numeric truth. No distinct center curve is established. Shaded regions represent spatial inhomogeneity, not statistical error. Stroke thickness and marker occlusion affect bounds; missing columns remain absent. See selection overlay and audit JSON.\n\nFig10b=blue without probe, Fig10c=red with probe. Common Ti does not give separate TH/TC. Fig11 has no flow curve; prose small flow does not mean exactly zero. 3.5e20 cm^-3 is fully-ionized fill baseline only.\n\nReproduce: /workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python '+str(Path(__file__).resolve())+'\n')
print(json.dumps(audit['rows_per_curve']))
print('54 points; calibration agrees with preexisting CSV to <1e-10 eV.')
