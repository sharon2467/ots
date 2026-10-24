"""Read-only native figure extraction and colorbar-coordinate audit; no spectral forward calculation."""
from pathlib import Path
import hashlib,json
import fitz
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PDF=ROOT/'Zhao 等 - 2019 - Progress in optical Thomson scattering diagnostics for ICF gas-filled hohlraums.pdf'
doc=fitz.open(PDF)
fig=doc.extract_image(307)
(OUT/'fig10_native.jpg').write_bytes(fig['image'])
im=Image.open(OUT/'fig10_native.jpg').convert('RGB')
im.save(OUT/'pdf10-000.png')
a=np.asarray(im)
values=np.arange(1,9,dtype=float)*.05
bars={}
for panel,x,box in [('b',1232,(1170,0,1390,510)),('c',1947,(1880,0,2076,510))]:
    im.crop(box).save(OUT/f'colorbar_{panel}_native_crop.png')
    dark=np.max(a[:510,x-2:x+1].mean(axis=1),axis=1)<180
    groups=[]
    for y in np.flatnonzero(dark):
        if groups and y==groups[-1][-1]+1: groups[-1].append(int(y))
        else: groups.append([int(y)])
    assert len(groups)==10,groups
    centers=np.array([np.mean(g) for g in groups[1:-1]][::-1])
    linear=np.polyfit(values,centers,1)
    logarithmic=np.polyfit(np.log(values),centers,1)
    bars[panel]={
      'tick_values_visually_transcribed':values.tolist(),
      'tick_y_native_pixels':centers.tolist(),
      'measurement':'Mean y of dark tick groups at meanRGB(x-2:x+1)<180; excludes top/bottom borders. Tick values read visually from native embedded image.',
      'native_tick_detection_x':x,
      'linear_fit_y_intercept':float(linear[1]),
      'linear_fit_y_slope':float(linear[0]),
      'linear_fit_rms_pixel_residual':float(np.sqrt(np.mean((np.polyval(linear,values)-centers)**2))),
      'log_fit_rms_pixel_residual':float(np.sqrt(np.mean((np.polyval(logarithmic,np.log(values))-centers)**2))),
      'power_at_top_border_y13':float((13-linear[1])/linear[0]),
      'power_at_bottom_border_y496':float((496-linear[1])/linear[0]),
      'interpretation':'Linear scattered-power colorbar. Extrapolated endpoint powers are approximate, not printed endpoints.'}

claims=[
 {'topic':'Maxwellian simulation model','status':'explicit linkage','pdf_pages':[4,5], 'text_lines':[228,272,318,323],
  'evidence':'Sec II B gives Maxwellian distribution functions Eqs (5),(6), susceptibilities Eqs (7)-(9); Sec II C: The spectra are calculated according to Eqs. (1)-(9), using the plasma conditions given by the 2D radiation-hydrodynamic code LARED.',
  'limit':'A volume sum of local Maxwellians need not equal a single homogeneous Maxwellian spectrum.'},
 {'topic':'single common ion temperature','status':'explicit fitting/reporting convention; simulation implementation incompletely specified','pdf_pages':[4,10], 'text_lines':[221,228,600,602],
  'evidence':'Seven fitting variables (Te, Ti, uik, uek, A, B1, B2); Fig11 reports one ion temperature Ti. Eq8 itself has generic species temperature Tj.',
  'limit':'No published independent TH/TC values or explicit statement that separate H/C temperatures were advanced by LARED. Treat common Ti for fitting as established, equality of all simulated species temperatures as strongly implied rather than a verbatim statement.'},
 {'topic':'volume integration','status':'explicit','pdf_pages':[5,6], 'text_lines':[324,336,356,366],
  'evidence':'The scattering volume is the overlap of probe beam and diagnostic aperture; each element receives local (Te,Ti,ne,etc.), spectra are calculated separately and summed. Spectral response, spectral resolution, inverse bremsstrahlung along optical path and temporal resolution enter the calculation.',
  'limit':'This is a finite scattering-volume integral, not an unlimited line-of-sight integral. Exact weights/meshes/kernels are not disclosed.'},
 {'topic':'instrument resolution','status':'explicit inclusion; kernel unspecified','pdf_pages':[6,9], 'text_lines':[356,366,507,510],
  'evidence':'SecII C explicitly includes spectral resolution; SecIV gives spherical experiment spectral resolution 0.057nm FWHM.',
  'limit':'Paper does not explicitly state a Gaussian kernel, a numerical 0.057nm convolution procedure, or exact sequence of all instrument operations. Gaussian convolution is a reasonable modeling hypothesis, not a published implementation fact.'},
 {'topic':'density','status':'explicit local versus baseline distinction','pdf_pages':[5,9], 'text_lines':[330,334,537,539],
  'evidence':'0.35atm C5H12 corresponds to electron density about3.5e20cm^-3 when fully ionized. Fig1b uses4.0e20cm^-3 at4.3ns; SecII B states simulated density spans3e20-5e20cm^-3 across volume for that example.',
  'limit':'3.5e20 is fill baseline, not fixed ne for every time and local element.'},
 {'topic':'species charges and mixture','status':'gas composition explicit; detailed simulation charge state unreported','pdf_pages':[3,4,9], 'text_lines':[181,185,210,212,537,539],
  'evidence':'Pentane C5H12; generic multi-ion theory carries Zj,Nj; density statement qualifies when fully ionized; ion charge is an assumed fitting input.',
  'limit':'Stoichiometric H:C=12:5 and fully stripped ZH=1,ZC=6 are justified fully-ionized pentane assumptions. Paper does not explicitly publish time-/cell-resolved charge state or any gold contamination in Fig10 simulations.'},
 {'topic':'flows','status':'explicit finite flow convention and small center flow','pdf_pages':[4,9], 'text_lines':[207,212,289,293,524,529],
  'evidence':'Ion susceptibilities/distributions shifted by k.u and electrons by k.(u+ud). Geometry makes center plasma flow quite small, giving small Doppler shift.',
  'limit':'Quite small does not establish exactly zero; one ion flow and electron drift in published fitting differs from independently sampled ion flows.'},
 {'topic':'simulation versus experiment temperatures','status':'explicit','pdf_pages':[9,10], 'text_lines':[524,545,565,584,600,602],
  'evidence':'Fig10b without probe, c with probe; Fig11 blue/red lines are corresponding LARED temperatures; squares are experiment. Major increase-in-Te discrepancy after2.8ns is discussed; code uses local Spitzer-Harm conduction.',
  'limit':'Sampling only around experimental Fig11 squares does not automatically cover the simulation temperature trajectory.'}
]
evidence={
 'source_pdf':str(PDF),'source_pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
 'figure10':{'pdf_page':10,'journal_page':'055201-9','xref':307,'native_width':2076,'native_height':600,
 'encoded_image_file':'fig10_native.jpg','encoded_image_sha256':hashlib.sha256(fig['image']).hexdigest(),
 'caption':'Comparison of experimental and simulated Thomson scattering spectra for octahedral spherical hohlraum (diagnosed at hohlraum center). (a) Measured spectra in experiment. (b) Simulated spectra without probe beam. (c) Simulated spectra with probe beam.'},
 'comparison_source':{'path':'OTS_code_share/NN_hotE_large_lmRange/data.py','sha256':hashlib.sha256((ROOT/'OTS_code_share/NN_hotE_large_lmRange/data.py').read_bytes()).hexdigest(),'relevant_lines':[213,296,335,456]},
 'colorbars':bars,'claims':claims,
 'coordinates_note':'Native embedded image integer pixel coordinates; no rendered-page scaling. Paper text line references refer to paper_layout.txt produced by pdftotext -layout.'}
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(evidence['colorbars'],indent=2))
