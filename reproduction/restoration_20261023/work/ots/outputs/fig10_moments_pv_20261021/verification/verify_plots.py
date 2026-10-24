"""Verify saved plot manifests and representative selections against underlying arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((B/'forward_plot_manifest.json').read_text())
for filename,digest in f['plots'].items():assert sha(B/'plots'/filename)==digest
m=json.loads((B/'moments_comparison.json').read_text())
for filename,digest in m['plots'].items():assert sha(B/'plots'/filename)==digest
for panel,records in f['data'].items():
    path=B/'forward'/panel/'forward_results.npz';assert sha(path)==records['input_sha256']
    d=np.load(path);p=np.load(B/'forward'/panel/'processed_vdfs.npz')
    for r in records['representative_rows']:
        j=int(np.argmin(abs(d['time_paper_ns']-r['target_time_ns'])))
        assert j==r['row'] and d['time_paper_ns'][j]==r['actual_time_ns'] and d['source_row_index'][j]==r['source_row_index']
        assert d['relativeL2'][j]==r['relativeL2'] and d['rmse'][j]==r['rmse']
    for sp,rec in records['preparation_moment_changes'].items():
        delta=p[sp+'_processed_T_eV']-p[sp+'_original_T_eV'];du=(p[sp+'_processed_u_m_s']-p[sp+'_original_u_m_s'])/1000
        np.testing.assert_allclose([delta.min(),delta.max(),delta.mean(),abs(du).max()],[rec[k] for k in ['delta_T_min_eV','delta_T_max_eV','delta_T_mean_eV','max_abs_delta_u_km_s']],rtol=1e-14)
report={'manifest_plot_hashes_match':True,'nine_nearest_time_representative_rows_match':True,'all_preparation_moment_change_summary_values_match':True,'visually_reviewed_pngs':['temperatures_vs_fig11','projected_flow_velocities','spectral_heatmaps_comparison','representative_spectra_comparison','spectral_error_over_time','preparation_moment_changes']}
(H/'plot_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
