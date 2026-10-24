import json
from pathlib import Path
root=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange/runs')
out=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020/exploration/final_diagnostics')
result={}
for name in ('study_fixed_c','study_fig_c'):
 d=json.loads((out/name/'detailed_test.json').read_text())
 c=json.loads((root/name/'evaluation/metrics.json').read_text())['metrics']
 if name=='study_fixed_c': c={'e':c}
 print(name, 'macro', d['macro_model_relative_l2_mean'], 'ci95', d['bootstrap']['macro_model_relative_l2_mean_ci95'], 'baseline',d['macro_baseline_relative_l2_mean'], 'baseline_minus_model_ci95',d['bootstrap']['macro_baseline_minus_model_mean_ci95'])
 differences={}
 for index,(species,s) in enumerate(d['species'].items()):
  m=s['model']
  pairs={'evdf_relative_l2_mean':'evdf_relative_l2_mean','evdf_relative_l2_median':'evdf_relative_l2_median','negative_mass_fraction_mean':'negative_mass_fraction_mean','evdf_global_flattened_r2':'evdf_r2'}
  differences[species]={k:m[k]-c[species][ck] for k,ck in pairs.items()}
  assert all(abs(v)<1e-6 for v in differences[species].values()), differences
  print(species, json.dumps({'model':m,'baseline_mean':s['train_mean_baseline']['evdf_relative_l2_mean'],'mean_ci95':d['bootstrap']['species_model_relative_l2_mean_ci95'][index],'negative_temperature_count':s['moments']['model']['negative_temperature_count'],'nonpositive_mass_count':s['moments']['model']['nonpositive_mass_count'],'temperature_comparisons':s['temperature_comparisons']},indent=2))
 print('canonical differences',differences)
 result[name]={'canonical_comparison_absolute_tolerance':1e-6,'differences_diagnostic_minus_canonical':differences,'passed':True}
path=out/'canonical_consistency.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(path)
