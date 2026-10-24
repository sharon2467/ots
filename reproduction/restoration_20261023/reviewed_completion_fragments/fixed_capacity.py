from pathlib import Path
import json,numpy as np
b=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_scale_20261020/exploration')
a=json.loads((b/'scale_fixed_30k_common_validation_metrics.json').read_text());w=json.loads((b/'scale_fixed_30k_wide_common_validation_metrics.json').read_text())
x=np.array([v['evdf_relative_l2'] for v in a['species']['e']['model']['per_sample']]);y=np.array([v['evdf_relative_l2'] for v in w['species']['e']['model']['per_sample']]);d=x-y;r=np.random.default_rng(20261020);ci=np.quantile([d[r.integers(0,len(d),size=len(d))].mean() for _ in range(2000)],[.025,.975])
result={'test_metrics_read':False,'interpretation':'Repeated common-development comparison; not an unbiased test confidence statement','wide_metrics':w['species']['e']['model']['metrics'],'base_relative_l2_mean':float(x.mean()),'wide_relative_l2_mean':float(y.mean()),'relative_mean_improvement':float(1-y.mean()/x.mean()),'base_minus_wide_difference_mean':float(d.mean()),'paired_percentile_bootstrap_2000_difference_ci95':ci.tolist(),'fraction_samples_wide_better':float((y<x).mean()),'common_index_hash_equal':a['indices']['common_indices_sha256']==w['indices']['common_indices_sha256']}
p=b/'fixed_30k_capacity_comparison.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
