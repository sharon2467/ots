import json,pathlib,numpy as np
base=pathlib.Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020')
a={stage:json.loads((base/'exploration'/f'study_fig_{stage}_validation_metrics.json').read_text()) for stage in 'abc'}
v={stage:np.mean([[x['evdf_relative_l2'] for x in a[stage]['species'][sp]['model']['per_sample']] for sp in ['e','i1','i2']],axis=0) for stage in 'abc'}
rng=np.random.default_rng(20261020);dif=(v['b']-v['c'])[rng.integers(0,750,(2000,750))].mean(1)
comparison={'validation_only':True,'test_metrics_read':False,'macro_means':{k:float(x.mean()) for k,x in v.items()},'B_minus_C_mean':float((v['b']-v['c']).mean()),'B_minus_C_paired_CI95':np.quantile(dif,[.025,.975]).tolist()}
(base/'exploration/fig_abc_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n');print(comparison)
