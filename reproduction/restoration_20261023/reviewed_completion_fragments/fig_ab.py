import json,pathlib,hashlib,numpy as np
base=pathlib.Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020');source=pathlib.Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange')
vals={s:json.loads((base/'exploration'/f'study_fig_{s}_validation_metrics.json').read_text()) for s in ['a','b']}
a=np.mean([[x['evdf_relative_l2'] for x in vals['a']['species'][s]['model']['per_sample']] for s in ['e','i1','i2']],axis=0);b=np.mean([[x['evdf_relative_l2'] for x in vals['b']['species'][s]['model']['per_sample']] for s in ['e','i1','i2']],axis=0)
rng=np.random.default_rng(20261020);diff=(a-b)[rng.integers(0,750,(2000,750))].mean(1)
out={'validation_only':True,'test_data_read':False,'A_macro':float(a.mean()),'B_macro':float(b.mean()),'paired_A_minus_B_95_CI':np.quantile(diff,[.025,.975]).tolist(),'mean_improvement':float((a-b).mean()),'relative_reduction':float(1-b.mean()/a.mean())}
for tag in ['train','test']:
 files=[source/'runs'/('study_fig_'+s)/'data'/('dataset_'+tag+'.parquet') for s in ['a','b']]
 hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files];out[tag+'_parquet_sha256']=hashes;assert hashes[0]==hashes[1]
(base/'exploration/fig_ab_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

