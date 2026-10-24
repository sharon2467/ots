import hashlib,json,re
from pathlib import Path
import numpy as np
root=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/OTS_code_share/NN_hotE_large_lmRange/runs')
out=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020/exploration')
def hashed(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): d.update(block)
 return d.hexdigest()
result={'test_metrics_read':False,'test_parquet_access':'binary SHA256 only; no parsing','runs':{}}
audits={}
for label in ('a','b','c'):
 name='study_fixed_'+label
 audit=json.loads((out/(name+'_validation_metrics.json')).read_text())
 audits[label]=audit
 history=json.loads((root/name/'training/training_history.json').read_text())
 candidates=[]
 for candidate in history['candidates']:
  records=candidate['history']
  candidates.append({**{key:candidate[key] for key in ('hidden_sizes','dropout','best_epoch','best_validation_loss','elapsed_seconds','stop_reason','parameter_count')},'epochs_completed':len(records),'first_epoch':records[0],'best_epoch_record':min(records,key=lambda r:r['validation_loss']),'last_epoch':records[-1],'last10_validation_mean':float(np.mean([r['validation_loss'] for r in records[-10:]]))})
 log=Path('/workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/training_study_20261020/logs')/(name+'.log')
 timing=[line for line in log.read_text().splitlines() if re.match(r'^(real|user|sys)\s',line)]
 result['runs'][label]={'run_id':name,'selected_candidate':history['selected_candidate'],'validation_metrics':audit['species']['e']['model']['metrics'],'validation_baseline':audit['baseline_selection_score'],'validation_index_hash':audit['split']['validation_indices_sha256'],'hashes':{split:hashed(root/name/'data'/('dataset_'+split+'.parquet')) for split in ('train','test')},'candidates':candidates,'wall_timing':timing}
result['all_train_hash_equal']=len({v['hashes']['train'] for v in result['runs'].values()})==1
result['all_test_hash_equal']=len({v['hashes']['test'] for v in result['runs'].values()})==1
assert all(audits[label]['split']['validation_indices']==audits['a']['split']['validation_indices'] for label in audits)
values={label:np.array([r['evdf_relative_l2'] for r in audit['species']['e']['model']['per_sample']]) for label,audit in audits.items()}
result['validation_paired_comparisons']={}
for left,right in (('a','b'),('b','c'),('a','c')):
 delta=values[left]-values[right]
 rng=np.random.default_rng(20261020)
 means=np.array([delta[rng.integers(0,len(delta),size=len(delta))].mean() for _ in range(2000)])
 result['validation_paired_comparisons'][left+'_minus_'+right]={'mean_relative_l2_difference':float(delta.mean()),'ci95_percentile_bootstrap_2000':np.quantile(means,[.025,.975]).tolist(),'fraction_samples_right_better':float((values[right]<values[left]).mean()),'relative_mean_improvement':float(1-values[right].mean()/values[left].mean())}
result['selected_by_common_validation_relative_l2']='study_fixed_'+min(values,key=lambda label:values[label].mean())
result['unpersisted_candidate_limitation']='C nonwinning candidates have training loss and epoch histories only; VDF validation metrics apply only to the saved winning checkpoint.'
path=out/'fixed_abc_comparison.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(path)
