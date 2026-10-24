#!/usr/bin/env python3
"""Plot common-development error against pool size and charged full wall time.

Reads trials.json, old-C common-development audits and old Bash-time logs only.
Does not read test metrics, prediction arrays, or raw Parquets.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import argparse
import hashlib
import json
import math
import re
import sys
sys.dont_write_bytecode=True
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent.parent
OLD=ROOT/'outputs/training_study_20261020'
CONFIG={'fixed_ion':('fixed','Fixed ion: electron', [1024,1024]),
        'fig1_fig11':('fig','Fig. 1/11: equal e/H/C macro', [1024,512])}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_old_wall(path):
    # Parse authoritative Bash real, without consuming any evaluation outputs.
    matches=re.findall(r'^real\s+(\d+)m([\d.]+)s\s*$',path.read_text(),re.MULTILINE)
    if len(matches)!=1:raise ValueError('Expected one Bash real timing in '+str(path))
    minutes,seconds=matches[0]
    return int(minutes)*60+float(seconds)


def rows(args):
    trials=json.loads(args.trials.read_text());data=[];sources=[args.trials]
    for kind,(short,title,width) in CONFIG.items():
        audit_path=args.old_audit_dir/('study_'+short+'_c_common_validation_metrics.json')
        log=args.old_logs/('study_'+short+'_c.log')
        audit=json.loads(audit_path.read_text())
        if audit['dataset_kind']!=kind or audit['counts']['common_validation_count']!=750 or not audit['overlap_check']['zero_overlap']:
            raise ValueError('Old common-development audit gate failed')
        data.append({'kind':kind,'run':'study_'+short+'_c','category':'original',
                     'label':'Old C','pool':audit['counts']['train_pool_count'],
                     'error':audit['selection_score'],'wall_seconds':actual_old_wall(log)})
        sources.extend([audit_path,log])
    for trial in trials:
        if trial.get('status')!='completed' or 'common_development' not in trial:continue
        kind=trial['dataset_kind'];width=CONFIG[kind][2]
        common=trial['common_development']
        if common['count']!=750 or common['zero_overlap_with_optimizer'] is not True:
            raise ValueError('Trial common-development gate failed: '+trial['run_id'])
        options=trial['training_options'];hidden=trial['selected_candidate']['hidden_sizes']
        tail='parent_all_run' in trial
        if hidden!=width:category='width';label='Wide '+str(hidden[0])+' x '+str(hidden[1])
        elif options['learning_rate']!=.001:category='low_lr';label='LR '+format(options['learning_rate'],'.1g')
        elif tail:category='other';label=trial['run_id']
        else:category='original';label=str(trial['train_pool']//1000)+'k'
        wall=trial['full_wall_seconds']
        if tail:
            charged=trial['charged_parent_all_wall_seconds']+trial['tail_train_evaluate_wall_seconds']
            if not math.isclose(wall,charged,rel_tol=0,abs_tol=.05):
                raise ValueError('Tail full wall must include parent all plus train/evaluate wall: '+trial['run_id'])
        data.append({'kind':kind,'run':trial['run_id'],'category':category,'label':label,
                     'pool':trial['train_pool'],'error':common['selection_score'],'wall_seconds':wall})
    if any(not math.isfinite(r['error']) or r['error']<0 or not math.isfinite(r['wall_seconds']) or r['wall_seconds']<=0 for r in data):
        raise ValueError('Invalid error/timing value')
    return data,sources


def main(args):
    paths=[args.output_prefix.with_suffix(suffix) for suffix in ('.png','.pdf','.json')]
    if any(p.exists() for p in paths):raise FileExistsError('Refusing to overwrite plot artifacts')
    data,sources=rows(args)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),sharey='row')
    colors={'original':'#2563a6','width':'#b34a18','low_lr':'#6c4ca0','other':'#287d53'}
    markers={'width':'^','low_lr':'s','other':'D'}
    for row,(kind,(_,title,_)) in enumerate(CONFIG.items()):
        branch=[r for r in data if r['kind']==kind]
        original=sorted((r for r in branch if r['category']=='original'),key=lambda r:r['pool'])
        if len(original)<3:raise ValueError('Old C,12k,and larger original architecture results required')
        for col,key in enumerate(('pool','wall_seconds')):
            ax=axes[row,col];scale=1000 if col==0 else 60
            ax.plot([r[key]/scale for r in original],[r['error']*100 for r in original],
                    color=colors['original'],marker='o',lw=1.7,label='Original architecture / LR')
            for r in branch:
                x=r[key]/scale;y=r['error']*100
                if r['category']!='original':
                    ax.scatter(x,y,s=65,marker=markers[r['category']],color=colors[r['category']],
                               edgecolors='white',linewidth=.7,zorder=4,label=r['label'])
                # Offset equal-N tail labels so point names stay readable.
                offset=(7,-16) if r['category']=='low_lr' else (7,8)
                ax.annotate(r['label'],(x,y),xytext=offset,textcoords='offset points',fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('Candidate training pool N (thousands)' if col==0 else 'Conservative complete workflow wall (minutes)')
            if col==0:ax.set_ylabel('Common-development VDF relative-L2 (%)')
            else:
                ax.axvline(120,color='.55',ls=':',lw=1,label='120-minute limit')
                ax.set_xlim(left=0)
            ax.grid(alpha=.2)
            ax.margins(x=.12,y=.22)
            ax.legend(fontsize=7,loc='best')
    fig.suptitle('Development only: common old 750 held-out states',fontsize=15,y=.98)
    fig.text(.06,.035,'Lines: old C, 12k and larger pools at each branch\'s original selected architecture/LR. Tail variants are separate points.\n'
             'Old C generation/split seed 20261020; expanded runs seed 20261030. This is not a pure controlled N-only experiment.\n'
             'Tail cost = parent complete generate/train/evaluate wall + tail train/evaluate wall. Old C cost includes its three-candidate search.',
             fontsize=9,va='bottom',color='.3')
    fig.tight_layout(rect=(0,.14,1,.95))
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(paths[0],dpi=180);fig.savefig(paths[1]);plt.close(fig)
    payload={'purpose':'Development only, no test reads','points':data,
             'sources':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in sources],
             'test_artifacts_read':False,'outputs':[str(p) for p in paths[:2]],
             'cautions':['Adaptive reuse of common old 750 development states','Old C seed differs from expanded runs',
                         'Original architecture line does not connect width/low-LR tail variants',
                         'Tail timing conservatively includes entire parent all; old C includes three-candidate search']}
    with paths[2].open('x') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'outputs':[str(p) for p in paths],'points':len(data)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=Path,default=BASE/'trials.json')
    p.add_argument('--old-audit-dir',type=Path,default=Path(__file__).parent)
    p.add_argument('--old-logs',type=Path,default=OLD/'logs')
    p.add_argument('--output-prefix',type=Path,default=Path(__file__).parent/'development_scaling')
    main(p.parse_args())
