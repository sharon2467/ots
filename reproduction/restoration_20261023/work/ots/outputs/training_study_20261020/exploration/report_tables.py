#!/usr/bin/env python3
"""Render final report tables from saved results; never select or train models."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.mplconfig'))
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent.parent
EX=BASE/'exploration'
read=lambda p:json.loads(p.read_text())
pct=lambda x:f'{100*x:.3f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'

def main():
    selection=read(BASE/'frozen_selection.json');summary=read(BASE/'results_summary.json')
    assert selection['frozen'] is True and summary['test_metrics_read'] is True
    sections=[]
    rows=[]
    for c in summary['network_candidates']:
        rows.append([c['run_id'],'×'.join(map(str,c['hidden_sizes'])),c['dropout'],c['raw_loss_weight'],c['parameter_count'],f"{c['best_epoch']}/{c['epochs_completed']}",f"{c['best_validation_loss']:.6f}",f"{c['best_epoch_history']['train_loss']:.6f}",f"{c['elapsed_seconds']:.1f}s",'是' if c['selected'] else '否'])
    sections.append('## 全部网络尝试\n\n'+table(['Run','隐藏层','Dropout','raw权重','参数量','最佳/停止epoch','最佳val loss','对应train loss','训练耗时','保留checkpoint'],rows))
    rows=[]
    for r in summary['runs']:
        val=' / '.join(pct(v['evdf_relative_l2_mean']) for v in r['validation_species'].values())
        test=' / '.join(pct(v['evdf_relative_l2_mean']) for v in r['test_species'].values())
        rows.append([r['run_id'],'×'.join(map(str,r['selected_candidate']['hidden_sizes'])),val,pct(r['validation_macro_relative_l2_mean']),test,pct(r['test_macro_relative_l2_mean']),f"{r['total_wall_seconds']/60:.2f}min"])
    sections.append('## 六次完整流水线\n\n'+table(['Run','保留网络','Val各物种L2','Val宏平均','Test各物种L2','Test宏平均','完整耗时'],rows))
    finalrows=[];temprows=[];originalrows=[];plotrows=[]
    for kind in ['fixed_ion','fig1_fig11']:
        name=selection['best'][kind]['run_id']
        final=read(EX/'final_diagnostics'/name/'detailed_test.json')
        aux=read(EX/'auxiliary_temperature_final'/name/'auxiliary_temperature_test.json')
        original=read(EX/'original_vdf_final'/name/'original_vdf_test.json')
        for j,s in enumerate(final['species_order']):
            label=('fixed e' if kind=='fixed_ion' else 'Fig '+{'e':'e','i1':'H','i2':'C'}[s])
            entry=final['species'][s];m=entry['model'];b=entry['train_mean_baseline'];ci=final['bootstrap']['species_model_relative_l2_mean_ci95'][j]
            finalrows.append([label,pct(m['evdf_relative_l2_mean']),f'{pct(ci[0])}–{pct(ci[1])}',pct(m['evdf_relative_l2_median']),pct(m['evdf_relative_l2_p95']),f"{m['evdf_global_flattened_r2']:.6f}",pct(m['negative_mass_fraction_mean']),pct(b['evdf_relative_l2_mean'])])
            estimators=aux['species'][s]['estimators'];raw=estimators['raw_signed_vdf_model'];tay=estimators['fixed_taylor3_model'];bias=estimators['fixed_taylor3_true_fourier_bias']
            temprows.append([label,f"{raw['all_finite']['mae_ev']:.2f}",f"{tay['all_finite']['mae_ev']:.2f}",pct(tay['all_finite']['median_absolute_relative_error']),f"{raw['negative_temperature_count']} / {tay['negative_temperature_count']}",f"{raw['nonfinite_count']} / {tay['nonfinite_count']}",f"{bias['all_finite']['mae_ev']:.2f}"])
            o=original['species'][s];model=o['model_vs_original']['raw_vdf_relative_l2'];target=o['target_fourier_vs_original']['raw_vdf_relative_l2']
            originalrows.append([label,pct(model['mean']),pct(model['p95']),pct(target['mean']),pct(target['p95']),pct(target['max'])])
            plotrows.append((label,m['evdf_relative_l2_mean'],b['evdf_relative_l2_mean'],ci))
    sections.append('## 最佳模型的独立测试\n\n'+table(['目标','平均L2','95% CI','中位L2','p95 L2','全局R²','平均负质量','均值基线L2'],finalrows))
    sections.append('## 温度诊断\n\n'+table(['目标','直接积分MAE/eV','Taylor3 MAE/eV','Taylor3中位相对误差','负T 直接/Taylor','非有限 直接/Taylor','真Fourier经Taylor3偏差MAE/eV'],temprows))
    sections.append('## 原始分布与低通标签\n\n'+table(['目标','模型→原VDF均值L2','模型→原VDF p95','标签→原VDF均值L2','标签→原VDF p95','标签→原VDF最大L2'],originalrows))
    (BASE/'report_tables.md').write_text('\n'.join(sections))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    x=np.arange(3)
    for kind,label,offset,color in [('fixed_ion','Fixed ion',-.18,'#31688e'),('fig1_fig11','Fig 1 / Fig 11',.18,'#35b779')]:
        runs=[r for r in summary['runs'] if r['dataset_kind']==kind]
        axes[0].bar(x+offset,[r['validation_macro_relative_l2_mean']*100 for r in runs],.34,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=['A: raw 0.1','B: raw 1','C: capacity'],ylabel='Validation mean relative L2 (%)',title='Selection used validation only')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    x=np.arange(len(plotrows));pred=np.array([r[1] for r in plotrows])*100;prior=np.array([r[2] for r in plotrows])*100;ci=np.array([r[3] for r in plotrows])*100
    axes[1].bar(x-.18,pred,.34,color='#31688e',label='Frozen model',yerr=np.vstack([pred-ci[:,0],ci[:,1]-pred]),capsize=3)
    axes[1].bar(x+.18,prior,.34,color='#b7b7b7',label='Training mean baseline')
    axes[1].set(xticks=x,xticklabels=[r[0] for r in plotrows],ylabel='Test mean relative L2 (%)',title='512 independent test states; 95% bootstrap CI')
    axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    fig.savefig(BASE/'study_summary.png',dpi=170);plt.close(fig)
    print(json.dumps({'tables':str(BASE/'report_tables.md'),'plot':str(BASE/'study_summary.png')}))

if __name__=='__main__':main()
