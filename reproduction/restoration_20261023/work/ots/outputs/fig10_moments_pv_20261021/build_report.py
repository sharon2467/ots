#!/usr/bin/env python3
"""Assemble a readable report from saved numerical results."""
import csv
import json
from pathlib import Path

R=Path(__file__).resolve().parent
def main():
    f=json.loads((R/'forward/summary.json').read_text())
    m=json.loads((R/'moments_comparison.json').read_text())
    p=json.loads((R/'forward_plot_manifest.json').read_text())
    rows=list(csv.DictReader((R/'moments/signed_moments_all_rows.csv').open()))
    lines=[
        'Fig10 最新推理：温度、流速、Fig11 对照及现有 PV 光谱闭环',
        '日期：2026-10-21；所有时间单位为 ns，温度为 eV，流速为 km/s。',
        '',
        '结果：当前模型在这些实验/抠图数据上的物理一致性不足。实验 a 的温度明显偏高，尤其 H；',
        'b/c 电子温度只部分跟随论文变化，离子温度整体仍有显著偏离。固定密度下的重建光谱也有明显残差。',
        '合成测试集的 VDF 宏误差 5.457244% 不能作为这三组数据的实验反演精度。',
        '',
        '一、冻结来源与本次范围',
        '模型：depth_fig_d4_s0，4 隐藏层 2048/1024/1024/1024，5,354,976 参数；Fig26k 池，22,100 优化样本。',
        '本次没有重训或重新推理神经网络。读取 fig10_nearest_d4_inference_20261021 中最新 VDF。',
        '光谱对比读取 fig10_nearest_row_fill_20261021：a 已扣背景1691；b/c 刻度/标签已用本行最近原正常值填补。',
        '没有重复扣底、填补、平移或改变时间轴。a/b/c 分别 556/480/480 条，原波长点数 527/319/320。',
        '原始生成、训练、评估与物理代码均未改。所有新脚本、派生副本及图保存在本目录。',
        '',
        '二、原始推理矩定义（温度对比主结果）',
        'A = integral f(v)dv；u = integral v*f(v)dv / A；T[eV] = m/e * integral (v-u)^2*f(v)dv / A。',
        '各物种用原保存的4097点完整速度轴与有符号VDF，float64梯形积分；一维投影温度，没有1/3因子。',
        'm_e=9.1093837015e-31 kg；m_H=1.67262192369e-27 kg；m_C=12*m_H；e=1.602176634e-19 C。',
        '未裁尾、拟合Maxwellian、平滑或调整温度。全部4548个物种行温度有限且为正。',
        '但原VDF每行都有负旁瓣，所以这些是原始有符号函数的代数矩，不能仅凭为正就宣称物理分布已验证。',
        '流速是保存速度轴方向上的分量，不是三维流速大小；符号沿现有轴约定。',
        '',
        '三、与 Fig11 的比较',
        '论文：Zhao et al. 2019, Progress in optical Thomson scattering diagnostics for ICF gas-filled hohlraums，PDF第10页。',
        'Fig10a 对应 Fig11 实验紫色方块；b 对应无探针蓝色模拟；c 对应有探针红色模拟。',
        '论文只有电子温度与单一离子温度，没有独立H/C真值，因此将H和C分别与同一个Ti参考对照。',
        '实验Te/Ti各27点沿用已有取点CSV；计算时仅将已求出的矩时序线性插值到这些参考时刻以比较。',
        '模拟缺原始数值表，只读取原生PDF位图颜色墨迹外包络，缺列不补；不把中点冒充模拟真实中心线。',
        '图中阴影是散射体积内空间不均匀，不是统计置信区间。图像读取约±2px：Te±10.82eV、Ti±8.66eV、时间±0.01247ns。',
        '不对模拟墨迹报“准确率”。实验数值比较亦为与论文拟合温度的差异，不是独立VDF真值误差。',
        '',
        'a 对论文27个相应实验点的差异：',
        '物种       MAE(eV)      RMSE(eV)       偏差均值(eV)']
    for s,x in m['experimental_errors'].items():
        lines.append(f'{s:4s} {x["MAE_eV"]:14.3f} {x["RMSE_eV"]:14.3f} {x["bias_eV"]:18.3f}')
    lines += ['', '原始推理流速全时段范围(km/s)：', '组别              e                       H                       C']
    for g in ('fig10a','fig10b','fig10c'):
        vals=[]
        for s in ('e','H','C'):
            v=[float(x['u_km_s']) for x in rows if x['panel']==g and x['species']==s]
            vals.append(f'{min(v):.3f} 至 {max(v):.3f}')
        lines.append(g+'    '+'    '.join(vals))
    lines += [
        'Fig11没有定量流速参考。论文“中心流动较小”的定性描述不能作为u=0的测量真值。',
        '',
        '代表时刻（均取最接近2/3/4ns的原数据行）：',
        '组别  目标/实际ns  物种  T(eV)  u(km/s)']
    for r in m['representative_rows']:
        lines.append(f'{r["panel"]} {r["target_time_ns"]:.0f}/{float(r["time_paper_ns"]):.6f} {r["species"]:2s} {float(r["T_eV"]):9.3f} {float(r["u_km_s"]):9.3f}')
    lines += [
        '',
        '尾部检查：a/H在2/3/4ns，|v|>1500km/s部分仅占中心二阶矩约2.54%/2.64%/3.01%，',
        '因此a/H约2keV高温不是主要由该远端尾部造成。b/e在2ns窗外二阶矩占比约27.31%，显示个别行对尾部敏感。',
        '',
        '四、现有 OTS_spectrum_PV 光谱重建',
        '唯一正向实现：OTS_code_share/NN_hotE_large_lmRange/phyEqns/ots_full_fourier.py::OTS_spectrum_PV。',
        '原样输入被其网格/非负/单位积分校验拒绝；原函数未被修改或绕过。',
        '用户随后明确允许独立副本处理及固定密度。严格四步：',
        '1) 原速度端点不变，float64 linspace生成4097点；2) 原VDF线性插值到该网格，无外推；',
        '3) 只把负值置0；4) 调用原normalize_distribution使梯形积分为1。',
        '修复网格最大位移e/H/C分别为8.46094/0.16968/0.05463 m/s；全量processed_vdfs.npz保留副本。',
        '原有符号VDF及原始矩不被覆盖。小负质量不等于矩变化小，裁负对二阶矩的影响单独作图和记录。',
        '采用现有Fig物理配置：lambda0=263.3nm，90°，H:C数目比12:5，Z_H=1、Z_C=6。',
        'ne固定3.5e26 m^-3 = 3.5e20 cm^-3，是论文完全电离充气基准，不是推理出的逐时刻局部密度。',
        'VDF已含漂移，正向的附加u_e/u_i1/u_i2均为0，避免重复添加流速。',
        '直接在每组最新版原生波长轴计算，没有再次光谱插值。没有密度拟合、额外背景、波长位移或仪器卷积。',
        '原PV函数输出逐谱maxabs归一化强度；观测谱也仅除逐行maxabs后比较。',
        'a原始为counts；b/c为抠图色条位置0–1，不是已标定的绝对散射功率，不能声称绝对功率重建。',
        'b/c最早期一些行的原峰值仅0.004175，逐行归一化后微小近常数值会显示为亮色横带；未擅自删除或再扣底。',
        '',
        '光谱差异指标 relativeL2 = ||PV - I/maxabs(I)||2 / ||I/maxabs(I)||2，先逐行算，再在本组平均。',
        '组别 成功/总数 平均relL2(%) 中位relL2(%) 最大relL2(%) 平均RMSE']
    for g,x in f['panels'].items():
        lines.append(f'{g} {x["success_count"]}/{x["rows"]} {100*x["relativeL2_mean"]:13.4f} {100*x["relativeL2_median"]:13.4f} {100*x["relativeL2_max"]:13.4f} {x["rmse_mean"]:.6f}')
    lines += [f'完整正向及保存墙钟：{f["total_wall_seconds"]:.3f}秒；单CPU线程；没有失败行插补。',
              '', '副本处理引起的温度与流速变化：', '组别/物种 平均DeltaT(eV) 最大DeltaT(eV) 最大|Delta u|(km/s)']
    for g,x in p['data'].items():
        for s,z in x['preparation_moment_changes'].items():
            lines.append(f'{g}/{s} {z["delta_T_mean_eV"]:13.4f} {z["delta_T_max_eV"]:13.4f} {z["max_abs_delta_u_km_s"]:17.5f}')
    lines += [
        '',
        '五、结论的边界',
        '本轮直接展示了与论文温度、以及条件正向谱之间的较大偏差，尚不能把当前模型的输出当作已经验证的实验VDF。',
        '裁负等副本处理最多改变温度约202eV、电子流速约90.9km/s；即使负质量低于0.43%，物理矩仍可能敏感。',
        '重建误差是固定密度、给定组分与现有干净PV模型下的条件检验。神经网络不输出密度，论文还报告0.057nm仪器FWHM，',
        '本轮按范围没有加仪器响应，也没有做密度或强度校准拟合，故不能把所有残差单独归因于VDF误差。',
        '后续若改进，应先核对实验响应/密度与训练分布的匹配及VDF物理约束，而不是仅继续增加网络层数。此次未执行额外拟合或训练。',
        '',
        '六、产物与复核',
        'complete_figures.pdf：六页图集（温度、流速、光谱二维比较、代表谱、误差时序、处理的矩变化）。',
        'plots/：对应PNG与PDF单图。references/：Fig11原图、实验CSV、模拟墨迹边界、取点脚本和读取限制。',
        'moments/signed_moments_all_rows.csv：4548行原始矩及尾部诊断。representative_moments.csv：2/3/4ns代表数据。',
        'experimental_comparison.csv：逐点温度差异。forward/fig10*/row_metrics.csv：逐时刻光谱差异。',
        'forward/fig10*/forward_results.npz：原生坐标、观测值/归一化值、PV光谱和残差。',
        'forward/fig10*/processed_vdfs.npz：实际输入原PV的修复副本及处理前后矩。',
        'analysis_plan.json：199份保护文件与48+35个registry链接快照及明确授权。verification/：独立数值复核。',
        '全部原文件保持不变；没有新训练、修改原物理实现或另写正向方程。',
    ]
    (R/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    readme='''本目录是独立事后分析，不进入原模型生成/训练/评估流程。

主要结果见 REPORT.txt 与 complete_figures.pdf。

已执行的一键命令（仓库 /workspace/ots 下）：
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/moments/compute_signed_moments.py
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/references/prepare_fig11_references.py
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/forward/run_forward.py --smoke
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/forward/run_forward.py
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/plot_moments_comparison.py
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/plot_forward_comparison.py
/workspace/ots_rebuild_20261024T025750_CST/work/ots/.venv/bin/python -B outputs/fig10_moments_pv_20261021/build_report.py

完整forward已结束，脚本拒绝覆盖已有summary；不要直接删除完成产物以重跑。
只复核可用 verification/README.txt 中的命令；只重画可执行上述两条plot命令。
合并PDF：
pdfunite outputs/fig10_moments_pv_20261021/moments_fig11_comparison.pdf outputs/fig10_moments_pv_20261021/spectral_comparison.pdf outputs/fig10_moments_pv_20261021/complete_figures.pdf

核心公式及所有处理限制见REPORT.txt。原始推理矩用于论文比较；裁负归一化副本仅用于正向可接受输入，两套不能混称。
没有训练，不需要恢复训练状态；本轮推理后处理与正向均已完成，无自动继续任务。
'''
    (R/'README.txt').write_text(readme)
    print('REPORT.txt and README.txt saved')

if __name__=='__main__':main()
