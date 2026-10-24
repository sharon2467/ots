本目录独立分析冻结 VDF 的温度、流速和 PV 光谱闭合。

结果：REPORT.txt；图集：complete_figures.pdf；单图：plots/。
原始有符号 VDF 的代数矩与裁负、归一化正向副本的矩分别报告。
固定 ne=3.5×10²⁰ cm⁻³ 是论文充气基准假设，不是模型预测密度。

复现入口位于本目录：moments/compute_signed_moments.py、references/prepare_fig11_references.py、forward/run_forward.py、plot_moments_comparison.py、plot_forward_comparison.py。
使用匹配的输入、模型结果和分析环境，在项目根目录以 python -B outputs/fig10_moments_pv_20261021/<脚本> 运行；外部绝对路径需先映射。forward/run_forward.py 拒绝覆盖已有完成结果，复现应使用独立输出目录。
正向计算只调用现有 OTS_spectrum_PV。方法与适用边界见 REPORT.txt。
