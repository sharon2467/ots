# OTS 科学复现代码与研究结果

本分支沿用已有 fixed_ion / fig1_fig11 训练、规模与深度实验、统一测试、Fig10 推理、温度与光谱闭环主线。发布代码、配置、必要方法说明、研究报告和图表；数据集、原始实验输入、权重、完整数值数组及操作/对话记录不随此分支发布。

## 阅读入口

| 阶段 | 方法与结果 |
|---|---|
| 原生产生成/训练/评估 | [CLI说明](OTS_code_share/NN_hotE_large_lmRange/README.md)；[参数化命令](reproduction/publication/training_commands.json) |
| 初始调参 | [研究报告](reproduction/restoration_20261023/work/ots/outputs/training_study_20261020/RECOVERY_REPORT.md) |
| 扩大数据与容量 | [研究报告](reproduction/restoration_20261023/work/ots/outputs/training_scale_20261020/RECOVERY_REPORT.md) |
| Fig深度控制实验 | [报告与缺项](reproduction/restoration_20261023/work/ots/outputs/depth_study_20261021/RECOVERY_REPORT.md)；同目录protocol.json、adapter.py |
| 统一1024条理论Test | [排名](reproduction/restoration_20261023/work/ots/outputs/fig_unified_test_20261021/ranking.csv)；[已核对标题的图表](reproduction/restoration_20261023/work/ots/outputs/fig_unified_test_20261021/reviewed_rendering/) |
| 最新Fig10 VDF | [输入和模型说明](reproduction/restoration_20261023/work/ots/outputs/fig10_nearest_d4_inference_20261021/inference_plan.json)；同目录plots/ |
| 温度、流速与原PV光谱闭环 | [报告](reproduction/restoration_20261023/work/ots/outputs/fig10_moments_pv_20261021/REPORT.txt)；同目录plots/与complete_figures.pdf |
| 偏差机制控制 | [诊断报告](outputs/bias_diagnosis_20261023/REPORT.txt)；[正远尾敏感性](outputs/bias_diagnosis_20261023/representation_audit/POSITIVE_TAIL_REPORT.md) |

## 结果边界

24个正式模型中21个重新计算的checkpoint与已记录历史SHA-256一致；这属于字节复现，不是旧磁盘原件找回。D3_s0、D4_s2、W4_s2的历史最终权重仍缺，现有部分模型不能替代它们；8条训练尾段未完成。完整性见[模型身份与限制](reproduction/publication/model_identity_and_limits.json)。预算中断项不证明完整流程能在2小时内完成。

同一现存1024条理论Test上，D4_s0的三物种宏平均VDF相对L2为5.45724444%。它有4个隐藏层（2048/1024/1024/1024）、5,354,976参数、26,000条训练池（22,100优化、3,900验证）。这是合成数据指标，不能当作实验准确率。

最新版Fig10共1516行，输出电子/H/C各4097点VDF。原OTS_spectrum_PV在固定ne=3.5×10²⁰ cm⁻³和明确的独立副本处理下，三组平均谱相对L2分别55.913367%、36.504808%、31.259989%，仍有明显偏差。原始有符号VDF矩与裁负、归一化后的正向副本分开保存/解释。没有另写正向模型。

偏差诊断报告中的“当时缺失D4”等表述对应其研究时点；后续已有D4 checkpoint字节复现和完整Fig10重算。该控制研究仍不能当作D4实际误差份额分解。Fourier标签重建、负旁瓣/远尾、仪器响应与空间平均的限制仍保留。

## 复现条件

见[复现说明](reproduction/publication/REPRODUCTION.md)。原工程21份Python源码未改；公开副本仅将data.py中的个人默认锚点路径改为paper_experimental_points.csv，数值算法未改。本分支没有运行训练或生成新数据。原相对目录结构保留；部分独立分析脚本与配置记录原执行绝对路径，迁移时必须在新运行副本中显式映射路径并核验外部输入，不能声称克隆后无需数据即可重现全部图。

本分支为单独的干净发布历史；既有master和未发布的本地完整成果保留。
