# Fig10 三物种 VDF 推理

采用冻结模型 depth_fig_d4_s0：4 个隐藏层（2048/1024/1024/1024），5,354,976 参数，训练池为 26,000 条。模型输入光谱，输出 Fourier 系数，经既有逆变换得到电子、H、C 的一维 VDF。

[10页图集](fig10_best_d4_complete_figures.pdf) · [输入和模型定义](inference_plan.json)

保留全部1516行：a/c 按各自波长轴插值到模型输入轴，b 已匹配输入轴；忽略 mask，逐行 maxabs 归一化。原有符号 VDF 保留。没有实验 VDF 真值，不能把理论 Test 误差视作实验反演误差。

本目录对应原始三组光谱的推理。背景扣除和标记区域填补后的结果见相邻 fig10_nearest_d4_inference_20261021 目录；温度与光谱闭合见 fig10_moments_pv_20261021。D4 使用已选定 checkpoint，完整训练尾段尚未完成，不据此证明该完整流程符合两小时预算。
