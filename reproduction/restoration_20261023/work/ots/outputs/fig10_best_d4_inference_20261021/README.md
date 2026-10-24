# Fig10 D4 推理恢复结果

本目录由原推理和绘图流程重算，不是磁盘原件找回。D4_s0 checkpoint 与历史SHA一致，训练尾段未完成。

[10页合并PDF](fig10_best_d4_complete_figures.pdf)、[输入与模型冻结信息](inference_plan.json)、[本轮核验](completion.json)、[逐组数组与统计](results/summary.json)。

全部1516行保留，a/c仅按此前授权插值到模型波长；忽略mask、逐行maxabs归一化。原有符号VDF保留；没有VDF真值，不把理论test误差视作实验误差。

命令及阶段日志见独立恢复根README.md。3个未达到历史最终权重的模型已在统一比较标为partial；D4最佳checkpoint身份经SHA核实。原历史summary的completed字样由completion.json中的scope_corrections修正。
