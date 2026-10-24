# Fig26k 深度与参数量控制实验

固定协议结构、种子、数据划分、损失和优化器，以 D2 基线、D3/D4 深模型及参数量匹配的 W3/W4 宽模型比较深度效应。每次完整流程的预算上限为两小时；未完成项的完整耗时不能据中断时间推定。

结果与未完成项见 [研究报告](REPORT.md)。下列为预注册候选命令，并不表示每个候选均已执行；实际范围以报告为准。在独立运行副本中准备匹配数据与路径后执行。

```bash
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s0
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s0
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s0
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s0
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s0
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s1
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s1
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s1
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s1
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s1
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s2
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s2
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s2
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s2
python -B outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s2
```

数据与原生基线准备成本按本轮实际 5871.605902 秒计回，每个新尾流程最多 1208 秒加30秒保存余量。

## 适配验证与续训接口

```bash
python -B outputs/depth_study_20261021/validate_adapter.py
python -B outputs/depth_study_20261021/validate_resume.py
python -B outputs/depth_study_20261021/resume_trial.py --run-id depth_fig_d2_s1
python -B outputs/depth_study_20261021/resume_trial.py --run-id depth_fig_d4_s1
python -B outputs/depth_study_20261021/freeze_selection.py final
```

续训须使用完整的模型、optimizer、scheduler、早停、随机状态与 history；累计计入全部活动计算时间。仅有权重不足以构成等价续训，已完成结果不得覆盖。
