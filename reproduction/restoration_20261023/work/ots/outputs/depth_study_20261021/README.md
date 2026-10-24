# 深度实验原流程重新计算

这是恢复过程的新计算，不是历史权重。原协议结构、种子、数据划分、损失、优化器不变。串行启动，每次保守完整流程小于2小时。

原件与文本证据保存在恢复根目录 recovered_text；旧报告数字不作本轮结论。

```bash
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s0
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s0
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s0
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s0
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s0
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d2_s2
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d3_s2
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w3_s2
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_d4_s2
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/run_trial.py --run-id depth_fig_w4_s2
```

数据与原生基线准备成本按本轮实际 5871.605902 秒计回，每个新尾流程最多 1208 秒加30秒保存余量。

## 恢复验证及原暂停/续训流程重放

```bash
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/validate_adapter.py
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/validate_resume.py
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/resume_trial.py --run-id depth_fig_d2_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/resume_trial.py --run-id depth_fig_d4_s1
python -B /workspace/ots_rebuild_20261024T025750_CST/work/ots/outputs/depth_study_20261021/freeze_selection.py final
```

程序化重放暂停仅保存本轮状态，不冒充历史暂停原件。实际完成轮次以归档provenance为准；真实额度/用户暂停绝不自动恢复。
