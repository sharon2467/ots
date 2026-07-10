# OTS 训练恢复报告

## 结论摘要

旧作业正常完成，但“训练完成”不等于“模型泛化理想”。目前证据显示，主要问题来自训练/评估流程，而不是一次随机失败：

1. `myscript.slurm` 固定使用 `cpu` 分区，日志明确显示 `NVIDIA Driver was not detected`；约 3200 万参数的网络完全在 CPU 上训练。
2. `main.py` 对 4999 条样本执行 119000 次全批量更新，只记录训练集 L1，没有验证集、早停、最佳权重或独立测试指标。
3. 第四阶段从训练 L1 `1.7293e-4` 缓慢降到 `1.4102e-4`，100000 步换来的收益有限，且没有证据表明测试误差同步改善。
4. `readData_dataset()` 的默认 `i_e=-1` 会遗漏最后一条物理样本，因此日志只加载 4999 条而不是 5000 条。
5. 评估日志大量出现 `ComplexWarning: Casting complex values to real discards the imaginary part`；旧评估脚本还会重复把 Tensor 包装成 Tensor，难以确认图像异常来自模型还是评估。
6. 训练目标的 160 个 Fourier 分量量级不同，旧流程直接对原值做统一 L1，容易让大幅分量支配梯度。

## 修复版设计

`train_v2.py` 不覆盖旧代码，新增一条可回滚的训练路径：

- 读取全部物理样本，固定随机种子，90% 训练 / 10% 验证，独立的 100 条测试集只用于最终评估；
- 按训练集统计量标准化 1224 维输入和每个 Fourier 输出分量；
- 使用 mini-batch、AdamW、验证集调度、梯度裁剪、早停和最佳权重保存；
- 比较 `512×256` 与 `1024×512` 两个紧凑网络，按同一验证集选择最佳模型；
- 将预测投影到 Hermitian 对称，保证实 EVDF 所需的 `F(-k)=conj(F(k))`；
- 用显式 `Re(F)cos(kv)-Im(F)sin(kv)` 逆变换，避免复数转实数告警；
- 在同一测试集上输出旧模型原始指标、旧模型物理投影指标和新模型指标；
- 保存 checkpoint、JSON 指标、训练曲线、预测数组和 EVDF 对比图。

## 作业配置

修复作业 `train_v2.slurm` 申请 π 2.0 的 `dgx2` 分区 1 张 GPU、8 个 CPU 核和 64 GB 内存，最长 4 小时。脚本保留旧模型和旧结果，输出写入 `runs/v2_<job_id>/`，成功后把最佳 checkpoint、指标和对比图复制到当前目录。

资源配置依据上海交大超算平台的[队列说明](https://docs.hpc.sjtu.edu.cn/job/partition.html)：当前工程位于 π 2.0 的 `lustre` 文件系统，应使用 `dgx2` GPU 队列；思源一号的 `a100` 队列使用独立的 `dssg` 文件系统，不能直接把本目录的 Slurm 脚本改成 `a100` 后提交。

## 基线证据

- 旧数据形状：`Input: torch.Size([4999, 1224]), Labels: torch.Size([4999, 160])`
- 旧训练初始 L1：`1.0408e-1`
- 第三阶段末 L1：约 `1.7548e-4`
- 第四阶段末 L1：`1.4102e-4`
- 旧训练只在训练集上打印首样本预测与标签，没有测试集数值指标。
- 作业配置：`--partition=cpu`、`--ntasks=32`，但训练代码是单进程全批量 PyTorch，32 个 Slurm task 没有被显式利用。

## 新训练结果

训练完成后，将从 `metrics_v2.json` 回填：

- Slurm Job ID 与实际设备；
- 入选网络和最佳 epoch；
- 旧/新模型 coefficient MAE、RMSE、R²；
- 旧/新模型 EVDF RMSE、R²、相对 L2；
- 负质量比例与 Hermitian 对称误差；
- 最终模型和图表路径。

## 版本控制

- 本地分支：`codex/ots-training-recovery-20260710`
- 修复采用新增文件，未覆盖现有的 `main.py`、`train.py`、`model.py` 或用户未提交改动。
- 训练数据、checkpoint、日志与图片属于运行产物，不应与源代码混入同一提交。
