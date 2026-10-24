# 全 Fourier 离子1反演训练报告

## 结论

本轮已完成三项工作：

1. 新增 `phyEqns/ots_full_fourier.py`，电子、离子1和离子2的速度分布与响应函数全部使用任意 Fourier 表示，`OTS_spectrum_full_fourier` 不再接收任何温度；
2. 新增确定性的离子1数据生成与 v2 训练路径：电子和离子2固定为 Maxwellian，离子1为采样分布，网络标签为 `Re(F_i1), Im(F_i1)`；
3. 北京超算正式 作业 已 `COMPLETED (0:0)`，全部核心产物通过远端及本地 SHA-256，独立复载验证 作业 也已 `COMPLETED (0:0)`。

正式模型选择 `1224 -> 5000 -> 5000 -> 160`，测试集离子1 VDF 的总体 R2 为 `0.9848470`，逐样本 relative L2 均值/中位数为 `0.07352/0.04659`。大多数样本的主峰、宽度和卫星峰恢复良好；最差样本仍会把尖锐多峰结构平滑化，最差 relative L2 为 `0.45368`。

## 实现范围

新增或修改的正式源码：

- `phyEqns/ots_full_fourier.py`
- `genDataSet_full_fourier.py`
- `train_v2.py`
- `slurm/train_full_fourier_ion1_bscc.slurm`

`train_v2.py` 的默认电子目标保持不变。新增 `--target-kind ion1` 后，数据读取、训练/验证划分、标准化、Hermitian 投影、候选比较、测试指标和图表逻辑仍沿用 v2，只把标签解释、指标名、图轴和 checkpoint 名改为离子1。

## 全 Fourier 物理与数值约定

三个物种统一使用：

```text
F_s(K) = integral f_s(v) exp(-i K v) dv
f_s(v) = (1/2*pi) integral F_s(K) exp(+i K v) dK
R_s(v) = integral_0^infinity K F_s(K) exp(+i K v) dK
chi_s = (omega_ps^2/k^2) R_s(v_s)
```

离子数密度由准中性条件得到：

```text
n_i,j = n_e * N_i,j / (Z_i1*N_i1 + Z_i2*N_i2)
```

因此正向模型只需要三组 `K_s,F_s`、各物种质量/电荷/比例、电子密度、几何和漂移。1370 eV 与 171.3 eV 只在数据生成器上游用于构造固定 Maxwellian，不进入 `OTS_spectrum_full_fourier`。

有限均匀 K 网格隐含无混叠速度范围 `|v| <= pi/delta_K`。实现会把该范围外的 VDF 与响应置零，防止离子窄速度网格的周期延拓混叠到电子峰。未加此保护时，离子表示会在红/蓝电子峰产生约 100% 的伪差；加保护后两段电子峰的 Maxwellian 对照差异降到 `1.82e-4` 和 `3.26e-4`。

Maxwellian 正半轴响应还用 Mathematica 和独立 NumPy 梯形积分交叉检查。对标准差 `sigma=2.3`、速度 `v=1.7`，两者给出
`0.1026657144 + 0.1332584684i`，绝对差为 `1.58e-12`。

## 正向模型自检

BSCC 自检 作业 与正式 作业 得到相同结果：

| 检查 | 结果 |
|---|---:|
| 离子1 Maxwellian Fourier 往返 relative L2 | `8.2667e-7` |
| 离子1 Fourier Hermitian 最大绝对误差 | `0` |
| 离子1 Fourier 响应与 legacy `w_fast` relative L2 | `0.010313` |
| 全 Maxwellian 离子峰光谱 relative L2 | `0.095549` |
| 全 Maxwellian 红电子峰光谱 relative L2 | `0.0001821` |
| 全 Maxwellian 蓝电子峰光谱 relative L2 | `0.0003259` |

离子峰对离子响应细节很敏感。这里的约 9.55% 光谱差异同时伴随极小的 Fourier 往返误差和稳定的直接响应积分，反映的是 legacy `w_fast` 近似与直接 Fourier 响应的差别，不是全 Fourier 积分发散。新数据、训练和评估始终使用同一全 Fourier 模型，没有混用两条正向路径。

## 数据集合同

远端版本目录：

```text
datasets/full_fourier_ion1_v1_seed20260716/
```

固定物理量：

- 探针 351 nm、散射角 123°、电子密度 `1.81e26 m^-3`；
- H/C 数量比例 `0.55/0.45`，`Z=1/6`，质量 `m_p/12m_p`；
- 电子固定 1370 eV Maxwellian；
- 离子2固定 171.3 eV Maxwellian；
- 三物种漂移均为零；
- 三个波段仍为 349–353 nm（579 点）、375.3552–599.7064 nm（304 点）、100.1144–339.6504 nm（341 点），各段独立归一化。

离子1在 `[-12,12]` 个参考热速度的 1000 点网格上采样：

- 3 个核心 Gaussian：中心 `[-2,2]`、sigma `[0.35,1.50]`、幅度 `[0.2,2.5]`；
- 1 个卫星 Gaussian：中心 `[-7,7]`、sigma `[0.25,1.00]`、幅度 `[0.02,0.50]`；
- 最后按速度积分归一化，并保留 80 个复 Fourier 模式。

Parquet 仍为 1384 列 metadata-first 合同，但标签是离子1：

```text
0:1224      OTS 光谱
1224:1304   Re(F_i1)
1304:1384   Im(F_i1)
```

| 文件 | 形状 | seed | 生成耗时 | 字节数 | SHA-256 |
|---|---:|---:|---:|---:|---|
| `dataset_train.parquet` | 5001×1384 | 20260716 | 349.34 s | 21,515,991 | `41977b6f0bd30e4946e145f007f800608a926b882b6be7085cbd19566cad1519` |
| `dataset_test.parquet` | 101×1384 | 20260717 | 7.56 s | 1,090,434 | `ebf63b51df5353ea3ae1f20eae8ec2d19eaaf737582bda83e67a48c2a57250f8` |

两个 Parquet 只保存在 BSCC，未下载到本地。

## 正式训练

- 正式 Job：`1348835`
- 状态：`COMPLETED (0:0)`
- 节点：`paraai-n32-h-01-agent-227`
- GPU：NVIDIA A100-PCIE-40GB
- 环境：Python 3.10.14、PyTorch 2.4.0+cu121、NumPy 1.26.4、Pandas 2.3.3、PyArrow 22.0.0、Matplotlib 3.10.9
- Slurm 总耗时：8 分 28 秒
- 训练/验证/测试物理样本：4500/500/100
- batch size：128
- 优化器：AdamW，初始学习率 `1e-3`，weight decay `1e-5`
- 损失：raw Fourier coefficient normalized MSE，`raw_loss_weight=1.0`
- 最大 epoch：250，patience 25，验证损失驱动降学习率

候选比较：

| 隐藏层 | 参数量 | 最佳 epoch | 最佳验证损失 | 训练耗时 |
|---|---:|---:|---:|---:|
| 1024×512 | 1,864,352 | 182 | `0.0603201` | 39.97 s |
| 5000×5000 | 31,950,160 | 250 | `0.0529796` | 68.14 s |

正式 checkpoint 选择 `5000×5000`。其最佳值出现在最大 epoch 250，说明验证损失仍有很小的下降空间；但学习率已降到 `1e-6`，曲线后段基本进入平台区。本报告不外推未实际运行的更长训练结果。

## 测试指标

| 指标 | 数值 |
|---|---:|
| Fourier coefficient MAE | `0.0073790` |
| Fourier coefficient RMSE | `0.0194694` |
| Fourier coefficient R2 | `0.9874143` |
| Hermitian symmetry MAE | `0` |
| 离子1 VDF MAE | `2.5714e-8` |
| 离子1 VDF RMSE | `8.0532e-8` |
| 离子1 VDF R2 | `0.9848470` |
| VDF relative L2 mean | `0.0735231` |
| VDF relative L2 median | `0.0464640` |
| 负质量比例均值 | `0.0033618` |

逐样本诊断：

| 统计 | 数值 |
|---|---:|
| relative L2 p90 / p95 / max | `0.12055 / 0.24996 / 0.45368` |
| sample R2 mean / median / min | `0.98500 / 0.99727 / 0.75611` |
| 负质量比例 mean / max | `0.003362 / 0.02113` |
| 重建积分 mean ± std | `0.999871 ± 0.000898` |
| 重建积分 min / max | `0.997112 / 1.002005` |

六个最差测试索引为 `1, 19, 12, 44, 46, 99`。图像目视检查显示，主要误差来自尖锐双峰之间的深凹口、很窄的峰和弱卫星峰幅值；模型常把这些结构平滑为更宽的峰。这与 80 模式低通标签、训练分布复杂度和未施加非负硬约束一致。普通宽峰与较平滑多峰样本总体贴合良好。

## 产物、哈希与复载验证

远端正式目录：

```text
runs/full_fourier_ion1_1348835/
models/full_fourier_ion1_v1_seed20260716/LATEST.txt = job_1348835
```

正式作业对 checkpoint、指标、预测、历史、两张原 v2 图、数据 manifest、离子1诊断、最差样本图、run manifest 和源码哈希共 11 个条目执行了 `sha256sum -c`，全部通过。

独立复载验证 作业：

- `torch.load(..., map_location="cpu")` 成功；
- `state_dict` 严格加载成功；
- checkpoint 的目标、数据集、隐藏层和测试指标与 JSON 一致；
- 单样本前向输出形状为 `1×160`，全部有限；
- checkpoint SHA-256：`a1f7ab4a3fa25cd6fd85e69d77118025c89b92808cba5bd11100cde31d2597f6`。

本地结果目录：

```text
bscc_results/full_fourier_ion1_1348835/
```

已下载 checkpoint、预测、JSON、history、三张图、两套哈希、正式日志和复载验证日志；没有下载训练/测试 Parquet。本地重新计算正式产物、复载产物和四个正式源码的 SHA-256，全部与远端一致。

## 调试审计轨迹

- `1348815/1348818`：发现 Slurm spool 工作目录和旧 NVIDIA driver 节点问题；
- `1348822/1348824`：通过 Maxwellian 对照定位离子 K 网格周期混叠，加入无混叠速度域保护；
- `1348826`：最终全 Fourier 自检通过；
- `1348835`：正式数据生成、训练、评估、哈希和注册全部完成；
- `1348845/1348846`：独立复载脚本的共享依赖路径和 profile 初始化顺序错误，均在读取模型前快速失败；
- `1348858`：修正后的独立复载验证完成。

这些前置失败和验证脚本失败没有修改正式数据或正式模型；正式 作业 的结果目录和注册项保持不变。

## 限制与后续建议

1. 当前任务只改变离子1，电子和离子2、密度、混合比例、温度上游参数、几何与漂移均固定；指标不能外推到同时变化的多参数等离子体。
2. 三个波段独立归一化，跨波段相对幅度信息仍被删除。
3. 80 个 Fourier 模式会主动平滑窄峰和深凹口；最差样本已经显示这一上限。
4. Hermitian 对称有硬投影，但 VDF 非负性和单位积分没有硬投影；当前积分统计很好，仍存在小幅负质量。
5. 测试集与训练集来自同一离子1生成分布。这证明了同分布反演能力，不等于实验、PIC 或新型离子分布的 OOD 泛化。
6. 若继续提高质量，建议另开版本比较更多 Fourier 模式、离子峰加权输入损失、非负/归一投影，以及不泄漏当前测试集的更复杂多峰验证集；不要覆盖本次 `full_fourier_ion1_v1_seed20260716/job_1348835` 基线。
