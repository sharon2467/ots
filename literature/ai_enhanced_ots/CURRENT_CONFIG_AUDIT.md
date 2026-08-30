# 当前 AI Enhanced OTS 物理配置审计

更新时间：2026-08-24

## 1. 定位

当前 canonical 配置是一个明确的 **HEDP/ICF 风格、弱到中等集体散射** 合成问题，不是覆盖磁约束、低温放电、推进和 HEDP 的通用 OTS 配置。

代码证据：

- `T_e` 参考值 1370 eV、`T_i` 171.3 eV：`data.py:28-29`；
- EVDF 网格 `[-0.8c,0.8c]`，三背景 Gaussian + 一弱热电子 Gaussian：`data.py:72-98`；
- 三个谱窗：`349–353 nm`、`375.3552–599.7064 nm`、`100.1144–339.6504 nm`：`data.py:153-160`；
- `351 nm`、`123°`、`n_e=1.81e26 m^-3`、H/C=`0.55/0.45`：`data.py:258-278`；
- 每个谱窗各自除以本窗最大绝对功率，绝对 prefactor 被省略：`ots_full_fourier.py:417-427`。

对应本地文件：

- `C:/Users/win/OneDrive/Desktop/ots/OTS_code_share/NN_hotE_large_lmRange/data.py`
- `C:/Users/win/OneDrive/Desktop/ots/OTS_code_share/NN_hotE_large_lmRange/phyEqns/ots_full_fourier.py`

## 2. 派生物理量

用中心波长近似：

```text
k = 2 * (2π/λ0) * sin(θ/2)
lambda_D = sqrt(epsilon0 * k_B*T_e / (n_e*e^2))
alpha = 1 / (k*lambda_D)
```

得到：

| 量 | 值 | 含义 |
|---|---:|---|
| `k` | `3.146×10^7 m^-1` | 351 nm、123° 的中心散射波数 |
| `lambda_D` | `2.045×10^-8 m` | 1370 eV、`1.81e26 m^-3` 的电子 Debye 长度 |
| `alpha` | `1.55` | 弱到中等集体；电子/离子屏蔽特征都重要 |
| `n_critical(351 nm)` | `9.05×10^27 m^-3` | 当前 `n_e/n_c≈0.0200`；传播密度本身与 HEDP corona/hohlraum 相容 |
| `sqrt(kT_e/m_e)` | `1.55×10^7 m/s = 0.0518c` | 一维 Maxwellian 标准差约定下的热速度 |
| `sqrt(2kT_e/m_e)` | `0.0732c` | 另一常用热速度约定；比较文献时须注明定义 |

因此，当前问题不是 Huang 2000 一类理想非集体“谱形直接等于 EEDF 投影”的极限；反演必须保留介电响应和离子屏蔽。

## 3. 波长窗口实际覆盖的相速度

按当前代码的散射矢量/频移配套符号，用 `v_phase = omega_shift/k` 映射：

| 波长 | 代码相速度 |
|---:|---:|
| 100.1144 nm | `+0.60582c` |
| 200 nm | `+0.30844c` |
| 339.6504 nm | `+0.01870c` |
| 349 nm | `+0.00325c` |
| 351 nm | `0` |
| 353 nm | `-0.00323c` |
| 375.3552 nm | `-0.03815c` |
| 500 nm | `-0.19834c` |
| 599.7064 nm | `-0.29472c` |

这表明谱窗并不对称：蓝侧延伸到约 `+0.61c`，红侧只到约 `-0.30c`。如果目标 EVDF 允许热电子中心在 `±0.4c`，红向高速度尾可能没有对称观测覆盖；模型可能依赖 Fourier/Hermitian 先验推断不可见部分。

## 4. 速度网格与物理模型的一致性

| 速度 | 非相对论动能 | 相对论动能 |
|---:|---:|---:|
| `0.4c` | 约 40.9 keV | 约 46.5 keV |
| `0.8c` | 约 163.5 keV | 约 340.7 keV |

当前正向模型是非相对论介电响应，而速度支持到 `0.8c`。这不意味着所有样本都有大量 `0.8c` 电子，但一旦训练分布在高速度端保留可观概率，能量映射、散射运动学和响应都应做相对论一致性检查。Ross 2010 在 `v_phase=0.12c` 已观察到显著相对论 form-factor 效应；当前 100 nm 端约 `0.606c`，属于一级模型风险。最低限度应统计每个样本在 `|v|>0.3c`、`0.5c` 的概率质量，而不能只看网格边界。

## 5. 主要风险与改进优先级

| 优先级 | 当前假设 | 实践风险 | 建议 |
|---|---|---|---|
| P0 | 三窗独立归一化 | 删除绝对密度/功率与窗间强度比；噪声被变成相关、异方差 | 同时保存 raw/calibrated counts 与兼容归一化谱；传播协方差 |
| P0 | 电子窗下探 `100.1144 nm` | 已进入 VUV；普通 OTS 镀膜、空气光路、光栅和探测器不能直接覆盖 | 将其标为理论窗口；按真实仪器把短波端限制在可标定范围，或单独设计 VUV 观测链 |
| P0 | 351 nm 同时可能是 ICF 驱动波长 | 驱动散射和自发光可淹没 probe；成熟 OMEGA OTS 常用 263/266 或 526/527 nm 避开背景 | 显式加入驱动背景/stray-light 模型，或选择独立 probe 波长 |
| P0 | 无 IRF、notch、stray light、背景 | 合成谱比实验谱过于理想，反演会学习不可实现的细节 | 在正向链中加入实测 IRF、传输、像素积分、背景与 mask |
| P0 | 噪声仅以简单百分比扰动表示 | 不代表 photon counting 或 streak-camera 相关噪声 | 使用计数/读出/相关协方差模型，并保留 1/5/10% 兼容支线 |
| P1 | `n_e/T_i/H:C/Z/u` 全固定 | 网络可利用固定谱形捷径；部署时对参数漂移脆弱 | 按实验先验联合随机化，或作为条件输入；设置固定与联合两层任务 |
| P1 | EVDF 仅 Gaussian mixture + 弱 Gaussian hot tail | 不覆盖真实机制；也会生成缺少实验产生机制的任意多峰 | 分层加入 Maxwellian、`m=2–5` bulk+Maxwellian tail、bi-Maxwellian、奇对称 heat-flux、beam+bulk、counterstream 与 anisotropy |
| P1 | 单一视角 | 只能约束沿 `k` 的一维投影，3D 各向异性不可辨识 | 明确输出为 projected EVDF；多 `k`/多角度数据另建任务 |
| P1 | 蓝/红速度覆盖不对称 | 可能把未观测一侧交给对称性/训练先验 | 给每个窗口计算 phase-velocity coverage；设计对称或多角度采集 |
| P1 | H/C=`0.55/0.45` 且 `Z_C/T_i/u` 固定 | 与 CH 的 0.5/0.5、CH2 的 0.667/0.333 离子数分数都不相同；多流/电离度易和 EVDF 退化 | 写明 fraction 定义；把 composition、`Z_C,T_i,u` 作为 nuisance/条件量并做可辨识性审计 |
| P1 | 平衡动态结构因子被用于所有输入 | IAW/EAW burst、SBS、return-current/reconnection wave burst 不是热涨落谱 | 加 posterior-predictive/OOD score 与拒识类，不强制输出某个 EVDF |
| P2 | 忽略有限接受角、散射体积和时间积分 | 文献接受角可达 3° 或 7–9°，空间 50 µm–mm、积累 5–6000 shots；平均会抹平结构 | 把角度、空间、时间和 shot integration 纳入观测算子与元数据 |
| P2 | 80 个 Fourier 模式固定 | 分辨率由谱窗、IRF、噪声和 Jacobian 决定，不只由重建网格决定 | 用 singular spectrum/resolution kernel 与验证集选择模式/正则强度 |
| P2 | 非相对论响应 + `0.8c` 支持域 | 高能尾的物理一致性不清楚 | 对高速度质量做阈值审计；必要时缩域或引入相对论模型 |

## 6. 推荐的训练域分层

不要把所有等离子体制度混成一个无标签的大范围。建议按实验任务建域：

1. **HEDP-current**：围绕当前 `351 nm/123°/H-C/keV/10^26 m^-3`，加入 IRF、噪声与有限参数漂移；用于对现有网络的最小现实化。
2. **HEDP-generalization**：改变探针波长、角度、密度、温度、离子组分/漂移与热尾；检验域外鲁棒性。
3. **noncollective-electron-feature**：面向低密度/低温或磁约束电子特征，`alpha << 1`，重点比较 Maxwellian、Druyvesteyn、bi-Maxwellian/κ。
4. **collective-IAW/EPW**：`alpha ≳ 1`，联合离子参数、流速和非 Maxwellian 电子屏蔽；不能只随机化电子 EVDF。

每个域必须记录 `lambda0, theta, k, alpha, ne, Te, Ti, species/Z, drifts, IRF, SNR/count level, phase-velocity coverage`，并把实验范围与训练范围分开标注。

## 7. 第一轮最小改造顺序

1. 保持现有 F 层任务不变，补 B0 与 B2-Fourier-equal；
2. 保存 raw/calibrated 谱，并加入 IRF、像素积分和物理噪声；
3. 给 `n_e/T_i/u/H:C/Z` 加有限随机化或条件输入；
4. 增加盲 OOD 分布族；
5. 报告 Jacobian 可辨识模式、EVDF moments、negative mass 和 UQ；
6. 再扩展到联合 J 层与多视角。
