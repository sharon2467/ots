# AI Enhanced OTS：由浅入深的文献脉络与知识框架

更新时间：2026-08-24

## 0. 如何使用这份路线

目标不是“把所有 Thomson scattering 论文堆在一起”，而是依次回答四个问题：

1. 光谱如何由速度分布和介电响应产生？
2. 实验真正测到了哪些波长、角度、空间/时间尺度与噪声？
3. 在给定观测算子下，哪些 EVDF 特征可辨识？
4. AI 相比传统优化改进了速度、鲁棒性还是可辨识性？

主线只把 **直接光学电子 Thomson scattering** 当作同域证据。微波/毫米波 fast-ion collective Thomson scattering 只作为速度空间层析和物理约束机器学习的相邻方法证据，并统一标记 `evidence::adjacent`。

## 1. 一张知识地图

```text
实验设置
  λ0, θ, polarization, volume, duration, collection optics
  ne, Te, Ti, species/Z, drifts
       │
       ▼
散射运动学与制度
  k = ki - ks,  ω = ωi - ωs
  alpha = 1/(k λD)
  noncollective  <──── transition ────>  collective
       │                                  │
       │ electron feature ≈ projected    │ dielectric screening,
       │ EVDF under restrictive limits   │ EPW + IAW features
       ▼                                  ▼
任意 VDF 正向模型
  fe(v·k), fi(v·k) -> χe, χi -> ε -> S(k,ω)
       │
       ▼
真实观测算子
  absolute power -> throughput/notch -> IRF -> pixels
  -> stray/background -> photon/read/correlated noise -> mask/normalization
       │
       ├── B0 Maxwellian NLLS/MLE
       ├── B1 parametric non-Maxwellian + model selection
       ├── B2 regularized/AD nonparametric EVDF
       ├── B3 Bayesian posterior + nuisance/UQ
       └── AI amortized inference / transfer / learned likelihood
```

关键边界：AI 可以加快对既定正向模型和训练先验的反演，但不能凭空恢复观测算子没有携带的信息。单视角测到的是沿散射波矢的一维投影；绝对标定、窗间强度比或多角度没有进入输入，就不能要求网络稳定恢复它们对应的物理量。

## 2. 第一层：教材与散射理论

### 建议阅读顺序

1. J. Sheffield, *Plasma Scattering of Electromagnetic Radiation*：把散射功率、集体性、电子/离子特征和仪器连成主干。
2. D. H. Froula et al., *Plasma Scattering of Electromagnetic Radiation: Theory and Measurement Techniques*：作为现代实验/分析手册；用户已有条目直接复用。
3. E. E. Salpeter (1960), DOI `10.1103/PhysRev.120.1528`：经典散射谱与集体效应起点。
4. D. E. Evans & J. Katzenstein (1969)：实验方法早期综述。
5. K. Muraoka & A. Kono (2011), DOI `10.1088/0022-3727/44/4/043001`：低温等离子体激光 Thomson scattering 综述。
6. E. Carbone & S. Nijdam：大气压/低温等离子体中的 Thomson scattering 综述。

### 必须掌握

- `S(k,ω)` 不是在所有制度下都等于裸 `f_e(ω/k)`；屏蔽项通过 `χ/ε` 改变谱形。
- `alpha = 1/(kλ_D)` 的定义、热速度定义和 Fourier 符号在不同文献中可能不同，比较数值前要统一约定。
- 电子特征主要约束电子速度尺度/尾部，离子声特征同时受 `T_e/T_i`、离子组分、电离度与相对漂移影响。
- finite collection angle、IRF、notch 和背景会改变高频细节的实际分辨率。

## 3. 第二层：仪器、标定与噪声

优先阅读以下“容易被纯合成代码忽略”的工作：

- Froula et al. (2006)：HEDP Thomson scattering 仪器/分析综述；
- Swadling et al. (2022), DOI `10.1063/5.0083195`：streak-camera 数据的相关协方差；
- Washeleski et al. (2013), DOI `10.1063/1.4821980`：低光子数下的最大似然分析；
- Friss & Yalin (2018), DOI `10.1364/OL.43.005343`：1064.5 nm 腔增强低温等离子体 TS，把弱信号能力推进到 `n_e=2.8×10^18 m^-3`、`T_e=2.7 eV`、SNR≈1100 的实验例；
- Katz et al. (2024), DOI `10.1063/5.0219182`：角分辨、多散射矢量系统；
- Lopez-Uricoechea & Walker (2025), DOI `10.1063/5.0281503`：IRF/Raman 标定和贝叶斯分析。

阅读时为每篇论文提取：探针波长/能量/脉宽、散射角与固体角、散射体积、时间/空间分辨、谱仪与像元宽度、IRF、notch/mask、背景、自发光、计数水平、重复次数与绝对/相对标定。

## 4. 第三层：按等离子体场景理解参数与预期分布

下面是“选定代表论文的证据锚点”，不是 OTS 的全局硬边界。完整证据表见 `OTS_EXPERIMENTAL_PARAMETER_EVIDENCE.md`。

| 场景 | 典型诊断问题 | 常见制度 | 预期沿 k 投影分布 |
|---|---|---|---|
| 磁约束聚变 | `T_e/n_e` 剖面、边缘/芯部参数、快速在线分析 | 多为 noncollective electron feature；大型系统多点、多通道 | 核心常近 Maxwellian；加热/输运可导致 bi-Maxwellian、各向异性或弱非热尾，但常规系统未必能辨识完整 EVDF |
| 低温/放电/大气压 | `n_e/T_e/EEDF`、碰撞加热、放电动力学 | 稀薄放电常为强 noncollective；大气压射流可进入过渡区，金属蒸气电弧可达 collective；信号弱、stray light 与 probe 扰动关键 | Maxwellian、Druyvesteyn、bi-Maxwellian、耗尽尾或漂移/各向异性；Huang 2000、Kono 2002、Koiso 2024 等给出直接 EEDF/EVDF 形变例 |
| 推进/稀薄等离子体 | 低密度电子参数、流动/非平衡 | noncollective，photon-counting 常重要 | 非平衡 EEDF、热/冷电子混合、漂移；统计似然比 plain MSE 更重要 |
| ICF/HEDP/激光产生 | EPW/IAW、激光吸收、热流、激波、组分与流速 | `alpha` 常在过渡到 collective；ns/ps、强自发光 | super-Gaussian、热流偏斜、弱热电子尾、束流/平台、双流；离子可多组分/多流 |
| 高重复率/实验室天体 | 二维/时序 `n_e/T_e/u`，冲击与磁重联 | noncollective 与 collective 均可能，取决于平台 | 局部近 Maxwellian 到多流/非热；系统误差与 sim-to-real 是 AI 核心问题 |

### 已有直接实验辨识较强的分布

- super-Gaussian bulk 与近 Maxwellian tail：Milder et al. 2019/2020/2021；
- 热流导致的奇对称扰动/非局域电子分布：Henchen et al. 2018/2019；
- 径向—切向联合的非 Maxwellian 凹陷：van Lammeren et al. 的 TORTUR 实验；
- 多散射矢量投影与各向异性：Milder et al. 2021、PHASMA 2023、Katz et al. 2024；
- 低温等离子体 EEDF 差分、振动能损弯折和低流量非 Maxwellian：Huang 2000、Kono 2002、Koiso 2024；
- 对向流、热电子尾和非热波爆发：Ross 2013、Turnbull 2020、Zhang 2023 等提供机制明确的 HEDP 压力测试。

### 主要作为模型族或合成 OOD 的分布

- κ tail、任意 beam/hole/plateau、非常细的多峰结构；
- 当前项目的三背景 Gaussian + 弱热 Gaussian 组合；
- 单视角下的完整 3D 各向异性。

这些形态可以作为压力测试，但报告时必须区分“实验已见”“物理上合理”“纯合成先验”。

## 5. 第四层：传统反演谱系

1. **理想非集体解析/差分**：Williamson & Clarke 1971；Zhuravlev et al. 1974；Huang et al. 2000。只在相应近似成立时可直接映射 `spectrum -> projected distribution`。
2. **Maxwellian 非线性拟合**：长期实验标准基线；必须卷积 IRF、处理背景并使用正确 likelihood。
3. **非 Maxwellian 参数族**：super-Gaussian、Druyvesteyn、multi-Maxwellian、hot-tail、heat-flux template；Milder 2019 说明用错 Maxwellian 可产生很大的参数偏差。
4. **全局优化 + MCMC**：Foo et al. 2023 对任意 VDF 正向模型采用低维族、differential evolution 和 `emcee`。
5. **可微非参数优化**：Milder et al. 2024/TSADAR 直接优化 256 点 EVDF，是当前 B2 首选依据。
6. **Bayesian model selection/UQ**：Fischer 2002、Escalona 2023、Suazo Betancourt 2024、Lopez-Uricoechea 2025。

详细可执行合同见 `OTS_TRADITIONAL_BASELINES.md`。

## 6. 第五层：AI/ML 发展脉络

| 阶段 | 代表工作 | 实际任务 | 与当前项目关系 |
|---|---|---|---|
| 2016 | Lee et al., DOI `10.1063/1.4961079` | NN 快速回归 KSTAR `T_e` | 参数 surrogate，不是 EVDF |
| 2018 | Fujii et al., DOI `10.1080/15361055.2017.1396179` | 学习复杂多通道噪声/likelihood | 可用于现实噪声建模，不是 EVDF |
| 2020 | Nishiura et al., DOI `10.1088/1748-0221/15/01/C01002` | cGAN fast-ion velocity tomography | 相邻微波 CTS，非直接光学电子 |
| 2024 | Pokornik et al., DOI `10.1063/5.0201148` | CNN 回归 collective IAW 参数 | 快，但真实噪声/OOD 下 Bayes 更稳 |
| 2024 | Milder et al., DOI `10.1088/2632-2153/ad2493` | AD 逐点 EVDF 与参数优化 | 非神经网络，却是最强直接 B2 |
| 2025 | Eisenbach et al., DOI `10.1088/1361-6463/ad89d6` | sim-to-real/transfer，二维谱的 `n_e/T_e` | 实验 teacher 来自传统 NLLS；非 EVDF |
| 2026 | Van Hoomissen et al., DOI `10.1063/5.0304922` | collective/noncollective transfer + UQ | 参数回归；强调少量实验微调 |
| 2026 | Jiang et al., DOI `10.1088/1741-4326/ae4fdc` | 物理约束 2D fast-ion VDF | 相邻 CTS；方法可借鉴，观测核不同 |

当前项目的真正新颖性应谨慎表述为：在直接光学 OTS 的任意非 Maxwellian electron EVDF 任务上，采用低通 Fourier 标签做 amortized full-distribution inversion。证据空白本身不是性能证明；必须通过 B0–B3、公平 OOD、仪器化合成和实验/高保真数据验证。

## 7. 第六层：推荐阅读包

### 7.1 10 篇最小入门包

1. Sheffield 教材；
2. Froula 等 2012；
3. Salpeter 1960；
4. Muraoka & Kono 2011；
5. Froula et al. 2006；
6. Milder et al. 2019（非 Maxwellian 拟合偏差）；
7. Milder et al. 2021 PRL（直接分布证据）；
8. Foo et al. 2023（任意 VDF 正向 + 传统 inference）；
9. Milder et al. 2024（AD 非参数基线）；
10. Pokornik 2024 或 Eisenbach 2025（AI 参数反演现实边界）。

### 7.2 实现传统基线包

Foo 2023、Swadling 2022、Washeleski 2013、Milder/TSADAR 2024、Suazo Betancourt 2024、Lopez-Uricoechea 2025。

### 7.3 当前代码 OOD 包

Milder 2019/2020/2021（super-Gaussian 与 tail）、Henchen 2018/2019（heat-flux odd component）、TORTUR（投影各向异性）、Ross 2013/Turnbull 2020（对向流与热尾）、Zhang 2023（非热波爆发）、PHASMA 2023/Katz 2024（多视线），再加相邻 Salewski/Nishiura/Jiang 作为 regularization 与物理约束方法参照。

## 8. 建议的研究判断框架

对每个新结果依次问：

1. 它是直接光学电子 OTS，还是相邻 CTS/其他光谱？
2. 它恢复的是参数、模型族、沿 `k` 的 1D EVDF，还是完整 2D/3D 分布？
3. 正向模型和仪器模型是否与实验一致？
4. 分辨率来自数据还是正则/训练先验？
5. 是否与 B0–B3 在相同观测和计算预算下比较？
6. OOD、噪声、校准漂移和失败率是否报告？
7. 不确定度是否有 repeated-synthetic coverage 或实验交叉验证？

只有这七项都可审计，AI Enhanced OTS 才从“合成谱回归”走向可部署诊断。
