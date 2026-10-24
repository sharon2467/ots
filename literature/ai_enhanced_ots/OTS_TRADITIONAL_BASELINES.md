# 传统 OTS 反演基线：从参数拟合到非参数/贝叶斯 EVDF 反演

更新时间：2026-08-24

## 0. 先说结论

当前 AI 项目不应只与“单个 Maxwellian 的最小二乘拟合”比较。建议建立四层基线：

- **B0：Maxwellian NLLS/MLE**，验证正向模型、仪器模型与噪声模型是否正确；
- **B1：扩展参数化分布拟合**，覆盖 super-Gaussian、双温/多组分、弱热尾、κ 与热流偏斜；
- **B2：确定性非参数/MAP 反演**，在与网络相同的 Fourier 表示上直接优化 EVDF；
- **B3：贝叶斯反演**，把可辨识性、校准与仪器不确定度写进后验。

其中，Milder 等 2024 年的 TSADAR 工作是目前找到的最强“直接光学 Thomson 散射、非神经网络、逐点 EVDF 优化”依据。检索未发现已经同行评议、与本项目任务完全等价的“神经网络从直接光学 OTS 光谱输出完整电子 EVDF”先例。微波 fast-ion collective Thomson scattering 的速度空间重建是重要相邻证据，但不能冒充直接光学电子 EVDF 证据。

## 1. 共同观测合同

所有 B0–B3 与 AI 必须调用同一个观测算子：

```text
f_e(v), f_i(v), plasma parameters
  -> arbitrary-VDF dielectric/structure-factor forward model
  -> absolute scattered power
  -> optical throughput and notch transmission
  -> measured instrumental response function (IRF) convolution
  -> detector pixel integration
  -> background/stray-light mean
  -> Poisson photon statistics + read/streak-camera correlated noise
  -> fit mask
  -> three-window normalization (compatibility branch only)
```

建议最小接口：

```python
def predict_raw(f_e, f_i, plasma, observation):
    """Return expected raw/calibrated detector signal before noise."""

def observe(mu_raw, observation, seed):
    """Return y_raw, y_norm, mask, and propagated covariance/likelihood metadata."""

def fit_baseline(level, observation, context, seed):
    """Fit B0/B1/B2/B3 and return a uniform FitResult."""
```

统一输出：

```text
FitResult = {
  f_hat, F_hat, theta_hat, nuisance_hat, spectrum_hat,
  nll, chi2, iterations, forward_evaluations, wall_seconds, peak_memory,
  status, failure_code, constraint_residuals, covariance_or_samples
}
```

公平比较必须固定：1224 个谱 bin、三段窗口、mask、IRF、噪声 realization、随机种子、train/validation/test 划分和可见 nuisance 信息。传统方法不能获得更干净或更窄的谱；AI 也不能暗中使用真实 `n_e/T_e/T_i/u/species`。

建议保留两个难度层：

- **F（fixed/fair）**：沿用当前训练合同，固定密度、离子、漂移与几何，只反演 EVDF；
- **J（joint/experimental）**：联合改变等离子体参数、波长零点、IRF、背景和标定。AI 只有在同一域重新训练或显式适配后才能参加此层比较。

当前三窗独立归一化会删除绝对幅度和窗间相对强度。兼容现有网络时应照旧处理；实验现实支线必须同时保存 raw/calibrated spectrum，并传播归一化后的相关协方差。

## 2. B0：Maxwellian NLLS/MLE

### 2.1 模型

F 层最小未知量：`T_e`、电子漂移 `u_e`，必要时加入每窗 amplitude。J 层再逐步开放：`n_e`、`T_i`、离子漂移、组分/平均电离度、激光波长偏移、低阶背景和 IRF 宽度。

目标函数不是默认的 plain MSE，而是与探测器一致的似然：

```text
argmin_theta  -log p(y | H[S_Maxwellian(theta)])
```

- streak camera：使用测得或估计的相关协方差，做 generalized least squares；
- photon counting：用 Poisson/count likelihood；
- 已归一化数据：用 Monte-Carlo 或 delta method 传播协方差。

算法建议：bounded multi-start nonlinear least squares；必要时 differential evolution 做全局初始化，再用局部 trust-region/LM 精修。约束包含正 `T/n`、quasi-neutrality 和物理参数边界。

### 2.2 验收

- 高 SNR、正向一致的 Maxwellian 合成样本上无系统偏差；
- 输出 Jacobian/profile/bootstrap 不确定度；
- 显式记录 nonconvergence、bound-hit、nonfinite、rank-deficient 和 residual/χ² failure。

## 3. B1：扩展参数化分布拟合

### 3.1 候选族

- super-Gaussian / Druyvesteyn，阶数 `m` 可拟合；
- bi-/multi-Maxwellian；
- bulk + 弱 hot Gaussian tail；
- κ 分布；
- skew/heat-flux correction；
- 有实验依据时加入 VFP/SNB 物理模板或多流组分。

混合权重必须非负且和为 1；组分按温度或漂移排序，避免 label switching。可以把 electron/ion feature 分步拟合作为初始化，但最终必须回到完整介电响应联合重算。

用 AICc/BIC、交叉验证或 Bayes evidence 选择分布族，禁止根据 test truth 挑选正确族。B1 的结论应表述为“模型族辨识”，不是 arbitrary VDF recovery。

### 3.2 验收

- 正确族下覆盖真实参数；
- 错误族能被 whitened residual、model selection 或 posterior predictive check 暴露；
- 输出完整隐含 `f(v)`，与 AI 使用同一 coefficient/EVDF 指标比较。

## 4. B2：确定性非参数/MAP 反演

### 4.1 首选表示

为最公平地对照当前网络，首选与网络相同的 80 个复 Fourier 模式/160 个实数、相同 K-grid 和 Hermitian 投影。另做 positivity-oriented spline 或 softmax-grid 敏感性分支。

```text
argmin_f  NLL(y | H[S(f)])
        + lambda_2 ||D2 f||^2
        + lambda_K sum_K w_K |F_K|^2
        + lambda_ent KL(f || m)
```

最小实现至少含二阶平滑与高-K惩罚。物理约束分别审计：

- `F(-K)=conj(F(K))`；
- `integral f(v) dv = 1`；
- `f(v) >= 0`；
- 有限低阶矩。

为避免“传统基线用了更强物理约束”的不公平，至少报告两版：

1. **B2-Fourier-equal**：仅 Hermitian，与当前 AI 输出约束一致；
2. **B2-physical**：再加 positivity、normalization 和正则项。

求解使用 automatic differentiation + L-BFGS/trust-region/Adam，多启动。正则强度只能在 train/validation 上按 likelihood、discrepancy principle、GCV 或 L-curve 选择，禁止查看 test `f_true`。

### 4.2 诊断

输出 effective degrees of freedom、Jacobian singular spectrum/resolution kernel、restart spread 和每项约束 residual。单视角只称为沿散射波矢 `k` 的一维投影 EVDF，不能声称恢复完整 3D 分布。

## 5. B3：贝叶斯反演

对 B1 参数族或降维后的 B2 basis 建立：

```text
p(z, nuisance | y) ∝ likelihood(y | H[S(z)], nuisance)
                      × physics_prior(z)
                      × calibration_prior(nuisance)
```

nuisance 至少包括 wavelength zero、background、IRF 和标定。噪声按实际探测器采用相关 Gaussian 或 Poisson/counting likelihood。

不建议直接对 160 维做朴素 MCMC。先在 B2 MAP 点用 Jacobian/SVD 保留约 10–30 个可辨识模式，或使用 Laplace approximation；再以 HMC/NUTS、ensemble MCMC 或 nested sampling 探索。贝叶斯方法不会自动消除不可辨识性。

必须报告 R-hat、ESS、divergence、多模态、posterior predictive residual；在重复合成观测上做 coverage 或 simulation-based calibration，而不只给一条 posterior mean。

## 6. 统一测试矩阵与指标

### 6.1 分布/域偏移

- Maxwellian sanity；
- 训练内 Gaussian mixtures；
- OOD super-Gaussian `m=3.5, 5`；
- κ/长尾；
- 0.1–1% hot tail 或 beam；
- skew/heat-flow distribution；
- non-Maxwellian ions；
- `n_e/T_e/geometry/IRF/wavelength/background` 域偏移；
- 多视角作为独立的 identifiability upgrade。

### 6.2 噪声

- 0/1/5/10% 合成扰动仅作兼容测试；
- photon-counting 和 streak-camera 实际噪声模型作主测试；
- stray light、notch、背景与 IRF mismatch 单独消融。

### 6.3 指标

- spectrum：NLL、χ²、whitened residual/相关性；
- Fourier：coefficient RMSE、R²；
- EVDF：relative/weighted L2、negative mass、归一化误差；
- moments：漂移、有效温度、heat flux/skew/kurtosis、tail fraction；
- uncertainty：95% coverage、interval width、calibration；
- engineering：wall time、forward calls、memory、收敛/失败/bound-hit；
- AI：训练总成本与 amortized inference latency 分开报告，同时给 fixed-compute 与 accuracy-matched runtime。

## 7. 直接证据与相邻证据

### 7.1 直接光学 OTS

- Milder et al., 2024：automatic differentiation、直接 256-point EVDF 优化及 Hessian/移动窗口不确定度；B2 的主要实现依据。DOI: `10.1088/2632-2153/ad2493`。
- Foo et al., 2023：任意速度分布正向模型，但反演采用低维分布族、differential evolution 与 ensemble MCMC；属于 B1/B3，不是自由网格 EVDF。DOI: `10.1063/5.0169393`。
- Lopez-Uricoechea & Walker, 2025：多 super-Gaussian 组件、NUTS/神经传输、IRF 与 Raman 标定；属于 B1/B3。DOI: `10.1063/5.0281503`。
- Suazo Betancourt et al., 2024：Maxwellian/Druyvesteyn/混合模型的 Bayesian model selection。DOI: `10.1063/5.0158749`。
- Swadling et al., 2022：streak-camera 相关协方差。DOI: `10.1063/5.0083195`。
- Washeleski et al., 2013：低信号 photon-counting MLE。DOI: `10.1063/1.4821980`。

### 7.2 相邻微波 fast-ion CTS

Salewski 2018、Nishiura 2020 和 Jiang 2026 支持 Tikhonov/速度空间层析、cGAN 与物理约束数据驱动 VDF 的方法论，但其粒子种类、频段和测量核与直接光学电子 OTS 不同。它们只能作为 `evidence::adjacent`，不能用来填补“直接光学神经网络完整 EVDF”证据空白。

