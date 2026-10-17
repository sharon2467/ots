# SPEC: 逐式实现 Zhao et al. (2019) 式 (7)–(9) 的双离子 Maxwellian OTS 光谱

## 0. 审阅状态与结论摘要

状态：**已按用户反馈改为解析路径，待审阅，尚未实施。**

本 SPEC 不再采用“温度生成速度分布后委托 full-Fourier 内核”的方案。拟在 `phyEqns/ots_full_fourier.py` 中新增一条彼此独立但接口风格一致的解析 Maxwellian 路径，逐式计算 Zhao et al. (2019) 的：

```text
式 (7): chi_e = alpha^2 w(x_e)
式 (8): chi_j = alpha^2 [Z_j^2 N_j / sum(Z_j N_j)] [T_e / T_j] w(x_j)
式 (9): w(x) = 1 - 2 x exp(-x^2) integral_0^x exp(p^2) dp
                  + i sqrt(pi) x exp(-x^2)
```

两种离子分别使用 `T_i1` 和 `T_i2`。新函数不构造速度网格、不计算 Fourier 系数、不调用 `ft_x_n`、`fourier_landau_response` 或 `OTS_spectrum_full_fourier`。

## 1. Requirements（需求）

### 1.1 文献依据

源文献：Hang Zhao et al., *Progress in optical Thomson scattering diagnostics for ICF gas-filled hohlraums*, Matter and Radiation at Extremes 4, 055201 (2019), DOI `10.1063/1.5090971`。

采用的原文公式和边界：

1. PDF 第 3 页（论文页 055201-2）式 (1)：散射功率由频移修正、绝对幅度/偏振因子和谱密度 `S(k, omega)` 组成。
2. 同页式 (2)：多离子谱密度由一个电子项和各离子项相加，`epsilon = 1 + chi_e + sum_j chi_j`。
3. PDF 第 4 页（论文页 055201-3）式 (5)、(6)：电子和每种离子均为归一化一维 Maxwellian 分布。
4. 同页式 (7)、(8)：电子和各离子的极化率分别由 `alpha^2 w(x_e)` 及带组分/温度权重的 `alpha^2 w(x_j)` 给出。
5. 同页式 (9)：`w(x)` 的实部包含 `exp(-x^2) integral_0^x exp(p^2) dp`，虚部为正号的 `sqrt(pi) x exp(-x^2)`。
6. 同页漂移说明：离子使用 `omega_s-omega_i-k dot u_i`，电子使用 `omega_s-omega_i-k dot (u+u_d)`；新接口以独立 `u_e/u_i1/u_i2` 表达该约定。
7. 同页式 (10)：拟合信号 `P_signal` 先按最大值归一化，之后才乘幅度 `A` 和叠加线性背景。新函数只返回 `P_signal`。
8. PDF 第 5–6 页（论文页 055201-4 至 055201-5）的仪器响应、分辨率、逆韧致吸收、时间分辨、梯度、`k` 展宽和多体元求和属于更外层的实验合成流程，不进入本次单体元基础函数。

### 1.2 功能目标

在 `phyEqns/ots_full_fourier.py` 中新增公开函数 `OTS_spectrum_maxwellian_two_ion(...)`：

- 输入电子温度 `T_e_ev`；
- 输入两种离子的独立温度 `T_i1_ev/T_i2_ev`；
- 直接使用解析一维 Maxwellian 分布；
- 直接按式 (7)–(9) 计算 `chi_e/chi_i1/chi_i2`；
- 按式 (2) 计算双离子动态结构因子；
- 保留式 (1) 中随波长变化的频移修正；
- 返回单个均匀体元、单个波长段的归一化光谱。

### 1.3 用户故事

作为 OTS 正向模型使用者，我希望给定电子温度和两种离子温度后，直接走论文的 Maxwellian 解析极化率路径计算光谱，不经过 full-Fourier 的速度网格、低通模式或数值 Landau 响应。

### 1.4 范围内

- 无碰撞、无磁、非相对论 Maxwellian 等离子体；
- 两种离子有独立的温度、质量、电荷和相对数密度；
- 温度参数统一使用 eV；
- 三种物种分别支持漂移速度；
- 介质内入射/散射光波数、截止频率检查和当前工作树的成对 `k/omega` 符号约定；
- 单体元归一化物理信号 `P_signal(lambda)`；
- float32/float64、CPU/CUDA tensor 语义与现有模块保持一致；
- 为式 (9) 提供无溢出的数值实现，并用独立高精度基准验收。

### 1.5 范围外

- 不调用或近似复用 full-Fourier 计算路径；
- 不使用 legacy `w_fast` 的五系数有理拟合冒充式 (9)；
- 不修改现有 `OTS_spectrum_full_fourier(...)` 的签名、函数体或返回值；
- 不实现式 (10) 的 `A/B1/B2`、最小二乘拟合或参数反演；
- 不实现仪器卷积、体元积分、梯度、`k` 展宽、逆韧致吸收或时间合成；
- 不输出绝对功率；
- 不实现碰撞、磁化、相对论或非 Maxwellian 修正；
- 不修改 `data.py`、Parquet、manifest、训练、评估、CLI、registry、Slurm、`ots_fourier.py` 或 `ots_superGaussian.py`。

### 1.6 验收目标

- 生产数据流明确出现式 (7)、(8)、(9) 的各个因子；
- 两种离子分别使用自己的 `T_ij`，不存在共享 `T_i`；
- 新函数不调用任何 Fourier helper；
- 式 (9) 在 `x=0`、正负 `x`、中等 `x` 和大 `|x|` 区间稳定、finite，并满足奇偶性与渐近行为；
- 新函数输出 shape、dtype、device、finite 性和归一化合同明确；
- 现有用户未提交的波矢符号修改不被覆盖；
- 目标测试和完整测试均现场执行并如实报告。

## 2. Design（设计）

### 2.1 架构决策

新增解析路径与现有 full-Fourier 路径并列：

```text
显式 v/f ----------------------> OTS_spectrum_full_fourier

T_e, T_i1, T_i2 [eV]
        |
        +--> Maxwellian f_e, f_i1, f_i2 at phase velocity
        +--> Eq. (9) w(x_e), w(x_i1), w(x_i2)
        +--> Eq. (7) chi_e
        +--> Eq. (8) chi_i1, chi_i2
        +--> Eq. (2) S(k, omega)
        +--> OTS_spectrum_maxwellian_two_ion
```

为避免改变已有 canonical 行为，本次不把现有大函数重构成新的共享核心。新解析函数只复用当前模块已有的常数、`_as_real_tensor(...)` 和 `_projected_phase_velocity(...)`；输入验证、几何和式 (2) 的少量表达式在新函数内显式写出。这样虽然存在少量重复，但改动边界最小、审阅时可逐式对照论文，也不会意外改变数据生成主线。

### 2.2 温度、热速度和 Maxwellian 分布

温度统一以 eV 输入，并先转成每粒子热能：

```text
E_e  = T_e_ev  * ELEMENTARY_CHARGE
E_i1 = T_i1_ev * ELEMENTARY_CHARGE
E_i2 = T_i2_ev * ELEMENTARY_CHARGE

v_te  = sqrt(E_e  / ELECTRON_MASS)
v_ti1 = sqrt(E_i1 / m_i1)
v_ti2 = sqrt(E_i2 / m_i2)
```

对应论文式 (5)、(6)：

```text
f_s(v) = exp[-v^2/(2 v_ts^2)] / [sqrt(2*pi) v_ts]
```

这里的 `v_ts` 是一维 rms 热速度。漂移不写入 Maxwellian 中心，而是通过相速度 `v_phase_s = omega_shift/k - k_hat dot u_s` 处理，与论文正文一致。

### 2.3 Debye 长度、散射参数和无量纲相速度

```text
lambda_De = sqrt(VACUUM_PERMITTIVITY * E_e /
                 (n_e * ELEMENTARY_CHARGE^2))
alpha = 1 / (k_magnitude * lambda_De)

x_e  = v_phase_e  / (sqrt(2) * v_te)
x_i1 = v_phase_i1 / (sqrt(2) * v_ti1)
x_i2 = v_phase_i2 / (sqrt(2) * v_ti2)
```

`T_e/T_ij` 在式 (8) 中可直接使用 eV 比值，因为单位相消。

### 2.4 式 (9) 的稳定计算

定义 Dawson 积分：

```text
D(x) = exp(-x^2) integral_0^x exp(p^2) dp
w(x) = 1 - 2 x D(x) + i sqrt(pi) x exp(-x^2)
```

直接计算原积分中的 `exp(p^2)` 会在大 `|x|` 溢出。生产实现使用完全等价的变量代换 `p=x*t`：

```text
D(x) = x integral_0^1 exp[-x^2 (1-t^2)] dt
```

此表达式的指数恒不大于 0。中小 `|x|` 使用固定 64 点 Gauss-Legendre 求积；大 `|x|` 使用同一 Dawson 积分的渐近展开。节点和权重用已存在的 NumPy 在模块加载时生成，不新增环境依赖：

```text
D(x) ~ 1/(2x) * [
    1
    + 1/(2x^2)
    + 3/(2x^2)^2
    + 15/(2x^2)^3
    + ...
]
```

固定内部参数：

| 参数 | 值 | 原因 |
|---|---:|---|
| Gauss-Legendre 阶数 | 64 | 对 `|x| <= 8` 提供高精度且矩阵规模小 |
| 渐近切换阈值 | `|x| > 8` | 避免边界层求积精度下降 |
| 渐近项数 | 10 | 在切换点保留充分余量 |

这些参数不是公共 API。若独立高精度验收未达到第 4 节容差，停止实施并回到 SPEC 审阅，不静默更换算法或容差。

### 2.5 双离子极化率：逐式实现 (7)、(8)

```text
Z_bar_weight = Z_i1*N_i1 + Z_i2*N_i2

chi_e = alpha^2 * w(x_e)

chi_i1 = alpha^2
         * (Z_i1^2*N_i1 / Z_bar_weight)
         * (T_e_ev/T_i1_ev)
         * w(x_i1)

chi_i2 = alpha^2
         * (Z_i2^2*N_i2 / Z_bar_weight)
         * (T_e_ev/T_i2_ev)
         * w(x_i2)

epsilon = 1 + chi_e + chi_i1 + chi_i2
```

这一路径不通过离子等离子体频率间接重写式 (8)，以便代码审阅能够逐项看到论文原式。

### 2.6 式 (2) 与归一化信号

```text
electron_screening = abs(1 - chi_e/epsilon)^2
ion_screening = abs(chi_e/epsilon)^2

S = 2*pi/k * [
      electron_screening * f_e(v_phase_e)
      + ion_screening * (Z_i1^2*N_i1/Z_bar_weight) * f_i1(v_phase_i1)
      + ion_screening * (Z_i2^2*N_i2/Z_bar_weight) * f_i2(v_phase_i2)
    ]

power = real[(1 + 2*omega_shift/omega_0) * S]
intensity = power / max(abs(power))
```

论文式 (1) 中 `I_0 r_e^2 V n_e sin^2(phi)/(2*pi)` 对一个固定体元和固定几何是整体常数，在 `P_signal` 最大值归一化时消去。为保持与现有接口一致，`E_i0` 和 `V` 仍验证但不进入返回值。绝对功率需要独立 SPEC。

### 2.7 新增内部函数 1

#### `_dawson_integral_eq9(x) -> torch.Tensor`

接口：

```text
_dawson_integral_eq9(x: torch.Tensor) -> torch.Tensor
```

输入/输出：

- `x` 必须是 finite 实 tensor；
- 输出与 `x` 同 shape/device/dtype；
- 保持奇函数性质；
- 不接受或返回 NumPy array。

伪代码：

```text
if x 为 complex 或包含 non-finite:
    raise ValueError

把模块级 64 点 Gauss-Legendre nodes/weights
转换为 x.device/x.dtype 上的 tensor

# 对所有 x 计算稳定求积候选；指数 <= 0，不会溢出
t = nodes mapped from [-1,1] to [0,1]
quadrature = x * sum_t(weights * exp[-x^2*(1-t^2)])

# 只在大 |x| 使用渐近候选；对其他元素用 safe_x=1 防止除零
large = abs(x) > 8
safe_x = where(large, x, 1)
term = 1
series = 1
for n from 1 to 9:
    term = term * (2*n - 1) / (2*safe_x^2)
    series = series + term
asymptotic = series / (2*safe_x)

return where(large, asymptotic, quadrature)
```

### 2.8 新增内部函数 2

#### `_maxwellian_response_eq9(x) -> torch.Tensor`

接口：

```text
_maxwellian_response_eq9(x: torch.Tensor) -> complex torch.Tensor
```

伪代码：

```text
D = _dawson_integral_eq9(x)
real_part = 1 - 2*x*D
imag_part = sqrt(pi)*x*exp(-x^2)
response = real_part + 1j*imag_part

if response 包含 non-finite:
    raise ValueError("Equation (9) response is non-finite")
return response
```

该函数不调用 `w_fast`，也不改变式 (9) 的虚部符号。

### 2.9 新增公开函数

#### `OTS_spectrum_maxwellian_two_ion(...) -> torch.Tensor`

接口：

```text
OTS_spectrum_maxwellian_two_ion(
    lms,
    ss,
    k_sh,
    E_i0,
    lm_0,
    n_e,
    T_e_ev,
    Z_i1,
    N_i1,
    m_i1,
    T_i1_ev,
    Z_i2,
    N_i2,
    m_i2,
    T_i2_ev,
    V,
    *,
    u_e,
    u_i1,
    u_i2,
) -> torch.Tensor
```

接口决策：

- 三个温度紧邻对应物种参数，且名称带 `_ev`；
- 漂移保持 keyword-only，防止三个三分量向量错位；
- 不提供 `fourier_modes`、速度网格或 Fourier 系数参数；
- 只返回光谱，不新增 `return_chi`、配置对象或字典返回。

完整伪代码：

```text
1. 按现有 OTS 函数规则把 lms 转成实 tensor，并由 lms 确定 device/dtype。
2. 验证 lms 是非空一维、finite、正波长。
3. 验证 ss/k_sh/E_i0/u_e/u_i1/u_i2 是 finite 三分量实向量。
4. 验证 lm_0/n_e/T_e_ev/Z_i1/N_i1/m_i1/T_i1_ev/
   Z_i2/N_i2/m_i2/T_i2_ev/V 是实 finite scalar。
5. 要求 lm_0、n_e、三个温度、两种质量、V > 0；N_i1/N_i2 >= 0；
   Z_i1*N_i1 + Z_i2*N_i2 > 0。
6. 归一化 ss 与 k_sh；拒绝零向量。
7. 计算 omega_0、omega_pe^2、介质内 k_i；检查入射 cutoff。
8. 逐 lms 计算 omega_s、介质内 k_s；检查散射 cutoff。
9. 按当前批准的 k_s-k_i 构造 k_components，计算 k_magnitude；
   omega_shift = omega_s-omega_0。
10. 分别调用 _projected_phase_velocity 得到电子、离子 1、离子 2 相速度。
11. 由三个温度和三种质量计算 v_te/v_ti1/v_ti2。
12. 直接按式 (5)、(6) 计算三个 phase-velocity Maxwellian 值。
13. 计算 lambda_De、alpha、x_e/x_i1/x_i2。
14. 三次调用 _maxwellian_response_eq9。
15. 逐项按式 (7)、(8) 计算 chi_e/chi_i1/chi_i2。
16. 计算 epsilon，并逐项按式 (2) 计算结构因子。
17. 乘式 (1) 的波长相关频移修正，取实部。
18. 用 finite 且正的 max(abs(power)) 归一化；否则 ValueError。
19. 验证输出全部 finite，返回 intensity。
```

### 2.10 现有未提交修改与符号约定

现场基线中 `phyEqns/ots_full_fourier.py` 已有用户未提交修改：

```text
k_components: k_i-k_s -> k_s-k_i
omega_shift:  omega_s-omega_i（未变）
```

这与 Zhao et al. 的 `k=k_s-k_i`、`omega=omega_s-omega_i` 成对定义一致。整体翻转 `k` 不改变 `|k|`，但会改变非零漂移的投影。现有测试仍按旧分量符号断言。

本 SPEC 保留该用户修改，不重写它。实施时，新解析函数明确采用相同的 `k_s-k_i`；同时把受影响的既有几何测试期望同步到已批准约定，并新增非零漂移方向测试。

### 2.11 依赖决策

生产实现继续依赖 PyTorch，并新增 `import numpy as np`，只用于模块加载时生成固定 Gauss-Legendre 节点和权重。NumPy 已是 canonical 主线依赖，因此不需要新增安装包、锁文件或 Slurm 模块。

不引入 SciPy。原因是 `scipy.special.dawsn` 虽可简化 CPU 实现，但会新增未锁定依赖、CPU/NumPy 往返、CUDA 兼容问题和 autograd 断点。测试若本机恰有 SciPy，可把它作为附加 oracle，但正式验收不能依赖 SciPy 存在。

## 3. Tasks（任务拆解）

1. `T0`：用户审阅并批准本修订 SPEC，确认解析式 (7)–(9)、eV 单位、归一化输出及当前 `k_s-k_i` 基线。依赖：无。
2. `T1`：在 `ots_full_fourier.py` 添加固定 Gauss-Legendre 内部常数及 `_dawson_integral_eq9(...)`。依赖：T0。
3. `T2`：实现 `_maxwellian_response_eq9(...)`，保持论文实部、虚部和符号。依赖：T1。
4. `T3`：实现 `OTS_spectrum_maxwellian_two_ion(...)` 的验证、几何、Maxwellian、式 (7)、式 (8)、式 (2) 和归一化。依赖：T2。
5. `T4`：补充 docstring，逐项注明公式、单位、漂移约定、适用域和范围外实验效应。依赖：T3。
6. `T5`：在 `tests/test_ots_full_fourier_t1.py` 添加式 (9)、式 (7)–(8)、双温度独立性、无 Fourier 调用、错误、dtype/device 和归一化测试；同步经批准的既有波矢测试符号。依赖：T3。
7. `T6`：静态审查 diff，确认旧公开函数和 canonical 数据路径未改，用户既有修改保留。依赖：T4、T5。
8. `T7`：运行目标测试和完整 `unittest` discovery，记录现场环境和数量。依赖：T6。
9. `T8`：用论文 263.3 nm/90 度/pentane 风格 IAW 条件和当前 351 nm/123 度条件生成数值对照，记录式 (9) 误差、各 `chi` 和三温度扰动。依赖：T7。
10. `T9`：实施完成后更新根 `AGENTS.md` 的实现事实、依赖、测试、风险、维护记录和顶部同步日期；SPEC 审阅阶段不更新长期事实。依赖：T8。

## 4. Acceptance Criteria / Tests（验收标准）

### 4.1 逐式静态验收

- 新公开函数源码中可直接定位 `alpha^2*w(x_e)`；
- 两个离子项分别可直接定位 `Z_j^2*N_j/Z_bar_weight`、`T_e_ev/T_ij_ev` 和 `w(x_ij)`；
- `_maxwellian_response_eq9` 明确实现 `1-2*x*D(x) + i*sqrt(pi)*x*exp(-x^2)`；
- 新函数直接实现 Maxwellian `f_e/f_i1/f_i2`、`epsilon` 和式 (2) 三项；
- mock 证明没有调用 `ft_x_n`、`fourier_landau_response` 或 `OTS_spectrum_full_fourier`；
- 不导入 legacy `ots_fourier`、`ots_superGaussian` 或 SciPy；
- 旧 `OTS_spectrum_full_fourier(...)` 的签名、函数体和返回值不变。

### 4.2 式 (9) 数值验收

至少验证：

- `D(0)=0`、`w(0)=1+0i`；
- `D(-x)=-D(x)`；`Re[w(-x)]=Re[w(x)]`；`Im[w(-x)]=-Im[w(x)]`；
- 在 `x in [-8,8]` 的稠密样本上，与测试内独立 256 点 Gauss-Legendre 参考相比：float64 最大绝对误差不超过 `1e-11`，float32 不超过 `2e-5`；
- 在 `|x|=10,20,100,1000` 上，与更高阶渐近参考相比相对误差不超过 `1e-10`（float64）或 `2e-5`（float32）；
- 阈值两侧 `x=8*(1+-1e-6)` 的响应连续，差值符合上述 dtype 容差；
- 所有 finite 实输入产生 finite 复响应；
- 与论文式 (9) 的虚部正号一致。

若 64 点/阈值 8/10 项未达到以上结果，停止实施并提交实测误差，不静默改变参数或容差。

### 4.3 式 (7)、(8) 与双离子温度验收

- 用测试内独立标量实现计算代表点，逐点核对 `chi_e/chi_i1/chi_i2`；
- `T_i1_ev != T_i2_ev` 时，两者分别使用各自温度和热速度；
- 只改变 `T_e_ev`、`T_i1_ev`、`T_i2_ev` 时，在两个离子组分均非零的 IAW 波段，三种扰动均产生 finite 且可检测的谱变化；
- 当 `N_i2=0` 时，只改变 `T_i2_ev` 不改变输出；离子 1 同理；
- 交换两套完整离子参数 `(Z,N,m,T,u)` 后输出不变；
- `T_i1_ev=T_i2_ev` 但质量不同时，`x_i1/x_i2` 仍由各自热速度计算。

### 4.4 光谱与接口验收

- 输出 shape 与 `lms` 相同，dtype/device 跟随 `lms`；
- 输出全部 finite，且 `max(abs(intensity))=1`；
- float32/float64 均通过；CUDA 可用时在 CUDA 上执行且不发生 NumPy 数据往返；
- 非零漂移测试验证当前 `k_s-k_i` 和 `omega_s-omega_i` 的成对符号；
- 论文 263.3 nm、90 度代表条件及当前 351 nm、123 度条件均得到 finite 光谱；
- 新接口不存在 `fourier_modes/v_e/f_e/K/F_e` 参数；
- `E_i0/V` 的兼容性和“归一化谱不使用绝对 prefactor”在 docstring 中明确。

### 4.5 非法输入验收

以下输入必须抛出带字段名的清晰 `ValueError`：

- 三个温度任一为 0、负数、NaN、Inf、复数或非标量；
- 两种离子质量任一为 0、负数、NaN、Inf、复数或非标量；
- 非法波长、密度、体积、组分权重、方向或漂移；
- 入射光或任一散射波长低于电子等离子体截止频率；
- `k`、`epsilon`、功率或归一化比例非 finite/无效。

### 4.6 回归与范围验收

- 运行 `tests/test_ots_full_fourier_t1.py`；
- 运行仓库完整 `unittest` discovery；
- 报告现场通过/失败/跳过数量，不能沿用历史记录；
- `git diff` 证明本任务只涉及：本 SPEC、`ots_full_fourier.py` 新增解析函数/内部 helper、目标测试和实施完成后的 `AGENTS.md`；
- `data.py` 仍调用原 full-Fourier 函数，默认数据生成和既有模型不会因新增接口自动改变；
- 不执行数据生成、训练、远端同步或 Slurm 作业。

## 5. 审阅确认项

用户已经明确选择“逐式实现式 (7)–(9) 的解析计算路径”。本轮仍需确认以下实现边界：

1. 温度公开单位固定为 eV；
2. 输出是均匀单体元的归一化 `P_signal`，不含仪器卷积、幅度或背景；
3. 式 (9) 使用数学等价的稳定 Dawson 求值：64 点 Gauss-Legendre + 大 `|x|` 渐近展开，不使用 `w_fast`；
4. 新解析函数独立实现，不重构或调用现有 full-Fourier 函数；
5. 保留当前工作树已存在的 `k_s-k_i` / `omega_s-omega_i` 成对符号，并同步相应测试。
