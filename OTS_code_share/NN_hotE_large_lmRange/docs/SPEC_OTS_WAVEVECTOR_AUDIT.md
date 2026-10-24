# SPEC: OTS 入射/散射光介质内波矢审计

## 1. Requirements（需求）

### 1.1 功能目标

只读核查 `phyEqns/ots_fourier.py` 与 `phyEqns/ots_full_fourier.py` 中入射光、散射光及涨落波矢的计算，逐项对照 Zotero 书目 *Plasma Scattering of Electromagnetic Radiation: Theory and Measurement Techniques* 的原始公式，回答：

1. 两份实现是否对入射光使用了等离子体内波矢，而对散射光错误使用了真空波矢；
2. 正确的介质内色散关系、符号约定和适用近似是什么；
3. 若存在错误，哪些下游量受到影响（`k`、`omega/k`、极化率、动态结构因子、频移修正和归一化谱形）；
4. 该错误在当前默认参数及三个波段上的数值量级；
5. 两份文件是否同错、是否存在实现差异，以及 canonical 主线的实际影响。

本轮不修改源码、不覆盖数据、不重训练模型。若确认需要修复，另开实施 SPEC。

### 1.2 用户故事

作为 OTS 正向模型维护者，我希望确认入射与散射电磁波都遵守一致的等离子体色散关系，从而避免把真空中的 `omega/c` 与介质内的 `sqrt(omega^2-omega_pe^2)/c` 混用，导致散射几何和相速度系统性偏移。

### 1.3 验收标准

- 给出书中可定位的章节、页码或公式编号及必要的短摘录/释义；
- 对两份文件的实际代码行逐一建立“物理量—代码表达式—理论表达式”映射；
- 明确区分单束光的 `k_i/k_s`、涨落波矢 `k = k_s-k_i`（或相反号约定）和频移 `omega = omega_s-omega_i`；
- 明确判断“正确 / 错误 / 仅在某近似下成立”，不能只指出表达式看起来不一致；
- 至少用当前默认 `lambda_0=351 nm`、`n_e=1.81e26 m^-3`、`theta=123 deg` 做一个数值对照；
- 说明结论对 canonical `ots_full_fourier.py` 与非 canonical `ots_fourier.py` 的影响边界；
- 不修改用户已有的 `ots_full_fourier.py` 工作树改动。

## 2. Design（设计）

### 2.1 架构决策

采用“原始文献证据 + 静态数据流审阅 + 独立数值复算”的三角验证：

```text
Zotero 原书相关公式
        |                 当前源码表达式
        |                       |
        +---- 统一符号/单位 -----+
                    |
         独立计算 k_i, k_s, |Δk|
                    |
       判断错误、近似条件与影响范围
```

不以仓库演示稿或既有知识库摘要替代原书；不因两个实现彼此一致就推断物理正确。

### 2.2 理论数据模型

对每个散射波长样本记录以下概念量：

| 量 | 定义 |
|---|---|
| `omega_i` | 入射光角频率 `2*pi*c/lambda_0` |
| `omega_s` | 散射光角频率 `2*pi*c/lambda_s` |
| `omega_pe` | 电子等离子体角频率 |
| `k_i_mag` | 由书中介质色散关系得到的入射光波数模 |
| `k_s_mag` | 由同一介质色散关系得到的散射光波数模 |
| `k_vec` | 按代码符号约定组成的 `k_i_vec-k_s_vec` 或其相反数 |
| `k_mag` | `|k_vec|` |
| `omega_shift` | 与 `k_vec` 配套的频率差 |
| `v_phase` | `omega_shift/k_mag`，含漂移投影时按实现修正 |

所有光学波矢均先在同一介质、同一局域电子密度假设下计算；若书中允许 `omega >> omega_pe` 的真空近似，则单独量化该近似误差，不能只对其中一束光使用。

### 2.3 审计接口与伪代码

本轮不新增生产函数。以下是需要审计的现有具体函数接口及其波矢相关伪代码。

#### `OTS_spectrum_fft(...) -> normalized_spectrum`

接口：

```text
OTS_spectrum_fft(
  lms, ss, k_sh, E_i0, lm_0, n_e, T_e,
  Z_i1, N_i1, m_i1, Z_i2, N_i2, m_i2,
  T_i, V, K, F_e, u_e, u_i1, u_i2
) -> tensor/array
```

审计伪代码：

```text
读取函数内 omega_i、omega_s、omega_pe 的定义
提取入射光波矢模及方向
提取散射光波矢模及方向
构造代码实际的涨落波矢和频移
与原书同一符号约定下的表达式比较
追踪 k_mag 到 phase velocity、chi、S(k,omega) 和输出谱
```

#### `OTS_spectrum_fft_real(...) -> normalized_spectrum`

接口：

```text
OTS_spectrum_fft_real(
  lms, ss, k_sh, E_i0, lm_0, n_e, T_e,
  Z_i1, N_i1, m_i1, Z_i2, N_i2, m_i2,
  T_i, V, K, F_er, F_ei, u_e, u_i1, u_i2
) -> tensor/array
```

审计伪代码：

```text
重复 OTS_spectrum_fft 的波矢审计
确认实/虚 Fourier 拆分没有引入另一套几何表达式
比较两个 legacy 接口的波矢代码是否逐项等价
```

#### `OTS_spectrum_full_fourier(...) -> normalized_spectrum`

接口：

```text
OTS_spectrum_full_fourier(
  lms, ss, k_sh, E_i0, lm_0, n_e,
  Z_i1, N_i1, m_i1, Z_i2, N_i2, m_i2, V,
  *, v_e, f_e, v_i1, f_i1, v_i2, f_i2,
  u_e, u_i1, u_i2, fourier_modes=80
) -> torch.Tensor
```

审计伪代码：

```text
验证输入波长、密度、散射方向的单位与形状
计算/提取 omega_i、omega_s、omega_pe
提取 k_i_vec 与 k_s_vec 的实际表达式
形成 k_components、k_magnitude、omega_shift
调用 _projected_phase_velocity 得到各物种共振速度
追踪 k_magnitude 到三物种 Landau 响应、chi、epsilon 和谱功率
与原书公式及 legacy 实现对照
```

#### `_projected_phase_velocity(omega_shift, k_magnitude, k_components, drift) -> velocity`

接口：

```text
_projected_phase_velocity(
  omega_shift: tensor,
  k_magnitude: tensor,
  k_components: tensor,
  drift: vector-like
) -> tensor
```

审计伪代码：

```text
projected_drift = dot(k_components, drift) / k_magnitude
return omega_shift / k_magnitude - projected_drift
```

检查该函数是否仅消费上游 `k`，还是内部又改变了介质/真空约定。

### 2.4 独立数值复算

```text
输入 lambda_i, lambda_s, n_e, theta
omega_i/s = 2*pi*c/lambda_i/s
omega_pe = sqrt(n_e*e^2/(m_e*epsilon_0))
按原书色散关系计算 k_i_mag 与 k_s_mag
按代码方向向量构造 k_vec_medium
复现代码表达式得到 k_vec_code
比较每个波段：
  relative_error(k_mag)
  relative_error(omega_shift/k_mag)
  必要时比较代表性 chi 或谱形
```

## 3. Tasks（任务拆解）

1. `T1`：通过 Zotero local API 定位书籍条目、附件和相关全文；依赖：无。
2. `T2`：提取介质内电磁波色散、散射波矢与频移的原书证据；依赖：T1。
3. `T3`：静态审阅 `ots_fourier.py` 两个公开谱函数及相关帮助函数；依赖：无。
4. `T4`：静态审阅 `ots_full_fourier.py` 的波矢到谱输出数据流；依赖：无。
5. `T5`：统一书和代码的符号约定，避免把整体反号误判为错误；依赖：T2、T3、T4。
6. `T6`：用默认参数及代表性波段独立复算并量化差异；依赖：T5。
7. `T7`：交叉复核，输出分文件结论、影响范围和修复建议；依赖：T6。

## 4. Acceptance Criteria / Tests（验收标准）

- 文献测试：每个核心理论判断都能回指 Zotero 原书位置，而非仅依赖二手资料；
- 静态测试：列出精确代码行，且覆盖 `OTS_spectrum_fft`、`OTS_spectrum_fft_real`、`OTS_spectrum_full_fourier`、`_projected_phase_velocity`；
- 符号测试：同时翻转 `k_vec` 与 `omega_shift` 时，说明 `omega/k` 是否保持一致；
- 一致性测试：分别判断 `k_i` 与 `k_s` 是否使用同一色散关系；
- 数值测试：报告至少一个离子特征附近波长和电子特征代表波长的 `k`/相速度差异；
- 影响测试：区分“绝对功率 prefactor 变化”和“归一化后仍会保留的谱形变化”；
- 边界测试：明确 `omega_s <= omega_pe` 时介质电磁波不能传播/波数转为虚数的处理要求；
- 变更测试：`git diff` 显示本轮没有改动两份源码或覆盖用户现有修改。
