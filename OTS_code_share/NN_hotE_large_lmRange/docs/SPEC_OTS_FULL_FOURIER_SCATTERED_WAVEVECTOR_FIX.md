# SPEC: `ots_full_fourier` 散射光介质内波矢修复

## 1. Requirements（需求）

### 1.1 功能目标

仅修复 `phyEqns/ots_full_fourier.py` 中散射光波数错误使用真空色散 `omega/c` 的问题，使入射光与散射光都在同一均匀、无磁、无碰撞冷等离子体近似下满足：

```text
k_medium(omega) = sqrt(omega^2 - omega_pe^2) / c
```

本次明确不修改 `phyEqns/ots_fourier.py`，不修改公开函数签名，不修改 Fourier、Landau 响应、动态结构因子、归一化或当前波矢/频移符号约定。

### 1.2 用户故事

作为 OTS 正向模型维护者，我希望 canonical `ots_full_fourier` 在散射体积内使用物理一致的入射/散射电磁波矢，并拒绝低于电子等离子体截止频率的散射波长，从而避免 `|Delta k|`、相速度和介电响应的系统性偏移。

### 1.3 验收标准

- `ots_full_fourier.py` 中入射与散射波数都包含同一个 `omega_pe_squared`；
- 对每个散射波长验证 `omega_s^2 - omega_pe^2 > 0`；不传播的散射波立即抛出清晰 `ValueError`；
- 在弹性点 `omega_s=omega_i`、同一介质中，入射与散射波数模相等；
- `k_components` 使用介质内散射波数，不再使用 `omega/SPEED_OF_LIGHT`；
- 保留现有 `k_components = k_i - k_s` 与 `omega_shift = omega_s - omega_i`，符号统一属于后续独立任务；
- `OTS_spectrum_full_fourier` 的签名、返回形状、dtype/device 行为与分段归一化合同不变；
- 不修改 `phyEqns/ots_fourier.py`；
- 保留工作树中 `ots_full_fourier.py` 已有的 `__main__` 演示块及其他用户修改；
- 相关回归测试及现有测试通过。

## 2. Design（设计）

### 2.1 架构决策

采用最小内联修复，不新增抽象层或公共接口。现有函数已经计算 `omega_pe_squared`，直接复用它计算散射波数数组，以避免不必要的辅助类、配置项或数据结构。

```text
lambda_i ----------------> omega_i --> k_i_medium
lambda_s[] --------------> omega_s --> k_s_medium[]
                                       |
directions i_hat, s_hat ---------------+
                                       v
                          Delta k = k_i - k_s
                                       |
                         后续相速度/chi/S 保持不变
```

### 2.2 数据与接口

不新增持久化数据或配置字段。

#### `OTS_spectrum_full_fourier(...) -> torch.Tensor`

接口保持：

```text
OTS_spectrum_full_fourier(
  lms,
  ss,
  k_sh,
  E_i0,
  lm_0,
  n_e,
  Z_i1,
  N_i1,
  m_i1,
  Z_i2,
  N_i2,
  m_i2,
  V,
  *,
  v_e,
  f_e,
  v_i1,
  f_i1,
  v_i2,
  f_i2,
  u_e,
  u_i1,
  u_i2,
  fourier_modes=80,
) -> torch.Tensor
```

修复部分伪代码：

```text
omega_i = 2*pi*c/lambda_i
omega_pe_squared = n_e*e^2/(m_e*epsilon_0)

incident_argument = omega_i^2 - omega_pe_squared
if incident_argument <= 0:
    raise ValueError(probe below plasma cutoff)
k_i = sqrt(incident_argument)/c * incident_direction

omega_s = 2*pi*c/lambda_s
scattered_argument = omega_s^2 - omega_pe_squared
if any(scattered_argument <= 0) or any(nonfinite):
    raise ValueError(scattered light below plasma cutoff)
k_s_magnitude = sqrt(scattered_argument)/c

k_components = k_i - k_s_magnitude * scattering_direction
k_magnitude = norm(k_components)

# 以下现有流程不变
omega_shift = omega_s - omega_i
phase_velocity = projected_phase_velocity(...)
responses = fourier_landau_response(...)
chi, dielectric, structure_factor, power = existing_formulas(...)
return independently_normalized(power)
```

#### `_projected_phase_velocity(...) -> torch.Tensor`

接口与实现不变；测试只用于确认它接收到修正后的 `k_magnitude/k_components`：

```text
drift_projection = dot(k_components, drift) / k_magnitude
return omega_shift / k_magnitude - drift_projection
```

### 2.3 错误处理

- 入射截止继续使用现有错误；
- 新增散射截止错误，必须区分于普通非有限 `k_magnitude`；
- 不引入复波数/衰减模型。低于 cutoff 的情况直接拒绝，因为当前下游只支持实传播波。

## 3. Tasks（任务拆解）

1. `T1`：code agent 在 `OTS_spectrum_full_fourier` 内计算并验证 `scattered_argument`；依赖：无。
2. `T2`：code agent 用 `k_s_magnitude` 替换三个 `omega/c` 分量，保持其余代码不变；依赖：T1。
3. `T3`：主 agent 审查 diff，确认 `ots_fourier.py` 未变且用户演示块保留；依赖：T2。
4. `T4`：test agent 增加介质散射波矢、弹性点和散射 cutoff 回归测试；依赖：T3。
5. `T5`：test agent 运行目标测试模块和完整测试套件，报告命令、通过/失败数量及警告；依赖：T4。
6. `T6`：主 agent 汇总验收与剩余边界；依赖：T5。

## 4. Acceptance Criteria / Tests（验收标准）

### 4.1 静态验收

- `git diff -- phyEqns/ots_fourier.py` 为空；
- `ots_full_fourier.py` 的生产代码 diff 只包含散射介质色散、cutoff 校验及必要注释；
- 既有未提交 `__main__` 演示块没有被删除或重写；
- 公开签名没有变化。

### 4.2 数值验收

- `omega_pe -> 0` 时修复结果退化为 `k_s=omega_s/c`；
- `lambda_s=lambda_i` 时 `|k_s|=|k_i|`，且 `|Delta k|=2*k_i*sin(theta/2)`；
- 默认 `n_e=1.81e26 m^-3`、`lambda_i=351 nm`、`theta=123 deg` 下，捕获到的 `k_components` 与书中介质公式一致；
- 选择 `lambda_s > 2*pi*c/omega_pe` 时抛出散射 cutoff `ValueError`；
- 默认三波段仍产生 finite、形状不变、可归一化的输出。

### 4.3 回归验收

- 运行 `tests/test_ots_full_fourier_t1.py`；
- 运行仓库现有完整 `unittest` 测试发现流程；
- test agent 不修改生产代码；若发现问题，只报告给主 agent，不自行修复。
