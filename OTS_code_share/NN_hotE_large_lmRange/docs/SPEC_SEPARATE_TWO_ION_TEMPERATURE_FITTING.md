# SPEC：双离子独立温度光谱拟合

## 0. 状态与范围

已批准并实施（2026-09-06）。

本次只修改：

```text
spectral_fitting.py
```

实施完成后，按仓库协议同步 `AGENTS.md` 中的接口说明和同步日期。不新增模块、CLI、数据文件或输出文件。

## 1. Requirements（需求）

### 1.1 功能目标

把 `fit_spectrum(...)` 从共同离子温度的七参数拟合：

```text
(T_e, T_i, u_ik, u_ek, A, B1, B2)
```

改成两种离子温度相互独立的八参数拟合：

```text
(T_e, T_i1, T_i2, u_ik, u_ek, A, B1, B2)
```

两种离子仍共用投影漂移速度 `u_ik`；电子使用 `u_ek`。拟合函数继续输入一条光谱并返回最优参数字典。

### 1.2 用户故事

调用者给出波长、测量光谱、固定电子密度、仪器 FWHM 和双离子物性后，可以分别得到第一、第二种离子的温度，而不是把两者约束成同一个 `T_i`。

### 1.3 接口变化

- `initial_guess` 从 7 个值变为 8 个值，顺序固定为：

  ```text
  T_e_ev, T_i1_ev, T_i2_ev, u_ik_cm_s, u_ek_cm_s, A, B1, B2
  ```

- 返回字典删除 `T_i_ev`，新增：

  ```text
  T_i1_ev
  T_i2_ev
  ```

- 两个离子温度分别使用 `10–6000 eV` 的边界。
- 这是明确的接口替换，不保留旧 `T_i_ev` 别名，避免同一结果出现含义重叠的三个离子温度键。

### 1.4 范围外

本次不改变电子密度、离子组分、共同离子漂移、背景和仪器展宽的现有约定；不增加权重、误差条、多初值、全局优化、绘图、文件保存或 canonical CLI 集成。

## 2. Design（设计）

### 2.1 参数模型

八参数向量及边界为：

| 参数 | 下界 | 上界 |
|---|---:|---:|
| `T_e_ev` | 10 | 5000 |
| `T_i1_ev` | 10 | 6000 |
| `T_i2_ev` | 10 | 6000 |
| `u_ik_cm_s` | `-1.5e8` | `1.5e8` |
| `u_ek_cm_s` | `-3.0e8` | `3.0e8` |
| `A` | 0 | 2 |
| `B1` | 0 | 2 |
| `B2` | 0 | 2 |

未给初值时仍使用各边界中点。优化器仍为一次 `scipy.optimize.least_squares(method="trf")`，不引入新的优化层。

### 2.2 正向调用适配

```text
八参数
  |
  +-- T_e_ev  ------> 电子温度
  +-- T_i1_ev ------> OTS_spectrum_maxwellian_two_ion.T_i1_ev
  +-- T_i2_ev ------> OTS_spectrum_maxwellian_two_ion.T_i2_ev
  +-- u_ik ---------> 两种离子的共同漂移
  +-- u_ek ---------> 电子漂移
  +-- A/B1/B2 ------> 信号幅度和线性背景
```

已有双离子 Maxwellian 正向函数已经接受独立的 `T_i1_ev` 和 `T_i2_ev`，因此不修改物理模块。

### 2.3 `main()` 自洽验证适配

保留当前九种组分状态及报告中的基准 `T_i`。为真正检验两条独立温度通道，每个状态的无噪声合成真值设为：

```text
T_i1 = 0.8 * T_i_base
T_i2 = 1.2 * T_i_base
```

这些不相等温度只用于数值闭环验证，不声称来自报告。`main()` 分别检查 `T_i1` 和 `T_i2` 的反演误差，并在终端分别打印两者的拟合值/真值；仍不保存数据。

## 3. 函数接口与伪代码

### 3.1 `fit_spectrum(...)`

```python
def fit_spectrum(
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
    initial_guess=None,
    *,
    probe_wavelength_nm=263.3,
    scattering_angle_deg=90.0,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
) -> dict[str, float]:
```

伪代码：

```text
验证波长、光谱、密度、FWHM 和双离子物性
将参数名、上下界扩展为八项
若无 initial_guess，取八项边界中点
否则验证 initial_guess.shape == (8,) 且位于边界内
用 least_squares 最小化 _residuals
验证优化状态和结果 finite
按八个参数名返回 float 字典
```

### 3.2 `_residuals(...)`

```python
def _residuals(
    parameters,
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
    probe_wavelength_nm,
    scattering_angle_deg,
    ion1_charge,
    ion1_mass_amu,
    ion1_number_fraction,
    ion2_charge,
    ion2_mass_amu,
) -> np.ndarray:
```

伪代码：

```text
P_fit = _fit_model(八参数及固定物理输入)
return P_fit - measured_power
```

### 3.3 `_fit_model(...)`

```python
def _fit_model(
    parameters,
    wavelength_nm,
    electron_density_cm3,
    instrument_fwhm_nm,
    probe_wavelength_nm=263.3,
    scattering_angle_deg=90.0,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
) -> np.ndarray:
```

伪代码：

```text
验证 parameters 是八个 finite 值
拆分为 Te, Ti1, Ti2, uik, uek, A, B1, B2
由电子密度、离子数目分数和电荷计算两种离子密度
建立散射几何并把 uik/uek 转为沿 k 的 m/s 向量
调用 OTS_spectrum_maxwellian_two_ion:
    electron temperature = Te
    ion-1 temperature = Ti1
    ion-2 temperature = Ti2
    both ion drifts = uik
    electron drift = uek
执行现有 Gaussian 仪器展宽和最大值归一化
按现有公式添加 A 和 B1→B2 线性背景
验证结果 finite 并返回
```

### 3.4 `_gaussian_convolve(...)`

```python
def _gaussian_convolve(
    wavelength_nm,
    signal,
    fwhm_nm,
) -> np.ndarray:
```

伪代码：

```text
由 FWHM 计算 sigma
用波长两两间距构造 Gaussian kernel
逐行归一化 kernel
broadened = kernel @ signal
验证结果 finite 并返回
```

此函数接口和算法不变，只作为完整实现合同列出。

### 3.5 `main()`

```python
def main() -> int:
```

伪代码：

```text
建立当前 351 nm、0.3 nc、仪器 FWHM 和波长网格
建立当前九种 H-He、H-C、Au-B 组分状态
对每个状态:
    Ti1_true = 0.8 * Ti_base
    Ti2_true = 1.2 * Ti_base
    组成八参数真值并调用 _fit_model 生成无噪声光谱
    从偏离真值的八参数初值调用 fit_spectrum
    用拟合八参数重新正向计算光谱
    分别检查 Te、Ti1、Ti2、速度、幅度/背景和光谱误差
    打印 Te/Ti1/Ti2 的 fitted/true 及 relative L2
全部状态通过后返回 0；任一失败抛出 RuntimeError
```

## 4. Tasks（任务拆解）

1. 审阅并批准本 SPEC。
2. 修改参数名、边界、初值长度和返回合同为八参数。
3. 修改 `_fit_model`，把两个温度分别传给两种离子。
4. 适配文档字符串、错误信息和 `main()` 的闭环输入、误差检查及打印。
5. 用仓库指定 Python 3.11.9 运行 `spectral_fitting.py`，验证九种状态。
6. 检查 Git diff，确认没有改动无关文件。
7. 同步 `AGENTS.md` 中的接口事实和日期。

## 5. Acceptance Criteria / Tests（验收标准）

- `fit_spectrum(...)` 接受恰好 8 项初值，7 项或其他长度会明确报错。
- 返回键恰好为 `T_e_ev/T_i1_ev/T_i2_ev/u_ik_cm_s/u_ek_cm_s/A/B1/B2`。
- `_fit_model` 分别把 `T_i1_ev` 和 `T_i2_ev` 传给正向模型，不再复制同一个温度。
- 单独改变 `T_i1_ev` 或 `T_i2_ev` 均能独立影响对应双离子正向谱。
- 当前九种双离子状态全部完成无噪声正向→反向→正向闭环。
- 每个状态的 `T_e/T_i1/T_i2` 相对误差均不超过 `1e-4`，速度绝对误差不超过 `1e3 cm/s`，幅度/背景绝对误差不超过 `1e-5`，重建光谱 relative L2 不超过 `1e-6`。
- 验证过程只打印结果，不保存数据或图片。
- 不修改 `phyEqns/ots_full_fourier.py`、canonical CLI 或其他无关实现文件。
