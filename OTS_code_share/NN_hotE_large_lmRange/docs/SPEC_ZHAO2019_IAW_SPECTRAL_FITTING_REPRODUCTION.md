# SPEC：Zhao et al. (2019) IAW 光谱拟合函数

## 0. 状态

已批准并实施（2026-09-05）。

只计划在现有目录 `NN_hotE_large_lmRange/` 下新增一个文件：

```text
spectral_fitting.py
```

该文件只完成一件事：输入一条 IAW 光谱，返回拟合得到的七个参数。

## 1. Requirements（需求）

### 1.1 功能目标

按照 Zhao et al. (2019) 的 spectral fitting 方法，最小化：

```text
r = sum_lambda [P_fit(lambda) - P_exp(lambda)]^2
```

其中论文式 (10) 为：

```text
P_fit(lambda)
  = A * P_signal(lambda, T_e, T_i, u_ik, u_ek)
    + B1
    + (B2-B1) * (lambda-lambda1)/(lambda2-lambda1)
```

返回参数：

```text
T_e_ev
T_i_ev
u_ik_cm_s
u_ek_cm_s
A
B1
B2
```

### 1.2 输入

公开函数接收：

- `wavelength_nm`：一维波长数组，单位 nm；
- `measured_power`：与波长逐点对应的实验相对强度；
- `electron_density_cm3`：预先假定的电子密度，单位 cm^-3；
- `instrument_fwhm_nm`：仪器谱分辨率 FWHM，单位 nm；
- `initial_guess`：可选的七参数初值。

### 1.3 固定物理条件

第一版固定为论文 IAW 实验条件：

```text
probe wavelength = 263.3 nm
scattering angle = 90 deg
ion mixture = C5H12
Z_H = 1
Z_C = 6
T_H = T_C = T_i
```

电子密度由调用者提供，不参与拟合。H/C 数密度由 `C5H12` 化学计量和电中性计算。

### 1.4 范围外

本次不实现：

- CSV 或其他文件读取；
- 命令行接口；
- 绘图和输出文件；
- Figure 1 合成数据；
- `R`、`Delta R` 等高线和不确定度；
- Monte Carlo、时间序列、梯度和 `k` 展宽；
- 仪器相对响应曲线；
- EPW 拟合；
- 新测试文件；
- canonical 数据、训练、评估或 CLI 的修改。

## 2. Design（设计）

### 2.1 实现方式

复用当前已有的：

```python
phyEqns.ots_full_fourier.OTS_spectrum_maxwellian_two_ion
```

该函数负责 Zhao et al. 式 (2)、(5)–(9) 的双离子 Maxwellian OTS 正向计算。新文件只增加拟合外层，不复制正向物理公式。

流程：

```text
wavelength_nm + measured_power
        |
        +--> 计算 H/C 数密度和固定 90 deg 几何
        +--> 正向函数生成 P_signal
        +--> Gaussian FWHM 仪器展宽
        +--> 最大值归一化
        +--> 式 (10) 添加 A、B1、B2
        +--> scipy.optimize.least_squares
        +--> 返回七参数 dict
```

### 2.2 优化约定

使用 `scipy.optimize.least_squares(method="trf")` 做有界无权最小二乘。

默认参数边界：

| 参数 | 下界 | 上界 |
|---|---:|---:|
| `T_e` | 10 eV | 5000 eV |
| `T_i` | 10 eV | 6000 eV |
| `u_ik` | `-1.5e8 cm/s` | `1.5e8 cm/s` |
| `u_ek` | `-3.0e8 cm/s` | `3.0e8 cm/s` |
| `A` | 0 | 2 |
| `B1` | 0 | 2 |
| `B2` | 0 | 2 |

若未提供初值，使用边界中点。第一版只运行一次优化，不增加多初值或全局优化。

### 2.3 返回合同

拟合成功返回：

```python
{
    "T_e_ev": float,
    "T_i_ev": float,
    "u_ik_cm_s": float,
    "u_ek_cm_s": float,
    "A": float,
    "B1": float,
    "B2": float,
}
```

输入非法、正向谱非 finite 或优化失败时抛出清晰的 `ValueError` 或 `RuntimeError`，不返回部分结果。

## 3. 函数接口与伪代码

文件中只新增以下四个函数。

### 3.1 `fit_spectrum(...)`

唯一公开函数：

```python
def fit_spectrum(
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
    initial_guess=None,
) -> dict[str, float]:
```

伪代码：

```text
把 wavelength_nm 和 measured_power 转为 float64 ndarray
验证两者为同长度一维数组、长度 >= 16、全部 finite
验证波长为正且严格递增
验证电子密度和 FWHM finite 且 > 0

设置七参数 bounds
若 initial_guess is None:
    使用 bounds 中点
否则:
    验证 shape=(7,)、finite 且严格位于 bounds 内

调用 scipy.optimize.least_squares:
    fun = _residuals
    method = "trf"
    bounds = 默认边界
    x_scale = "jac"

若优化失败或结果 non-finite:
    raise RuntimeError

按固定键名返回七个 float
```

### 3.2 `_residuals(...)`

```python
def _residuals(
    parameters,
    wavelength_nm,
    measured_power,
    electron_density_cm3,
    instrument_fwhm_nm,
) -> np.ndarray:
```

伪代码：

```text
P_fit = _fit_model(...)
return P_fit - measured_power
```

### 3.3 `_fit_model(...)`

```python
def _fit_model(
    parameters,
    wavelength_nm,
    electron_density_cm3,
    instrument_fwhm_nm,
) -> np.ndarray:
```

伪代码：

```text
拆分 parameters = Te, Ti, uik, uek, A, B1, B2

按 C5H12 和电中性计算 N_H、N_C
建立 263.3 nm、90 deg 固定几何
把 uik/uek 从 cm/s 转为沿散射 k 方向的 m/s 三维向量

调用 OTS_spectrum_maxwellian_two_ion:
    electron temperature = Te
    H temperature = Ti
    C temperature = Ti
    electron drift = uek
    H/C drift = uik

把正向结果转为 NumPy float64
P_signal = _gaussian_convolve(...)
P_signal /= max(abs(P_signal))

t = (wavelength-lambda1)/(lambda2-lambda1)
background = B1 + (B2-B1)*t
P_fit = A*P_signal + background

验证 P_fit 全部 finite
return P_fit
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
sigma = fwhm / (2*sqrt(2*ln(2)))
根据 wavelength 点间距离构造 Gaussian kernel
逐行归一化 kernel
result = kernel @ signal
验证 result finite
return result
```

该写法允许输入波长网格不等间距，不额外增加重采样函数。

## 4. Tasks（任务拆解）

1. 用户审阅并批准本 SPEC。
2. 验证指定 Python 3.11.9 中已有 `numpy/scipy/torch`；缺少依赖时先报告。
3. 新建 `NN_hotE_large_lmRange/spectral_fitting.py`。
4. 实现四个函数和必要的输入验证。
5. 用一条由正向模型生成的光谱做最小数值闭环检查。
6. 确认除新 `.py` 和实施后的 `AGENTS.md` 同步外，没有修改其他文件。

## 5. Acceptance Criteria / Tests（验收标准）

- 只新增一个实现文件 `spectral_fitting.py`；
- `fit_spectrum(...)` 输入波长、光谱、固定电子密度和仪器 FWHM，返回七参数字典；
- `_fit_model(...)` 数值上严格满足论文式 (10)；
- 首末波长处的线性背景分别等于 `B1/B2`；
- `u_ik` 只用于离子漂移，`u_ek` 只用于电子漂移；
- H/C 使用同一个拟合离子温度 `T_i`；
- 仪器展宽后的 `P_signal` 归一化最大绝对值为 1；
- 合成无噪声光谱的优化结果 finite，且最终残差平方和小于初值；
- 非法数组、非递增波长、非正密度/FWHM、非法初值或优化失败会明确报错；
- 不增加 CLI、图、输出文件、demo、等高线、Monte Carlo 或独立测试模块；
- 不修改现有正向函数和 canonical 流水线。
