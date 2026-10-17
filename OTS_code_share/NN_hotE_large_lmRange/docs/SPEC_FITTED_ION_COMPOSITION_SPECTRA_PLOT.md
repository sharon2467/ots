# SPEC：拟合结果的四组分 OTS 光谱图

## 0. 状态与范围

已批准并实施（2026-09-08）。

实现代码只改 `spectral_fitting.py` 中两处：

1. 新增一个绘图函数 `plot_ion_composition_spectra(...)`；
2. 适配现有 `main()`，为每个拟合结果调用该函数。

不修改 `fit_spectrum`、`_fit_model`、`plot_fit_result`、正向物理模块、测试模块或 canonical CLI。实施完成后仅按仓库协议同步本 SPEC 状态与 `AGENTS.md` 事实记录。

## 1. Requirements（需求）

### 1.1 功能目标

对 `main()` 中每个双离子拟合结果生成一张 2×2 四联图：

1. 纯 ion-1：`x_i1=1`；
2. 纯 ion-2：`x_i2=1`；
3. 数目分数加权的平均离子描述；
4. 原始双离子组分。

例如 H-He、`x_H=0.5`：

```text
H=1 | He=1
Zbar=1.5 average ion | H-He, x_H=0.5
```

### 1.2 物理约定

- 四幅图沿用同一拟合结果的 `Te/uik/uek`。
- 纯 ion-1 使用 `Ti1`，纯 ion-2 使用 `Ti2`。
- 平均离子按离子数目分数 `x=x_i1` 计算：

  ```text
  Zbar  = x*Z1  + (1-x)*Z2
  mbar  = x*m1  + (1-x)*m2
  Tibar = x*Ti1 + (1-x)*Ti2
  ```

- 平均离子调用现有双离子 API 时，把两个离子槽都设为相同的 `Zbar/mbar/Tibar`，各占 0.5；这在电中性和谱形上等价于一个平均离子物种。
- 四条谱直接使用 `OTS_spectrum_maxwellian_two_ion(...)` 返回的理论归一化信号；Gaussian 仪器展宽设为 0，不调用 `_gaussian_convolve`，也不加入拟合幅度 `A` 或背景 `B1/B2`。
- 各面板展示归一化相对 OTS 功率，不把四条独立归一化谱解释为绝对贡献的可加分解。
- 该“无实验因素”约定只用于新增的四联组分图；现有拟合和 `plot_fit_result(...)` 的仪器模型不在本次修改范围内。

### 1.3 用户故事

用户查看任一拟合状态时，可以直接比较两种纯离子假想等离子体、平均离子近似和真实双离子描述的 IAW 谱形差异。

### 1.4 范围外

不计算新的物理公式，不拆解介电函数内部的逐项功率，不对四条谱重新拟合，不增加 CLI、数据保存、梯度、误差分析或新的数据结构。不修改 `_fit_model` 或 `_gaussian_convolve` 来支持零 FWHM。

## 2. Design（设计）

### 2.1 架构决策

只调用现有公开正向 API `OTS_spectrum_maxwellian_two_ion(...)`：

```text
fitted_result
  |
  +-- 准备现有 API 所需的几何、漂移和电中性离子密度
  |
  +-- OTS API(N1=ne/Z1, N2=0) ------------> pure ion-1
  +-- OTS API(N1=0, N2=ne/Z2) ------------> pure ion-2
  +-- OTS API(Z1=Z2=Zbar,
  |           m1=m2=mbar,
  |           Ti1=Ti2=Tibar,
  |           N1=N2=ne/(2*Zbar)) ---------> average ion
  +-- OTS API(N1=x*ne/Zbar,
              N2=(1-x)*ne/Zbar) ----------> two-ion mixture
  |
  +-- Matplotlib 2x2 figure -> optional PNG
```

新函数只负责把已有拟合参数转换成公开 API 的几何、漂移和满足电中性的离子密度输入；动态结构因子、极化率、谱功率和归一化全部由现有 API 完成，不复制光谱计算公式。由于直接调用理论正向 API，四张谱不包含仪器卷积、幅度或背景。

### 2.2 输出布局

四个面板共享横纵坐标范围：

```text
[ pure ion-1 ] [ pure ion-2 ]
[ average ion] [ two-ion mix ]
```

横轴为相对探针波长的 `Delta lambda [nm]`，纵轴为 `Normalized OTS power`。标题显示物种名、纯组分/混合分数；平均离子标题至少显示 `Zbar`。

### 2.3 文件输出

函数在 `output_path=None` 时只返回 Matplotlib Figure；传入路径时保存 PNG。`main()` 为九种状态分别保存到：

```text
spectral_fitting_composition_plots/
```

## 3. 函数接口与伪代码

### 3.1 新函数 `plot_ion_composition_spectra(...)`

```python
def plot_ion_composition_spectra(
    wavelength_nm,
    fitted_result,
    electron_density_cm3,
    *,
    ion1_label,
    ion2_label,
    title=None,
    output_path=None,
    probe_wavelength_nm=263.3,
    scattering_angle_deg=90.0,
    ion1_charge=1.0,
    ion1_mass_amu=1.0,
    ion1_number_fraction=12.0 / 17.0,
    ion2_charge=6.0,
    ion2_mass_amu=12.0,
):
```

伪代码：

```text
验证 wavelength 是 finite 一维数组
验证 fitted_result 含八个 finite 拟合参数
验证标签非空、电子密度/双离子物性为 finite 正值、fraction 位于 (0,1)
建立探针/散射方向和沿 k 的电子、离子漂移向量

pure_ion1 = OTS_spectrum_maxwellian_two_ion(..., N1=ne/Z1, N2=0)
pure_ion2 = OTS_spectrum_maxwellian_two_ion(..., N1=0, N2=ne/Z2)

计算 Zbar、mbar、Tibar
average = OTS_spectrum_maxwellian_two_ion(
    ...,
    Z1=Z2=Zbar,
    m1=m2=mbar,
    Ti1=Ti2=Tibar,
    N1=N2=ne/(2*Zbar),
)

mixture = OTS_spectrum_maxwellian_two_ion(
    ...,
    N1=x*ne/Zbar,
    N2=(1-x)*ne/Zbar,
)

把四个 API 输出转为 NumPy 并验证 finite/shape

创建共享坐标的 2x2 Figure
依次绘制四条谱并设置标题、坐标、网格
若 title 非空，设置总标题
若 output_path 非空，创建父目录并保存 PNG
return figure
```

### 3.2 修改函数 `main()`

接口保持不变：

```python
def main() -> int:
```

伪代码：

```text
保留现有九种状态、正向/反向闭环和三联拟合图逻辑
给每个状态补充 ion1_label 与 ion2_label
建立 spectral_fitting_composition_plots 输出目录

对每个拟合成功的状态:
    保留现有 plot_fit_result 调用
    调用 plot_ion_composition_spectra(
        fitted_result=fitted,
        labels/charges/masses/fraction=当前状态,
        output_path=当前状态对应 PNG,
    )
    关闭返回的 Figure

打印两类绘图输出目录
return 0
```

## 4. Tasks（任务拆解）

1. 用户审阅并批准本 SPEC。
2. 在 `spectral_fitting.py` 新增 `plot_ion_composition_spectra(...)`。
3. 只在 `main()` 中补充物种标签、调用和输出目录。
4. 用指定 Python 3.11.9 运行语法检查及 `main()`。
5. 确认九种状态各生成一张四联图，并抽查 H-He、H-C、Au-B 图像布局。
6. 运行相关 38 项正向物理测试。
7. 核对 Git diff，并同步必要文档事实。

## 5. Acceptance Criteria / Tests（验收标准）

- 除新增绘图函数和 `main()` 适配外，不修改 `spectral_fitting.py` 其他函数。
- 四种谱全部通过现有 `OTS_spectrum_maxwellian_two_ion(...)` 生成，不复制动态结构因子、极化率、谱功率或归一化公式。
- 新四联图不调用 `_fit_model` 或 `_gaussian_convolve`，Gaussian 仪器 FWHM 等效为 0，且不含 `A/B1/B2`。
- H-He `x_H=0.5` 的平均离子面板明确显示 `Zbar=1.5`。
- 纯组分面板分别使用对应物种的拟合离子温度。
- 平均离子使用数目分数加权的 `Zbar/mbar/Tibar`。
- 双离子面板使用原始两种离子参数和原始组分分数。
- 所有谱一维、finite，长度与输入波长一致。
- 四联图无坐标裁切、文字重叠或空面板。
- `main()` 的九种正向/反向闭环继续通过，并额外生成九张四联 PNG。
- 不修改 `phyEqns/ots_full_fourier.py`、测试源码、canonical CLI 或其他实现文件。
