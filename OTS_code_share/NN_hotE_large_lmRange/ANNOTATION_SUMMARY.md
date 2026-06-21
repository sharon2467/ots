# OTS 神经网络项目 - 注释总结与文档

## 项目概述

**项目名称**：OTS（光学汤姆逊散射）光谱频域重建神经网络  
**目标**：使用深度学习从测量的光学汤姆逊散射光谱，反演推断等离子体中电子的速度分布  
**时间范围**：2024年12月 - 2026年3月  

---

## 文件结构与依赖关系

```
NN_hotE_large_lmRange/
├── phyEqns/                    # 物理模型层
│   ├── ots_fourier.py          # ★★★ 傅里叶展开法OTS光谱（最完整）
│   ├── ots_superGaussian.py    # 超高斯分布OTS光谱
│   └── ots_arb_gaussian.py     # 任意高斯混合OTS光谱
│
├── readData.py                 # ★ 数据I/O层
├── train.py                    # ★ 训练循环
├── model.py                    # ★ 神经网络架构
│
├── main.py                     # ★★ 主训练脚本（程序入口）
├── genDataSet.py               # ★★ 数据集生成
├── evaluateModel.py            # 统一模型评估与单样本推理
│
└── [generated files]
    ├── dataset_train.parquet   # 生成的训练数据（5000样本）
    ├── dataset_test.parquet    # 生成的测试数据（100样本）
    ├── OTS_Te_model.pickle     # 训练好的模型权重
    └── result*.png             # 评估结果图像
```

**调用链路**：
```
genDataSet.py (生成数据)
    ↓ 使用
ots_fourier.py, ots_superGaussian.py (物理模型)
    ↓
main.py (训练)
    ├→ readData.py (读数据)
    ├→ model.py (构建网络)
    └→ train.py (优化训练)
    ↓ 生成
OTS_Te_model.pickle
    ↓ 用于
evaluateModel.py (评估推理)
```

---

## 核心模块详解

### 1. 物理模型层 (`phyEqns/`)

#### ots_fourier.py (★★★ 最重要)
**用途**：使用傅里叶展开的非Maxwell电子分布计算OTS光谱  
**关键函数**：

| 函数 | 功能 | 参数说明 |
|------|------|--------|
| `ft_x(x, f)` | 正傅里叶变换 | x: 速度网格, f: 分布函数 |
| `ft_x_n(x, f, n)` | 限制带宽傅里叶变换 | n: 保留的模式数（通常=80） |
| `ift_k(K2, F2)` | 逆傅里叶变换 | K2: 波数, F2: 傅里叶系数 |
| `w_fast(x)` | Maxwellian 响应函数快速近似 | 用于离子响应计算 |
| `OTS_spectrum_fft(...)` | **完整正向模型** | 输入：光谱参数、傅里叶系数; 输出：光谱强度I(λ) |

**物理方程** (OTS正向模型核心)：
```
1. 色散函数：ε(ω,k) = 1 + χ_e + χ_i1 + χ_i2
   其中 χ_α = (1/(k*λ_d))² * w_α(ξ)，λ_d = Debye长度

2. 散射函数：S(ω,k) = (2π/k) * |1 - χ_e/ε|² * f_e(v_s)
                    + (2π/k) * |χ_e/ε|² * (Z_i²*N_i/Z_N) * f_i(v_si)

3. 光谱强度：I(λ) ∝ (1 + 2ω_s/ω_0) * S(ω,k)
```

#### ots_superGaussian.py
**用途**：针对超高斯电子分布的快速计算  
**应用**：当电子分布明确满足超高斯形式时使用（速度快但灵活性低）  
**注意**：m=2 为 Maxwell/Gaussian；m>2 表示中心更平顶、尾部更陡，m<2 才是更长尾分布。

#### ots_arb_gaussian.py
**用途**：任意多峰高斯混合电子分布  
**应用**：处理有多个分离电子束的情况

---

### 2. 数据处理层

#### readData.py
**核心函数**：
- `readData_dataset(filename, i_s=1, i_e=-1)`
  - 从Parquet文件读取数据
  - 输入形状：[样本数, 1224]  →  输出形状：[样本数, 160]
  - 数据切片机制用于mini-batch训练

**数据格式**：
```
Parquet表结构：
  列 0:1224      ← 输入特征（OTS光谱，3个波段拼接）
  列 1224:1384   ← 标签（频域表示：80个傅里叶模式的实部+虚部）
```

#### genDataSet.py
**核心流程**：
```python
for sample in range(num_samples):
    1. v, f_v = random_gaussians(1000)     # 生成非Maxwell分布
    2. K, F_e = ft_x(v, f_v)              # 傅里叶正变换
    3. K2, F_e2 = ft_x_n(v, f_v, 80)      # 限制为80个低频模式
    4. I1, I2, I3 = OTS_spectrum_fft(...)  # 三个波段的光谱
    5. 拼接 → 1224维输入特征
    6. Re(F_e2) + Im(F_e2) → 160维标签
    7. 存入Parquet
```

**生成数据统计**：
- 训练集：5000样本
- 测试集：100样本
- 耗时：10-30分钟

---

### 3. 神经网络层

#### model.py - ModelOTS架构
```
输入层 (1224维)
    ↓
隐层1: Linear(1224 → 5000) + LeakyReLU
    ↓
隐层2: Linear(5000 → 5000) + LeakyReLU
    ↓
输出层: Linear(5000 → 160)，无激活
    ↓
输出 (160维)
```

**设计特点**：
- 参数量 ~32M（大模型，需足量数据）
- LeakyReLU而非ReLU避免梯度消失
- 全连接网络适合将全局光谱特征映射到频域
- 输出无激活函数允许任意实数

#### train.py - 训练循环
```python
Loss_func(xs, ys, model):
    # 当前代码：Fourier 系数域 L1 损失（平均绝对误差）
    # 论文文字：EVDF 域 MSE；若要严格复现论文 loss，需要修改 Loss_func

train(xs, ys, model, loss_rate, N):
    # Adam优化器（自适应学习率）
    # 梯度下降 N 步
    # 每200步打印loss监控训练进度
```

---

### 4. 主程序

#### main.py - 训练管道
```
读数据 → 初始化模型 → 多阶段分层训练 → 保存模型 → 验证
```

**分层训练策略**（粗→细调）：
```python
train(..., loss_rate=1.e-4, N=3000)      # 阶段1：粗调
train(..., loss_rate=1.e-5, N=6000)      # 阶段2：中调
train(..., loss_rate=1.e-6, N=10000)     # 阶段3：细调
train(..., loss_rate=1.e-7, N=100000)    # 阶段4：超细调（很长）
```

**优点**：
- 初期大步长快速降低loss
- 后期小步长精细调整权重
- 避免过早陷入局部极值

---

#### genDataSet.py - 数据生成
**两个数据集生成**：
1. 训练集（5000样本）→ `dataset_train.parquet`
2. 测试集（100样本）→ `dataset_test.parquet`

**重要参数固定**：
- 电子温度：T_e = 1370 eV（代码中转换为 K 传给正向模型）
- 离子温度：T_i = 171.3 eV（代码中转换为 K 传给正向模型）
- 电子密度：n_e = 1.81×10²⁶ m⁻³
- 入射光波长：λ_0 = 351 nm

**变动**：每个样本的电子速度分布不同（随机非Maxwell）

**与论文基准数据集的差异**：论文正文描述的通用数据集使用 10 个 Gaussian 叠加；
当前 `NN_hotE_large_lmRange` 分支使用 3 个背景 Gaussian 加 1 个弱热电子 Gaussian，
并扩展电子峰波段到约 100-600 nm，更接近热电子增强数据集。

---

#### evaluateModel.py - 模型评估
```
读测试数据 → 前向推理 → 傅里叶逆变换恢复速度分布 → 绘图对比 → 保存结果图
```

**工作流**：
```python
for i in range(6):  # 处理6个测试样本
    xs, ys_true = readData_dataset(test_path, i, i+1)
    ys_pred = model(xs)                    # 前向推理
    
    # 分离频域表示的实部与虚部
    Fe_true = ys_true[0:80]
    Fe_pred = ys_pred[0:80]
    
    # 傅里叶逆变换恢复速度空间分布
    v, f_true = ift_k(K2, Fe_true + 1j*ys_true[80:160])
    v, f_pred = ift_k(K2, Fe_pred + 1j*ys_pred[80:160])
    
    # 绘图对比
    plt.plot(v, f_true, label='Ground Truth')
    plt.plot(v, f_pred, label='Model Prediction')
```

输出：`result1.png` （2×3网格显示6个样本）

---

`evaluateModel.py` 通过 `--detail-index` 提供单样本光谱、Fourier 系数和 EVDF 可视化。
通过 `--source parquet` 读取已有测试集，通过 `--source synthetic` 现场生成测试样本。

---

## 关键物理参数与常数

### 物理常数（CODATA 2018）
```python
e = 1.602176634e-19    # 基本电荷 (C)
k_B = 1.380649e-23     # 玻尔兹曼常数 (J/K)
c = 2.99792458e8       # 光速 (m/s)
m_e = 9.10938356e-31   # 电子质量 (kg)
m_p = 1.67262192e-27   # 质子质量 (kg)
ε_0 = 8.854187817e-12  # 真空介电常数 (F/m)
```

### 仿真等离子体参数
```
T_e = 1370 eV      → 电子温度；代码内部转换为 K
T_i = 171.3 eV     → 离子温度；代码内部转换为 K
n_e = 1.81×10²⁶    → 电子密度 (m⁻³)
λ_0 = 351 nm       → 入射光波长（紫外线）
λ_d                → Debye长度，由 ε₀k_BT_e/(n_e e²) 计算

离子组成：C1H1混合
  H⁺：Z=1, 相对浓度 55%
  C⁶⁺：Z=6, 相对浓度 45%
```

### 数据维度统计
```
输入光谱（1224维）：
  - 波段1 (UV): 349-353 nm   → 579点
  - 波段2 (Vis): 375-600 nm  → 304点
  - 波段3 (UV): 100-340 nm   → 341点

输出标签（160维）：
  - 傅里叶模式数：80
  - 实部 (80维) + 虚部 (80维)
```

---

## 数据流与推理过程

### 训练阶段
```
随机初始化权重
    ↓
for epoch in [3000, 6000, 10000, 100000]:
    ↓
  y_pred = model(xs)                # 全批量前向传播
  loss = |y_pred - y_true|_mean     # 当前代码：Fourier 系数域 L1
  loss.backward()                   # 反向传播
  optimizer.step(lr)                # 参数更新
  ↓
保存model.pickle
```

### 推理阶段
```
加载 model.pickle
    ↓
x_spectrum ∈ R^1224
    ↓
y_freq = model(x_spectrum) ∈ R^160   # 预测频域系数
    ↓
y_real = y_freq[0:80]      # 实部
y_imag = y_freq[80:160]    # 虚部
    ↓
F_complex = y_real + i*y_imag
    ↓
v, f_velocity = ift_k(K, F_complex)   # 傅里叶逆变换
    ↓
f_velocity(v) 即为推断的电子速度分布
```

---

## 注释覆盖详情

| 文件 | 模块级注释 | 函数docstring | 行级注释 | 覆盖率 |
|------|----------|--------------|--------|--------|
| ots_fourier.py | ✓ 详细 | ✓ 完整 | ✓ 物理方程和算法步骤 | 95% |
| ots_superGaussian.py | ✓ 完整 | ✓ 完整 | ~ 部分 | 80% |
| ots_arb_gaussian.py | ✓ 完整 | ~ 部分 | ~ 部分 | 70% |
| readData.py | ✓ 详细 | ✓ 完整 | ✓ 数据格式说明 | 95% |
| train.py | ✓ 详细 | ✓ 完整 | ✓ 训练流程 | 100% |
| model.py | ✓ 详细 | ✓ 完整 | ✓ 架构设计 | 100% |
| main.py | ✓ 详细 | ✓ 完整 | ✓ 控制流 | 100% |
| genDataSet.py | ✓ 详细 | ✓ 完整 | ✓ 数据生成流程 | 95% |
| evaluateModel.py | ✓ 详细 | ✓ 完整 | ✓ 双数据源与可视化流程 | 100% |

---

## 关键概念术语表

| 术语 | 中文 | 说明 |
|------|------|------|
| OTS | 光学汤姆逊散射 | 通过激光与等离子体中自由电子散射诊断等离子体 |
| Fourier展开 | 傅里叶展开 | 用傅里叶基函数表示任意函数（本项目用于非Maxwell分布） |
| Debye长度 | 德拜长度 | λ_d = √(ε₀k_BT_e/(n_e·e²))，等离子体屏蔽特征距离 |
| 色散函数 | Dielectric Function | ε(ω,k) = 1 + χ_e + χ_i，描述等离子体的电磁响应 |
| Faddeeva函数 | Faddeeva Function | w(ζ) = exp(-ζ²)erfc(-iζ)，等离子体色散关系的核心函数 |
| 极化率 | Susceptibility | χ_α = (1/kλ_d)²·w_α(ξ)，各物种对电磁波的响应强度 |
| 散射函数 | Scattering Function | S(ω,k)，描述光谱强度的波长/频率依赖 |
| 超高斯分布 | Super-Gaussian | m=2 为 Gaussian；m>2 中心更平顶且尾部更陡，m<2 更长尾 |
| Parquet格式 | Parquet | 列式存储数据格式，高效读写 |
| 傅里叶逆变换 | Inverse Fourier Transform | 从频域恢复实空间（从傅里叶系数恢复速度分布） |

---

## 使用工作流

### 第一次使用（完整流程）
```bash
# 1. 生成数据（10-30分钟）
python genDataSet.py
# 输出：dataset_train.parquet (5000样本), dataset_test.parquet (100样本)

# 2. 训练模型（30分钟-2小时，取决于硬件）
python main.py
# 输出：OTS_Te_model.pickle

# 3. 批量评估并输出第一个样本的详细视图
python evaluateModel.py --source parquet --samples 6 --detail-index 0
# 输出：evaluation.png, evaluation_detail.png
```

### 后续使用（推理阶段）
```bash
# 跳过数据生成和训练，直接评估
python evaluateModel.py --source parquet
python evaluateModel.py --source synthetic --synthetic-profile random
```

---

## 扩展与改进建议

### 短期改进
1. ✓ **模型保存改进**：改用 `torch.save/load` 替代 pickle
2. ✓ **GPU支持**：添加 `.to(device)` 支持CUDA
3. **Mini-batch训练**：当数据量>10k时，改用DataLoader
4. **学习率调度**：动态调整学习率而非固定阶段
5. **早停机制**：基于验证集loss的早停

### 中期扩展
1. **参数推断**：输出不仅仅是速度分布，还有温度、密度等
2. **多波长通道**：同时处理多个诊断信号
3. **不确定性量化**：贝叶斯神经网络估计推理置信度
4. **物理约束**：在损失函数中加入物理可行性约束

### 长期规划
1. **迁移学习**：在其他OTS装置间迁移
2. **实验数据**：从仿真转向真实实验数据
3. **反演优化**：联合神经网络与物理模型的混合反演
4. **实时诊断**：部署到等离子体诊断系统进行实时推理

---

## 故障排查

### 常见问题

**Q：生成数据时很慢**
- A：物理模型计算复杂（Faddeeva函数、色散关系），正常现象。可改用GPU或减少样本数。

**Q：训练loss下降很慢或不下降**
- A：检查：
  - 数据是否正确加载（验证 xs.shape = [5000, 1224], ys.shape = [5000, 160]）
  - 学习率是否太小（试试1e-3）
  - 是否梯度消失（试试batch normalization）

**Q：推理结果的速度分布不合理（负值、发散等）**
- A：可能原因：
  - 模型未收敛，训练不足（增加epoch）
  - 数据标准化不够（可添加归一化层）
  - 傅里叶逆变换的数值误差（增加采样点）

**Q：evaluateModel.py报错找不到模型文件**
- A：确认 `OTS_Te_model.pickle` 是否存在且路径正确

---

## 参考文献与来源

### 物理模型参考
- Maxwellian/Faddeeva 响应函数快速近似：Weideman (1994)
- OTS诊断理论：Sheffield et al., Plasma Diagnostic Techniques
- 超高斯分布：Zheng et al. 论文

### PyTorch/深度学习
- 官方文档：https://pytorch.org/docs
- Adam优化器：Kingma & Ba (2014)

---

**文档更新日期**：2026-06-13  
**项目主要贡献者**：[待填充]  
**许可证**：[待定]

