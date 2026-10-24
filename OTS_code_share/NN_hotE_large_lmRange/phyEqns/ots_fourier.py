# -*- coding: utf-8 -*-
"""
OTS光谱计算模块 - 傅里叶变换与谱分析工具集

创建于: 2025-01-15 Wed 18:27:11
更新于: 2025-04-19 Sat 23:16:53

本模块提供：
1. 傅里叶变换工具函数（实空间 <-> k空间）
2. OTS（Thomson散射）光谱正向模型（基于Fourier展开的非Maxwellian电子速度分布）
3. 色散关系与权重函数计算

与论文的关系：
- 论文公式 (2) 给出 OTS 动态结构因子 S(k, omega)，公式 (3)
  给出任意 EVDF 下的电子极化率 chi_e。
- 本模块的 `OTS_spectrum_fft` 对应论文公式 (5)：用 EVDF 的
  Fourier 系数 F_e(K) 计算 chi_e，再代入公式 (2) 生成三段光谱。
- 训练标签使用前 80 个 Fourier 模式的实部和虚部，网络输出后再通过
  `ift_k` 或 `ift_k_x` 还原一维电子速度分布 f_e(v)。

关键变量约定：
- x, v: 速度空间坐标，单位 m/s。
- K: 与速度共轭的 Fourier 波数，单位 s/m。
- F_e(K) = ∫ f_e(v) exp(-i K v) dv。
- f_e(v) = (1/2π) ∫ F_e(K) exp(i K v) dK。
- 复数系数中，实部主要编码偶对称分量，虚部主要编码奇对称分量。
"""

import torch
import math
import matplotlib.pyplot as plt

# 常数定义（物理单位）
e = torch.tensor(1.602176634e-19)              # 基本电荷 (C)
k_B = torch.tensor(1.380649e-23)               # 玻尔兹曼常数 (J/K)
c = torch.tensor(2.99792458e8)                 # 光速 (m/s)
m_e = torch.tensor(9.10938356e-31)             # 电子质量 (kg)
m_p = torch.tensor(1.67262192369e-27)          # 质子质量 (kg)
epsilon_0 = torch.tensor(8.854187817e-12)      # 真空介电常数 (F/m)


def ift_k(K2, F2):
    """
    离散傅里叶逆变换 (Inverse Fourier Transform)：从k空间恢复实空间
    
    参数：
        K2 (Tensor): 波数数组 [K_0, K_1, ..., K_{N-1}]
        F2 (Tensor): 频域复数值 [F(K_0), F(K_1), ..., F(K_{N-1})]，对应各波数的傅里叶系数
    
    返回：
        x (Tensor):   实空间坐标（对应原始物理空间，如速度）
        f2 (Tensor):  实空间函数值（复数）
    
    数学原理：
        f(x) = (1/2π) * ∫ F(k) * exp(i*k*x) dk  (连续情形)。

        这里采用梯形积分近似。给定等间隔 K 网格后，代码用
        dx = 2π/(N*dk) 构造一组与 K 网格互补的速度坐标；这与
        `ft_x_n` 中保留有限低频 Fourier 模式的做法一致。
    
    物理应用：
        - 电子速度分布 f(v) 从其傅里叶展开系数 F(k) 恢复
        - 用于 OTS 光谱反演问题中的频域→速度域转换
    """
    # 计算波数步长（k 空间的离散间隔）
    dk = K2[2] - K2[1]
    
    # 获取采样点数
    N = len(K2)
    
    # 计算总的k空间范围
    K = N * dk
    
    # 根据傅里叶变换对偶关系计算速度空间步长：Δx = 2π/(N ΔK)
    dx = 2 * math.pi / K
    
    # 生成实空间坐标网格（-N*dx/2 到 +N*dx/2）
    x = torch.linspace(-1, 1, N) * N * dx / 2
    
    # 初始化实空间函数值（复数形式）
    f2 = 0 * 1j * K2  # 与 K2 同形状的零复数张量
    
    # 对每个实空间点 x_i 进行傅里叶逆变换求和
    for i in range(len(x)):
        # 使用 torch.trapz 进行梯形数值积分
        # 被积函数：F(k) * exp(i*k*x)
        # 归一化因子：1/(π*2)
        f2[i] = torch.trapz(F2 * torch.exp(1j * K2 * x[i]), K2) / math.pi / 2
    
    return x, f2


def ift_k_x(K2, F2, x):
    """
    离散傅里叶逆变换 - 在指定的实空间点上计算
    
    参数：
        K2 (Tensor):  波数数组 [K_0, K_1, ..., K_{N-1}]
        F2 (Tensor):  频域复数值（傅里叶系数）
        x (Tensor):   指定的实空间坐标点（与 K2 可长度不同）
    
    返回：
        x (Tensor):   返回输入的 x（便于与 ift_k 保持一致的接口）
        f2 (Tensor):  在 x 处的函数值（复数，长度与 x 相同）
    
    区别于 ift_k：
        - ift_k 自动生成均匀网格的 x 坐标
        - ift_k_x 允许指定任意的 x 坐标点
    
    应用场景：
        已有非均匀采样的实空间网格时使用
    """
    # 初始化输出张量（与 x 同长度的零复数张量）
    f2 = 0 * 1j * x
    
    # 对每个指定的实空间点进行傅里叶逆变换求和
    for i in range(len(x)):
        f2[i] = torch.trapz(F2 * torch.exp(1j * K2 * x[i]), K2) / math.pi / 2
    
    return x, f2


def ft_x(x, f):
    """
    离散傅里叶正变换：从实空间到k空间
    
    参数：
        x (Tensor):  实空间坐标 [x_0, x_1, ..., x_{N-1}]
        f (Tensor):  实空间函数值 [f(x_0), f(x_1), ..., f(x_{N-1})]
    
    返回：
        K2 (Tensor):  波数数组（范围：-Ndk/2 到 +Ndk/2）
        F2 (Tensor):  频域复系数
    
    数学原理：
        F(k) = ∫ f(x) * exp(-i*k*x) dx
        
        通过梯形法则数值积分实现离散版本
    
    注意事项：
        - 返回的波数 K2 与输入 x 的长度相同
        - 波数范围由实空间范围自动计算
    """
    # 计算实空间步长
    dx = x[2] - x[1]
    
    # 获取采样点数
    N = len(x)
    
    # 计算实空间范围
    L = N * dx
    
    # 根据采样定理计算k空间步长
    # Δk = 2π/L，其中 L = N*Δx
    dk = 2 * math.pi / L
    
    # 生成k空间网格（-Ndk/2 到 +Ndk/2）
    K2 = torch.linspace(-1, 1, N) * N * dk / 2
    
    # 初始化频域张量（与 x 同形状的零复数张量）
    F2 = 0 * x * 1j
    
    # 对每个波数 K_j 进行傅里叶正变换求和
    for i in range(len(K2)):
        # 被积函数：f(x) * exp(-i*K*x)
        F2[i] = torch.trapz(f * torch.exp(-1j * K2[i] * x), x)
    
    return K2, F2


def ft_x_n(x, f, n):
    """
    限制带宽的傅里叶正变换：只保留前 n 个频率成分
    
    参数：
        x (Tensor):  实空间坐标数组
        f (Tensor):  实空间函数值
        n (int):     要计算的频率成分数（限制在 [-ndk/2, +ndk/2]）
    
    返回：
        K2 (Tensor):  被限制的波数数组（长度为 n，而非 len(x)）
        F2 (Tensor):  对应的傅里叶系数（长度为 n）
    
    物理意义：
        - 论文通过 Fourier 低通约束把任意 EVDF 的高频不连续细节截断，
          使“EVDF -> OTS 光谱”的前向映射更平滑。
        - 这里的 n=80 对应网络 160 维输出中的 80 个复数 Fourier 系数。
        - 这不是普通 FFT 的“前 n 项切片”，而是在以零波数为中心的有限
          K 区间上重新积分求系数。
    
    应用场景：
        - genDataSet.py 中使用 n=80 来降低频域维度
        - 原始采样点 1000 个 → 只保留 80 个低频模式
    """
    # 计算实空间步长
    dx = x[2] - x[1]
    
    # 获取采样点数
    N = len(x)
    
    # 计算实空间范围
    L = N * dx
    
    # 计算k空间步长
    dk = 2 * math.pi / L
    
    # 生成 n 个波数点（限制在 -ndk/2 到 +ndk/2）
    # 这实际上是在 k 空间中进行"采样"，只取 n 个均匀分布的点
    K2 = torch.linspace(-1, 1, n) * dk * n / 2
    
    # 初始化频域张量（长度为 n 的零复数张量）
    F2 = 0 * K2 * 1j
    
    # 对每个被选中的波数点进行傅里叶正变换
    for i in range(len(K2)):
        F2[i] = torch.trapz(f * torch.exp(-1j * K2[i] * x), x)
    
    return K2, F2



def w_Zheng_sg(xi, m):
    """
    超高斯分布的电子响应函数 - Zheng 公式
    
    参数：
        xi (Tensor):  规范化的频移参数（ξ = v_s / (√2 * v_te)），其中 v_s 是频移相关的速度
        m (float):    超高斯指数；m=2 为 Maxwell/Gaussian，
                      m>2 时中心更平顶、尾部比 Gaussian 更陡。
    
    返回：
        f_w (Tensor):  复数形式的电子响应函数。
    
    物理意义：
        - 色散关系中的电子响应函数，用于构造 chi_e。
        - 超高斯常用于描述逆韧致吸收加热后的非 Maxwellian EVDF。
        - 若 m<2 才会出现比 Gaussian 更长的尾部；m>2 是更陡尾部。
    
    数学形式：
        基于Faddeeva函数与离子质量效应的组合
        涉及 Gamma 函数（阶乘在复数域的推广）
    """
    # 参数定义
    z1 = 100                                            # 积分区间参数
    N1 = 1000                                           # 积分采样点数
    
    # 超高斯归一化因子
    a = math.sqrt(3 * math.gamma(3/m) / math.gamma(5/m) / 2)
    
    # 对频移参数进行归一化
    xi = xi / a
    
    # 生成积分网格
    z = torch.linspace(-1, 1, N1) * z1 + xi
    
    # 计算 Zheng 公式
    # 实部：与被动响应相关，使用第一类积分
    # 虚部：与阻尼相关，使用指数项和 Kramers-Kronig 关系
    f_w = (math.gamma(1/m) * math.gamma(5/m)) / (3 * math.gamma(3/m)**2) * (
        1 + (m) / (2 * math.gamma(1/m)) * xi * torch.trapz(
            (torch.exp(-torch.abs(z)**(m))) / (z - xi), z
        ) + 1j * (m * math.pi) / (2 * math.gamma(1/m)) * xi * 
        torch.exp(-torch.abs(xi)**(m))
    )
    
    return f_w


def w_ft(x, K, F_e):
    """
    通过 Fourier 展开计算电子极化率中的权重积分
    
    参数：
        x (Tensor):   相位空间坐标（规范化的频移）
        K (Tensor):   波数（傅里叶模式索引）
        F_e (Tensor):  电子速度分布的傅里叶系数（复数）
    
    返回：
        f_e3 (Tensor): 加权后的电子响应函数（复数）
    
    物理意义：
        - 对应论文公式 (5) 中 chi_e 的 Fourier 表达式。
        - 积分范围是 K>0，因此实现里用 `relu(K)` 只保留正半轴贡献。
        - 输出还会在 `OTS_spectrum_fft` 中乘以尺度因子，得到 chi_e 所需
          的无量纲响应函数。
    """
    # 获取波数采样数
    N = int(len(K) / 2)
    
    # 计算波数步长
    dK = K[2] - K[1]
    
    # 初始化输出（与 x 同形状的零张量）
    f_e3 = 0 * x
    
    # Fourier 求和：论文公式 (5) 的正 K 积分核 K*F_e(K)*exp(i*K*v)。
    for i in range(len(K)):
        # `relu(K[i])` 实现 0..∞ 的积分下限；负 K 不参与该响应积分。
        f_e3 = f_e3 + torch.relu(K[i]) * F_e[i] * torch.exp(1j * K[i] * x) * dK
    
    # 归一化因子来自傅里叶变换理论和权重函数的定义
    ratio = 2 * math.sqrt(2) * math.pi
    f_e3 = f_e3 / ratio
    
    return f_e3


def w_ft_real(x, K, F_er, F_ei):
    """
    通过实部/虚部分离的 Fourier 展开计算电子响应权重
    
    参数：
        x (Tensor):    相位空间坐标
        K (Tensor):    波数数组
        F_er (Tensor): 傅里叶系数的实部（余弦分量）
        F_ei (Tensor): 傅里叶系数的虚部（正弦分量）
    
    返回：
        f_e3 (Tensor): 加权的电子响应函数
    
    与 w_ft 的区别：
        - w_ft 直接使用复数系数。
        - w_ft_real 接收网络输出形式的实部、虚部，适合无复数张量的调用方。
        - 与 `w_ft` 不同，本函数没有截断到正 K；调用时通常只传正半轴 K。
    """
    # 获取波数采样数（仅为兼容性，实际未使用）
    N = int(len(K) / 2)
    
    # 计算波数步长
    dK = K[2] - K[1]
    
    # 初始化输出
    f_e3 = 0 * x
    
    # 分别处理实部和虚部
    for i in range(len(K)):
        # 构造复数傅里叶系数：F = F_r + i*F_i
        F_e = F_er[i] + 1j * F_ei[i]
        
        # 傅里叶求和（使用欧拉公式分离出 exp(i*K*x)）
        f_e3 = f_e3 + K[i] * F_e * torch.exp(1j * K[i] * x) * dK
    
    # 同样的归一化因子
    ratio = 2 * math.sqrt(2) * math.pi
    f_e3 = f_e3 / ratio
    
    return f_e3


def w_fast(x):
    """
    Maxwellian 响应函数的快速近似
    
    参数：
        x (Tensor):  归一化相速度 ξ = v/(√2 * v_t)。
    
    返回：
        y (Tensor):  用于 Maxwellian 物种极化率的复响应函数。
    
    物理意义：
        - OTS 正向模型需要 Maxwellian 离子响应。
        - 该实现使用有理函数拟合近似实部，并保留
          i*sqrt(pi)*x*exp(-x^2) 的 Landau 阻尼项。
        - 它不是完整复平面 Faddeeva 函数求值器，只覆盖本代码使用的实 x。
    
    数学形式：
        w(x) = 1 - [A1项 + A2项 + A4项] + i*√π*x*exp(-x²)
        
        其中各项由分式表示，系数 A1-A5 是通过最小二乘拟合获得的
    """
    # 拟合系数（来自文献或数值优化）
    A1 = 5.329459916821784
    A2 = 0.2697000298919308
    A3 = 5.488464853264165
    A4 = 0.48922975317802353
    A5 = 0.36044367719523124
    
    # 计算分式项的总和（实部近似）
    # 这是 Padé 近似或有理函数拟合的形式
    real_part = 2 * x**2 / (A1 + 2 * x**2) + \
                2 * A2 * x**2 / (1 + A3 * x**4) + \
                2 * A4 * x**2 / (1 + A5 * x**4)
    
    # 虚部：通过 Kramers-Kronig 关系得到的标准形式
    # √π * x * exp(-x²) 是 w 函数虚部的主要贡献
    imag_part = math.sqrt(math.pi) * x * torch.exp(-x**2)
    
    # 组合实部和虚部
    y = 1 - real_part + 1j * imag_part
    
    return y
# Main Fourier-based OTS forward model used for dataset generation.
def OTS_spectrum_fft(lms, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1, m_i1, Z_i2, N_i2, m_i2, T_i, V, K, F_e, u_e, u_i1, u_i2):
    """
    OTS 光谱正向模型 - 基于傅里叶展开的非Maxwell电子分布
    
    核心功能：
        给定等离子体参数与非Maxwell电子速度分布的傅里叶系数，
        计算对应的光学汤姆逊散射光谱强度 I(λ)
    
    输入参数分类：
    ─────────────────────
    1. 几何与检测参数：
        lms (Tensor):      测量的波长数组 [λ_1, λ_2, ..., λ_M] (m单位)
        ss (Tensor):       散射方向（单位向量），即观察器相对于入射光的方向
        k_sh (Tensor):     入射波矢方向（单位向量）
        E_i0 (Tensor):     入射光极化方向（单位向量）
        lm_0 (float):      入射光波长 (m)
    
    2. 等离子体参数：
        n_e (float):       电子密度 (m⁻³)
        T_e (Tensor):      电子温度 (K)。上游若以 eV 给出，需用 T_eV*e/k_B 转换。
        T_i (Tensor):      离子温度 (K)。代码内部通过 k_B*T 得到热能。
    
    3. 离子种类参数（支持两种离子）：
        Z_i1, Z_i2 (float):  离子核电荷数
        N_i1, N_i2 (float):  离子相对浓度（相对数密度）
        m_i1, m_i2 (Tensor): 离子质量 (kg)
        V (float):         等离子体体积 (m³)
    
    4. 电子速度分布（傅里叶形式）：
        K (Tensor):        傅里叶波数 [K_1, ..., K_n]
        F_e (Tensor):      电子速度分布傅里叶系数（复数）
    
    5. 漂移速度（流动效应）：
        u_e (Tensor):      电子漂移速度向量 (m/s)
        u_i1, u_i2 (Tensor): 各种离子漂移速度向量 (m/s)
    
    返回值：
        I (Tensor):        散射谱强度 I(λ)，按 max(abs(P_s)) 归一化。
    
    物理模型：
    ─────────
    1. 入射波矢：
        k_0 = √(ω₀² - ωₚ²)/c，用冷等离子体折射率修正入射光波数。
    
    2. 动力学参数计算：
        - 三个物种（电子、离子1、离子2）的热速度和分布
        - 规范化的频移参数 ξ_s = v_s/(√2 * v_t)
    
    3. 色散函数（电介质函数）：
        ε(ω,k) = 1 + χ_e + χ_i1 + χ_i2

        其中 χ_i 使用 Maxwellian 离子响应；χ_e 使用论文公式 (5) 的
        Fourier 表达式计算。
              λ_d = √(ε₀*k_B*T_e/(n_e*e²)) 是Debye长度
    
    4. 散射函数（光谱形状）：
        S(ω,k) = (2π/k) * |1 - χ_e/ε|² * f_e(v_s)
                + (2π/k) * |χ_e/ε|² * (Z_i² * N_i / Z_N) * f_i(v_si)
        
        多种散射机制：
        a) 电子-光子散射（主要）
        b) 离子贡献的Thomson散射
    
    5. 最终光谱：
        P(ω) ∝ (1 + 2*(ω-ω₀)/ω₀) * S(ω,k)  --- 频移校正项
        
        I(λ) = real(P(λ))/max(abs(P))。
    
    特殊说明：
        - 本函数假设 T_e、T_i 单位为 K；`e/k_B` 是 eV -> K 的转换因子。
        - 使用 Fourier 展开表示非 Maxwellian EVDF，允许捕获复杂速度结构。
        - 通过 K 和 F_e 可以编码任意的平衡外或非平衡速度分布
    """
    # ═══════════════════════════════════════════════════════════════════
    # 物理常数（CODATA 2018）
    # ═══════════════════════════════════════════════════════════════════
    e = torch.tensor(1.602176634e-19)              # 基本电荷 (C)
    k_B = torch.tensor(1.380649e-23)               # 玻尔兹曼常数 (J/K)
    c = torch.tensor(2.99792458e8)                 # 光速 (m/s)
    m_e = torch.tensor(9.10938356e-31)             # 电子质量 (kg)
    m_p = torch.tensor(1.67262192369e-27)          # 质子质量 (kg)
    epsilon_0 = torch.tensor(8.854187817e-12)      # 真空介电常数 (F/m)
    
    # ═══════════════════════════════════════════════════════════════════
    # 第一步：基本参数与缩放因子计算
    # ═══════════════════════════════════════════════════════════════════
    
    # Debye 长度（表征等离子体屏蔽距离）
    lm_d = torch.sqrt((epsilon_0 * k_B * T_e) / (n_e * e**2))
    
    # Thomson 散射截面的经典电子半径的贡献因子
    r_e = (1) / (4 * math.pi * epsilon_0) * (e**2) / (m_e * c**2)
    
    # 总离子核电荷加权（用于归一化离子的相对贡献）
    ZN = Z_i1 * N_i1 + Z_i2 * N_i2
    
    # ═══════════════════════════════════════════════════════════════════
    # 第二步：波矢与频率的计算
    # ═══════════════════════════════════════════════════════════════════
    
    # 归一化单位向量（确保数值稳定性）
    ss = ss / torch.sqrt(torch.sum(ss**2))
    
    # 入射光角频率
    omega_0 = 2 * math.pi * c / lm_0
    
    # 等离子体频率（电子集体振荡的特征频率）
    omega_p = torch.sqrt((n_e * e**2) / (m_e * epsilon_0))
    
    # 入射波矢：冷等离子体中电磁波满足 k^2 c^2 = omega_0^2 - omega_p^2
    k_0 = torch.sqrt(omega_0**2 - omega_p**2) / c * k_sh
    
    # ═══════════════════════════════════════════════════════════════════
    # 第三步：热速度与分布函数参数
    # ═══════════════════════════════════════════════════════════════════
    
    # 电子热速度 v_te = sqrt(k_B*T_e/m_e)，T_e 的单位为 K
    v_te = torch.sqrt(k_B * T_e / m_e)
    
    # 离子热速度
    v_ti1 = torch.sqrt(k_B * T_i / m_i1)
    v_ti2 = torch.sqrt(k_B * T_i / m_i2)
    
    # 离子分布函数（Maxwell-Boltzmann，权重函数 w_i）
    f_i1 = lambda v: 1 / math.sqrt(2 * math.pi) / v_ti1 * torch.exp(-v**2 / (2 * v_ti1**2))
    f_i2 = lambda v: 1 / math.sqrt(2 * math.pi) / v_ti2 * torch.exp(-v**2 / (2 * v_ti2**2))
    
    # 离子权重函数（使用快速 Faddeeva 函数）
    w_i = lambda x: w_fast(x)
    
    # ═══════════════════════════════════════════════════════════════════
    # 第四步：对于每个测量波长，计算散射参数
    # ═══════════════════════════════════════════════════════════════════
    
    # 测量光的角频率数组
    omega = 2 * math.pi * c / lms
    
    # 中间波矢的各分量
    kx_s = k_0[0] - ss[0] / c * omega
    ky_s = k_0[1] - ss[1] / c * omega
    kz_s = k_0[2] - ss[2] / c * omega
    
    # 中间波矢幅度
    k_s = torch.sqrt(kx_s**2 + ky_s**2 + kz_s**2)
    
    # 频移（Doppler shift）
    omega_s = omega - omega_0
    
    # ═══════════════════════════════════════════════════════════════════
    # 第五步：电子贡献
    # ═══════════════════════════════════════════════════════════════════
    
    # 电子漂移调节后的频移相关速度
    v_s = omega_s / k_s - (kx_s * u_e[0] + ky_s * u_e[1] + kz_s * u_e[2]) / k_s
    
    # 规范化的频移参数 ξ_e = v_s / (√2 * v_te)
    x_s = v_s / v_te / math.sqrt(2)
    
    # 电子分布函数（通过傅里叶展开在采样点上计算）
    f_es = 0 * lms
    dK = K[2] - K[1]
    for j in range(len(K)):
        # 傅里叶反变换：从 F_e(K) 恢复 f(v)
        f_e = F_e[j] * torch.exp(1j * K[j] * v_s) * dK / math.pi / 2
        f_es = f_es + f_e
    
    # 电子响应函数：Fourier 形式的 chi_e 积分，等价于论文公式 (5)
    w_es = w_ft(v_s, K, F_e) * 2 * v_te**2 * math.pi * math.sqrt(2)
    
    # ═════════════════════════════════════════════════════════════════
    # 第六步：离子贡献（两种离子）
    # ═════════════════════════════════════════════════════════════════
    
    # 离子1的频移相关速度与权重
    v_si1 = omega_s / k_s - (kx_s * u_i1[0] + ky_s * u_i1[1] + kz_s * u_i1[2]) / k_s
    x_si1 = v_si1 / v_ti1 / math.sqrt(2)
    f_is1 = f_i1(v_si1)
    w_is1 = w_i(x_si1)
    
    # 离子2的频移相关速度与权重
    v_si2 = omega_s / k_s - (kx_s * u_i2[0] + ky_s * u_i2[1] + kz_s * u_i2[2]) / k_s
    x_si2 = v_si2 / v_ti2 / math.sqrt(2)
    f_is2 = f_i2(v_si2)
    w_is2 = w_i(x_si2)
    
    # ═════════════════════════════════════════════════════════════════
    # 第七步：色散函数（电介质函数）计算
    # ═════════════════════════════════════════════════════════════════
    
    # 电子的极化率
    chi_e = (1 / k_s / lm_d)**2 * w_es
    
    # 离子1的极化率（包含温度比与核电荷修正）
    chi_i1 = (1 / k_s / lm_d)**2 * Z_i1**2 * N_i1 / ZN * (T_e / T_i) * w_is1
    
    # 离子2的极化率
    chi_i2 = (1 / k_s / lm_d)**2 * Z_i2**2 * N_i2 / ZN * (T_e / T_i) * w_is2
    
    # 总的相对介电函数
    epsilon = 1 + chi_e + chi_i1 + chi_i2
    
    # ═════════════════════════════════════════════════════════════════
    # 第八步：散射函数（光谱形状）计算
    # ═════════════════════════════════════════════════════════════════
    
    # Thomson 散射函数
    # 项1：电子对光子的直接散射（主导项）
    S_e = 2 * math.pi / k_s * torch.abs(1 - chi_e / epsilon)**2 * f_es + \
          2 * math.pi / k_s * torch.abs(chi_e / epsilon)**2 * Z_i1**2 * N_i1 / ZN * f_is1 + \
          2 * math.pi / k_s * torch.abs(chi_e / epsilon)**2 * Z_i2**2 * N_i2 / ZN * f_is2
    
    # ═════════════════════════════════════════════════════════════════
    # 第九步：频移校正与归一化
    # ═════════════════════════════════════════════════════════════════
    
    # 功率谱（包含 1+2ω_s/ω_0 的频移修正）
    P_s = torch.real((1 + 2 * (omega_s) / omega_0) * S_e)
    
    # 按最大绝对值归一化；通常谱为非负，因此数值范围约为 [0, 1]
    I = P_s / torch.max(torch.abs(P_s))
    
    return I


def OTS_spectrum_fft_real(lms, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1, m_i1, Z_i2, N_i2, m_i2, T_i, V, K, F_er, F_ei, u_e, u_i1, u_i2):
    """
    OTS 光谱正向模型 - 实部/虚部分离版本
    
    与 OTS_spectrum_fft 的区别：
        - 输入分离的傅里叶系数实部 F_er 和虚部 F_ei
        - 在计算中分别处理 cos 和 sin 项，可能在某些实现中更高效或稳定
        - 数值精度或计算顺序可能略有差异
    
    参数和返回值同 OTS_spectrum_fft
    
    物理模型：同上，只是数值实现方式不同
    """
    # 物理常数
    e = torch.tensor(1.602176634e-19)
    k_B = torch.tensor(1.380649e-23)
    c = torch.tensor(2.99792458e8)
    m_e = torch.tensor(9.10938356e-31)
    m_p = torch.tensor(1.67262192369e-27)
    epsilon_0 = torch.tensor(8.854187817e-12)
    
    # 基本参数计算（与 OTS_spectrum_fft 相同；T_e/T_i 单位为 K）
    lm_d = torch.sqrt((epsilon_0 * k_B * T_e) / (n_e * e**2))
    r_e = (1) / (4 * math.pi * epsilon_0) * (e**2) / (m_e * c**2)
    ZN = Z_i1 * N_i1 + Z_i2 * N_i2
    
    ss = ss / torch.sqrt(torch.sum(ss**2))
    omega_0 = 2 * math.pi * c / lm_0
    omega_p = torch.sqrt((n_e * e**2) / (m_e * epsilon_0))
    k_0 = torch.sqrt(omega_0**2 - omega_p**2) / c * k_sh
    
    v_te = torch.sqrt(k_B * T_e / m_e)
    v_ti1 = torch.sqrt(k_B * T_i / m_i1)
    v_ti2 = torch.sqrt(k_B * T_i / m_i2)
    
    f_i1 = lambda v: 1 / math.sqrt(2 * math.pi) / v_ti1 * torch.exp(-v**2 / (2 * v_ti1**2))
    f_i2 = lambda v: 1 / math.sqrt(2 * math.pi) / v_ti2 * torch.exp(-v**2 / (2 * v_ti2**2))
    w_i = lambda x: w_fast(x)
    
    # 波长相关的参数
    omega = 2 * math.pi * c / lms
    kx_s = k_0[0] - ss[0] / c * omega
    ky_s = k_0[1] - ss[1] / c * omega
    kz_s = k_0[2] - ss[2] / c * omega
    k_s = torch.sqrt(kx_s**2 + ky_s**2 + kz_s**2)
    omega_s = omega - omega_0
    
    # 电子贡献（使用实部/虚部分离方式；调用方应传入正半轴 K）
    v_s = omega_s / k_s - (kx_s * u_e[0] + ky_s * u_e[1] + kz_s * u_e[2]) / k_s
    x_s = v_s / v_te / math.sqrt(2)
    
    # 电子分布函数（分离的 Fourier 反变换）
    f_es = 0 * lms
    dK = K[2] - K[1]
    for j in range(len(K)):
        # 使用 cos 和 sin 分别计算实部和虚部贡献
        f_e = F_er[j] * torch.cos(K[j] * v_s) * dK / math.pi
        f_e += F_ei[j] * torch.sin(K[j] * v_s) * dK / math.pi
        f_es = f_es + f_e
    
    # 使用分离版本的 Fourier 响应权重
    w_es = w_ft_real(v_s, K, F_er, F_ei) * 2 * v_te**2 * math.pi * math.sqrt(2)
    
    # 离子贡献（同 OTS_spectrum_fft）
    v_si1 = omega_s / k_s - (kx_s * u_i1[0] + ky_s * u_i1[1] + kz_s * u_i1[2]) / k_s
    v_si2 = omega_s / k_s - (kx_s * u_i2[0] + ky_s * u_i2[1] + kz_s * u_i2[2]) / k_s
    x_si1 = v_si1 / v_ti1 / math.sqrt(2)
    x_si2 = v_si2 / v_ti2 / math.sqrt(2)
    f_is1 = f_i1(v_si1)
    f_is2 = f_i2(v_si2)
    w_is1 = w_i(x_si1)
    w_is2 = w_i(x_si2)
    
    # 色散函数
    chi_e = (1 / k_s / lm_d)**2 * w_es
    chi_i1 = (1 / k_s / lm_d)**2 * Z_i1**2 * N_i1 / ZN * (T_e / T_i) * w_is1
    chi_i2 = (1 / k_s / lm_d)**2 * Z_i2**2 * N_i2 / ZN * (T_e / T_i) * w_is2
    epsilon = 1 + chi_e + chi_i1 + chi_i2
    
    # 散射函数
    S_e = 2 * math.pi / k_s * torch.abs(1 - chi_e / epsilon)**2 * f_es + \
          2 * math.pi / k_s * torch.abs(chi_e / epsilon)**2 * Z_i1**2 * N_i1 / ZN * f_is1 + \
          2 * math.pi / k_s * torch.abs(chi_e / epsilon)**2 * Z_i2**2 * N_i2 / ZN * f_is2
    
    # 功率谱与归一化
    P_s = (1 + 2 * (omega_s) / omega_0) * S_e
    I = P_s / torch.max(torch.abs(P_s))
    
    return I




if __name__=="__main__":
    """
    测试与演示代码：验证 OTS 光谱计算正向模型
    
    流程：
    1. 设定等离子体参数（温度、密度、离子组成）
    2. 生成超高斯速度分布（参数化后进行傅里叶变换）
    3. 调用正向模型计算光谱
    4. 绘图对比两个版本（复数模式 vs 分离实部虚部）
    """
    # ─────────────────────────────────────────────────────────────
    # 物理常数与参数设置
    # ─────────────────────────────────────────────────────────────
    k_sh = torch.tensor([1., 0., 0.])                # 入射波矢方向（沿x轴）
    ss = torch.tensor([0., 1., 0.])                 # 散射方向（沿y轴）
    E_i0 = torch.tensor([0., 0., 1.])               # 入射光极化（沿z轴）
    lm_0 = 263e-9                                    # 入射光波长 (263 nm)
    
    # 等离子体参数：以 eV 设定，再转换为 K 供正向模型使用
    T_e = 1.4e3 * e / k_B                           # 电子温度 1400 eV -> K
    T_i = 0.5e3 * e / k_B                           # 离子温度 500 eV -> K
    n_e = 1e26                                       # 电子密度 (m⁻³)
    m = 2.                                           # 超高斯指数
    
    # 测量参数
    P = m / 2                                        # （未使用，可删除）
    lms = torch.linspace(-1, 1, 5000) * 120e-9 + lm_0  # 测量波长范围
    
    # 离子种类（C1H1 混合物）
    Z_i1 = 1                                         # 氢离子：核电荷数 = 1
    N_i1 = 1                                         # 相对浓度
    Z_i2 = 6                                         # 碳离子：核电荷数 = 6
    N_i2 = 0                                         # 不存在
    ZN = Z_i1 * N_i1 + Z_i2 * N_i2
    
    # 离子质量
    m_i1 = m_p                                       # 质子
    m_i2 = m_p * 12                                  # 碳12
    
    # 参数化速度分布
    R = 1e-3                                         # 等离子体尺度
    V = R**3                                         # 体积
    I_0 = 1
    v_te = torch.sqrt(k_B * T_e / m_e)              # 电子热速度
    
    # ─────────────────────────────────────────────────────────────
    # 生成超高斯速度分布
    # ─────────────────────────────────────────────────────────────
    omega = 2 * math.pi * c / lms
    omega_0 = 2 * math.pi * c / lm_0
    omega_p = torch.sqrt((n_e * e**2) / (m_e * epsilon_0))
    k_0 = torch.sqrt(omega_0**2 - omega_p**2) / c * k_sh
    
    kx_s = k_0[0] - ss[0] / c * omega
    ky_s = k_0[1] - ss[1] / c * omega
    kz_s = k_0[2] - ss[2] / c * omega
    k_s = torch.sqrt(kx_s**2 + ky_s**2 + kz_s**2)
    omega_s = omega - omega_0
    v_s = omega_s / k_s
    x_s = v_s / v_te / math.sqrt(2)
    
    # 参数化速度分布（超高斯）
    u_para = ss - k_sh
    u_perp = torch.tensor([u_para[2], -u_para[1], 0])
    u_e = torch.tensor([0., 0., 0.]) * c
    u_i1 = torch.tensor([0., 0., 0.]) * c
    u_i2 = torch.tensor([0., 0., 0.]) * c
    
    # 超高斯分布参数
    a = lambda m: (2 * math.gamma(5 / m)) / (3 * math.gamma(3 / m))
    g = lambda x, m: torch.exp(-(a(m) * x**2)**(m / 2))
    C = lambda v_te, m: m / (4 * math.pi * math.gamma(3 / m)) * (a(m) / 2 / v_te**2)**(3 / 2)
    f_ev = lambda v: C(v_te, m) * g(v / math.sqrt(2) / v_te, m)
    
    # 在实空间进行傅里叶变换
    vxs = torch.linspace(-1, 1, 1000) * 10 * v_te * math.sqrt(2)  # 速度网格
    vrs = torch.linspace(0, 1, 1000) * 10 * v_te * math.sqrt(2)   # 径向速度
    
    # 二维积分得到一维速度分布
    f_es1 = 0 * 1j * vxs
    for i in range(len(vxs)):
        vx = vxs[i]
        v = torch.sqrt(vx**2 + vrs**2)
        f_es1[i] = 2 * math.pi * torch.trapz(vrs * f_ev(v), vrs)
    
    # 进行傅里叶正变换获得频域表示
    [K, F_e] = ft_x(vxs, f_es1)
    
    # 提取正 K 半轴，用于 real/imag 分离版本的响应积分
    K_r = K[500:]
    F_er = torch.real(F_e[500:])
    F_ei = torch.imag(F_e[500:])
    
    # ─────────────────────────────────────────────────────────────
    # 计算光谱并对比两个版本
    # ─────────────────────────────────────────────────────────────
    
    # 版本1：复数傅里叶系数
    P_fft = OTS_spectrum_fft(lms, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1, m_i1, 
                              Z_i2, N_i2, m_i2, T_i, V, K, F_e, u_e, u_i1, u_i2)
    
    # 版本2：分离的实部/虚部
    P_fft_2 = OTS_spectrum_fft_real(lms, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1, m_i1, 
                                     Z_i2, N_i2, m_i2, T_i, V, K_r, F_er, F_ei, u_e, u_i1, u_i2)
    
    # 绘制对比图
    plt.plot(lms, P_fft, label='Complex FFT')
    plt.plot(lms, P_fft_2, '--', label='Real/Imag FFT')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Spectral Intensity')
    plt.legend()
    plt.show()



