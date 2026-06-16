# -*- coding: utf-8 -*-
"""
多高斯参数化 OTS 光谱模块 - 早期的非 Maxwellian EVDF 正向模型

创建于: 2024-12-16 Mon 17:46:17
更新于: 2024-12-16 Mon 20:18:06

本模块提供：
1. w_1(x)           - Maxwellian 响应函数的快速近似
2. w(x, ...)        - 多个 Maxwellian/Gaussian 电子组分的加权响应
3. OTS_spectrum_arb - 基于参数化多高斯电子分布的 OTS 光谱

应用场景：
- 当电子速度分布可近似为若干 Gaussian 组分叠加时使用。
- 例如：背景电子 + 一个或多个热电子束。
- 这是 `ots_fourier.py` 之前的参数化版本；论文主方法改用 Fourier
  表示以避免预设 EVDF 的具体函数族。

与其他模块的关系：
- ots_fourier.py：Fourier 展开，主方法，适合任意一维 EVDF。
- ots_superGaussian.py：单一超高斯，用于泛化测试或对照。
- ots_arb_gaussian.py：多峰 Gaussian 混合，便于构造有热电子峰的样例。
"""

# 前向模型实现模块
import numpy as np
import os
# import matplotlib.pyplot as plt

import random
import torch
import math


def w_1(x):
    """
    Maxwellian 响应函数在实轴上的快速近似。

    参数：
        x (Tensor): 归一化相速度 v/(sqrt(2)*v_t)。

    返回：
        Tensor: 复响应函数，用于构造物种极化率。
    """
    A1 = 5.329459916821784
    A2 = 0.2697000298919308
    A3 = 5.488464853264165
    A4 = 0.48922975317802353
    A5 = 0.36044367719523124

    y = 1 - (2 * x**2 / (A1 + 2 * x**2) + 2 * A2 * x**2 / (1 + A3 * x**4) + 2 * A4 * x**2 / (1 + A5 * x**4)) + 1j * math.sqrt(math.pi) * x * torch.exp(-x**2)
    return y

def w(x,v_te,v_es,n_es):
    """
    组合多个 Gaussian 电子组分的响应函数。

    参数：
        x:    以总电子热速度 v_te 归一化的相速度。
        v_te: 参考电子热速度，单位 m/s。
        v_es: 各 Gaussian 组分的速度宽度，单位 m/s。
        n_es: 各组分的相对权重；调用方负责保证物理上合理。

    返回：
        Tensor: 多组分加权后的电子响应函数。
    """
    f_w=0*x
    for j in range(len(v_es)):
        W=w_1(x*v_te/v_es[j])
        f_w=f_w+W*n_es[j]/v_es[j]**2

    f_w=f_w*v_te**2
    return f_w


def OTS_spectrum_arb(T_e, T_i, n_e, Z_i, lms, n_es, v_es):
    """
    计算多 Gaussian 电子分布的一维 OTS 光谱。

    参数：
        T_e, T_i: 电子/离子温度，单位 eV；函数内部转换为 K。
        n_e:      电子数密度，单位 m^-3。
        Z_i:      单一离子物种的电荷数。
        lms:      散射光波长数组，单位 m。
        n_es:     电子 Gaussian 组分的相对权重。
        v_es:     电子 Gaussian 组分的速度宽度，单位 m/s。

    返回：
        Tensor: 按最大值归一化后的光谱强度。

    注意：
        该函数固定了 263 nm 入射光、直角散射几何和单一离子质量，
        因此主要用于早期参数化测试，不是论文主训练数据的生成入口。
    """

    e = torch.tensor(1.602176634e-19)  # Elementary charge (C)
    k_B = torch.tensor(1.380649e-23)   # Boltzmann constant (J/K)
    c = torch.tensor(2.99792458e8)     # Speed of light in vacuum (m/s)
    m_e = torch.tensor(9.10938356e-31) # Electron mass (kg)
    epsilon_0 = torch.tensor(8.854187817e-12)
    
    T_e = T_e * e / k_B
    T_i = T_i * e / k_B
    
    k_s = torch.tensor([1., 0., 0.])
    s = torch.tensor([0., 1., 0.])
    E_i0 = torch.tensor([0., 0., 1.])
    lm_0 = 263e-9
#     s = s / np.sqrt(2)
    omega_0 = 2 * torch.pi * c / lm_0
    # k_0 = 2 * torch.pi / lm_0 * k_s
    omega_p = torch.sqrt((n_e * e ** 2) / (m_e * epsilon_0))
    k_0 = torch.sqrt(omega_0 ** 2 - omega_p ** 2) / c * k_s
    m_p = 1.6726219e-27  # Proton mass (kg)
    
    lm_d = torch.sqrt(epsilon_0 * k_B * T_e / (n_e * e ** 2))
    # lm_d = 50e-9
    
    v_te = torch.sqrt(k_B * T_e / m_e)
    v_ti = torch.sqrt(k_B * T_i / m_p)
    
    f_i = lambda v: \
            1 / math.sqrt(2 * torch.pi) / v_ti * torch.exp(-v ** 2 / (2 * v_ti ** 2))
    
    chi_i = lambda omega, k: \
            (1 / k / lm_d) ** 2 * Z_i * (T_e / T_i) * w_1(omega / (math.sqrt(2) * k * v_ti))
        
    
    # Convert requested wavelengths to angular frequencies.
    #     lms = torch.linspace(220e-9, 310e-9, 1000)
    omega = 2 * torch.pi * c / lms
    
    kx_s = k_0[0] - s[0] / c * omega
    ky_s = k_0[1] - s[1] / c * omega
    kz_s = k_0[2] - s[2] / c * omega
    k_s = torch.sqrt(kx_s ** 2 + ky_s ** 2 + kz_s ** 2)
    omega_s = omega - omega_0
    v_s = omega_s / k_s
    x_s = v_s / v_te / math.sqrt(2)
        
#     f_e = 1 / math.sqrt(2 * torch.pi) / v_te * torch.exp(-v_s ** 2 / (2 * v_te ** 2))
    f_e = lms * 0
    for j in range(len(v_es)):
#         f_0 = torch.exp(-v_s ** 2 / v_es[j] ** 2 / 2 *np.sqrt(2))
        f_0 = torch.exp(-v_s ** 2 / v_es[j] ** 2 / 2)
        f_e = f_e+f_0 * n_es[j] / math.sqrt(torch.pi) / v_es[j] ** 3 * v_te ** 2 * 2;
    w_es = w(x_s, v_te, v_es, n_es) / torch.sum(n_es)

    chi_e = (1 / k_s / lm_d) ** 2 * w_es
    epsilon = 1 + chi_e + chi_i(omega_s, k_s)
    S_e = 2 * torch.pi / k_s * torch.abs(1 - chi_e / epsilon) ** 2 * f_e \
            + 2 * torch.pi / k_s * torch.abs(chi_e / epsilon) ** 2 * f_i(omega_s / k_s)
    
    P_s = S_e * (1 + 2 * omega_s / omega_0)
    I = P_s / torch.max(P_s)

    return I
#     return I, f_e, w_es

'''
T_e = 1.4e3
T_i = 0.5e3
n_e = 1.e26
Z_i = 1

lm_0 = 263e-9
lms = torch.linspace(-1, 1, 5000) * 70e-9 + lm_0
# lms = torch.linspace(220e-9, 310e-9, 1000)
# lms = torch.linspace(260e-9, 270e-9, 1000)


v_es=torch.tensor([49622051.70164801, 20429075.387794092, 15099509.481489016, 12523476.082536925, 10933044.271508662, 9826630.513969654, 8999904.99781918, 8351939.97087401, 7826393.689694841, 7389013.2605174435, 7017617.850924954])
n_es=torch.tensor([0.003490698882540102, -0.04351464008122033, 0.3013492943563613, 2.3434975098599056, -11.676982740238893, 32.90861879890078, -59.29990163158405, 68.58635933810157, -49.29569216878118, 20.061683856351117, -3.5359065442006417])

I_0, f_e, w_es = OTS_spectrum_arb(T_e, T_i, n_e, Z_i, lms, n_es, v_es)


plt.figure(figsize=(5, 5))
# plt.plot(lms / 1e-9, np.imag(w_es))
# plt.plot(lms / 1e-9, f_e)
plt.plot(lms / 1e-9, I_0)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [a.u.]")
# plt.axis([260, 266, 0, 1.1])
plt.axis([200, 330,0,0.1])
plt.savefig("E:/作业/等离子体大创/tomsonscattering/fig1")
'''
