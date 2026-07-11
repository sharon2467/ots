# -*- coding: utf-8 -*-
"""
数据集生成模块 - 使用 Fourier 正向模型合成 OTS 训练/测试数据

创建于: 2025-06-24 Tue 18:40:16
更新于: 2026-03-17 Tue 23:56:14

核心功能：
1. 随机生成一维电子速度分布 f_e(v)。
2. 用 `ft_x_n(..., n=80)` 得到低通 Fourier 表示 F_e(K)。
3. 用 `OTS_spectrum_fft` 生成三段独立归一化的 OTS 光谱。
4. 将光谱和 Fourier 系数保存为 Parquet，供 `main.py` 训练。

与论文描述的对应关系：
- 论文 Fig. 1 使用三段光谱拼成 1224 维输入，网络输出 80 个复数
  Fourier 模式（160 个实数），本文件保持这一接口。
- 论文正文描述的通用训练集是 10 个 Gaussian 的叠加；当前
  `NN_hotE_large_lmRange` 分支实际使用 3 个背景 Gaussian 加 1 个
  弱热电子 Gaussian，并把电子峰波长范围扩展到约 100-600 nm。
  因此本文件更接近论文热电子增强数据集的实现，而不是最简的
  10-Gaussian 基准数据集。

数据生成流程：
    1. 在 [-0.8c, 0.8c] 的速度网格上生成非 Maxwellian EVDF。
    2. 归一化 EVDF，使 ∫ f_e(v) dv = 1。
    3. 对 EVDF 做低通 Fourier 变换，保留 80 个 K 模式。
    4. 分别计算离子峰、红移电子峰、蓝移电子峰三段 OTS 光谱。
    5. 拼接光谱作为 1224 维输入；拼接 Re(F_e)、Im(F_e) 作为 160 维标签。

数据统计：
    - 训练集：5000 个样本
    - 测试集：100 个样本
    - 每个样本输入维度：1224
    - 每个样本标签维度：160 (80 傅里叶模式 × 2 [Re, Im])

生成时间：约 10-30 分钟（取决于硬件和物理计算复杂度）。
"""
import sys
if __name__=='__main__':
    sys.path.append(sys.path[0]+'/../..')
import torch, math
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import time

import pyarrow as pa
import pyarrow.parquet as pq

from phyEqns.ots_superGaussian import OTS_spectrum_sg
from phyEqns.ots_fourier import OTS_spectrum_fft, ft_x, ft_x_n, ift_k, ift_k_x


e = torch.tensor(1.602176634e-19)  # Elementary charge (C)
k_B = torch.tensor(1.380649e-23)   # Boltzmann constant (J/K)
c = torch.tensor(2.99792458e8)     # Speed of light in vacuum (m/s)
m_e = torch.tensor(9.10938356e-31) # Electron mass (kg)
m_p = torch.tensor(1.67262192369e-27) # Proton mass (kg)
epsilon_0 = torch.tensor(8.854187817e-12)


def random_gaussians(num_points):
    """
    生成当前热电子数据分支使用的随机 EVDF。

    参数：
        num_points: 速度网格点数。

    返回：
        v:   速度网格，范围 [-0.8c, 0.8c]，单位 m/s。
        f_v: 归一化后的一维电子速度分布，满足 ∫f_v dv = 1。

    实际构造：
        - 3 个背景 Gaussian：中心在 ±0.06c 内，宽度 0.01-0.05c，
          高度 0.2-2.5。
        - 1 个弱热电子 Gaussian：中心在 ±0.4c 内，宽度 0.01-0.05c，
          高度为 0.001-0.045，用于增加远端电子峰样本。

    注意：
        这与论文中“10 个 Gaussian，中心 ±0.2c，宽度 0.02-0.08c，
        振幅 0.5-1.5”的基准描述不同；这里对应本目录名中的
        `hotE_large_lmRange` 变体。
    """
    v = np.linspace(-1, 1, num_points)*0.8

    f_v = np.zeros_like(v)

    N=3
    for _ in range(N):
        center = np.random.uniform(-1,1)*0.06
        sigma = np.random.uniform(0.01, 0.05)
        height = np.random.uniform(0.2, 2.5)
        f_v += height * np.exp(-((v - center) ** 2) / (2 * sigma ** 2))

    N=1
    for _ in range(N):
        center = np.random.uniform(-1,1)*0.4
        sigma = np.random.uniform(0.01, 0.05)
        height = np.random.uniform(0.1, 4.5)*0.01
        f_v += height * np.exp(-((v - center) ** 2) / (2 * sigma ** 2))

    v=torch.tensor(v)*c
    f_v /= np.trapezoid(f_v, v)
    f_v=torch.tensor(f_v)

    return v, f_v



# if __name__=="__main__":

# OTS geometry and plasma parameters.

an_s=123/180*math.pi;
ss=torch.tensor([math.cos(an_s),math.sin(an_s),0])
# ss=torch.tensor([0,1,0])

k_sh=torch.tensor([1,0,0])
E_i0=torch.tensor([0,0,1])
lm_0=351e-9
T_e=1.37e3*e/k_B     # 1370 eV converted to K for the forward model.
T_i=0.223e3 *e/k_B   # 223 eV converted to K; generate_data overwrites it below.
n_e=1.81e26
m=2.

# CH plasma mixture used in this dataset branch.
Z_i1=1
N_i1=0.55
Z_i2=6
N_i2=0.45

m_i1=m_p
m_i2=m_p*12
R=1e-3
V=R**3
I_0=1
v_te=torch.sqrt(k_B*T_e/m_e)

u_e=torch.tensor([0,0,0])*c
u_i1=torch.tensor([0,0,0])*c
u_i2=torch.tensor([0,0,0])*c

# OTS wavelength grids. Each segment is normalized inside OTS_spectrum_fft.
# x1: ion feature near the 351 nm probe.
# x2/x3: red- and blue-shifted electron features; this hot-electron branch
# expands the paper's schematic 375-500 nm and 200-340 nm ranges to about
# 375-600 nm and 100-340 nm so faster electrons remain in range.
x1=torch.linspace(349,353,579)*1.e-9
x2=torch.linspace(375.3552,599.7064,304)*1.e-9
x3=torch.linspace(100.1144,339.6504,341)*1.e-9

def generate_data(num_samples):
    """
    Generate a Parquet-ready table containing spectra and Fourier labels.

    Row 0 is metadata, not a physical sample:
        - columns lm_0..lm_1223 store the wavelength grid in nm.
        - columns K_0..K_159 store the K grid repeated for Re/Im labels.

    Rows 1..num_samples are physical samples:
        - first 1224 columns: concatenated normalized spectra [x1, x2, x3].
        - last 160 columns: [real(F_e[0:80]), imag(F_e[0:80])].
    """
    data = []
    labels = []
    label_Fe = []
    label_idx = torch.cat((x1,x2,x3),0).numpy()/1e-9
    label_K = []

    # Temperatures are specified in eV here and converted to K before calling
    # the OTS forward model. This script keeps them fixed across samples.
    T_es=np.linspace(1370,1370,num_samples)
    T_is=np.linspace(171.3,171.3,num_samples)
#     Dvs=torch.linspace(-1,1,num_samples)*0.2*c

    for i in range(num_samples):
        T_e0 = T_es[i]
        T_i0 = T_is[i]
#         Dv = Dvs[i]
        T_e = T_e0 *e/k_B
        T_i = T_i0 *e/k_B

        # Generate one non-Maxwellian EVDF and its low-pass Fourier label.
        [vxs,f_es1]=random_gaussians(1000)
        
        # Full transform is useful for debugging; the network label keeps n=80.
        [K,F_e]=ft_x(vxs,f_es1)
        [K2,F_e2]=ft_x_n(vxs,f_es1,80)
        label_K = torch.cat((K2,K2),0).numpy()

        y1=OTS_spectrum_fft(x1,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,K2,F_e2,u_e,u_i1,u_i2)
        y2=OTS_spectrum_fft(x2,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,K2,F_e2,u_e,u_i1,u_i2)
        y3=OTS_spectrum_fft(x3,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,K2,F_e2,u_e,u_i1,u_i2)

        # The three calls normalize their own spectral segment, matching the
        # paper's per-band normalization before concatenation.
        I_0=torch.cat((y1,y2,y3),0)
        data.append(I_0.numpy())
        labels.append([T_i0])
        F_e=torch.cat((torch.real(F_e2),torch.imag(F_e2)),0).numpy()
        label_Fe.append(F_e)

#     df_data = pd.DataFrame(data, columns=label_idx)
#     df_labels2 = pd.DataFrame(label_Fe, columns=label_K)

    data.insert(0,label_idx)
    label_Fe.insert(0,label_K)
    labels.insert(0,[0])  # Retained for historical T_i labels; not written below.

    df_data = pd.DataFrame(data, columns=["lm_%d"%i for i in range(len(I_0))])
    df_labels2 = pd.DataFrame(label_Fe, columns=["K_%d"%i for i in range(len(F_e))])

    table_con=[]
    table_con.append(df_data)
    table_con.append(df_labels2)
#     table_con.append(df_labels)

    df = pd.concat(table_con, axis=1)

    return df

print("################# Start! ###########################")
t_start=time.time()
df=generate_data(5000)
t_end=time.time()
print("################# Finished! ########################")
print("Total time: %.2f"%(t_end-t_start))

table = pa.Table.from_pandas(df);
pq.write_table(table, './dataset_train.parquet')


df=generate_data(100)
table = pa.Table.from_pandas(df);
pq.write_table(table, './dataset_test.parquet')

