# -*- coding: utf-8 -*-
"""
模型评估与可视化模块 - 对测试集进行批量推理并还原 EVDF

创建于: 2025-06-24 Tue 18:40:16
更新于: 2026-03-17 Tue 23:56:14

核心功能：
1. 加载训练好的神经网络模型
2. 在测试集上进行批量前向推理
3. 从模型输出的 160 维 Fourier 系数恢复速度分布（傅里叶反变换）
4. 绘制真实值与预测值的对比图
5. 计算模型性能指标（可扩展）

评估流程：
    for i in range(num_test_samples):
        1. 读取第i个测试样本（光谱输入）
        2. 前向推理获取模型预测的频域系数
        3. 傅里叶反变换恢复速度空间分布
        4. 与真实速度分布对比并绘图

输出结果：
    - result1.png：6个测试样本的速度分布对比（2×3网格）
    - 定量误差分析（可添加）

与论文的关系：
    论文 Fig. 3 使用蓝线/红线对比真实 EVDF 与网络重建 EVDF；本脚本
    执行同类可视化。Parquet 第 0 行保存 K 网格，所以脚本先读取第 0 行
    取得 `Ks`，再对每个样本的 Re/Im 系数组成复数 F_e(K) 并调用 `ift_k`。

应用场景：
- 模型训练后的定性与定量评估
- 物理可信度检查（恢复的速度分布是否合理）
- 超参数调优的参考
"""

import torch, math
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy

import pyarrow as pa
import pyarrow.parquet as pq

from model import ModelOTS as DynamicNet
from model import load_model
from readData import readData_dataset
from train import Loss_func

from phyEqns.ots_superGaussian import OTS_spectrum_sg
from phyEqns.ots_fourier import OTS_spectrum_fft, ft_x, ft_x_n, ift_k, ift_k_x



### constants ############################################################
e = torch.tensor(1.602176634e-19)  # Elementary charge (C)
k_B = torch.tensor(1.380649e-23)   # Boltzmann constant (J/K)
c = torch.tensor(2.99792458e8)     # Speed of light in vacuum (m/s)
m_e = torch.tensor(9.10938356e-31) # Electron mass (kg)
m_p = torch.tensor(1.67262192369e-27) # Proton mass (kg)
epsilon_0 = torch.tensor(8.854187817e-12)



### OTS parameters ######################################################
an_s=123/180*math.pi;
ss=torch.tensor([math.cos(an_s),math.sin(an_s),0])
# ss=torch.tensor([0,1,0])

k_sh=torch.tensor([1,0,0])
E_i0=torch.tensor([0,0,1])
lm_0=351e-9
T_e=1.4e3*e/k_B      # 1400 eV converted to K; retained for synthetic checks.
T_i=0.223e3 *e/k_B   # 223 eV converted to K.
n_e=1.81e26
m=2.

# C1H1
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


foldername_data="."
foldername_model="."

data_path = '%s/dataset_test.parquet'%(foldername_data)
model_path="%s/OTS_Te_model.pickle"%(foldername_model)

# Read metadata row: first 1224 columns are wavelength grids, label columns
# store the K grid used by the 80 retained Fourier modes.
[lms,Kxs]=readData_dataset(data_path,0,1)
Ks=Kxs[0][0:80]
[vxs,_]=ift_k(Ks,Ks*0)

def random_gaussians(num_gaussians, v_range, v):
    """
    Construct a normalized-looking Gaussian mixture on an existing velocity grid.

    This helper is kept for ad-hoc synthetic checks; the main loop below evaluates
    samples already stored in `dataset_test.parquet`.
    """
    f_v = 0*v
    for _ in range(num_gaussians):
        center = np.random.uniform(*v_range)*c  # 随机中心
        sigma = np.random.uniform(0.02, 0.08)*c  # 控制半高宽（决定电子速度的扩展）
        height = np.random.uniform(0.5, 1.5)  # 高斯核的高度
        f_v += height * np.exp(-((v - center) ** 2) / (2 * sigma ** 2))

    return f_v

lms1=lms[0,0:579]
lms2=lms[0,579:579+304]
lms3=lms[0,579+304:1224]
model=load_model(model_path)

fv1s=[]
fv2s=[]
Sis=[]

f_es1=random_gaussians(10, [-0.2,0.2], vxs)
start=1

fv1s=[]
fv2s=[]
model=load_model(model_path)
for i0 in range(6):
    i=i0+start

    # Read one physical test sample. Row 0 is metadata, so start=1 selects
    # the first actual EVDF/spectrum pair.
    [xs,ys]=readData_dataset(data_path,i,i+1)
    
    # Re-read K metadata to make the EVDF inverse transform explicit.
    [lms,Kxs]=readData_dataset(data_path,0,1)
    Ks=Kxs[0][0:80]
    
    ys=ys.detach().numpy();
    ys2=model(xs).detach().numpy()
        
    # Ground-truth EVDF from stored Fourier labels.
    Fe=torch.Tensor(ys[0][0:80])+1j*torch.Tensor(ys[0][80:160])

    [vxs,fv1]=ift_k(torch.Tensor(Ks),Fe)

    # Predicted EVDF from the network output.
    Fe=torch.Tensor(ys2[0][0:80])+1j*torch.Tensor(ys2[0][80:160])
    [vxs,fv2]=ift_k(torch.Tensor(Ks),Fe)

    fv1s.append(fv1)
    fv2s.append(fv2)



    plt.subplot(2,3,i0+1)
    plt.plot(vxs,fv1)
    plt.plot(vxs,fv2)

# plt.show()
plt.savefig("result1.png")




