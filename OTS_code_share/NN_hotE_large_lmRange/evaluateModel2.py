# -*- coding: utf-8 -*-
"""
参数扫描评估脚本 - 扫描单峰 EVDF 的宽度并比较网络重建

Created on 2025-10-29 Wed 16:32:12
Updated on 2026-03-17 Tue 23:56:15

本脚本不直接读取测试集中已有 EVDF，而是：
1. 从 `dataset_test.parquet` 的第 0 行读取波长网格与 K 网格元数据。
2. 构造一组不同宽度 Dv1 的单 Gaussian EVDF。
3. 用 `OTS_spectrum_fft` 生成对应光谱。
4. 调用已训练模型预测 Fourier 系数，再用 `ift_k` 还原 EVDF。
5. 保存 `result2.png`，用于检查模型对训练分布外宽度变化的泛化。

温度变量约定：
    T_e/T_i 传给正向模型时单位为 K；脚本中用 eV 数值乘以 e/k_B 完成转换。
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
T_e=1.4e3*e/k_B      # 1400 eV converted to K.
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

# Metadata row: wavelengths and Fourier K grid, not a physical sample.
[lms,Kxs]=readData_dataset(data_path,0,1)
Ks=Kxs[0][0:80]
[vxs,_]=ift_k(Ks,Ks*0)

def random_gaussians(num_gaussians, v_range, v):
    """
    Build a Gaussian-mixture EVDF on an existing velocity grid.

    Kept for manual experiments; the main loop below uses a controlled single
    Gaussian so width Dv1 is the only scanned parameter.
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

fv1s=[]
fv2s=[]

Dv1=0.08*c
vc1=0.*c
Dv2=0.06*c
vc2=-0.21*c

vs=torch.linspace(0.01,0.2,20)*c

for i in range(len(vs)):
    Dv1=vs[i]
    # Single Gaussian EVDF with fixed center and scanned width.
    f_v=lambda v: np.exp(-(v-vc1)**2/Dv1**2)
    f_es1=f_v(vxs)
    
    f_es1 /= np.trapz(f_es1, vxs)
    f_es1=torch.tensor(f_es1)
    [_,F_e]=ft_x(vxs,f_es1)
    
    xs1=OTS_spectrum_fft(lms1*1e-9,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,Ks,F_e,u_e,u_i1,u_i2)
    xs2=OTS_spectrum_fft(lms2*1e-9,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,Ks,F_e,u_e,u_i1,u_i2)
    xs3=OTS_spectrum_fft(lms3*1e-9,ss,k_sh,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,Ks,F_e,u_e,u_i1,u_i2)
    xs=torch.cat((xs1,xs2,xs3),0)
    
    # Ground-truth Fourier label and model prediction.
    ys=torch.cat((torch.real(F_e),torch.imag(F_e)),0).numpy()
    ys2=model(xs).detach().numpy()
    
    N=int(len(ys2)/2)
    
    Fer=ys[0:N]
    Fej=ys[N:2*N]
    Fer2=ys2[0:N]
    Fej2=ys2[N:2*N]
    
    Fe=torch.Tensor(ys[0:80])+1j*torch.Tensor(ys[80:160])
    [vxs,fv1]=ift_k(torch.Tensor(Ks),Fe)
    Fe=torch.Tensor(ys2[0:80])+1j*torch.Tensor(ys2[80:160])
    [vxs,fv2]=ift_k(torch.Tensor(Ks),Fe)
    
    vxs1=vxs*1.2
    fv1 /= np.trapz(fv1, vxs1)

    fv1s.append(fv1)
    fv2s.append(fv2)

    plt.subplot(4,5,i+1)
    plt.plot(vxs,fv1)
    plt.plot(vxs,fv2)

plt.savefig("result2.png")



