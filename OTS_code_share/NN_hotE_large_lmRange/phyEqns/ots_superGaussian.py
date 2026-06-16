# -*- coding: utf-8 -*-
"""
超高斯分布 OTS 光谱模块 - 使用解析超高斯 EVDF 计算光谱

创建于: 2025-01-15 Wed 19:07:33
更新于: 2025-02-13 Thu 23:26:39

本模块提供：
1. w_Zheng_sg      - 超高斯电子分布的 Zheng 响应函数
2. w_fast          - Maxwellian 离子响应的快速近似
3. OTS_spectrum_sg - 超高斯电子分布的 OTS 光谱正向模型

物理背景：
- 超高斯分布常用于描述逆韧致吸收加热后的非 Maxwellian 电子分布。
- m=2 时退化为 Maxwell/Gaussian；m>2 时中心更平顶、尾部更陡；
  m<2 才表示比 Gaussian 更长的高能尾部。
- 论文使用超高斯样例检验网络对训练集外 EVDF 形状的泛化能力。

与ots_fourier.py的区别：
- ots_fourier.py：使用 Fourier 展开，适合任意一维 EVDF。
- ots_superGaussian.py：直接使用解析超高斯形式，参数少但只覆盖超高斯族。
"""

import torch, math
import matplotlib.pyplot as plt

pi = math.pi
import torch,math
import matplotlib.pyplot as plt

pi=math.pi

def w_Zheng_sg(xi,m):
    """
    计算超高斯 EVDF 的电子响应函数。

    参数：
        xi (Tensor): 归一化相速度 v/(sqrt(2)*v_te)。
        m (float):  超高斯指数，m=2 为 Maxwellian。

    返回：
        Tensor: 复数响应函数，用于构造电子极化率 chi_e。
    """
    z1=100
    N1=1000
    a=math.sqrt(3*math.gamma(3/m)/math.gamma(5/m)/2)
    xi=xi/a
    z=torch.linspace(-1,1,N1)*z1+xi
    f_w=(math.gamma(1/m)*math.gamma(5/m))/(3*math.gamma(3/m)**(2))*(1+(m)/(2*math.gamma(1/m))*xi*torch.trapz((torch.exp(-torch.abs(z)**(m)))/(z-xi),z)+1j*(m*pi)/(2*math.gamma(1/m))*xi*torch.exp(-torch.abs(xi)**(m)))
    return f_w

def w_fast(x):
    """
    Maxwellian 响应函数的快速实轴近似，用于离子极化率。

    该形式只覆盖本代码传入的实数 xi，不是通用复平面的 Faddeeva 求值器。
    """
    A1 = 5.329459916821784
    A2 = 0.2697000298919308
    A3 = 5.488464853264165
    A4 = 0.48922975317802353
    A5 = 0.36044367719523124

    y = 1 - (2 * x**2 / (A1 + 2 * x**2) + 2 * A2 * x**2 / (1 + A3 * x**4) + 2 * A4 * x**2 / (1 + A5 * x**4)) + 1j * math.sqrt(math.pi) * x * torch.exp(-x**2)
    return y


def OTS_spectrum_sg(lms,s,k_i,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,m,u_e,u_i1,u_i2):
    """
    计算超高斯电子分布对应的归一化 OTS 光谱。

    参数约定：
        lms:  待计算的散射光波长数组，单位 m。
        s:    观察/散射方向，函数内部会归一化为单位向量。
        k_i:  入射激光传播方向，函数内部会归一化为单位向量。
        lm_0: 入射激光波长，单位 m。
        n_e:  电子数密度，单位 m^-3。
        T_e:  电子温度，单位 K；若以 eV 设定，上游应使用 T_eV*e/k_B 转换。
        T_i:  离子温度，单位 K。
        Z_i*, N_i*, m_i*: 两种离子的电荷数、相对浓度和质量。
        m:    超高斯指数。
        u_e/u_i*: 电子和离子漂移速度向量，单位 m/s。

    返回：
        Tensor: 每个输入波长处的谱强度，按本波段最大值归一化。

    说明：
        电子项使用超高斯解析响应 `w_Zheng_sg`；离子项仍假设 Maxwellian，
        与论文中“离子保持 Maxwellian，电子 EVDF 可非 Maxwellian”的假设一致。
    """

    e = torch.tensor(1.602176634e-19)  # Elementary charge (C)
    k_B = torch.tensor(1.380649e-23)   # Boltzmann constant (J/K)
    c = torch.tensor(2.99792458e8)     # Speed of light in vacuum (m/s)
    m_e = torch.tensor(9.10938356e-31) # Electron mass (kg)
    m_p = torch.tensor(1.67262192369e-27) # Proton mass (kg)
    epsilon_0 = torch.tensor(8.854187817e-12)

#     T_e=T_e * e / k_B
#     T_i=T_i * e / k_B

    I_0=1
    lm_d=torch.sqrt((epsilon_0*k_B*T_e)/(n_e*e**2))
    r_e=(1)/(4*pi*epsilon_0)*(e**2)/(m_e*c**2)
    ZN=Z_i1*N_i1+Z_i2*N_i2
    # lms=linspace(-1,1.2,1000)*80e-9+lm_0
    s=s/torch.sqrt(torch.sum(s**2))
    k_i=k_i/torch.sqrt(torch.sum(k_i**2))
    omega_0=2*pi*c/lm_0
    omega_p=torch.sqrt((n_e*e**2)/(m_e*epsilon_0))
    k_0=torch.sqrt(omega_0**2-omega_p**2)/c*k_i
    v_te=torch.sqrt(k_B*T_e/m_e)
    v_ti1=torch.sqrt(k_B*T_i/m_i1)
    v_ti2=torch.sqrt(k_B*T_i/m_i2)
    # Super-Gaussian parameters and one-dimensional projected EVDF.
    a=lambda m: (2*math.gamma(5/m))/(3*math.gamma(3/m))
    g=lambda x,m: torch.exp(-(a(m)*x**2)**(m/2))
    C=lambda v_te,m: m/(4*pi*math.gamma(3/m))*(a(m)/2/v_te**2)**(3/2)
    # C=@(v_te,m) m./(4.*gamma(3./m)).*(a(m)^3/2/v_te^2).^(1./2)
    f_e=lambda v: C(v_te,m)*g(v/math.sqrt(2)/v_te,m)
    f_i1=lambda v: 1/math.sqrt(2*pi)/v_ti1*torch.exp(-v**2/(2*v_ti1**2))
    f_i2=lambda v: 1/math.sqrt(2*pi)/v_ti2*torch.exp(-v**2/(2*v_ti2**2))
    w_i=lambda x: w_fast(x)
    # w_e=@(x) w_Cheng_sg(x,m)
    w_e=lambda x: w_Zheng_sg(x,m)
    omega=2*pi*c/lms
    kx_s=k_0[0]-s[0]/c*omega
    ky_s=k_0[1]-s[1]/c*omega
    kz_s=k_0[2]-s[2]/c*omega
    k_s=torch.sqrt(kx_s**2+ky_s**2+kz_s**2)
    omega_s=omega-omega_0
    # v_se=omega_s./k_s-dot(s,u_e)
    v_se=omega_s/k_s-(kx_s*u_e[0]+ky_s*u_e[1]+kz_s*u_e[2])/k_s
    x_se=v_se/v_te/math.sqrt(2)
    # Super-Gaussian electron distribution projected onto the scattering wave vector.
    # f_es=f_e(v_se)
    f_es=torch.zeros(lms.size())+1j
    w_es=torch.zeros(lms.size())+1j
    v_r=torch.linspace(0,1,1000)*100*v_te
    
    for i in range(len(v_se)):
        f_es[i]=2*pi*torch.trapz(v_r*f_e(torch.sqrt(v_r**2+v_se[i]**2)),v_r)
        w_es[i]=w_e(x_se[i])
    
    # Maxwellian ion contributions; drift is projected onto the scattering wave vector.
    # v_si1=omega_s./k_s-dot(s,u_i1)
    # v_si2=omega_s./k_s-dot(s,u_i2)
    v_si1=omega_s/k_s-(kx_s*u_i1[0]+ky_s*u_i1[1]+kz_s*u_i1[2])/k_s
    v_si2=omega_s/k_s-(kx_s*u_i2[0]+ky_s*u_i2[1]+kz_s*u_i2[2])/k_s
    x_si1=v_si1/v_ti1/math.sqrt(2)
    x_si2=v_si2/v_ti2/math.sqrt(2)
    f_is1=f_i1(v_si1)
    f_is2=f_i2(v_si2)
    w_is1=w_i(x_si1)
    w_is2=w_i(x_si2)
    chi_e=(1/k_s/lm_d)**2*w_es
    chi_i1=(1/k_s/lm_d)**2*Z_i1**2*N_i1/ZN*(T_e/T_i)*w_is1
    chi_i2=(1/k_s/lm_d)**2*Z_i2**2*N_i2/ZN*(T_e/T_i)*w_is2
    epsilon=1+chi_e+chi_i1+chi_i2
    S_e=2*pi/k_s*torch.abs(1-chi_e/epsilon)**2*f_es+2*pi/k_s*torch.abs(chi_e/epsilon)**2*Z_i1**2*N_i1/ZN*f_is1+2*pi/k_s*torch.abs(chi_e/epsilon)**2*Z_i2**2*N_i2/ZN*f_is2
    # The absolute Thomson prefactor is omitted because the training data uses
    # independently normalized spectral segments.
    # P_s=(1)/(2*pi)*I_0*r_e**2*V*n_e*(1+2*(omega_s)/omega_0)*torch.norm(torch.cross(s,(torch.cross(s,E_i0))))**2*S_e
#     P_s=(1)/(2*pi)*I_0*r_e**2*V*n_e*(1+2*(omega_s)/omega_0)*S_e
    P_s=torch.real((1+2*(omega_s)/omega_0)*S_e)
#     I=P_s
    I=P_s / torch.max(P_s)

    return I


# Manual demo below: compare the normalized spectrum generated by one
# super-Gaussian electron distribution.

if __name__=="__main__":
    e = torch.tensor(1.602176634e-19)  # Elementary charge (C)
    k_B = torch.tensor(1.380649e-23)   # Boltzmann constant (J/K)
    c = torch.tensor(2.99792458e8)     # Speed of light in vacuum (m/s)
    m_e = torch.tensor(9.10938356e-31) # Electron mass (kg)
    m_p = torch.tensor(1.67262192369e-27) # Proton mass (kg)
    epsilon_0 = torch.tensor(8.854187817e-12)

    m=2.6
    lm_0 = 263e-9
    # lm_0 = 351e-9
    lms=torch.linspace(-1,1.3,1000)*70e-9+lm_0
    # lms=torch.linspace(-1,1.3,1000)*1e-9+lm_0
    
    k_i = torch.tensor([1., 0., 0.])
    s = torch.tensor([0., 1., 0.])
    E_i0 = torch.tensor([0., 0., 1.])
    V=1
    
    T_e=1.4e3*e/k_B
    T_i=0.4e3*e/k_B
    n_e=1.e26
    P=1
    
    Z_i1=1
    N_i1=1
    Z_i2=6
    N_i2=0
    m_i1=m_p*Z_i1*2
    m_i2=m_p*Z_i2*2
    
    u_e=torch.tensor([0,0,0])
    u_i1=torch.tensor([0,0,0])
    u_i2=torch.tensor([0,0,0])
    
    
    
    P_s=OTS_spectrum_sg(lms,s,k_i,E_i0,lm_0,n_e,T_e,Z_i1,N_i1,m_i1,Z_i2,N_i2,m_i2,T_i,V,m,u_e,u_i1,u_i2)
    plt.plot(lms/1e-9,P_s)
    # plt.plot(x_se,w_es)
    plt.show()



