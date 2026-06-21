# -*- coding: utf-8 -*-
"""
神经网络模型模块 - 定义 OTS 到 EVDF Fourier 表示的反演网络

创建于: 2025-06-17 Tue 12:20:50
更新于: 2026-03-17 Tue 23:56:15

本模块包含：
1. ModelOTS - 多层全连接神经网络
2. load_model - 从文件加载模型

架构设计：
────────────
任务：从 1224 维输入光谱 → 160 维 EVDF Fourier 系数
  
    输入层（1224维） 
         ↓
    隐层1（5000维，LeakyReLU）
         ↓
    隐层2（5000维，LeakyReLU）
         ↓
    输出层（160维，线性）

设计选择：
- 隐层大小 5000×5000：足够大的容量捕捉光谱与频域的非线性映射
- LeakyReLU：相比 ReLU 有更好的梯度流动（避免 dying ReLU）
- 输出层线性激活：Fourier 系数可为任意实数，不限制在 [0,1] 范围
- 该结构对应论文 Fig. 1 的两层全连接 DNN，不使用 CNN 的平移不变性假设。

参数量估计：
- 层1：1224 × 5000 + 5000 ≈ 6.1M 参数
- 层2：5000 × 5000 + 5000 ≈ 25M 参数
- 层3：5000 × 160 + 160 ≈ 0.8M 参数
- 总计：约 32M 参数

模型保存：
- 使用 pickle 序列化 state_dict
- 建议改用 torch.save/torch.load 获得更好的兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# 隐层尺寸常数（网络配置）
N1 = 5000   # 第一隐层神经元数
N2 = 5000   # 第二隐层神经元数


class ModelOTS(nn.Module):
    """
    光学汤姆逊散射 EVDF 反演神经网络。
    
    功能：
        将测量的三段 OTS 光谱（1224维）映射到电子速度分布函数
        f_e(v) 的 Fourier 表示（160维）。
        
        这 160 维由以下构成：
        - 前 80 维：傅里叶系数的实部
        - 后 80 维：傅里叶系数的虚部
        
        共 80 个 Fourier 模式。评估脚本会把这 160 个实数重新组成
        复数 F_e(K)，再通过 `ift_k` 还原速度空间 EVDF。
    """
    
    def __init__(self):
        """
        初始化网络层
        
        架构：
            Input (1224) → Linear → LeakyReLU 
                       → Linear → LeakyReLU 
                       → Linear → Output (160)
        """
        super().__init__()
        
        # 激活函数：LeakyReLU 有正的负斜率，避免神经元完全失活
        self.relu = nn.LeakyReLU()
        
        # 第一全连接层：1224 → 5000
        # 将高维输入光谱特征投影到第一个隐空间
        self.A2 = nn.Linear(1224, N1)
        
        # 第二全连接层：5000 → 5000
        # 在隐空间中进行进一步的特征融合与非线性变换
        self.A1 = nn.Linear(N1, N2)
        
        # 输出层：5000 → 160
        # 最后投影到 EVDF 的 Fourier 系数空间。
        self.A0 = nn.Linear(N2, 160)
    
    def forward(self, x_A3):
        """
        前向传播
        
        参数：
            x_A3 (Tensor): 输入光谱，shape = [batch_size, 1224]
        
        返回：
            x_A0 (Tensor): 输出 Fourier 系数，shape = [batch_size, 160]
        
        计算步骤：
            1. x_A2 = ReLU(W_A2 @ x_A3 + b_A2)  → [batch, 5000]
            2. x_A1 = ReLU(W_A1 @ x_A2 + b_A1)  → [batch, 5000]
            3. x_A0 = W_A0 @ x_A1 + b_A0        → [batch, 160]
        """
        # 第一隐层：线性变换 + 激活
        x_A2 = self.relu(self.A2(x_A3))
        
        # 第二隐层：线性变换 + 激活
        x_A1 = self.relu(self.A1(x_A2))
        
        # 输出层：线性变换（无激活函数，保允许任意实数输出）
        x_A0 = self.A0(x_A1)
        
        return x_A0
    
    def save_model(self, PATH):
        """
        保存模型到文件
        
        参数：
            PATH (str): 文件路径
        
        实现方式：
            1. 提取模型参数字典（state_dict）
            2. 包装在外层字典中
            3. 使用 pickle 序列化
        
        改进建议：
            - 改用 torch.save(self.state_dict(), PATH)
            - 或 torch.save(self, PATH)
            - pickle 方式可能在不同 PyTorch 版本间兼容性较差
        """
        # 获取模型的所有参数和缓冲区
        pet1 = dict(self.state_dict())
        
        # 包装在字典中（便于日后扩展）
        pet = {"pet1": pet1}
        
        # 使用 pickle 序列化并保存
        with open(PATH, 'wb') as f:
            pickle.dump(pet, f, pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, PATH):
        """
        从文件加载模型参数（成员函数版本）
        
        参数：
            PATH (str): 模型文件路径
        
        副作用：
            修改当前对象的参数
        """
        # 读取 pickle 文件
        pet = {}
        with open(PATH, 'rb') as f:
            pet = pickle.load(f)
            pet1 = pet['pet1']
        
        # 加载参数到当前模型
        self.load_state_dict(pet1)


def load_model(PATH):
    """
    从文件加载模型（工厂函数）
    
    参数：
        PATH (str): 模型文件路径
    
    返回：
        net (ModelOTS): 加载完毕的模型对象
    
    使用场景：
        - evaluateModel.py：model = load_model(model_path)
    
    工作流程：
        1. 新建空的 ModelOTS 实例
        2. 读取 pickle 文件获取参数字典
        3. 将参数加载到模型
        4. 返回初始化完成的模型
    """
    # 读取 pickle 文件
    pet = {}
    with open(PATH, 'rb') as f:
        pet = pickle.load(f)
        pet1 = pet['pet1']
    
    # 新建模型实例
    net = ModelOTS()
    
    # 加载参数
    net.load_state_dict(pet1)
    
    return net







