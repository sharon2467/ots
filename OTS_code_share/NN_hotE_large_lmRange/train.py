# -*- coding: utf-8 -*-
"""
神经网络训练模块 - 实现损失函数与优化循环

创建于: 2025-06-17 Tue 12:20:49
更新于: 2026-03-17 Tue 23:56:15

本模块提供：
1. Loss_func - 计算 L1（绝对值）损失
2. train    - 执行 Adam 优化训练循环

设计特点：
- 当前实现使用 L1Loss（平均绝对误差）直接比较 160 维 Fourier 标签。
  * 这反映本代码的实际行为。
  * 论文正文描述的 loss 是 EVDF 预测值与真值的 MSE；若要严格复现实验
    文字，应在输出系数后做 inverse Fourier transform，再在 f_e(v) 域计算 MSE。
  
- Adam 优化器自适应学习率
  * 初始 lr=1e-4，逐步降低到 1e-7
  * 分阶段训练提高稳定性
  
- 全批次训练（无 mini-batch）
  * 当前数据量不大（<5000样本）
  * 若数据量增加需改用 DataLoader
"""

import torch


def Loss_func(xs, ys, model):
    """
    计算训练损失（当前代码：Fourier 系数域 L1 距离）。
    
    参数：
        xs (Tensor):   输入批次，shape = [batch_size, 1224]
        ys (Tensor):   目标标签，shape = [batch_size, 160]，
                       格式为 [Re(F_e[0:80]), Im(F_e[0:80])]
        model (nn.Module): 神经网络模型
    
    返回：
        loss (Tensor): 标量损失值
    
    计算流程：
        1. 前向传播：y_pred = model(xs)
        2. 计算 L1 损失：loss = mean(|y_pred - ys|)
    
    物理意义：
        - 最小化模型输出与真实 Fourier 系数的平均绝对差异。
        - 由于 Fourier 系数可逆地表示低通 EVDF，这会间接约束重建的
          速度分布；但它不是论文文字中“EVDF 域 MSE”的完全相同实现。
    """
    # 定义 L1 损失函数（绝对值误差）
    f_loss = torch.nn.L1Loss()
    
    # 前向传播：从输入光谱预测 EVDF 的 Fourier 系数。
    y = model(xs)
    
    # 计算损失
    loss = f_loss(y, ys)
    
    return loss


def train(xs, ys, model, loss_rate, N):
    """
    训练单阶段：指定学习率下运行 N 步梯度下降
    
    参数：
        xs (Tensor):      训练输入，shape = [样本数, 1224]
        ys (Tensor):      训练标签，shape = [样本数, 160]，即 80 个复系数的实/虚部
        model (nn.Module):神经网络模型（原地修改）
        loss_rate (float):学习率（如 1e-4, 1e-5 等）
        N (int):          迭代步数（如 3000, 6000 等）
    
    返回：
        model (nn.Module):训练后的模型
    
    训练流程（全批量 PyTorch 优化循环）：
        for t in range(N):
            1. 计算损失：loss = Loss_func(xs, ys, model)
            2. 清空梯度：optimizer.zero_grad()
            3. 反向传播：loss.backward()
            4. 参数更新：optimizer.step()
    
    特点：
        - 每 200 步打印一次损失（便于监控训练进度）
        - 使用科学计数法显示损失值（%.4e）
        - 全批次更新（整个传入张量作为一个批次，不做 DataLoader mini-batch）
    
    调用示例（main.py 中）：
        train(xs, ys, model, loss_rate=1.e-4, N=3000)  # 阶段1
        train(xs, ys, model, loss_rate=1.e-5, N=6000)  # 阶段2
        train(xs, ys, model, loss_rate=1.e-6, N=10000) # 阶段3
        ...
    """
    # 创建 Adam 优化器
    # Adam 自适应学习率优化算法，通常比标准 SGD 更快收敛
    optimizer = torch.optim.Adam(model.parameters(), lr=loss_rate)
    
    # 训练循环
    for t in range(N):
        # 计算当前批次的损失
        loss = Loss_func(xs, ys, model)
        
        # 每 200 步打印一次进度
        if t % 200 == 1:
            print(t, "%.4e" % loss.item())
        
        # 清空之前的梯度累积
        optimizer.zero_grad()
        
        # 反向传播：计算所有参数的梯度
        loss.backward()
        
        # 使用优化器更新参数
        # θ_{t+1} = θ_t - lr * ∇L(θ_t)
        optimizer.step()
    


