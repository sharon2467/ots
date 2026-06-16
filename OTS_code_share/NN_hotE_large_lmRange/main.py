# -*- coding: utf-8 -*-
"""
主程序 - OTS 光谱到 EVDF Fourier 系数的训练管道

创建于: 2025-06-17 Tue 12:20:49
更新于: 2026-03-17 Tue 23:56:15

功能：
- 从 Parquet 数据集加载训练数据
- 构建神经网络模型
- 多阶段全批量训练（学习率递减策略）
- 保存训练后的模型
- 快速验证（前向推理）

执行流程：
1. 加载数据（Dataset_train.parquet）
2. 初始化模型（ModelOTS）
3. 分阶段训练：
   ├─ 阶段1：lr=1e-4，3000 步
   ├─ 阶段2：lr=1e-5，6000 步
   ├─ 阶段3：lr=1e-6，10000 步
   └─ 阶段4：lr=1e-7，100000 步  (超长训练)
4. 保存模型到 OTS_Te_model.pickle
5. 前向推理验证

数据处理：
- 输入：1224 维光谱特征，对应三段独立归一化的 OTS 光谱。
- 输出：160 维 Fourier 表示，即 80 个 EVDF Fourier 模式的实部和虚部。
- 数据量：约 5000 个样本（可配置）

训练策略分析：
- 当前代码使用 Fourier 系数域 L1 损失；论文正文描述的是 EVDF 域 MSE。
  若需要严格复现论文的 loss 定义，应同步修改 `train.Loss_func`。
- Adam 优化器（自适应学习率）
- 学习率从大到小逐阶段衰减（coarse→fine调整）
- 后期阶段步数很多（100000），用于精细调优
"""

import torch, math
import matplotlib.pyplot as plt
import pickle

import pyarrow as pa
import pyarrow.parquet as pq

from model import ModelOTS as DynamicNet
from train import train, Loss_func
from readData import readData_dataset

# ═════════════════════════════════════════════════════════════════
# 配置与常数
# ═════════════════════════════════════════════════════════════════

# 数据集路径：由 genDataSet.py 生成，第一行为波长/K 网格元数据。
data_path = './dataset_train.parquet'

# ═════════════════════════════════════════════════════════════════
# 第一步：加载训练数据
# ═════════════════════════════════════════════════════════════════

print("Loading training dataset...")

# 从 Parquet 文件读取数据。
# xs: 训练输入 [样本数, 1224]，三段 OTS 光谱拼接。
# ys: 训练标签 [样本数, 160]，[Re(F_e), Im(F_e)]。
[xs, ys] = readData_dataset(data_path)

print(f"Data shape - Input: {xs.shape}, Labels: {ys.shape}")

# ═════════════════════════════════════════════════════════════════
# 第二步：初始化模型
# ═════════════════════════════════════════════════════════════════

print("Initializing model...")

# 创建神经网络模型实例
model = DynamicNet()

# ═════════════════════════════════════════════════════════════════
# 第三步：多阶段分层训练
# ═════════════════════════════════════════════════════════════════

print("Starting multi-stage training...")

# 阶段 1：粗调，学习率较大
print("Stage 1: lr=1e-4, N=3000")
train(xs, ys, model, loss_rate=1.e-4, N=3000)

# 阶段 2：中调，学习率中等
print("Stage 2: lr=1e-5, N=6000")
train(xs, ys, model, loss_rate=1.e-5, N=6000)

# 阶段 3：细调，学习率较小
print("Stage 3: lr=1e-6, N=10000")
train(xs, ys, model, loss_rate=1.e-6, N=10000)

# 阶段 4：超细调，学习率很小
# 注意：这个阶段的迭代次数很多（100000），训练时间较长
print("Stage 4: lr=1e-7, N=100000")
train(xs, ys, model, loss_rate=1.e-7, N=100000)

print("Training completed!")

# ═════════════════════════════════════════════════════════════════
# 第四步：保存模型
# ═════════════════════════════════════════════════════════════════

print("Saving model...")

model_path = './OTS_Te_model.pickle'
model.save_model(model_path)

print(f"Model saved to {model_path}")

# ═════════════════════════════════════════════════════════════════
# 第五步：快速验证（前向推理）
# ═════════════════════════════════════════════════════════════════

print("Verifying with forward pass...")

# 在训练集上做推理（验证模型是否正常工作）
ys2 = model(xs).detach().numpy()

print(f"Output shape: {ys2.shape}")
print(f"Output sample (first 10 values): {ys2[0, :10]}")
print(f"Label sample (first 10 values): {ys[0, :10].numpy()}")

# ═════════════════════════════════════════════════════════════════
# （可选）后续步骤
# ═════════════════════════════════════════════════════════════════

# 进一步的 EVDF 还原和可视化在 evaluateModel.py / singleSpectrum.py 中进行；
# 本脚本只负责训练并保存权重。







