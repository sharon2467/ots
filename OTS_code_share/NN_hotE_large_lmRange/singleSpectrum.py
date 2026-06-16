# -*- coding: utf-8 -*-
"""
单样本推理与可视化 - 对测试集中的单个光谱样本进行推理并绘图对比

创建于: 2025-06-24 Tue 18:40:16
更新于: 2026-03-17 Tue 23:56:14

功能：
- 从测试数据集中读取单个样本
- 使用训练好的模型进行前向推理
- 将模型输出的 160 维 EVDF Fourier 系数与真实标签进行对比
- 绘制 5 幅子图：三段输入光谱 + Fourier 实部/虚部对比

应用场景：
- 定性评估模型性能
- 调试与快速验证
- 可视化输入光谱与 Fourier 系数重建；若要查看 EVDF(v)，使用 evaluateModel.py
"""

import sys
# Legacy workstation path; local imports normally resolve from the script directory.
sys.path.append('文档/物理问题/OTS/NN')

import torch, math
import matplotlib.pyplot as plt
import pickle
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from model import ModelOTS as DynamicNet
from model import load_model
from readData import readData_dataset
from train import Loss_func

# ═════════════════════════════════════════════════════════════════
# 配置：数据集与模型路径
# ═════════════════════════════════════════════════════════════════

# 测试数据路径。第 0 行为波长/K 网格元数据，第 1 行开始为样本。
data_path = './dataset_test.parquet'

# 模型路径，由 main.py 保存。
model_path = "./OTS_Te_model.pickle"

# ═════════════════════════════════════════════════════════════════
# 第一步：加载单个测试样本
# ═════════════════════════════════════════════════════════════════

# 选择要评估的物理样本行；i=1 是第一个真实样本。
i = 2

# 读取第 i 个样本（范围 [i, i+1)）
[xs, ys] = readData_dataset(data_path, i, i+1)

# ═════════════════════════════════════════════════════════════════
# 第二步：加载训练好的模型并做推理
# ═════════════════════════════════════════════════════════════════

# 从文件加载模型
model = load_model(model_path)

# 前向推理，获取模型预测的 EVDF Fourier 表示。
ys = ys.detach().numpy()[0]          # 真实值，shape = [160]
ys2 = model(xs).detach().numpy()[0]  # 模型预测，shape = [160]

# ═════════════════════════════════════════════════════════════════
# 第三步：分离频域表示为实部和虚部
# ═════════════════════════════════════════════════════════════════

# 复数 Fourier 模式数量；前半是实部，后半是虚部。
N = int(len(ys2) / 2)

# 分离真实值和预测值的实部与虚部
# 前 N 个分量：傅里叶系数的实部
# 后 N 个分量：傅里叶系数的虚部

Fer = ys[0:N]      # 真实实部
Fej = ys[N:2*N]    # 真实虚部
Fer2 = ys2[0:N]    # 预测实部
Fej2 = ys2[N:2*N]  # 预测虚部

# ═════════════════════════════════════════════════════════════════
# 第四步：绘制结果
# ═════════════════════════════════════════════════════════════════

# 创建 2×3 的子图布局
plt.figure(figsize=(15, 8))

# ─────────────────────────────────────────────────────────────
# 子图 1-3：输入光谱的三个波段
# ─────────────────────────────────────────────────────────────

# 子图 1：离子峰，接近 351 nm 探针波长。
plt.subplot(2, 3, 1)
plt.plot(xs[0][0:579])
plt.title('Input Spectrum - Band 1 (λ: 349-353 nm)')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.grid(True)

# 子图 2：红移电子峰，当前 hotE 分支扩展到约 600 nm。
plt.subplot(2, 3, 2)
plt.plot(xs[0][579:579+304])
plt.title('Input Spectrum - Band 2 (λ: 375-600 nm)')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.grid(True)

# 子图 3：蓝移电子峰，当前 hotE 分支扩展到约 100 nm。
plt.subplot(2, 3, 3)
plt.plot(xs[0][579+304:1224])
plt.title('Input Spectrum - Band 3 (λ: 100-340 nm)')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.grid(True)

# ─────────────────────────────────────────────────────────────
# 子图 4：频域表示的实部对比
# ─────────────────────────────────────────────────────────────

plt.subplot(2, 3, 4)
plt.plot(Fer, label='Ground Truth', linewidth=2)
plt.plot(Fer2, label='Model Prediction', linewidth=2, linestyle='--')
plt.title('Real Part of Fourier Coefficients')
plt.xlabel('Fourier Mode Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# ─────────────────────────────────────────────────────────────
# 子图 5：频域表示的虚部对比
# ─────────────────────────────────────────────────────────────

plt.subplot(2, 3, 5)
plt.plot(Fej, label='Ground Truth', linewidth=2)
plt.plot(Fej2, label='Model Prediction', linewidth=2, linestyle='--')
plt.title('Imaginary Part of Fourier Coefficients')
plt.xlabel('Fourier Mode Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# ═════════════════════════════════════════════════════════════════
# 第五步：保存图像
# ═════════════════════════════════════════════════════════════════

plt.tight_layout()
plt.savefig("result0.png", dpi=150)

print("Visualization saved to result0.png")
print(f"Sample index: {i}")
print(f"Input shape: {xs.shape}")
print(f"True output shape: {ys.shape}")
print(f"Predicted output shape: {ys2.shape}")


