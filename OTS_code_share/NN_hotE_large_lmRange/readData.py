# -*- coding: utf-8 -*-
"""
数据 I/O 模块 - 从 Parquet/Excel 文件读取光谱与标签数据

创建于: 2025-06-17 Tue 12:20:49
更新于: 2025-06-17 Tue 12:20:14

本模块提供：
1. readData_dataset - 从 Parquet 文件读取训练/测试数据集
2. readData_paper - 从 Excel 文件读取实验数据（论文数据）

数据格式说明：
─────────────────
由 `genDataSet.py` 生成的 Parquet 数据表结构：
  列 0:1224    -> 输入特征，三段归一化 OTS 光谱拼接而成
                  [ion peak, red-shifted electron peak, blue-shifted electron peak]
  列 1224:1384 -> 标签，80 个 Fourier 模式的实部 + 80 个虚部

特殊行约定：
  第 0 行不是训练样本，而是元数据行：
  - 前 1224 列存放各光谱点的波长（单位 nm）。
  - 后 160 列存放对应的 K 网格，前后各重复一次以对齐 Re/Im 标签。

训练/测试样本：
  第 1 行开始才是真正样本；因此 `readData_dataset` 默认 i_s=1。
  
应用场景：
  - genDataSet.py 生成的数据集读取
  - main.py 中的训练数据加载
  - evaluateModel.py 中的测试数据加载
"""

import torch, math
import pandas as pd
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq


def readData_dataset(filename, i_s=1, i_e=-1):
    """
    从 Parquet 文件读取数据集
    
    参数：
        filename (str):  Parquet 文件路径
        i_s (int):       起始行索引（包含）。默认 1 跳过元数据行。
        i_e (int):       结束行索引（不含）。默认 -1 会按 pandas 语义排除最后一行。
    
    返回：
        In (Tensor):     输入张量，shape = [样本数, 1224]。
        labels (Tensor): 标签张量，shape = [样本数, 160]。
    
    数据处理步骤：
        1. 使用 pyarrow 读取 Parquet 文件
        2. 转换为 pandas DataFrame
        3. 按行范围 [i_s:i_e] 切片。
        4. 分离输入（前 1224 列）和标签（后续列）
        5. 转换为 PyTorch 张量（float32 精度）
    
    注意事项：
        - i_s=1 默认跳过第 0 行元数据，而不是跳过列名。
        - i_e=-1 在 pandas 切片中会漏掉最后一行；若要读取到物理文件末尾，
          调用方应显式传入 None 或 len(data)。当前脚本沿用历史默认值。
        - 输入和标签都转换为 float32，以便直接送入 PyTorch 模型。
    """
    # 使用 PyArrow 读取 Parquet 文件
    table_read = pq.read_table(filename)
    
    # 转换为 Pandas DataFrame 便于切片和处理
    data = table_read.to_pandas()
    
    # 提取输入特征（前 1224 列）：三段归一化 OTS 光谱。
    In = data.iloc[i_s:i_e, 0:1224].values
    
    # 转换为 PyTorch 张量
    In = torch.tensor(In, dtype=torch.float32)
    
    # 提取标签（第 1224 列之后）：[Re(F_e) 80维, Im(F_e) 80维]。
    labels = data.iloc[i_s:i_e, 1224:].values
    
    # 转换为 PyTorch 张量
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return In, labels


def readData_paper(filename):
    """
    从 Excel 文件读取实验光谱数据。
    
    参数：
        filename (str):  Excel 文件路径（.xlsx 或 .xls）
    
    返回：
        (lms_i2, Iex_i2, lms_e, Iex_e, lms_e2, Iex_e2, lms_e3, Iex_e3)
        
        其中：
          lms_*  = 波长数据
          Iex_*  = 对应波长处的强度
          
          分类说明：
            lms_i2, Iex_i2: 离子峰附近数据（λ > 349 nm）
            lms_e, Iex_e:   电子散射全波段
            lms_e2, Iex_e2: 红移电子峰（λ > 375 nm）
            lms_e3, Iex_e3: 蓝移电子峰（λ < 340 nm）
    
    功能：
        - 从 Excel 表格中提取多列实验数据
        - 自动过滤 NaN 值
        - 根据波长范围进行分类
        - 返回 PyTorch 张量格式
    
    应用场景：
        用于把真实或论文图表整理出的光谱拆成与网络输入一致的三段。
    """
    # 读取 Excel 文件
    dfs = pd.read_excel(filename)
    
    # 获取所有列名
    names = [name for name in dfs]
    
    # 初始化列表用于存储各类数据
    lms_i = []      # 离子线波长
    Iex_i = []      # 离子线强度
    lms_e = []      # 电子散射波长
    Iex_e = []      # 电子散射强度
    
    # 提取离子峰数据（通常在列 2-3）。
    name = names[2]
    for val in dfs[name]:
        lms_i.append(val)
    
    name = names[3]
    for val in dfs[name]:
        Iex_i.append(val)
    
    # 提取电子散射数据（通常在列 0-1）。
    name = names[0]
    for val in dfs[name]:
        if not math.isnan(val):  # 过滤 NaN 值
            lms_e.append(val)
    
    name = names[1]
    for val in dfs[name]:
        if not math.isnan(val):
            Iex_e.append(val)
    
    # 转换为 PyTorch 张量
    lms_i = torch.tensor(lms_i)
    Iex_i = torch.tensor(Iex_i)
    lms_e = torch.tensor(lms_e)
    Iex_e = torch.tensor(Iex_e)
    
    # 根据波长范围分类，匹配论文 Fig. 1 的三段输入设计。
    # 离子峰：λ > 349 nm
    lms_i2 = lms_i[lms_i > 349]
    Iex_i2 = Iex_i[lms_i > 349]
    
    # 红移电子峰：λ > 375 nm
    lms_e2 = lms_e[lms_e > 375]
    Iex_e2 = Iex_e[lms_e > 375]
    
    # 蓝移电子峰：λ < 340 nm
    lms_e3 = lms_e[lms_e < 340]
    Iex_e3 = Iex_e[lms_e < 340]
    
    return lms_i2, Iex_i2, lms_e, Iex_e, lms_e2, Iex_e2, lms_e3, Iex_e3






