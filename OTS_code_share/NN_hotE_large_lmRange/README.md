# 一键运行 OTS

`main.py all` 会依次完成：**生成数据 → 训练与验证 → 独立测试评估**。

在 PowerShell 中先设置目录和 Python：

```powershell
Set-Location 'C:\Users\win\OneDrive\Desktop\ots\OTS_code_share\NN_hotE_large_lmRange'
$otsPython = 'C:\Users\win\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe'
```

然后任选一条运行。以下采用 **16条训练样本、4条测试样本、32×16网络、最多3个epoch**，用于快速跑通流程。

**原 fixed_ion 数据集：**

```powershell
& $otsPython main.py all --run-id fixed_demo01 --run-dir runs/fixed_demo01 --dataset-id fixed_data01 --checkpoint-id fixed_model01 --dataset-kind fixed_ion --train-samples 16 --test-samples 4 --candidates '32,16' --max-epochs 3
```

**Fig.1/Fig.11 三物种数据集：**

```powershell
& $otsPython main.py all --run-id fig_demo01 --run-dir runs/fig_demo01 --dataset-id fig_data01 --checkpoint-id fig_model01 --dataset-kind fig1_fig11 --train-samples 16 --test-samples 4 --candidates '32,16' --max-epochs 3
```

Fig数据默认从本机74c1工作树中已有的`paper_experimental_points.csv`读取锚点。两个数据集均生成干净理论谱；fixed默认只输出电子，Fig默认输出e/H/C。

## 常用调整

- **扩大训练规模**：修改`--train-samples`、`--test-samples`、`--candidates`和`--max-epochs`。省略这四项时，默认5000/100条样本、比较`512,256;1024,512`两个网络、最多250个epoch；默认启用早停。
- **计算设备**：默认自动选择；追加`--device cpu`或`--device cuda`可指定训练和评估设备。
- **网格和输出物种**：追加`--generation-config 'config.json'`，省略时使用所选数据集的默认配置。Fig的锚点路径可在JSON中用`points_csv`覆盖。
- **再次运行**：更换`run-id`、`run-dir`、`dataset-id`和`checkpoint-id`中的名称；`--overwrite`不能覆盖已注册的ID。
- **查看全部参数**：`& $otsPython main.py all --help`。

## 结果在哪里

结果位于本目录下指定的`runs/<名称>/`：

- `data/`：训练/测试Parquet及`dataset_manifest.json`。
- `training/`：`model.pt`、训练历史和学习曲线。
- `evaluation/`：指标、预测和对比图；多物种结果分别在`e/`、`i1/`、`i2/`中。
- `pipeline_summary.json`：本次运行的产物路径汇总。

**Windows权限**：命令结束时需要创建registry符号链接。本机此前出现过`WinError 1314`；运行终端需具备创建符号链接的权限，否则即使数据和模型已生成，注册步骤仍会失败。
