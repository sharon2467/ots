独立温度/光谱偏差诊断；不接入训练工作流

入口报告：REPORT.txt
图：representation_audit/representation_control.png
    measurement_sensitivity/sensitivity.png
主要数值：input_and_verification_audit.json
          representation_audit/controlled_results.json
          representation_audit/positive_tail_results.json
          measurement_sensitivity/results.json

以下命令执行独立敏感性控制，不包含 D4 神经网络性能评估。
实验定义和解释边界见各子目录报告。

在 /workspace/ots 使用现有虚拟环境，按依赖顺序执行：
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/representation_audit/audit_representation.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/representation_audit/controlled_states.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/representation_audit/make_figure.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/representation_audit/positive_tail_sensitivity.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/measurement_sensitivity/run_sensitivity.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/bias_diagnosis_20261023/input_and_verification_audit.py

脚本记录输入的绝对路径；迁移环境时需显式映射路径并核对输入哈希。
脚本只写各自诊断输出；复现会更新诊断结果文件，不会覆盖原始数据/模型。
完整输入/源码来源：representation_audit/results.json 的sha256_before/after，
measurement_sensitivity/results.json 的protected_input_sha256，
input_and_verification_audit.json 的input/sha256。
论文来源与色标核查：paper_audit/REPORT.txt、evidence.json、manifest.json。
SHA256SUMS.txt 不含自身、缓存与__pycache__。D4 数值为独立模型研究的引用结果。
