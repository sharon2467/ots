# 一键运行 OTS

从本目录执行；Fig的generation-config必须把points_csv指向外部提供的论文锚点CSV。核心命令顺序不变：生成→训练/验证→测试评估。以下为原流程快速示例，不是已发布研究分数对应的训练规模。

```bash
python main.py all --run-id fixed_demo01 --run-dir runs/fixed_demo01 --dataset-id fixed_data01 --checkpoint-id fixed_model01 --dataset-kind fixed_ion --train-samples 16 --test-samples 4 --candidates '32,16' --max-epochs 3
python main.py all --run-id fig_demo01 --run-dir runs/fig_demo01 --dataset-id fig_data01 --checkpoint-id fig_model01 --dataset-kind fig1_fig11 --generation-config /path/to/fig_generation.json --train-samples 16 --test-samples 4 --candidates '32,16' --max-epochs 3
```

正式研究逐次参数见仓库根 `reproduction/publication/training_commands.json`。Fig默认输出e/H/C，fixed_ion只重建电子。运行目录保存data、training、evaluation和pipeline_summary.json；使用新run/dataset/checkpoint标识，不覆盖旧结果。全部参数见`python main.py all --help`；设备用`--device cpu`或`--device cuda`。本发布阶段未执行这些命令。
