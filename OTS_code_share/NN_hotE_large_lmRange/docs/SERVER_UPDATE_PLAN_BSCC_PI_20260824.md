# BSCC 与上海交大 π 服务器端更新计划

初版日期：2026-08-24  
BSCC 决策更新：2026-08-29  
状态：仅计划；远端审计全程只读，尚未删除、移动、上传、覆盖、提交或取消任何作业。  
当前优先级：先实施北京超级云计算中心 GPU 账号的原地更新；上海交大 π 集群更新延期。  
范围：BSCC `~/yhh/ots` 中的 canonical 源码、Slurm 入口、三个 ID 的符号链接索引、日志和哈希；不改动正式训练结果的科学结论。

## 1. 现场基线

| 项目 | 本地 | BSCC | π 集群 |
|---|---|---|---|
| 源码基线 | working tree；HEAD `6b974b4` 不能代表全部最终文件 | `/home/bingxing2/home/scx6090/yhh/ots`，旧源码平铺、无 Git | `/lustre/home/acct-phywsm/sharon2467/myfolder/OTS_code_share/NN_hotE_large_lmRange`，独立 Git HEAD `49fa64b`、无 remote |
| 当前队列 | 不适用 | `squeue` 为空 | `squeue` 为空 |
| 旧入口 | 本地已删除 6 个 legacy Python | 6 个均存在 | 其中 5 个存在，另有更多旧辅助入口 |
| canonical 缺口 | 发布包尚未冻结 | 缺 `data.py`、`evaluation.py`、当前六模块、tests 等 | 缺 `data.py`、`evaluation.py`、`ots_full_fourier.py`、tests 等 |
| 主要历史产物 | 本地回收副本 | 正式 hot-electron、paper Fig. 3、ion1；另有两个大 smoke run | legacy 数据/checkpoint、4 个 v2 run |
| registry / 独立 SHA 文件 | 新 canonical registry 尚未实现 | 现有 `models/` 是历史模型注册；正式 run 内有 SHA；将另建三个 ID 链接库 | 不存在 datasets/models/registry/LATEST/SHA 文件，哈希仅写在部分 stdout |

四个 π v2 作业 `59648208/59739405/59740000/59740036` 及 BSCC 正式作业 `1346661/1347041/1347309/1348835/1348858` 均已现场核实为 `COMPLETED 0:0`。

## 2. 总体决策

1. **BSCC 在现有工程根原地更新。** canonical 活动源码直接部署到 `/home/bingxing2/home/scx6090/yhh/ots`
2. **源码与历史产物分离。** 原地更新源码不等于迁移历史数据。新 canonical 源码不得覆盖或冒充旧数据、旧 pickle、论文 checkpoint、ion1 checkpoint 或旧 v2 checkpoint。
3. **只按 allow-list 同步。** 禁止在远端工程根执行 `rsync --delete`；`.gitignore` 也不能充当部署排除合同。
4. **不用 Git 拉取。** 本地无 remote、BSCC 无 Git，不得用 `git pull` 代替发布。
5. **先备份/staging、后原子替换、最后清理。** 在新源码通过哈希、单测、CLI 和计算节点 smoke 前，不删除任何旧入口、环境、日志或重复产物。
6. **历史 run 内哈希一律保留。** `runs/**/SHA256SUMS.txt`、`SOURCE_SHA256SUMS.txt` 和 stdout 中的正式哈希属于证据，不作为“残余哈希”清理。
7. **新 canonical registry 采用三个符号链接库。** 新增 `registry/runs/`、`registry/datasets/`、`registry/checkpoints/`，分别按 `run_id/dataset_id/checkpoint_id` 命名链接；不引入数据库、服务或 `LATEST` 语义。
8. **现有 `models/` 历史注册合并到 checkpoint 链接库。** hot-electron、paper 和 ion1 的三个历史 job 迁入 `registry/checkpoints/`；每个注册项仍直接指向它原来指向的 checkpoint、metrics 或 manifest，不补 `training/` 兼容层，不移动或重命名历史产物。完成逐链接核验和备份后，旧 `models/` 不再作为活动 registry。
9. **每个子命令都显式指定新的输出 run。** `generate/train/evaluate/all` 都必须接收 `--run-id` 和 `--run-dir`；`run_id` 只表示本次成功后要发布的 run 链接，从不用于解析既有输入。train 的既有输入只由 `dataset_id` 定位；checkpoint 加载和 evaluate 的既有输入只由 `checkpoint_id + dataset_id` 定位。
10. **所有远端删除另设执行确认门。** 本文给出推荐删除清单，但真正执行前再次向用户确认精确绝对路径。

## 3. 发布前门槛：先冻结本地发布候选

当前 working tree 是正确的同步来源，但还不能直接打包。正式部署前必须完成：

1. 修正 `build_fixed_ion_distributions` 新双参数接口与 `tests/test_data_t2.py`、相关文档中的旧单参数调用不一致。
2. 修正五份文档移入 `docs/` 后，`tests/test_legacy_cleanup_t6.py` 仍检查旧根路径和旧删除范围的问题。
3. 清理 `ots_full_fourier.py` 新演示块的尾随空格，使 `git diff --check` 通过。
4. 从发布包排除 7 个 tracked `pyc`、所有 `__pycache__`、本地结果目录、数据、模型和凭据。
5. 在 `main.py` 中实现最小 registry helper，不引入新服务、数据库、ORM 或“latest”选择逻辑：
   - 校验 ID 是安全的单一路径段；
   - 创建并解析三类相对符号链接；
   - 区分“本次待发布的 run/checkpoint/dataset ID”和“作为输入解析的 dataset/checkpoint ID”；
   - 分别校验 dataset registry/manifest 的 `dataset_id` 和 checkpoint registry/payload 的 `checkpoint_id`；允许 checkpoint 中记录的训练 `dataset_id` 与本次评估 `dataset_id` 不同；
   - 对 ID 冲突、普通文件冒充链接、断链和越过 `runs/` 根目录立即失败。
6. 调整 CLI 路径合同：
   - 四个子命令都必须显式提供 `--run-id` 和 `--run-dir runs/RUN_DIR`，所有新产物只写入这个新 run；
   - `generate` 在新 run 的 `data/` 生成数据，成功后发布本次 `run_id` 和新的 `dataset_id`；
   - `train` 用 `dataset_id` 解析既有 `data/` 输入，在新 run 的 `training/` 写模型，成功后发布本次 `run_id` 和新的 `checkpoint_id`；
   - `evaluate` 用 `dataset_id + checkpoint_id` 解析既有数据和模型，在新 run 的 `evaluation/` 写结果，成功后只发布本次 `run_id`；
   - `all` 在同一个新 run 中完成三阶段，成功后发布本次 `run_id/dataset_id/checkpoint_id`。
   - public CLI 删除 `--output-dir/--train-data/--test-data/--dataset-manifest/--checkpoint`；这些路径统一由当前 `run-dir` 或输入 artifact registry 得出。
7. 新写 `slurm/canonical_all_bscc.slurm`。脚本必须显式接收 `OTS_RUN_ID/OTS_DATASET_ID/OTS_CHECKPOINT_ID/OTS_RUN_DIR`，从 `~/yhh/ots` 调用 `python main.py all`；不自动生成 ID，不调用 legacy Python。提交前必须创建 `runs/$OTS_RUN_DIR`，并通过 `sbatch --output/--error` 把 Slurm 自身日志写为 `runs/$OTS_RUN_DIR/slurm-%j.out` 和 `slurm-%j.err`，不得再在工程根生成日志。
8. π 脚本本轮不写、不部署；`canonical_all_pi.slurm` 延期到 π 更新任务。
9. 当前本地 `slurm/` 中 8 个文件均不得原样作为 BSCC active 入口部署：它们调用已删除的 `train_v2.py/genDataSet.py/evaluateModel.py`，或只服务历史作业。
10. 增加 registry 测试并更新 CLI 测试，覆盖正常跨 run/跨 dataset 评估、输入维度或 K/波长网格不兼容、冲突、断链、越界目标、发布失败回滚，以及每个命令都必须有独立 `run-dir`。
11. 运行全部现有 unittest 模块、registry 新测试和必要语法检查；未通过前不得生成发布 manifest。
12. 测试通过后重新生成最终 `SOURCE_MANIFEST_<release-id>.sha256`。当前审计 manifest `3a7cd94f...63776` 只记录审计时 working tree，不是发布验收值。

最终源码 allow-list 应仅包含：

- 六个 canonical 模块：`data.py/evaluation.py/main.py/model.py/train.py/phyEqns/ots_full_fourier.py`；
- 保留物理模块 `phyEqns/ots_fourier.py`、`phyEqns/ots_superGaussian.py`；
- 修正后的六个测试模块；
- `docs/*.md`；
- `paper_reproduction/` 的 Python 与报告；BSCC 专用复现 Slurm 只放历史/复现区，不作为 canonical active 入口；
- 新的 BSCC canonical Slurm；
- registry 相关源码与测试；
- 最终源码 SHA manifest。

明确排除 `.git`、凭据、Parquet、checkpoint、pickle、NPZ、PNG、日志、runs、datasets、models、registry、本地回收结果和 cache。

### 3.1 三个 ID 链接库的固定布局

新 canonical registry 固定放在工程根的 `registry/`；迁移验收期间与历史 `models/` 临时并存，验收后由 `registry/checkpoints/` 取代旧活动注册：

```text
registry/
├── runs/
│   └── <run_id>        -> ../../runs/<RUN_DIR>
├── datasets/
│   └── <dataset_id>    -> ../../runs/<RUN_DIR>/data
└── checkpoints/
    └── <checkpoint_id> -> ../../runs/<RUN_DIR>/training
```

设计边界：

- 三个目录只是符号链接索引，不复制 Parquet、checkpoint 或图，不保存额外 JSON 数据库。
- 三类 ID 分别唯一。作为本次输出发布的 ID 在命令开始时必须尚未存在；作为输入使用的 dataset/checkpoint ID 必须已经存在并可解析。
- `--overwrite` 只允许覆盖本次流水线目标产物，绝不允许重绑定已有 ID。
- 链接一律使用相对目标，使整个 `ots/` 工程在同一文件系统内移动时仍可解析。
- ID 必须是安全的单一路径段；拒绝空值、`.`、`..`、斜杠、反斜杠和路径穿越。
- `run-dir` 必须位于工程根 `runs/` 下，并且不能已经被其他 run ID 注册。本次命令默认拒绝写入已有 pipeline 产物；提交时预先创建的 `slurm-%j.out/.err` 不算 pipeline 产物。失败重试只能对尚未发布的同一 `run-dir` 显式使用 `--overwrite`。
- 新 canonical 条目的解析目标必须真实存在并位于 `runs/` 下；每个 dataset/checkpoint 分别指向其自身生产 run 的 `data/` 与 `training/`，不要求它们和当前输出 run 是同一个目录。历史迁移条目保持原链接目标，不强制改造成 canonical 目录结构。
- evaluate 不比较 checkpoint 的训练 `dataset_id` 与本次评估 `dataset_id`。它只要求两类 registry 各自有效，并继续校验输入/输出维度、K 网格、波长网格、finite 值等真正影响推理正确性的结构合同。
- 不提供模糊查找、前缀匹配或自动选择最新版本；每个子命令实际使用的 ID 都必须精确匹配其输入/输出角色。

### 3.2 registry 生命周期与跨 run 数据流

| 子命令 | `run_id` | `run-dir` | `dataset_id` | `checkpoint_id` |
|---|---|---|---|---|
| `generate` | 待发布输出 ID | 必需的新输出目录 | 待发布输出 ID | 不使用 |
| `train` | 待发布输出 ID | 必需的新输出目录 | 解析既有输入 | 待发布输出 ID |
| `evaluate` | 待发布输出 ID | 必需的新输出目录 | 解析既有输入 | 解析既有输入 |
| `all` | 待发布输出 ID | 必需的新输出目录 | 待发布输出 ID | 待发布输出 ID |

```text
generate(run_id=NEW_RUN, run_dir=NEW_DIR, dataset_id=NEW_DATASET)
  -> 不解析 run_id
  -> 写 NEW_DIR/data
  -> 成功后发布 run_id -> NEW_DIR
  -> 成功后发布 dataset_id -> NEW_DIR/data

train(run_id=NEW_RUN, run_dir=NEW_DIR,
      dataset_id=EXISTING_DATASET, checkpoint_id=NEW_CHECKPOINT)
  -> 只解析 dataset_id 取得 train Parquet 和 manifest
  -> 写 NEW_DIR/training
  -> checkpoint.run_id 记录 NEW_RUN
  -> 成功后发布 run_id -> NEW_DIR
  -> 成功后发布 checkpoint_id -> NEW_DIR/training

evaluate(run_id=NEW_RUN, run_dir=NEW_DIR,
         dataset_id=EXISTING_DATASET, checkpoint_id=EXISTING_CHECKPOINT)
  -> 解析 dataset_id 取得 test Parquet 和 manifest
  -> 解析 checkpoint_id 取得 checkpoint
  -> 分别校验所选 dataset_id 与 checkpoint_id
  -> 允许 checkpoint.dataset_id != 本次 evaluation dataset_id
  -> 校验维度、K 网格和波长网格兼容
  -> 写 NEW_DIR/evaluation
  -> 成功后发布 run_id -> NEW_DIR

all(run_id=NEW_RUN, run_dir=NEW_DIR,
    dataset_id=NEW_DATASET, checkpoint_id=NEW_CHECKPOINT)
  -> 全部产物写入 NEW_DIR/{data,training,evaluation}
  -> 在 pipeline_summary.json 成功后发布三个输出 ID
```

发布前先预检所有待发布 ID、输入链接和 `run-dir`，再创建临时符号链接并原子改名；任一步失败时只回滚本次新建的临时/最终链接，不删除既有输入链接或 run 产物。`all` 的 `pipeline_config.json/pipeline_summary.json` 记录本次发布链接；standalone train/evaluate 则在既有 history/metrics 中记录输入 artifact 的 ID、路径和来源 run ID。

`run_id` 的语义改为“生成当前命令输出的 run”，不再要求它和输入 artifact 的生产 run 相同：

- `dataset_manifest.json.run_id` 是数据生成 run；
- `model.pt.run_id` 是训练 run；
- `evaluation/metrics.json.run_id` 是评估 run；
- train 只核对 manifest 的 `dataset_id`，不再要求 manifest `run_id == 当前 train run_id`；
- dataset loader 核对本次选择的 `dataset_id`；checkpoint loader 只核对本次选择的 `checkpoint_id`，不再把 checkpoint 中的训练 `run_id/dataset_id` 与当前 evaluation run/dataset 强制相等；
- evaluation 同时记录 evaluation dataset ID、dataset 生产 run ID、checkpoint ID、checkpoint 训练 dataset ID 和 checkpoint 训练 run ID，明确区分训练来源与评估对象。

ID 输入解析与显式输出 run 的使用方式：

```bash
python main.py train \
  --run-id NEW_TRAIN_RUN_ID \
  --run-dir runs/NEW_TRAIN_RUN_DIR \
  --dataset-id EXISTING_DATASET_ID \
  --checkpoint-id NEW_CHECKPOINT_ID

python main.py evaluate \
  --run-id NEW_EVALUATION_RUN_ID \
  --run-dir runs/NEW_EVALUATION_RUN_DIR \
  --dataset-id EXISTING_DATASET_ID \
  --checkpoint-id EXISTING_CHECKPOINT_ID
```

训练和评估超参数仍可显式传入；上面省略的是可由 registry 唯一确定的输入文件路径，而每次输出的 `run-id/run-dir` 始终显式存在。

### 3.3 现有 `models/` 历史注册迁入 checkpoint 链接库

现有三个历史模型注册使用 `models/<dataset-family>/job_<job-id>/`，每个 job 目录内已经有指向原 checkpoint、metrics 或 manifest 的链接。迁移只改变注册入口的位置，不改变这些链接最终指向的文件，也不要求历史 run 增加 `training/` 子目录。

`registry/checkpoints/` 因此允许两种明确条目：

1. **新 canonical checkpoint**：`registry/checkpoints/<checkpoint_id>` 是目录符号链接，指向 `runs/<RUN_DIR>/training`。
2. **历史 checkpoint 注册**：`registry/checkpoints/job_<job-id>/` 是轻量注册目录，内部保留原 `models/.../job_<job-id>/` 的文件链接；如果相对链接文本因目录层级改变而必须重写，只调整相对路径写法，解析后的绝对目标必须与迁移前完全相同。

固定历史注册为：

| 历史 checkpoint ID | 迁入的新位置 | 必须保持的原始目标 |
|---|---|---|
| `job_1346661` | `registry/checkpoints/job_1346661/` | `runs/bscc_v2_1346661/OTS_Te_model_v2.pt`、`metrics_v2.json` |
| `job_1347041` | `registry/checkpoints/job_1347041/` | `runs/paper_fig3_1347041/paper_fig3_model.pt`、训练/Figure-3 metrics |
| `job_1348835` | `registry/checkpoints/job_1348835/` | `runs/full_fourier_ion1_1348835/OTS_i1_model_v2.pt`、`metrics_v2.json`、`run_manifest.json` |

历史注册目录保持原有入口名，例如：

```text
registry/checkpoints/job_1346661/
├── model.pt    -> ../../../runs/bscc_v2_1346661/OTS_Te_model_v2.pt
└── metrics.json -> ../../../runs/bscc_v2_1346661/metrics_v2.json
```

这里的 `model.pt` 只是历史注册项沿用的链接名，不表示 payload 已变成 canonical checkpoint。由于旧 checkpoint 的 schema、数据合同和训练路径不同，canonical `python main.py evaluate` 仍应由严格 loader 拒绝它们；只有新流水线生成并通过当前 schema 的 checkpoint ID 才能作为 canonical evaluate 输入。

迁移步骤：

1. 记录旧 `models/` 的全部目录、`LATEST.txt` 内容、链接文本和解析后目标。
2. 把旧 `models/` 轻量注册树复制到 `remote_backups/<release-id>/models_registry/`；原始 `LATEST.txt` 作为历史证据保留，但不把 “latest” 语义带入新 registry。
3. 把三个旧 job 注册目录的链接迁入 `registry/checkpoints/job_1346661`、`job_1347041`、`job_1348835`；不修改历史 run 目录结构。
4. 对每个迁入链接同时比较迁移前后的链接名称、解析后绝对目标、目标文件大小和已有轻量哈希。
5. 完成链接、大小和已有轻量哈希核验后，旧 `models/` 从活动 registry 退役；实际删除或移动旧目录仍列入执行前确认清单。

历史 checkpoint 迁移不从文件名猜测 `run_id`。历史 dataset 则在第 4.1 节完成物理归并后，按已经存在的三个明确 dataset family ID 迁入 `registry/datasets/`；旧 manifest 若不满足 canonical 合同仍由 loader 正常拒绝，registry 只负责准确定位，不改变 artifact 格式。

### 3.4 BSCC Slurm stdout/stderr 合同

Slurm 在作业脚本 shell 启动前就决定 stdout/stderr 文件，因此不能在 `#SBATCH` 行中依赖 `$OTS_RUN_DIR` 的 shell 展开。提交命令负责先创建目录并传入最终路径：

```bash
cd /home/bingxing2/home/scx6090/yhh/ots

export OTS_RUN_ID=RUN_ID
export OTS_DATASET_ID=DATASET_ID
export OTS_CHECKPOINT_ID=CHECKPOINT_ID
export OTS_RUN_DIR=CURRENT_RUN

mkdir -p "runs/$OTS_RUN_DIR"
sbatch \
  --job-name="$OTS_RUN_DIR" \
  --output="$PWD/runs/$OTS_RUN_DIR/slurm-%j.out" \
  --error="$PWD/runs/$OTS_RUN_DIR/slurm-%j.err" \
  --export=ALL \
  slurm/canonical_all_bscc.slurm
```

`canonical_all_bscc.slurm` 不设置会把日志写到工程根的固定 `#SBATCH --output/--error`；命令行参数是唯一正式日志路径合同。`CURRENT_RUN` 必须是安全的 `runs/` 直接子目录名。

作业成功后才发布 `registry/runs/<run_id>`。作业失败时，该 run 目录保持未注册状态，但 `.out/.err` 和部分产物留在同一目录供诊断；后续可用同一未发布 `run-dir` 显式重试，或在确认后清理。

## 4. BSCC 处理清单

远端历史根：`/home/bingxing2/home/scx6090/yhh/ots`。

### 4.1 更新前全量文件盘点与 legacy run 归并

源码激活、registry 迁移和清理开始前，必须重新检查 `~/yhh/ots` 下的全部文件、目录和符号链接，不能只按已知文件名处理。盘点结果写入 `remote_backups/<release-id>/LEGACY_RELOCATION_MANIFEST.tsv`，每一项至少记录：

```text
原绝对路径
类型（文件/目录/符号链接）
大小与 mtime
链接文本与解析目标（如适用）
关联 Job ID / legacy run
是否被 SHA、manifest、registry 或 Slurm 引用
处置（保留活动区/移入既有 run/备份/待确认删除）
目标绝对路径
移动前后轻量 SHA 或链接核验结果
```

归并规则：

1. 每个文件都必须得到分类；存在未归属项时阻止源码更新，先人工确定其所有者和用途。
2. 只归入**已经存在**且所有权明确的 `runs/<legacy_run>/`，不为了容纳散落文件虚构新的 legacy run。
3. 已经位于正确 legacy run 内的文件保持原位；不为了统一外观重排其内部既有 checkpoint、metrics 或 hash。
4. 根目录或其他活动区中、能明确归属某个 legacy run 的散落文件按用途移入：

   ```text
   runs/<legacy_run>/data/        # 该 run 独占的散落数据
   runs/<legacy_run>/slurm/       # 对应 Job 的 .out/.err
   runs/<legacy_run>/provenance/  # 该 run 专用 Slurm、源码快照、LAST/验证记录
   ```

5. 已有正式映射至少包括：
   - Job `1346661` 相关文件 → `runs/bscc_v2_1346661/`；
   - Job `1347041` 相关文件 → `runs/paper_fig3_1347041/`；
   - Job `1347309` 相关文件 → `runs/paper_fig3_diagnostic_1347309/`；
   - Job `1348835` 及其验证 Job `1348858` 的文件 → `runs/full_fourier_ion1_1348835/`；
   - `1347027/1347038` 的文件只归入各自已经存在的 smoke run，随后仍按清理批次决定是否整体删除。
6. 失败、probe 或自检日志只有在存在明确对应 run 时才归入该 run；否则进入 manifest 的“未绑定清理候选”，不得随意塞入正式 run。
7. 三套历史数据物理归并到其生产 run：
   - `hot_e_large_range_job1346661` → `runs/bscc_v2_1346661/data/`；
   - `paper_10gaussian_v1_seed20260715` → `runs/paper_fig3_1347041/data/`；
   - `full_fourier_ion1_v1_seed20260716` → `runs/full_fourier_ion1_1348835/data/`。
8. 为上述三个明确 dataset ID 创建 `registry/datasets/<dataset_id>`，分别指向归并后的 `data/`；旧 `datasets/` 路径在迁移验证期间保留兼容链接，随后与旧 `models/` 一样退役。历史 manifest 不因迁移而改写成 canonical schema。
9. 旧 `models/` 的轻量注册按第 3.3 节迁入 `registry/checkpoints/`，不作为普通文件塞入 run；其实际 checkpoint/metrics 仍由原 run 持有。
10. 将被新 canonical 替换的共享旧源码先进入 `remote_backups/<release-id>/source_pre_update/`；只有明确属于单个历史 run 的源码或 Slurm 快照才另放入该 run 的 `provenance/`。
11. 移动前后分别检查路径、大小、轻量 SHA 和所有受影响符号链接。若目标已存在：内容相同则只保留 run 内副本并把源列为待确认删除；内容不同则停止，不覆盖。
12. 归并完成后再次全量扫描工程根。除活动源码、`runs/registry/remote_backups/slurm/docs/tests/paper_reproduction/phyEqns` 等明确活动目录及共享配置外，不应再有可归属 legacy run 的散落文件。

这一步是整理历史所有权，不是清空根目录。凭据、活动源码、共享配置和无法证明归属的文件不得强制移动。

### 4.2 更新 / 新增

canonical 活动源码直接更新到：

`/home/bingxing2/home/scx6090/yhh/ots`

不创建新的活动项目根。在现有根目录中：

- 原地更新 `main.py/model.py/train.py/phyEqns/ots_full_fourier.py`；
- 新增 `data.py/evaluation.py`、修正后的 `tests/`、`docs/` 和 `paper_reproduction/`；
- 新增 `registry/runs`、`registry/datasets`、`registry/checkpoints` 三个链接库；
- 将旧 `models/` 的三个历史 job 注册按第 3.3 节合并进 `registry/checkpoints/`；
- 只把 `slurm/canonical_all_bscc.slurm` 作为 canonical active 提交入口；
- `paper_fig3_bscc.slurm` 与 diagnostic 脚本若保留，仅标记为历史复现入口；再次运行前必须另行修复其 self-test 写入正式 run 和重复创建旧式 `models/` 注册的行为；
- 新数据和模型使用全新 ID，不能复用旧 manifest 或旧 `.pt`；
- 新 `run-dir` 固定位于现有根目录的 `runs/RUN_DIR`，不写入旧正式 run。

上传仍先进入现有 `remote_backups/<release-id>/staging/`，通过源码 SHA 后再用逐文件原子替换激活；staging 是临时发布区，不是新的活动源码目录。

原地更新会改变 ion1 `SOURCE_SHA256SUMS.txt` 所记录的旧绝对源码路径。实施前必须把这些旧源码按原字节保存到 `remote_backups/<release-id>/source_pre_update/`，并生成“旧绝对路径 → 备份相对路径”的映射及备份 SHA。原 `SOURCE_SHA256SUMS.txt` 保持不改，但更新后不能再假设它能在原绝对路径直接通过；历史核验改为在备份映射上执行。这是原地部署相对于新建活动目录的明确代价。

BSCC 根目录的 `AGENTS.md` 是旧版受保护文件，agent 不得修改；若用户需要同步，必须由用户本人处理。

### 4.3 明确保留

以下正式数据和产物保持原内容与原哈希；散落数据/日志按第 4.1 节归入既有 run，原引用通过 registry 或迁移期兼容链接保持可解析：

- hot-electron 正式族：
  - 根目录 `dataset_train.parquet`、`dataset_test.parquet` 归入 `runs/bscc_v2_1346661/data/`；
  - `registry/datasets/hot_e_large_range_job1346661` 指向该 `data/`；
  - 正式 stdout/stderr 归入 `runs/bscc_v2_1346661/slurm/`；
  - 原 checkpoint/metrics 文件保持原路径；其旧 `models/.../job_1346661` 注册迁入 `registry/checkpoints/job_1346661`。
- paper Fig. 3 正式族：
  - 论文数据归入 `runs/paper_fig3_1347041/data/`，`registry/datasets/paper_10gaussian_v1_seed20260715` 指向该目录；
  - 正式 stdout/stderr 归入 `runs/paper_fig3_1347041/slurm/`；
  - `runs/paper_fig3_diagnostic_1347309/`；
  - 原 checkpoint/metrics 文件保持原路径；旧 `LATEST.txt` 进入备份，`job_1347041` 迁入 `registry/checkpoints/job_1347041`。
- ion1 正式族，默认整体保留：
  - 数据归入 `runs/full_fourier_ion1_1348835/data/`，`registry/datasets/full_fourier_ion1_v1_seed20260716` 指向该目录；
  - Job `1348835` 和验证 Job `1348858` 的 stdout/stderr 归入 `runs/full_fourier_ion1_1348835/slurm/`；
  - 原 checkpoint/metrics/manifest 保持原路径；旧 `LATEST.txt` 进入备份，`job_1348835` 迁入 `registry/checkpoints/job_1348835`；
  - 专用源码、Slurm 和 LAST/验证记录归入该 run 的 `provenance/`。
- 上述 run 内全部 SHA、manifest、history、metrics、图和正式 checkpoint。
- 诊断 Job `1347309` 的 stdout/stderr 归入其自身 `runs/paper_fig3_diagnostic_1347309/slurm/`。
- `remote_backups/` 至少保留到 canonical 验收和清理确认完成。

现有三套旧式 `models/` 注册经现场检查均有效，因此它们是迁移输入而不是垃圾文件。迁移前完整记录，迁移后通过新旧链接解析到同一原始产物来验收；只有新 `registry/checkpoints/` 和备份均通过后，旧 `models/` 才可从活动区退役。

现有三套 `datasets/` 数据族同样是迁移输入。数据移入 legacy run 后，必须先验证新 `registry/datasets/`、迁移期兼容链接和原 SHA，再退役旧 `datasets/` 活动树。

### 4.4 建议删除：canonical 验收后、用户再次确认后执行

高价值清理：

- `runs/paper_fig3_1347027/`：10 步 smoke，约 123 MiB，无 registry 指向；
- `runs/paper_fig3_1347038/`：1 步 smoke，约 123 MiB，无 registry 指向；
- `runs/paper_fig3_diagnostic_1347306/`：失败作业空目录；
- 根目录 `.python-packages/`：约 215 MiB，仅被误放的 π Slurm 引用；必须等新 BSCC 计算节点环境自检通过后再删。

低容量但高混淆清理：

- 根目录 cache、`__pycache__` 和 `pyc`；
- 旧 active Slurm：`train_v2_bscc.slurm`、误放的 `train_v2.slurm`、旧平铺 `paper_fig3_bscc.slurm`、`paper_fig3_diagnostic_bscc.slurm`；属于单个 legacy run 的先移入该 run 的 `provenance/`，共享脚本进入发布备份，再从 active 根移走；
- smoke、失败、自检或 probe 的顶层日志先按第 4.1 节归入已经存在的对应 run；没有对应 run 的未绑定日志才进入待确认删除清单。涉及 Job ID：
  `1346659/1347027/1347038/1347306/1347364/1348815/1348818/1348822/1348824/1348826/1348831/1348832/1348845/1348846`；
- 旧 `datasets/` 活动树在数据全部归入对应 legacy run、`registry/datasets/` 和迁移期兼容链接验收后退役；只清理旧路径/链接，不删除已迁入 run 的 Parquet 或 manifest；
- 旧 `models/` 注册树在备份和新 `registry/checkpoints/` 逐项验收后退役；原 checkpoint 仍在 `runs/`，这里只清理轻量链接和 `LATEST.txt` 活动副本；
- `LAST_BSCC_*.txt` 有明确 legacy run 所有者时先移入其 `provenance/`；只有已被新 registry 完全替代且不属于保留运行证据的重复指针才列为删除候选。

禁止使用通配符盲删；实施时先把每个候选解析为绝对路径并再次核对。

上述默认清理预计释放约 461 MiB，其中两个 smoke run 约 246 MiB、旧 Python 包约 215 MiB。

### 4.5 单独决策：ion1 整族

默认保留。若用户决定释放约 145 MiB，必须把数据、run、`registry/datasets/full_fourier_ion1_v1_seed20260716`、`registry/checkpoints/job_1348835`、备份中的旧 datasets/models/LATEST、LAST、正式/验证日志、专用源码与 Slurm 作为一个完整证据族处理；不得只删 registry、只删 checkpoint 或只删绝对路径源码。

## 5. π 集群处理清单

本节保留 2026-08-24 审计结论，仅作为后续任务输入；本轮不修改 π 集群，不为 π 创建 registry，也不部署源码或 Slurm。

活动根：`/lustre/home/acct-phywsm/sharon2467/myfolder/OTS_code_share/NN_hotE_large_lmRange`。

### 5.1 更新 / 新增

- staging 验证后，按 allow-list 替换旧 `main.py/model.py/train.py`；
- 新增 `data.py/evaluation.py/phyEqns/ots_full_fourier.py`；
- 新增修正后的 `tests/`、`docs/` 和 paper reproduction 的 Python/报告；BSCC 专用 `.slurm` 不进入 π active 调度目录；
- `phyEqns/ots_fourier.py`、`ots_superGaussian.py` 已与本地哈希一致，可原样保留，仅纳入 manifest 校验；
- active `slurm/` 最终只保留新的 `canonical_all_pi.slurm`；
- 新 canonical run 使用独立 ID 和目录；根目录旧 Parquet 没有 canonical manifest，严禁静默复用；
- 首轮不同步或修改远端 `.git`，也不在远端独立历史上 commit、pull、push、reset 或 gc。

### 5.2 明确保留

- legacy 正式族：
  - `dataset_train.parquet`、`dataset_test.parquet`；
  - `OTS_Te_model.pickle`；
  - `result0.png`–`result3.png`；
  - Job `59149922` 对应 `stdout.txt/errlog.txt`。
- 四个完整 v2 run：
  - `runs/v2_59648208/`；
  - `runs/v2_capacity_59739405/`；
  - `runs/v2_rawloss_59740000/`；
  - `runs/v2_rawloss_59740036/`。
- 四个 v2 正式 stdout/stderr，尤其是含唯一远端哈希记录的后 3 个 stdout。
- `TRAINING_RECOVERY_REPORT.md` 的远端历史版本；更新文档前先归档。
- 独立 `.git` 历史，至少保留到新 canonical 完成验收并由用户另行决定；其 357 MiB 中含旧数据和 checkpoint 的唯一 Git 历史副本。
- 两个中间消融 run `59739405/59740000` 默认保留，它们构成容量与损失权重实验链。

π 当前不存在 `datasets/`、`models/`、`registry/`、`LATEST*` 或独立 SHA/checksum 文件，因此这里没有 registry 或 SHA 文件需要删除、更新或“补齐”。不能据此把 stdout 中的历史哈希删掉。

### 5.3 建议从活动源码区移走；归档后、用户再次确认后删除

旧入口与备份：

- `evaluateModel.py`、`evaluateModel2.py`、`evaluateModel3.py`；
- `genDataSet.py`、`quick_test.py`、`readData.py`、`train_v2.py`、`singleSpectrum.py`；
- `train_v2.py.pre_rawloss_4f17fb4`；
- `phyEqns/ots_arb_gaussian.py`。

旧调度入口：

- `myscript.slurm`；
- `train_v2.slurm`、`train_v2_capacity.slurm`、`train_v2_rawloss.slurm`；
- `install_pyarrow.slurm`；
- 根目录历史 `submit_v2.py/monitor_v2.py`（若实施前复核仍存在）。

重复或临时产物：

- 根目录 `OTS_Te_model_v2.pt`、`metrics_v2.json`、`comparison_v2.png`：已现场确认与 `runs/v2_59648208/` 副本逐字节相同，约 7.8 MiB；
- `quick_test_result.png`；
- 12 个活动源码区 `pyc/__pycache__`，约 129 KiB；
- 失败的 PyArrow 安装日志在归档失败原因后删除。

旧 Slurm 和正式训练日志先移入只读历史归档，不直接删除。根目录 legacy 与四个 v2 正式日志继续保留。

### 5.4 需用户单独确认

- `.git`：默认保留；无 remote 且含唯一旧历史，不自动删除或重写。
- `.venv` 约 12 MiB、`.python-packages` 约 87 MiB：等新 canonical 在 `dgx2` 计算节点 smoke 成功后，可删除以回收约 99 MiB 和约 1,628 个 inode。
- 是否重组根目录数据/checkpoint 到 `datasets/`、`runs/legacy_59149922/`：默认不移动，避免破坏历史路径引用。
- 是否删除两个中间 run `59739405/59740000`：默认保留。

π Lustre 当前整体 inode 使用率约 99%，因此环境/cache 清理的 inode 收益比删除少量日志更重要；但必须在新环境验收后执行。

## 6. 实施顺序

### 阶段 A：本地发布候选

1. 完成第 3 节全部门槛。
2. 完成 registry helper、基于 ID 的输入解析、四命令显式 `run-dir`、新 BSCC Slurm 和相应测试。
3. 运行全部相关测试和 `git diff --check`。
4. 生成精确 allow-list、最终 SHA manifest 和发布 ID。
5. 记录本地 working-tree 状态；不用 HEAD 代替发布 ID。

### 阶段 B：远端变更前快照

1. 再次现场检查 BSCC `squeue`，确认没有作业引用旧源码。
2. 按第 4.1 节全量枚举服务器文件、目录和链接，生成 `LEGACY_RELOCATION_MANIFEST.tsv`；任何未分类项都会阻止继续。
3. 记录待保留 artifact 的路径、大小、链接目标和已有轻量哈希。
4. 完整记录旧 `models/` 三个注册、两个 `LATEST.txt` 和所有符号链接解析目标。
5. 把将被覆盖的旧源码和旧 `models/` 轻量注册复制到 `remote_backups/<release-id>/`，生成备份 SHA 与路径映射。
6. 按 manifest 把有明确既有 run 所有者的散落数据、日志、脚本和指针归入对应 `runs/<legacy_run>/`；更新受影响链接并复核目标/哈希。
7. 再次全量扫描，确认没有遗漏的 legacy-owned 散落文件；未绑定项只能保留、备份或进入待确认删除清单。
8. 解析所有删除候选的绝对路径并提交用户确认。
9. 不重新计算 100 MiB 级大 checkpoint，除非用户另行要求。

### 阶段 C：staging 与静态验收

1. 上传 allow-list 到 BSCC 的临时 staging；暂不覆盖 active。
2. 执行 `sha256sum -c SOURCE_MANIFEST_<release-id>.sha256`。
3. 验证六个 canonical 文件存在、六个 canonical 已删除旧入口不存在。
4. 确认 active 新 Slurm 不引用 legacy Python。
5. 运行 `python main.py --help`，确认 `generate/train/evaluate/all`。
6. 运行全部现有测试、registry 新测试和依赖导入检查。
7. 在 staging 中模拟三 ID 注册、解析、冲突和回滚；不得触碰正式 `runs/`。

### 阶段 D：计算节点 smoke

1. BSCC 使用 `gpu`；不在登录节点训练。
2. 每次提交前创建对应的 `runs/$OTS_RUN_DIR`，并在 `sbatch` 命令中显式传入该目录下的 `slurm-%j.out/.err` 路径。
3. 在全新 smoke run ID 下运行极小 `main.py all`。
4. 检查 pipeline config/summary、manifest、checkpoint 三个 ID、测试隔离、产物路径和 finite 指标。
5. 检查三个新链接分别解析到 `runs/RUN_DIR`、`runs/RUN_DIR/data`、`runs/RUN_DIR/training`。
6. 分别用新的 train run-dir 和 evaluate run-dir 做跨 run smoke：train 不再要求 Parquet/manifest 路径，evaluate 不再要求 Parquet/manifest/checkpoint 路径，但两者都必须显式提供自己的 `run-id/run-dir`。
7. 检查每个 job 的 `.out/.err` 只出现在对应 run 目录，工程根没有新日志。
8. smoke 不写历史 run、旧 `models/` 或旧数据路径。

### 阶段 E：激活与清理

1. 按 allow-list 把 staging 文件逐项原子替换到现有 `~/yhh/ots`；不创建新的活动项目根。
2. 将旧 `models/` 的三个 job 注册按原有入口名和最终目标迁入 `registry/checkpoints/`；只在需要时改写相对链接文本，不改历史 run 目录结构。
3. 再次执行 source hash、CLI、tests、registry 解析和历史 artifact 路径检查。
4. 验证旧 `models/` 与新 checkpoint 链接解析到相同原始产物，再将旧 `models/` 标记为待退役。
5. 取得用户对精确删除路径的确认后，只执行第 4.4 节的 BSCC 清理；π 本轮不动。
6. `.python-packages` 只在 BSCC 计算节点 smoke 成功后删除。

## 7. 验收标准

- BSCC `squeue` 在激活时无引用旧路径的作业。
- 远端 source SHA 与最终本地发布 manifest 全部一致。
- canonical 六模块、全部相关测试和新 BSCC Slurm 均位于现有 `~/yhh/ots` 并可调用；不存在新的嵌套活动项目根。
- active Slurm 中不再出现 `train_v2.py/genDataSet.py/evaluateModel.py/readData.py`。
- `main.py --help` 显示四个 canonical 子命令。
- 全部相关 unittest 在目标依赖环境通过。
- BSCC 隔离 smoke 运行成功，产物合同与 ID 一致。
- `LEGACY_RELOCATION_MANIFEST.tsv` 覆盖更新前发现的每个文件、目录和链接；所有可归属 legacy run 的散落项均已进入既有对应 run，且移动前后目标/哈希核验通过。
- 无法归属既有 run 的项目均有显式保留、备份或待确认删除决定，没有被硬塞进正式 run。
- canonical Slurm 的 stdout/stderr 位于 `runs/<current_run>/slurm-<job-id>.out/.err`；工程根没有新产生的 `.out/.err`。
- `registry/runs/<run_id>` 精确解析为 `runs/RUN_DIR`。
- `registry/datasets/<dataset_id>` 精确解析为 `runs/RUN_DIR/data`。
- 新 canonical `registry/checkpoints/<checkpoint_id>` 精确解析为 `runs/RUN_DIR/training`；历史 `job_*` 注册保留原始 checkpoint/metrics/manifest 目标。
- train 仅由 `dataset_id` 定位既有输入，evaluate/ checkpoint loader 仅由 `dataset_id + checkpoint_id` 定位既有输入；两者的输出均严格写入各自显式的新 `run-dir`。
- generate → train → evaluate 可以使用三个不同的 run ID/目录，evaluate 也允许使用与 checkpoint 训练来源不同的 dataset ID；断链、越界、ID 重绑定或维度/K/波长网格不兼容仍失败。
- `job_1346661/job_1347041/job_1348835` 已从旧 `models/` 注册迁入新 checkpoint 链接库，每个内部链接解析后仍指向原始历史文件；历史 run 中没有人为补建 `training/`。
- 旧 `LATEST.txt` 已进入带 SHA 的备份，新 registry 不提供隐式 latest。
- 所有正式历史文件的内容、大小和轻量哈希与更新前一致；路径变化必须逐项出现在 `LEGACY_RELOCATION_MANIFEST.tsv`，checkpoint/metrics 等已在正确 run 内的核心产物不移动，旧源码 SHA 按第 4.2 节的备份映射核验。
- 没有把 legacy pickle、v2 checkpoint、paper checkpoint 或 ion1 checkpoint 冒充 canonical `training/model.pt`；它们仅作为历史 checkpoint registry 项定位。
- 清理报告列出每个实际删除绝对路径、释放空间、是否可恢复；未获确认的远端数据零删除。

## 8. 回滚

- BSCC 源码：从 `remote_backups/<release-id>/source_pre_update/` 按 allow-list 恢复被替换文件。
- BSCC legacy 归并：按 `LEGACY_RELOCATION_MANIFEST.tsv` 的反向映射恢复原路径和链接；已迁入 run 的数据只在旧路径重新建立兼容链接，不复制大文件。
- BSCC registry：删除本次新建的 canonical 链接，重新启用备份中的旧 `datasets/models` 轻量注册；原 checkpoint 始终未移动。
- π：本轮未修改，无需回滚。
- 新 smoke 和新 canonical run 使用独立 ID，不覆盖历史目录；失败时只清理本轮明确创建的临时/staging 目录。
- 在 BSCC 新版本完成至少一次计算节点 smoke、一次 source hash 和一次 registry 复核前，不清理旧环境、旧源码备份或旧 `models/`。

## 9. 执行前需要用户确认的删除批次

建议将实际删除拆成三个独立确认批次：

1. **低风险重复/临时项**：BSCC cache、pyc、失败空目录、未绑定非正式日志，以及迁移验收后的旧 `datasets/models` 活动链接树。
2. **大空间项**：BSCC 两个 paper smoke run 和旧 `.python-packages`。
3. **历史族项**：BSCC ion1 整族、任何正式数据/checkpoint 重组或删除。

默认只建议执行前两个批次中已经满足验收前提的项目；第三批默认全部保留。
