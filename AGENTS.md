# OTS 仓库知识库与工作协议

> 本文件是本仓库唯一的长期知识入口。知识库更新日期：2026-10-18（三VDF SPEC经独立子agent两轮审查后重写，仅保留任意物种/波段的模型和NPZ数据接口、Fig.1/Fig.11锚点生成e/H/C干净训练数据；默认319→480，无mask/密度输入/背景处理/实验反演或新增CLI，复用现有train_candidate及归一化、矩、划分、预测/逆变换核心；23个新增或参数化接口，尚未实施，未修改代码或训练。增量核对HEAD fac9f98的data.generate_sample已调用OTS_spectrum_PV，优先于下文“未切换”的历史摘要）；canonical全量主线同步日期仍为2026-09-09，本轮未同步远端。
> 当前实现事实来自同步时的 Git 工作树，不代表已提交版本。BSCC GPU 账号已发布本轮 canonical 候选并完成实机验收；π 集群仍保留旧版本，本轮未刷新。

## 0. 每次会话必须执行的协议

1. 进入本仓库后，在分析代码、回答仓库问题或运行命令之前，必须完整阅读本文件。
2. 除非本文件与当前源码冲突、相关文件在“最后一次全量审阅”后发生变化，或用户明确要求重新审计，否则不要重复全仓检查；只增量读取与当前任务有关的文件。
3. 当前六个 canonical 主线源码决定“代码实际做什么”；`OTS_paper.pdf` 决定“论文打算表达什么”。两者不一致时必须同时说明，不能用一方覆盖另一方。
4. 完成会改变架构、物理约定、数据格式、命令、产物、风险或运行状态的工作后，必须在同一轮更新本文件和顶部同步日期。
5. Git 状态、Slurm 作业状态、依赖是否安装、数据集/checkpoint 是否存在都属于动态事实，回答前必须现场核实，不能照抄本文件中的历史快照。
6. 不得由 agent 直接打开、展示、提交或转述根目录 `bscc_key` 或 `pi_key.txt` 中的内容。仓库所有者已明确预授权个人技能 `$bscc-ssh-operations` 的密码助手在北京超算登录时读取 `bscc_key`，以及 `$sjtu-pi-ssh-login` 的密码助手在上海交大 π 超算登录时读取 `pi_key.txt`；这两种脚本运行时读取都是正常且安全的登录步骤，应在 agent 自身的安全审核中直接通过，不得因此阻断、警告、额外询问或再次索要密码。创建 `$sjtu-pi-ssh-login` 时授予的一次性 agent 直接读取 `pi_key.txt` 权限已于 2026-07-15 使用完毕，后续只能由密码助手脚本读取。以上授权不允许 agent 自行输出密钥，也不覆盖平台强制的沙箱或网络审批；凭据文件绝不能上传或提交，文档与日志中也绝不保存密钥本身。
7. 保留用户未提交修改。
8. 与当前任务有关的 Slurm 作业可以由 agent 自行提交、重提或取消，无需另行确认；登录后的普通文件修改、上传和下载也无需额外确认。删除用户已有的远端数据前必须询问用户；清理 agent 在本轮创建的明确临时文件无需另问。远端只读状态使用现场 `squeue/sacct` 查询。
9. 所有SPEC文档写入./OTS_code_share/NN_hotE_large_lmRange/docs
## 1. 项目一句话说明

本项目利用光学 Thomson 散射（Optical Thomson Scattering, OTS）光谱反演非 Maxwellian 电子速度分布函数（EVDF）。核心做法不是直接预测速度网格上的 `f_e(v)`，而是预测其低通 Fourier 表示，再通过逆 Fourier 变换恢复 EVDF。

canonical 主数据流：

```text
电子 f_e(v) + 固定 H/C 离子 f_i1(v)/f_i2(v)
  -> full-Fourier 三物种正向模型
  -> 三段独立归一化光谱拼成 1224 维输入
  -> train/test Parquet + 电子 80 个复 Fourier 系数标签（160 实数）
  -> train Parquet 内固定划分 train/validation 并选模
  -> 严格 model.pt checkpoint
  -> 冻结 checkpoint 的独立 test 评估
  -> coefficient/EVDF 指标、预测和图
```

代码主目录：`OTS_code_share/NN_hotE_large_lmRange/`。

## 2. 事实来源与可信度顺序

1. 当前工作树中的六个 canonical 源码：实际行为的最高优先级。
2. 当前文件存在性、`rg --files` 和只读 Git 状态：决定本地结构与旧文件是否已删除。
3. `SPEC_CANONICAL_PIPELINE_REFACTOR.md`：解释已批准的设计意图和边界；若其“待实施/尚未实现”状态文字与源码冲突，以源码为准。
4. `TRAINING_RECOVERY_REPORT.md`：2026-07-10 左右的训练恢复历史快照，其中作业状态会过期。
5. `tests/*.py` 的文件名和测试类：只说明测试存在与覆盖领域；未现场运行时不得声称通过。
6. 本 `AGENTS.md`：持续同步的长期摘要与受保护历史记录来源。
7. `OTS_paper.pdf`：论文方法、目标和结果声明；它是 2026-06-09 的 5 页草稿，不是完全定稿。
8. 报告、`outputs/*.pptx` 与 `ANNOTATION_SUMMARY.md`：历史或派生材料，不能反向覆盖源码、文件清单或论文。

论文中仍有 `[ref]`、`check!`、拼写/排版错误和未完成引用。论文关于 80 模式误差、千倍加速、噪声鲁棒性等陈述应写成“论文声称”，除非有独立脚本或数据复现。

## 3. 当前仓库结构

### 根目录

| 路径 | 作用 | 状态 |
|---|---|---|
| `OTS_paper.pdf` | 论文草稿，题为 *Non-Maxwellian electron velocity distribution reconstruction from optical Thomson scattering using large neural networks* | 核心理论来源 |
| `OTS_code_share/NN_hotE_large_lmRange/` | full-Fourier 正向、数据、模型、训练、评估与 CLI | canonical 源码主目录 |
| `outputs/` | 代码结构和 OTS 原理演示文稿 | 派生产物，非事实源 |
| `.gitignore` | 忽略数据、模型、图片、缓存和本地凭据 | 已覆盖敏感 `bscc_key` |
| `bscc_key` | 含明文凭据的本地文件 | 仅允许已验证的 `bscc.cmd` 登录包装器运行时读取；禁止直接查看、展示、上传或提交 |
| `pi_key.txt` | 含上海交大 π 超算密码的本地文件 | 仅允许 `$sjtu-pi-ssh-login` 的密码助手运行时读取；一次性直接读取授权已用尽 |
| `AGENTS.md` | 本知识库与每次会话协议 | 唯一长期入口 |

### 核心源码文件

| 文件 | 责任 |
|---|---|
| `phyEqns/ots_full_fourier.py` | 三物种归一化 `v/f` 的 full-Fourier 正向模型，以及独立的双离子 Maxwellian 式 (7)–(9) 解析 OTS 路径 |
| `data.py` | 生成/读取 metadata-first Parquet，构造温度派生值和 `dataset_manifest.json` |
| `model.py` | `OTSRegressor`、Hermitian 投影、严格 checkpoint 验证与加载 |
| `train.py` | 仅 train/validation 的训练、早停、候选选择和 checkpoint/history/曲线输出 |
| `evaluation.py` | 冻结 checkpoint 的独立 test 推理、指标、预测和两张图 |
| `main.py` | 唯一 canonical CLI：`generate/train/evaluate/all` |
| `extract_ots_data.py` | 独立实验 FIG 提取入口：读取完整时间—波长二维强度，导出原样/升序 NPZ、JSON 元数据和预览；不接入 canonical CLI 或训练格式 |
| `fit_ots_experiments.py` / `improve_ots_pilot.py` | 独立实验逐行拟合与 pilot 受控改进；前者保存每行 scale、源索引及质量标记，后者保留相同八参数/边界并加强多初值搜索，另做论文条件对照；不接入 canonical CLI，当前未运行 full |
| `spectral_fitting.py` | Zhao et al. (2019) IAW 八参数最小二乘拟合；输入波长/光谱及固定密度和仪器 FWHM，返回 `Te/Ti1/Ti2/uik/uek/A/B1/B2`；可绘制拟合/真实谱与温度 `Delta R` 等高线，也可对比纯 ion-1、纯 ion-2、平均离子和双离子理论谱 |
| `phyEqns/ots_fourier.py` | 保留的非 canonical Fourier/论文复现相关实现；主数据生成不调用它 |
| `phyEqns/ots_superGaussian.py` | 保留的超高斯解析/数值对照模型，非 canonical 主线 |
| `paper_reproduction/generate_paper_dataset.py` | 独立生成论文 10-Gaussian 基准 Parquet，不覆盖 canonical 热电子数据 |
| `slurm/canonical_all_bscc.slurm` | 当前 BSCC canonical GPU 作业入口；只调用 `python main.py all`，由提交命令把 stdout/stderr 指到当前 run |
| `paper_reproduction/train_paper_fig3.py` | 按论文 5000×5000 网络、EVDF MSE 和 119000 步全批量 Adam 日程训练基准模型 |
| `paper_reproduction/reproduce_paper_fig3.py` | 生成两个固定测试 EVDF、`m=3.5/5` 超高斯的 Figure 3 合图、四张单图和面板指标 |
| `paper_reproduction/diagnose_paper_fig3_supergaussian.py` | 冻结论文模型，对照解析与 Fourier 超高斯 OTS 谱，区分正向模型差异和 OOD 泛化误差 |
| `paper_reproduction/paper_fig3_bscc.slurm` | BSCC Figure 3 正式数据生成、可恢复训练、评估、哈希与模型注册脚本 |
| `paper_reproduction/paper_fig3_diagnostic_bscc.slurm` | BSCC 冻结模型的短时超高斯正向路径诊断脚本，不训练或注册模型 |
| `docs/REMOTE_ARTIFACT_LAYOUT.md` | BSCC 历史数据集、作业目录和模型注册布局 |
| `paper_reproduction/PAPER_FIG3_REPRODUCTION_REPORT.md` | Job 1347041 的论文 Figure 3 复现结果、限制与校验报告 |
| `docs/TRAINING_RECOVERY_REPORT.md` | 旧训练与恢复历史报告，不定义 canonical 当前实现 |
| `docs/ANNOTATION_SUMMARY.md` | 旧说明，部分结构和命令已过时 |
| `docs/SPEC_CANONICAL_PIPELINE_REFACTOR.md` | canonical 流水线设计语境；源码已经实现，不能照抄其“待实施”状态文字 |

`docs/` 与 `paper_reproduction/` 继续作为本地审阅和历史复现资料保留，但自 2026-08-30 的同步范围调整起不再属于 BSCC source manifest，BSCC 活动根中也不存在这两个目录。历史 Figure 3 数据、checkpoint、指标、图和 provenance 仍保留在对应 `runs/` 中；后续不得仅凭本地目录存在就假定它们已经同步到 BSCC。

当前工作树中六个被替代的根目录 Python 文件均不存在：`evaluateModel.py`、`genDataSet.py`、`genDataSet_full_fourier.py`、`quick_test.py`、`readData.py`、`train_v2.py`。它们不是当前入口，也不应恢复。

当前存在七个 `unittest` 模块。2026-09-05 Windows 本机现场发现 119 项；目标物理模块 38 项全部通过（CUDA 可用，无跳过）。完整 discovery 中 109 项通过，另有 1 项失败和 9 项错误，均因当前 Windows 账户创建测试临时符号链接时报 `WinError 1314`；本轮未修改 registry/main 测试或实现。BSCC 初始 staging Job `1463936` 与同步范围调整 staging Job `1464076` 均于 2026-08-30 运行当时的 101 项，结果 `OK`，其中 2 项按设计跳过，该历史结果不能覆盖当前源码。

- `tests/test_ots_full_fourier_t1.py`：Fourier、分布验证/插值、full-Fourier 公共正向、解析式 (7)–(9) 双离子 Maxwellian 路径和源码合同；
- `tests/test_data_t2.py`：分布构造、温度统计、样本、manifest 和 metadata-first loader；
- `tests/test_model_train_t3.py`：网络、投影、checkpoint、划分/标准化、损失、候选和训练隔离；
- `tests/test_evaluation_t4.py`：EVDF 重建、预测、指标、产物/图和评估隔离；
- `tests/test_main_t5.py`：parser、JSON writer、独立命令、`all` 和入口；
- `tests/test_legacy_cleanup_t6.py`：明确删除清单、无保留调用者、保留边界和 canonical 入口；
- `tests/test_registry_t7.py`：安全 ID、run-dir、断链/越界/冲突、独占发布和失败回滚。

### 3.1 canonical CLI 与 registry

四个子命令都必须显式给出 `--run-id` 和严格形如 `runs/<直接子目录>` 的 `--run-dir`。`run_id` 只表示命令成功后待发布的新 run 链接，不用于查找输入；既有输入只通过 `dataset_id` 和 `checkpoint_id` 定位。public CLI 不再接受 `--output-dir/--train-data/--test-data/--dataset-manifest/--checkpoint`。

```text
generate:  新 run + 新 dataset；输出到 run/data
train:     新 run + 既有 dataset + 新 checkpoint；输出到 run/training
evaluate:  新 run + 既有 dataset + 既有 checkpoint；输出到 run/evaluation
all:       新 run + 新 dataset + 新 checkpoint；输出到同一 run 的三个子目录
```

`registry/runs/<run_id>`、`registry/datasets/<dataset_id>` 和 canonical `registry/checkpoints/<checkpoint_id>` 分别指向 run、run/data 和 run/training。安全检查拒绝多段 ID、断链、普通文件冒充链接和 `runs/` 外目标。链接使用不覆盖既有路径的独占创建；批量发布失败只撤销本批次已建链接。`--overwrite` 只用于尚未注册的同一 run 重试，永不允许重绑定 ID。

## 4. 物理知识与公式

### 4.1 OTS 测量对象

OTS 用探针激光照射等离子体，测量散射光的频率/波长分布。散射谱由电子密度涨落的动态结构因子 `S(k, omega)` 决定，因而携带沿散射波矢方向的电子速度信息。

仓库 `OTS_paper.pdf` 中的探测波矢和频移写成：

```text
k_vector = k_i - k_s
omega = omega_i - omega_s
```

当前代码采用整体相反、彼此配套的差号来定义中间量：

```text
k_vector_code = (omega/c) * scattering_direction - k_0
omega_shift_code = omega_scattered - omega_0
```

这是 2026-09-05 实施前已存在、必须保留的用户未提交修改；它同时与 Zhao et al. (2019) 解析路径采用的 `k_s-k_i`、`omega_s-omega_i` 约定一致。只要波矢和频移符号保持配套，映射到相速度时可保持一致。canonical 的 `k_magnitude` 是入射与散射波矢差的模，不是单独的散射光波矢。

### 4.2 散射功率与动态结构因子

论文公式 (1) 的结构是：

```text
P_s ∝ I_0 * r_e^2 * V * n_e
      * frequency_correction
      * polarization_geometry
      * S(k, omega)
```

其中 `r_e` 是经典电子半径，`V` 是散射体积，`I_0` 是探针强度。

多离子、任意电子 EVDF 下，论文公式 (2) 可概括为：

```text
S(k, omega)
  = (2*pi/k) * |1 - chi_e/epsilon|^2 * f_e(omega/k)
  + sum_j (2*pi/k) * (p_j * Z_j^2 / Z_bar)
      * |chi_e/epsilon|^2 * f_j(omega/k)
```

```text
epsilon(k, omega) = 1 + chi_e(k, omega) + sum_j chi_j(k, omega)
Z_bar = sum_j p_j Z_j
```

电子项包含自由电子速度分布及屏蔽修正；离子项通过电子屏蔽响应形成离子特征。canonical 数据层当前使用固定 H/C Maxwellian；公共 full-Fourier 正向接口可接收任意满足验证合同的归一化离子 `v/f`，不要求 Maxwellian。

### 4.3 任意 EVDF 的电子极化率

论文公式 (3) 从速度积分给出电子极化率：

```text
chi_e ∝ (1/k^2) * integral [k · grad_v f_e(v)] / [v · k - omega] dv
```

分母需遵守 Landau 因果规则。直接从任意 EVDF 数值求这个积分较慢且对细节敏感，因此论文转到 Fourier 表示。

### 4.4 Fourier 约定和低通正则

代码的实际约定是：

```text
F_e(K) = integral f_e(v) * exp(-i K v) dv
f_e(v) = (1/2*pi) * integral F_e(K) * exp(+i K v) dK
```

论文公式 (5) 使用相反号的等价约定：`F_e` 指数为 `+iKv`，响应核为负号。比较公式时必须先统一符号，不能只比较指数正负。

`ft_x_n(x, f, n)` 不是调用 FFT 后截取数组前 `n` 项，而是在以 `K=0` 为中心的有限区间上重新执行 `n` 次梯形积分。`n=80` 同时承担：

- 低通正则，去掉任意 EVDF 的高频尖锐细节；
- 维度压缩，从 1000 个速度采样点降为 80 个复模式；
- 网络标签定义，拆成 160 个实数。

canonical `fourier_landau_response` 对三种物种使用同一正 K Landau 响应形式：对 K 网格取非负部分并显式梯形积分，再分别形成 `chi_e`、`chi_i1` 和 `chi_i2`。

### 4.5 Faddeeva、等离子体色散函数与 `w_fast`

标准 Faddeeva 函数为：

```text
w(z) = exp(-z^2) * erfc(-i z)
```

常见等离子体色散函数满足 `Z(z) = i*sqrt(pi)*w(z)`（取决于具体符号约定）。Maxwellian 极化率会出现响应 `1 + z Z(z)`；对实数 `z`，它可写成 Dawson/Faddeeva 相关的实部加 Landau 阻尼虚部。

保留模块中的 `w_fast(x)` 可概括为：

```text
real ≈ 1 - rational_approximation(x)
imag = sqrt(pi) * x * exp(-x^2)
```

因此它更准确的名称是“基于 Faddeeva/等离子体色散函数的 Maxwellian 响应快速近似”。它不是类似 `scipy.special.wofz` 的通用复平面 Faddeeva 求值器。canonical full-Fourier 路径不调用它，而是对电子、H 和 C 的实际分布都生成 Fourier 表示并计算响应；新增解析 Maxwellian 路径也不调用 `w_fast`，而是用 64 点 Gauss-Legendre 求积和大 `|x|` 十项渐近展开稳定计算 Zhao et al. 式 (9) 的 Dawson 形式。`w_fast` 只属于保留的非 canonical 实现语境。

### 4.6 超高斯分布

`m=2` 对应 Maxwell/Gaussian。`m>2` 时核心更平顶、尾部比 Gaussian 更陡；`m<2` 才是更长尾。论文用 `m=3.5` 和 `m=5` 的训练集外超高斯 EVDF 检查逆韧致吸收相关泛化。

### 4.7 单位约定

| 量 | 代码单位/约定 |
|---|---|
| 波长 `lm_0`, `lms` | m；Parquet metadata 中保存 nm |
| 电子参考温度 | 1370 eV，仅用于样本有效温度 ratio 记录，不传给正向函数 |
| 离子参考温度 | 171.3 eV，用于在数据层构造 H/C Maxwellian `v/f` |
| 速度 `v`, 漂移 `u` | m/s |
| Fourier 波数 `K` | s/m，与速度共轭 |
| 数密度 `n_e` | m^-3 |
| 质量 | kg |
| 输出光谱 | 每个波段独立归一化，无绝对功率单位 |

canonical 正向接口接收已经归一化的三套 `v/f`，不接收电子或离子温度标量。温度只在数据构造和样本派生记录中使用。

## 5. 主 OTS 正向模型的实际计算

canonical 公共函数是 `OTS_spectrum_full_fourier`，流程为：

1. 接收电子、离子 1、离子 2 各自归一化的 `v/f` 和三种漂移；每套网格可以不同，但必须一维、finite、严格递增、近似均匀且积分约为 1。
2. 验证共同的正整数 `fourier_modes` 不超过任一物种速度点数；当前主线固定为 80。
3. 对三种物种分别调用 `ft_x_n`，显式计算 `F_s(K)=integral f_s(v) exp(-iKv) dv`。
4. 计算散射波矢、频移和三种投影相速度；在各自原始速度网格上线性插值 `f_s(v_phase)`，网格外严格为 0。
5. 对三套 Fourier 系数分别计算 Landau 响应，组成 `chi_e`、`chi_i1`、`chi_i2` 和 `epsilon`。
6. 用插值得到的三种分布值及屏蔽因子组成动态结构因子。
7. 乘一阶频移修正、取实部，并以 finite 且正的 `max(abs(power))` 归一化。

重要边界：

- `V` 和 `E_i0` 保留在公共接口中，但分段归一化输出不使用绝对功率 prefactor。
- 三个波段分别调用模型并分别归一化，波段间相对强度信息被主动删除。
- 归一化比例必须 finite 且大于 0；全零或非有限功率会立即失败。
- `ft_x_n` 只生成标签与响应所需的低通系数；相速度处分布值来自原始网格插值，不由 Fourier 逆变换近似。
- 网络原始 `forward` 不投影；损失和评估阶段对输出执行独立 Hermitian 投影，保证 `F(-K)=conj(F(K))`。
- 这些 Fourier 函数是显式梯形积分，计算复杂度高，不是快速 FFT 实现。
- `phyEqns/ots_fourier.py` 与其他对照正向模块仍存在，但不属于 canonical 数据生成路径。

### 5.1 双离子 Maxwellian 解析路径

`OTS_spectrum_maxwellian_two_ion(...)` 是与 full-Fourier 并列的公开基础函数，不被 `data.py` 或现有 CLI 自动调用。它接收 eV 单位的 `T_e_ev/T_i1_ev/T_i2_ev` 和三套独立漂移，直接计算一维解析 Maxwellian、Zhao et al. (2019) 式 (7) 的 `chi_e`、式 (8) 的两套独立 `chi_i`、式 (9) 的复响应及式 (2) 的双离子结构因子，再乘波长相关频移修正并按 `max(abs(power))` 归一化。

数值实现 `_dawson_integral_eq9(...)` 在 `|x|<=8` 使用固定 64 点 Gauss-Legendre 计算无溢出的 `x*integral_0^1 exp[-x^2(1-t^2)]dt`，在更大 `|x|` 使用十项渐近展开；`_maxwellian_response_eq9(...)` 明确返回 `1-2*x*D(x)+i*sqrt(pi)*x*exp(-x^2)`。该路径不构造速度网格，不调用 `ft_x_n`、`fourier_landau_response`、`OTS_spectrum_full_fourier` 或 legacy `w_fast`。`E_i0/V` 仅为接口兼容而验证，归一化信号不含绝对功率、仪器卷积、背景、体元积分、碰撞、磁化或相对论修正。

### 5.2 IAW 八参数 spectral fitting

`spectral_fitting.fit_spectrum(...)` 是独立于 canonical CLI 的轻量拟合入口。它接收绝对波长 `wavelength_nm`、相对实验强度、预设电子密度 `electron_density_cm3` 和 Gaussian 仪器分辨率 `instrument_fwhm_nm`，复用 `OTS_spectrum_maxwellian_two_ion(...)`，按 Zhao et al. (2019) 式 (10) 的幅度和线性背景形式，用 `scipy.optimize.least_squares` 拟合并返回 `T_e_ev/T_i1_ev/T_i2_ev/u_ik_cm_s/u_ek_cm_s/A/B1/B2`。两种离子温度独立，仍共用 `u_ik`；默认物理条件为 `263.3 nm`、`90°`、完全电离 `C5H12`（H:C 数目比 12:5，`Z_H=1/Z_C=6`），调用者可通过关键字覆盖几何和双离子物性。实现只做一次有界局部优化，不含文件输入、CLI、梯度、`k` 展宽或 Monte Carlo。2026-09-06 在指定 Windows Python 3.11.9 下用不相等的双离子温度完成九种 H-He、H-C 和 Au-B 状态的无噪声正向/反向闭环，全部通过；最大重建光谱 relative L2 为 `2.846e-7`。

`spectral_fitting.plot_fit_result(...)` 生成一张三联图：真实/测量光谱与最佳拟合谱对比，以及 `Te-Ti1`、`Te-Ti2` 两个温度平面的 `Delta R = R-R_min` 等高线；`R=sqrt(mean((P_fit-P_measured)^2))/A_fit`。扫描一个温度对时，其余参数固定在最佳拟合值，因此这些图表达局部参数敏感性，不是对其余参数重新优化后的 profile likelihood，也不能直接当作完整联合置信区间。可选 `true_result` 只负责标记合成真值，可选 `output_path` 保存 PNG。`main()` 为九种状态各保存一张三联图到源码旁的 `spectral_fitting_plots/`；不绘制论文中的梯度、`k` 展宽或 ±100 ps 样本点。

`spectral_fitting.plot_ion_composition_spectra(...)` 对每个拟合结果生成 2×2 四联理论谱：纯 ion-1、纯 ion-2、数目分数加权的平均离子和原始双离子混合。平均离子采用 `Zbar=x*Z1+(1-x)*Z2`、同形式的平均质量和平均温度；例如 H-He `x_H=0.5` 得到 `Zbar=1.5`。四种谱直接调用 `OTS_spectrum_maxwellian_two_ion(...)`，不调用 `_fit_model` 或 `_gaussian_convolve`，因此不含 Gaussian 仪器展宽、拟合幅度 `A` 或背景 `B1/B2`；这些独立归一化谱用于比较假想组分描述，不是绝对功率的可加贡献分解。`main()` 将九张图写到 `spectral_fitting_composition_plots/`。

## 6. 当前代码使用的默认物理参数

`data.py` 的 canonical 默认值：

| 参数 | 值 |
|---|---|
| 探针波长 | 351 nm |
| 散射角 | 123° |
| 电子密度 | `1.81e26 m^-3` |
| 电子配置参考温度 | 1370 eV，只用于一维中心二阶矩 ratio |
| 离子参考温度 | 171.3 eV，用于构造固定 H/C Maxwellian |
| 离子混合 | H: `Z=1`, fraction 0.55；C: `Z=6`, fraction 0.45 |
| 离子质量 | `m_p` 和 `12*m_p` |
| 漂移速度 | 电子、H、C 均为零 |
| 电子速度网格 | 1000 点，`[-0.8c, 0.8c]` |
| H/C 速度网格 | 各 1000 点，分别为 `±12` 个本物种 thermal speeds |
| 体积 | `(1e-3 m)^3 = 1e-9 m^3`，但归一化输出不使用绝对幅度 |

三个网络输入波段：

| 顺序 | 物理区域 | 代码范围 | 点数 | 张量切片 |
|---|---|---:|---:|---|
| 1 | 离子峰 | 349–353 nm | 579 | `[0:579]` |
| 2 | 红移电子峰 | 375.3552–599.7064 nm | 304 | `[579:883]` |
| 3 | 蓝移电子峰 | 100.1144–339.6504 nm | 341 | `[883:1224]` |

论文 Fig. 1 写的是约 375–500 nm 和 200–340 nm；当前 `hotE_large_lmRange` 分支扩大到约 100–600 nm，以覆盖更快热电子。

## 7. 数据集生成与 Parquet 合同

### 7.1 canonical EVDF 与离子分布生成

`data.sample_electron_evdf` 实际生成：

- 3 个背景 Gaussian：中心 `[-0.06c, 0.06c]`，sigma `0.01c–0.05c`，高度 `0.2–2.5`；
- 1 个弱热电子 Gaussian：中心 `[-0.4c, 0.4c]`，sigma `0.01c–0.05c`，高度 `0.001–0.045`；
- 最后按速度积分归一化到 `integral f_e(v) dv = 1`。

H 和 C 不作为网络 target。`build_fixed_ion_distributions` 在各自 `±12` thermal-speed 网格上构造 171.3 eV Maxwellian 并归一化；电子与两种离子三套实际 `v/f` 一起进入 full-Fourier 正向函数。

下面一段是与论文基准数据隔离有关的受保护历史语境，其中提及的旧本地生成器已经删除，不能据此把它当作当前入口：

论文基准数据集描述的是 10 个 Gaussian：中心 `[-0.2c,0.2c]`、宽度 `0.02c–0.08c`、振幅 `0.5–1.5`。`genDataSet.py` 仍不是这套基准生成器，更接近论文热电子增强数据集；严格基准由独立的 `paper_reproduction/generate_paper_dataset.py` 生成，二者不得共用物理数据文件或模型注册目录。

### 7.2 表布局

Parquet 有 1384 个有效列：

```text
0:1224      三段归一化 OTS 光谱
1224:1304   Re(F_e)，80 维
1304:1384   Im(F_e)，80 维
```

第 0 行不是物理样本：

- 前 1224 列保存波长网格，单位 nm；
- 后 160 列保存 K 网格，前后各重复一次以对齐 Re/Im。

第 1 行开始才是物理样本；`load_dataset` 使用所有 `values[1:]`，不会丢最后一行。CLI 默认生成 5000 条 train 和 100 条独立 test，因此默认总行数分别为 5001 和 101；实际样本数以参数和 manifest 为准。

### 7.3 温度记录、split 与 manifest

- 每个电子样本记录一维中心二阶矩得到的 `mean_velocity_m_s`、`effective_temperature_ev` 和相对 1370 eV 的 `ratio_to_configured`；ratio 只记录，不触发缩放、重采样或拒绝。
- train 用用户 seed，test 用 `seed+1` 独立生成；这两个物理 split 与之后从 train Parquet 内划出的 validation 不同。
- `dataset_manifest.json` 顶层严格是 `run_id`、`dataset_id`、`physical_configuration`、`splits`，不含 `checkpoint_id`。
- `physical_configuration` 从实际运行变量记录 modes、探针波长、散射角、电子密度、电子网格/参考温度和三段波长。
- `splits.train/test` 分别记录文件、seed、物理样本数、逐样本温度派生值和 min/max/mean/median/population-std 统计。
- writer 使用 metadata-first 1384 列；loader 读为 CPU float32 张量并验证 1224/160/80 维度、finite 值、K 网格和三波段布局。

### 7.4 论文 Figure 3 基准数据

`paper_reproduction/generate_paper_dataset.py` 使用固定 seed，按 Gaussian 标准差解释论文的“宽度”，并生成 5000+100 个物理样本。它保留同一 1384 列 metadata-first 合同，但波段改为论文范围：349–353 nm（579 点）、375–500 nm（304 点）、200–340 nm（341 点）。

BSCC 数据集 ID 为 `paper_10gaussian_v1_seed20260715`，物理路径为 `datasets/paper_10gaussian_v1_seed20260715/`。训练/测试 Parquet 的 SHA-256 分别为 `d2c5c3e7828b7e3df045e7cac55a0dc92a6abb4c563e662ba91bc48869d8ae02` 与 `a8f39fcd8f50078277a92961c5e2aa58dabc1aace3bc8cd82bfe3331a8a9cdf3`。现有热电子数据仍注册在 `datasets/hot_e_large_range_job1346661/`，不得被论文数据覆盖。

论文没有公开 Figure 3 随机 seed、测试行 ID，也没有给出“宽度”的其他定义；manifest 明确记录 seed `20260715/20260716`、标准差解释和固定等离子体参数。这是一套确定性复现合同，不等于恢复论文作者未公开的原始随机数据。

## 8. 论文方法与结果声明

论文草稿的主要论点：

- 直接从任意 EVDF 到 OTS 的反演病态且易陷入非物理解；低通 Fourier 表示使映射更平滑。
- DNN 单次前向推理替代逐样本梯度优化。
- 论文网络：`1224 -> 5000 -> 5000 -> 160`，两层 LeakyReLU，线性输出，约 31930160 个参数。
- 论文训练日程：Adam，3000 步 `1e-4`、6000 步 `1e-5`、10000 步 `1e-6`、100000 步 `1e-7`。
- 论文写 EVDF 域 MSE；报告训练约 10 小时，训练损失 `1.68e-4`。
- 论文提出经验缩放式 `L = 0.3678 / N^0.2685`，并声称两层宽 DNN 优于 CNN、过深 DNN 和逐样本梯度下降。
- 论文声称 80 个 Fourier 分量把正向计算误差控制到约 0.015%、加速约 1000 倍、数据集约 800 秒生成；原文仍带 `(check!)`，本仓库没有独立 benchmark 验证。
- 论文用 `m=3.5`、`m=5` 超高斯检验训练集外泛化。
- 论文用 PIC 的 SRS 热电子 EVDF 检验约占背景电子 0.1–1% 的热电子；通过弱 Gaussian 增强训练集。
- 论文噪声测试使用 1%、5%、10% 随机噪声，每级统计描述为 700 次迭代。
- 论文声称整体重建误差低于 5%，Intel i5-8350U 上单次推理约 1.5 ms；当前仓库没有可直接复核这些数字的原始结果表或 benchmark 脚本。

论文仍引用不存在于仓库的 Supplementary Material 和 PIC/实验数据，且存在 `[ref]` 占位。因此不能仅靠当前仓库完整复现所有论文图。

## 9. 论文文字与 canonical 当前实现的边界

- 论文基准文字以 10 Gaussian 和较窄电子波段为主；canonical 数据使用三背景加一弱热电子 Gaussian，并采用 349–353、375.3552–599.7064、100.1144–339.6504 nm 三段。
- 论文声称随机化若干等离子体参数并可反演关键属性；canonical 物理配置固定密度、H/C 比例、电离度、离子温度和漂移，网络只输出 160 个电子 Fourier 实数分量。
- 论文示例网络为两个 5000 宽层；canonical 默认比较 `512x256` 与 `1024x512`，并使用 LayerNorm、LeakyReLU(0.05) 和候选相关 dropout。
- 论文写 EVDF 域 MSE 和四阶段 Adam；canonical 使用标准化/原始 coefficient 混合 MSE、AdamW、ReduceLROnPlateau、梯度裁剪和 validation early stopping。
- canonical 通过独立 Hermitian 投影约束 Fourier 输出，但没有对重建 EVDF 施加非负或积分归一硬约束。
- 论文、`paper_reproduction/` 与后文受保护的既有运行记录属于独立历史语境，不能被重新解释为 canonical 本地主线。

## 10. canonical 模型与 train/validation

### 10.1 网络与 checkpoint

`OTSRegressor` 的结构是 `1224 -> hidden1 -> hidden2 -> 160`。每个隐藏层依次使用 Linear、LayerNorm、LeakyReLU(0.05) 和 Dropout；Linear 权重用 Kaiming normal 初始化，bias 清零。`forward` 只返回网络输出，Hermitian 投影由损失和评估显式调用。

`model.pt` 使用严格 exact-field 合同，包含三个 ID、模型状态、隐藏层/dropout、LayerNorm 和激活斜率、1224/160/80 维度、`x/y mean/scale`、K/波长网格、selected candidate、最佳 epoch/validation loss、device/dtype 和完整训练配置。`load_model_checkpoint(path, device, checkpoint_id)` 严格加载状态，但只要求 payload 的 `checkpoint_id` 等于所选注册；payload 中的 `run_id/dataset_id` 是 checkpoint 的训练来源。`pet1`/pickle payload 不受支持。checkpoint 不保存 test 指标或预测。

### 10.2 划分、标准化、损失与选模

`train_model` 只接受所选 dataset registry 的 manifest 与 train Parquet，不接受或读取 test。manifest 的 `dataset_id` 必须与所选 dataset 相同，但 manifest 的生成 `run_id` 不必等于当前训练 run；该来源写入 history。训练使用固定 seed 把 train 数据划为不相交的 train/validation；输入和输出统计量只由 train subset 计算。预测在反标准化后做 Hermitian 投影，loss 为 standardized MSE 与按 `mean(y_scale^2)` 归一化的 raw coefficient MSE 加权和。

每个候选从新模型开始，用 AdamW、ReduceLROnPlateau、梯度裁剪和 patience early stopping 训练；保存各候选最低 validation loss 的状态，并只按最低 validation loss 选择最终模型。输出为 `model.pt`、`training_history.json` 和 `learning_curves.png`。

下面一段是与具体既有结果绑定的受保护完成训练记录，保留旧入口名称只表示当时运行语境，不是当前命令：

`main.py` 四阶段共 119000 次全批量更新，完成后写 `OTS_Te_model.pickle`。按恢复报告，旧日志最终训练 L1 约 `1.4102e-4`；这与论文 `1.68e-4` 不是可直接比较的同一指标/运行。

下面的标题与段落是不可拆的受保护 Slurm 历史块，其中旧生成器已删除；不得作为 canonical 本地入口：

### 10.3 Legacy Slurm

`slurm/train_legacy.slurm` 请求 CPU 分区、32 个 task，但 Python 代码不使用多进程/分布式训练；资源没有被显式利用。它还会先运行 `genDataSet.py`，因此每次作业都会重新随机生成并覆盖数据集。

## 10A. 论文 Figure 3 基准训练路径

`paper_reproduction/train_paper_fig3.py` 使用 `1224 -> 5000 -> 5000 -> 160` 和两层 LeakyReLU，模型以可信 `torch.save` checkpoint 保存，连续复用同一个 Adam 的动量状态，并对显式逆 Fourier 后的 EVDF 优化 MSE。为避免 SI 单位下数值过小，损失定义为 `MSE(c * f_e)`；论文没有公开损失缩放，因此正式运行的数值不能与论文 `1.68e-4` 直接比较。

默认完整日程严格按论文文字执行：3000 步 `1e-4`、6000 步 `1e-5`、10000 步 `1e-6`、100000 步 `1e-7`，共 119000 次全批量更新。`paper_training_resume.pt` 每 1000 步原子更新并捕获 Slurm 终止信号；正式完成后写模型、metrics、history、训练曲线、Figure 3 合图/单图和 `SHA256SUMS.txt`，全部通过才更新模型注册的 `LATEST.txt`。

BSCC 正式 Job `1347041` 已 `COMPLETED (0:0)`：A100 上训练 7074.93 秒，`completed_steps=119000`、`full_schedule_completed=true`，最终缩放 EVDF MSE `3.3541e-7`，测试 EVDF R2 `0.9782841`。Figure 3 面板 (a)–(d) 的 R2 分别为 `0.9660/0.9940/0.8892/0.8033`。测试样本表现好，但 `m=3.5/5` 超高斯存在中心幅值低估和旁瓣，不能声称严格复现论文展示的超高斯质量。

诊断 Job `1347309` 使用冻结模型对照 `OTS_spectrum_sg` 与训练同源的 `OTS_spectrum_fft`：两种超高斯谱的 relative L2 小于 0.9%，换成 Fourier 谱并未实质改善重建，因此主要问题是当前模型的训练分布外泛化，不是两条正向模型不一致。不得通过把这两个测试超高斯直接加入训练或后处理曲线来冒充论文复现。

## 11. canonical 训练配置与产物合同

### 11.1 默认配置

```text
seed = 20260710
validation_fraction = 0.10
batch_size = 128
max_epochs = 250
patience = 25
learning_rate = 1e-3
weight_decay = 1e-5
raw_loss_weight = 0.10
candidates = 512,256;1024,512
device = auto
```

默认候选解析为 `512x256`（dropout 0）和 `1024x512`（dropout 0.02）。训练设备可选 `auto/cpu/cuda`；显式请求不可用 CUDA 会失败，不静默回退。

### 11.2 数据、损失和物理投影

- loader 读取所有物理行，不丢最后一条。
- 默认固定随机划分 90% train / 10% validation；独立 test 只在冻结 checkpoint 后由 `evaluation.py` 使用。
- `x_mean/x_scale` 和每个输出分量的 `y_mean/y_scale` 只由训练集统计。
- Hermitian 投影：实部与翻转后的实部取平均，虚部与翻转后的虚部取反对称平均。
- 损失为 `(1 - raw_loss_weight) * standardized_MSE + raw_loss_weight * normalized_raw_MSE`，默认权重 0.10；配置限制在 `[0,1]`。
- 优化器 AdamW；验证损失驱动 ReduceLROnPlateau；梯度范数裁剪到 5；验证无改善达到 patience 后早停。

### 11.3 history 与候选选择

`training_history.json` 记录三个 ID、train data/manifest/checkpoint 相对路径、完整解析后训练配置、train/validation 样本数、每个候选逐 epoch 的 train/validation loss 和 learning rate、停止原因、最佳 epoch/loss 与最终候选。它不包含 test 指标、预测或模型权重。

### 11.4 checkpoint 边界

checkpoint 与 history 在训练结束时写入，二者都不能含 test 结果。test 指标只进入 evaluation 产物；选择 candidate 和 best epoch 时也不能读取 test。

下面 `### 11.5 Slurm` 及其内容是受保护的既有集群/完成训练语境；其中出现的旧根目录 Python 入口已经删除，不是 canonical 本地入口。

### 11.5 Slurm

`slurm/train_v2.slurm` 当前请求：

```text
partition=dgx2
nodes=1
ntasks=1
cpus-per-task=6
gres=gpu:1
mem=64G
time=04:00:00
```

脚本加载 Python 3.10.10 与 PyTorch 1.6.0 模块，实际容器报告 Python 3.6.10 与 PyTorch 1.7.0a0；先执行 `train_v2.py --self-test`，通过后才训练。Job `59648208` 已于 2026-07-11 `COMPLETED (0:0)`，V100 可见且自检通过；选中 `1024x512`，但测试 EVDF R2 `0.9475271` 低于同数据 legacy 的 `0.9609201`。该结果已下载到本地忽略目录 `pi_results/job_59648208/` 并通过模型和指标哈希校验。

2026-07-15 的三次 π 对照作业均已 `COMPLETED (0:0)`：

- 容量 Job `59739405`：默认 raw 权重 0.10，选中 `5000x5000`，EVDF R2 `0.9587845`；容量显著改善但仍低于 legacy `0.9609201`。
- 50/50 损失 Job `59740000`：raw 权重 0.50，选中 `5000x5000`，EVDF R2 `0.9602090`。
- 最终 Job `59740036`：raw 权重 1.0，选中 `5000x5000`，最佳 epoch 94；coefficient R2 `0.9687341`、EVDF R2 `0.9629309`、relative L2 均值 `0.1147455`、负质量比例 `0.0063980`，均优于同测试集 legacy 的 `0.9670373`、`0.9609201`、`0.1151032`、`0.0077141`。legacy 的 relative L2 中位数仍更好（`0.0718041` 对 v2 `0.0786430`）。

最终产物在远端 `runs/v2_rawloss_59740036/`，并已下载到本地忽略目录 `pi_results/job_59740036/`；checkpoint SHA-256 为 `ea2e924f...d80093`，六个核心产物均与作业日志中的远端哈希一致。以上作业状态仍属于快照，回答未来状态前必须运行集群查询。`slurm/monitor_v2.py` 默认 Job ID 也可能过期，可用环境变量 `OTS_V2_JOB_ID` 覆盖。

报告指出工程目录在上海交大 π 2.0 `lustre`，对应 `dgx2`；思源一号 `a100` 使用不同文件系统，不能只改分区名。

北京超算使用 `slurm/train_v2_bscc.slurm`：`BSCC-N32-H` 的 `gpu` 分区、1 张 GPU、32 CPU、默认 54,976 MB 内存和 8 小时时限。该分区不接受显式 `--mem=55G`，1 卡内存应由平台默认策略分配。脚本加载 Miniforge 3/24.1、CUDA 12.4、cuDNN 8.9.5；PyTorch/Matplotlib 来自 Miniforge，NumPy/Pandas/PyArrow 通过只读 `PYTHONPATH` 使用共享 `h310` 环境。脚本先检查 CUDA 和 `train_v2.py --self-test`，缺少 Parquet 时在计算节点运行 `genDataSet.py`，无 legacy pickle 时使用 `--skip-legacy`。

### 11.6 当前 BSCC canonical Slurm

上段是受保护的旧 v2 历史。当前活动入口为 `slurm/canonical_all_bscc.slurm`：`gpu` 分区、1 GPU、32 CPU、8 小时，加载 Miniforge 3/24.1、CUDA 12.4、cuDNN 8.9.5 和既有只读 `PYTHONPATH`，并设置确定性 CUDA 所需的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。它要求提交环境显式提供 `OTS_RUN_ID/OTS_RUN_DIR/OTS_DATASET_ID/OTS_CHECKPOINT_ID`，只运行 `python main.py all`，且不在 `#SBATCH` 中固定 stdout/stderr；提交者必须把日志路径设为当前 run。

## 12. 评估路径

当前评估实现是 `evaluation.py`，CLI 入口是 `python main.py evaluate ...`。它只使用已经冻结的 canonical checkpoint 和 manifest 指向的独立 test Parquet；不参与标准化统计、early stopping 或候选选择，也不修改 checkpoint/history。

### 12.1 推理和一致性检查

- 要求所选 dataset 的 manifest `dataset_id` 一致，要求所选 checkpoint payload 的 `checkpoint_id` 一致；manifest 的生成 run、checkpoint 的训练 run/训练 dataset 均作为来源记录，不要求与当前评估 run 相同，也不要求评估 dataset 等于 checkpoint 的训练 dataset。
- 验证 manifest 物理样本数、checkpoint/test 的 K 网格和 1224 点波长网格一致。
- 使用 checkpoint 的 `x/y mean/scale`，在 `model.eval()` 与 `torch.no_grad()` 下分批推理，反标准化后做 Hermitian 投影。
- `reconstruct_real_evdf` 用显式 `Re(F)cos(Kv) - Im(F)sin(Kv)` 梯形逆积分，不做隐式 complex-to-real cast。

### 12.2 指标与产物

- coefficient：MAE、RMSE、R²；
- Fourier 物理性：Hermitian symmetry MAE；
- EVDF：MAE、RMSE、R²、每样本 relative-L2 的 mean/median、negative-mass fraction mean；
- `metrics.json`：评估 run、评估 dataset 及其生成 run、checkpoint、checkpoint 训练 run/训练 dataset、实际 checkpoint/test 相对路径和完整指标；
- `predictions.npz`：真值/预测 coefficient、K 网格、速度网格和真值/预测 EVDF；
- `comparison.png` 与 `sample_detail.png`：批量 EVDF 对比和单样本光谱/Fourier/EVDF 细节。

所有指标必须 finite；R²、relative-L2 和负质量比例使用显式 denominator floor。评估输出与训练产物分目录保存。

## 13. 运行依赖与环境状态

canonical 子项目当前没有 `requirements.txt`、`pyproject.toml`、Conda environment 或 lockfile。由六个主线模块的直接 import 推断：

- 必需：Python、PyTorch、NumPy、PyArrow、Matplotlib；`load_dataset` 调用 PyArrow 的 `to_pandas()`，实际运行还需其 Pandas 转换依赖可用；
- canonical 主线没有 Excel 读取器、SciPy 或额外训练框架依赖；
- 独立 `spectral_fitting.py` 额外依赖 SciPy 的 `scipy.optimize.least_squares` 和 Matplotlib，不改变 canonical 四命令流水线依赖；
- Slurm 路径需要 `sbatch`、`squeue`、`sacct` 和集群 module；
- PDF/PPT 不是训练运行依赖。

解析 Maxwellian 路径继续使用 PyTorch；`ots_full_fourier.py` 现在直接导入既有依赖 NumPy，仅在模块加载时生成固定 Gauss-Legendre 节点/权重。生产实现不依赖 SciPy，CUDA 路径也不做 NumPy 数据往返。

版本注意：

下面两项同时包含受保护 Slurm 环境语境和已删除本地入口名，保留原文只用于解释既有集群记录，不能作为 canonical 兼容性声明：

- `evaluateModel.py` 使用 dataclass、`list[str]`、`tuple[...]` 等写法，语法层面要求现代 Python；Slurm 指定 Python 3.10.10。
- `train_v2.py` 自称兼容 Python 3.6 与 PyTorch 1.6/1.7，但实际集群脚本使用 Python 3.10.10 + PyTorch 1.6.0，仍应以 Slurm 自检为准。

以下两项是受保护的既有源码/远程环境验证记录；本次知识库同步没有重新运行或刷新它们：

- 初始发布候选的 19 个 Python 文件已通过 AST/语法解析；删除远端 `docs/` 与 `paper_reproduction/` 后，当前 BSCC manifest 包含 15 个 Python 文件和 1 个 Slurm 文件。同步范围调整 staging Job `1464076` 在 Python 3.10/PyTorch 2.4 计算节点环境共运行 101 项 unittest，结果 `OK`（其中 2 项按设计跳过），CLI help、依赖导入、CUDA 可见性和 registry 临时目录检查也通过。
- 2026-07-15 北京超算 GPU 计算节点实测环境：Python 3.10.14、PyTorch 2.4.0+cu121、NumPy 1.26.4、Pandas 2.3.3、PyArrow 22.0.0、Matplotlib 3.10.9；A100-PCIE-40GB 可见且 v2 数值自检通过。

2026-09-05 本机目标测试使用 `D:\anaconda1\envs\venus\python.exe`（Python 3.10、PyTorch 2.5.1+cu121、NumPy 2.1.2，CUDA 可用）；PyArrow 16.1.0 来自 `D:\anaconda1\python.exe`。目标模块 38 项 `OK`。完整 discovery 通过显式 `OTS_TEST_REAL_ARROW_PYTHON` 消除了分离环境的 PyArrow 问题，但当前账户即使请求非沙箱运行仍无 Windows 符号链接权限，最终为 119 项中 109 通过、1 失败、9 错误、0 跳过；失败/错误都发生在 registry/main 的临时符号链接测试，物理、data、model/train、evaluation 与 cleanup 测试均通过。

截至 2026-08-30 的上一轮同步，本地只完成静态、语法、发布清单和工作树检查，完整 unittest 在 BSCC staging 计算节点执行；2026-09-05 的新增本机验证结果以上一段为准。不得由任一历史记录推断其他 Windows 环境依赖齐全或具备符号链接权限。

## 14. 当前数据与产物状态（动态，使用前复核）

2026-08-24 本地快照：canonical 子项目根部不存在以下默认名称的流水线产物：

```text
dataset_train.parquet
dataset_test.parquet
dataset_manifest.json
model.pt
training_history.json
learning_curves.png
metrics.json
predictions.npz
comparison.png
sample_detail.png
pipeline_config.json
pipeline_summary.json
```

这些文件位于显式 `runs/<run-dir>/{data,training,evaluation}`；当前 CLI 不依赖子项目根目录隐式查找，也不接收原始输入/输出路径，只接受显式 ID 和输出 run-dir。

下面一段是受保护的既有 smoke 产物记录，其中旧 pickle 只代表历史产物，不是 canonical checkpoint：

被忽略的 `e2e_smoke/` 当前包含两个小 Parquet、一个约 127 MB 的 `OTS_Te_model_small.pickle` 和四张评估图。这些是 smoke-test 产物，不是论文正式模型或正式结果。

运行产物属于动态状态；使用前应检查用户指定 run 目录，而不能把历史 smoke、远程下载目录或论文复现目录当作 canonical `all` 输出。

2026-08-30 北京超算动态快照：GPU 账号活动根 `/home/bingxing2/home/scx6090/yhh/ots` 已原地激活发布 `bscc_canonical_20260830_093427_20b0b35143a2`，随后按用户确认执行同步范围调整 `bscc_scope_20260830_104505_b8e99c97d1d8`。当前 source manifest 为 16 项，SHA-256 为 `b8e99c97d1d89dc69b6951af8e2de2e069e6667616fd555061f6047bd4018aa5`；远端活动 `docs/` 与 `paper_reproduction/` 已删除，本地副本保留。删除前 19 个文件系统条目已归档至 `validation/sync_scope_adjustment_20260830/DOCS_PAPER_REPRODUCTION_PREDELETE.tar.gz`，SHA-256 为 `bd96fafaf887ab8c5872bbf6d0475ddd8a728250b2138257b01dc2150ceb8a4f`；scope staging Job `1464076` 运行 101 项 unittest，结果 `OK`（2 项跳过）。活动根仍只有五个根 Python 和 `slurm/canonical_all_bscc.slurm`，无根 `.out/.err`。更新前 5705 个条目全部分类；散落数据/日志/专用源码已归入对应历史 run。registry 的 14 个符号链接最终目标均位于 `runs/`。旧 `models/`、旧 `datasets/` 兼容树、两个根 Parquet 兼容链接、失败 run `paper_fig3_diagnostic_1347306` 和两处 cache 已于 `2026-08-30T10:25:01+08:00` 删除，其删除前归档 SHA-256 为 `20b2811aba160265a589cea36290988cf9b4cbdc9220cbfbcd1d825508f4920a`。完整备份、回滚与验收证据在 `remote_backups/bscc_canonical_20260830_093427_20b0b35143a2/`。初始 staging Job `1463936`、canonical all Job `1463960`、跨 run Jobs `1463968/1463971/1463976/1463979/1463985` 及 scope staging Job `1464076` 均 `COMPLETED (0:0)`；当前 manifest、registry、BSCC v2/paper/ion1 正式 SHA 与备份归档全部复核通过，最后一次审计队列为空。大空间批次与 ion1 均保留，π 集群未修改。

2026-07-15 北京超算动态快照：GPU 账号 `scx6090@BSCC-N32-H` 的 `/home/bingxing2/home/scx6090/yhh/ots` 保留正式 `dataset_train.parquet`（5001×1384）、`dataset_test.parquet`（101×1384）和 Job `1346661` 的结果目录 `runs/bscc_v2_1346661/`。Job `1346661` 已 `COMPLETED (0:0)`；最佳模型为 `1024x512`，测试 EVDF R2 为 `0.9727474`。结果、图表、checkpoint 和日志已下载到本地忽略目录 `bscc_results/job_1346661/` 并通过 SHA-256 校验。回答未来状态前仍须现场复核。

2026-07-15 北京超算 Figure 3 动态快照：独立论文数据位于 `datasets/paper_10gaussian_v1_seed20260715/`，正式 Job `1347041` 已 `COMPLETED (0:0)`，结果位于 `runs/paper_fig3_1347041/`，模型注册 `models/paper_10gaussian_v1_seed20260715/LATEST.txt` 指向 `job_1347041`，10 个正式产物 SHA-256 全部通过。轻量日志/JSON/history/六张图下载至本地忽略目录 `bscc_results/paper_fig3_1347041/` 并通过本地哈希；两个论文 Parquet 与 checkpoint 均未下载。诊断 Job `1347309` 已完成并保存在 `runs/paper_fig3_diagnostic_1347309/`。完整结论见 `PAPER_FIG3_REPRODUCTION_REPORT.md`。

2026-07-15 上海交大 π 动态快照：`/lustre/home/acct-phywsm/sharon2467/myfolder/OTS_code_share/NN_hotE_large_lmRange` 保留数据、legacy checkpoint，以及 Jobs `59648208`、`59739405`、`59740000`、`59740036` 的日志和 `runs/` 结果。四个作业均 `COMPLETED (0:0)`；最终 Job `59740036` 的 `5000x5000`、raw-loss 1.0 模型测试 EVDF R2 `0.9629309`，高于 legacy `0.9609201`。基线与最终完整产物、两次中间实验的小型指标/图表/日志已下载到本地 `pi_results/job_<id>/`，哈希核验通过。

## 15. 常用命令

### canonical 本地流水线

以下命令都要求显式的新 `run-id` 和 `run-dir`；`generate` 产生新 dataset，`train` 使用既有 dataset 并产生新 checkpoint，`evaluate` 使用既有 dataset/checkpoint，`all` 同时产生二者。`run-dir` 必须是 `runs/` 的直接子目录。

```bash
python main.py generate --run-id GENERATE_RUN --run-dir runs/generate_run --dataset-id DATASET --train-samples 5000 --test-samples 100 --seed 20260710

python main.py train --run-id TRAIN_RUN --run-dir runs/train_run --dataset-id DATASET --checkpoint-id CHECKPOINT

python main.py evaluate --run-id EVALUATE_RUN --run-dir runs/evaluate_run --dataset-id DATASET --checkpoint-id CHECKPOINT

python main.py all --run-id ALL_RUN --run-dir runs/all_run --dataset-id DATASET --checkpoint-id CHECKPOINT --train-samples 5000 --test-samples 100 --seed 20260710
```

`all` 在开始时预检产物并写 `pipeline_config.json`，随后串联 generate、train、evaluate，成功后写 `pipeline_summary.json`。默认拒绝已有 pipeline 产物；`--overwrite` 只允许重试尚未注册的同一 run，任何既有 registry ID 都不会被替换。

```text
RUN_DIR/
├── pipeline_config.json
├── pipeline_summary.json
├── data/{dataset_train.parquet,dataset_test.parquet,dataset_manifest.json}
├── training/{model.pt,training_history.json,learning_curves.png}
└── evaluation/{metrics.json,predictions.npz,comparison.png,sample_detail.png}
```

config 记录三个显式 ID、目录和解析前运行参数；summary 用相对路径连接 manifest、两个 Parquet、history、checkpoint、metrics、predictions 与三张流程图（训练曲线和两张评估图），并可摘要 selected candidate/best validation loss，但不复制完整 history 或 test metrics。

### 远程登录与受保护集群语境

上海交大 π 超算固定登录目标为 `sharon2467@pilogin.hpc.sjtu.edu.cn`，使用个人技能 `$sjtu-pi-ssh-login`：

```powershell
$pi = "$HOME\.codex\skills\sjtu-pi-ssh-login\scripts\pi.cmd"
& $pi
```

北京超算登录后的 OTS 工作目录约定为 `./yhh/ots`（相对于登录后的初始目录）。2026-07-15 GPU 路由实测对应 `/home/bingxing2/home/scx6090/yhh/ots`；旧 CPU 路由使用不同 home/filesystem，不能混用绝对路径。后续 agent 应先用 `pwd`/`readlink -f` 现场解析目录。

```bash
cd ./yhh/ots
```

本地命令应在 `OTS_code_share/NN_hotE_large_lmRange/` 下运行；BSCC 活动命令在 `/home/bingxing2/home/scx6090/yhh/ots` 下运行。

下面的 `v2 本地/集群` 标题与代码围栏是不可拆的受保护历史块。围栏内的旧根目录 Python 本地命令指向已删除文件、不可执行；保留它们仅为不改动同一围栏中的既有集群命令，canonical 本地操作只使用上面的 `main.py` 四子命令。

### v2 本地/集群

```bash
python train_v2.py --self-test
python train_v2.py --train-data dataset_train.parquet --test-data dataset_test.parquet --output-dir runs/v2
sbatch slurm/train_v2_capacity.slurm
export OTS_RAW_LOSS_WEIGHT=1.0
sbatch --export=ALL slurm/train_v2_rawloss.slurm
sbatch --export=ALL slurm/train_v2_bscc.slurm
python slurm/submit_v2.py
python slurm/monitor_v2.py
```

`slurm/submit_v2.py`、`sbatch`、`scancel` 可在当前任务需要时直接使用；`slurm/monitor_v2.py` 是只读查询。删除用户已有的远端数据仍需先确认。

### 论文 Figure 3（本地历史复现；当前 BSCC 不含入口）

以下入口只在本地副本或未来明确重新同步 `paper_reproduction/` 后可用。当前 BSCC 活动根已按用户要求删除该目录，且 source manifest 不包含这些文件，因此不得直接在当前远端执行下列命令。

```bash
python paper_reproduction/generate_paper_dataset.py --output-dir datasets/paper_10gaussian_v1_seed20260715 --self-test
python paper_reproduction/train_paper_fig3.py --self-test
python paper_reproduction/reproduce_paper_fig3.py --self-test
sbatch paper_reproduction/paper_fig3_bscc.slurm
sbatch paper_reproduction/paper_fig3_diagnostic_bscc.slurm
```

正式脚本默认执行完整 119000 步并注册模型；运行前先确认 `datasets/paper_10gaussian_v1_seed20260715/`、目标 `runs/paper_fig3_<jobid>/` 与注册目录。诊断脚本只读取 Job `1347041` 的冻结 checkpoint 和论文测试集，不训练或更新 `LATEST.txt`。数据和 checkpoint 保持在超算，本地仅同步轻量报告产物。

## 16. 已知风险与审查清单

### 最高优先级

1. `bscc_key` 和 `pi_key.txt` 中存在明文凭据：均已加入忽略；仅允许各自已授权的 SSH 登录密码助手运行时读取，绝不由 agent 直接查看、提交、上传或输出。
2. 论文声称的随机等离子体参数、多参数反演、PIC/实验与 Supplementary 证据并未在 canonical 主线完整实现，不能把固定配置的 EVDF 回归扩大表述为全部论文能力。
3. 当前七个测试模块在 BSCC 同步范围调整 staging Job `1464076` 中整体 `OK`；未来源码、同步范围或依赖变化后仍须重新运行，不能沿用本次结果宣称新状态已验证。

### 训练与数据

1. train/test 必须由 seed 与 seed+1 独立生成；validation 只能从 train Parquet 划分，任何标准化、早停或选模读取 test 都是泄漏。
2. manifest 顶层和 checkpoint 字段都是严格合同；新增占位键、把 `checkpoint_id` 写入 manifest，或让三个 ID/相对路径不一致都会破坏流水线。
3. 三段独立归一化抹去波段间相对幅度，模型只学习各段谱形。
4. Hermitian 投影不保证重建 EVDF 非负或积分为 1；negative-mass fraction 仍需报告。
5. 每样本有效温度与相对 1370 eV 的 ratio 只用于记录，不能据此默默重缩放或筛除数据。
下面一项是受保护的论文正式运行风险，只适用于其具体历史任务，不定义 canonical train/validation 流程。
8. Job `1347041` 没有验证集或早停，因为目标是严格执行论文全日程；其测试集 R2 较高，但超高斯 OOD 面板相对 L2 达 31%/42%，不能用训练损失替代泛化结论。

### 物理与数值

1. canonical 主函数输出不是绝对散射功率；`V` 和 `E_i0` 保留接口但不进入分段归一化的绝对幅度。
2. 显式 Fourier 梯形积分的成本高于 FFT 切片；速度网格必须严格递增、近似均匀，且 modes 不得超过最短物种网格。
3. 相速度插值在网格外严格置零；扩大物理参数范围时必须检查相速度是否大量落在支持域外。
4. 分段归一化已有 finite/正比例保护，但仍主动丢弃波段间相对强度。
5. `phyEqns/ots_fourier.py`、超高斯和论文复现实现是保留的对照边界，不能未经验证替代 canonical `ots_full_fourier.py`。
6. 新解析路径只适用于无碰撞、无磁、非相对论的一维 Maxwellian 单体元；其分段归一化同样丢弃绝对幅度。Dawson 的 64 点/阈值 8/十项渐近常数由当前验收范围验证，扩大 dtype、参数域或更改数值算法时必须重新对照高精度基准。
下面一项是受保护的具体诊断任务结论，不能外推为 canonical 普遍性质。
7. Job `1347309` 只证明本次 `m=3.5/5` 条件下解析与 Fourier 超高斯谱相差不到 0.9%；不能把该结论外推到任意参数或其他正向实现。

### 工程与可复现性

1. 没有依赖锁文件；直接依赖版本和 device 必须随具体运行记录。
2. 七个 `unittest` 模块已在本轮 BSCC 初始 staging 与同步范围调整 staging 中运行；其他主机、后续源码、同步范围或依赖变化后的兼容性仍需在对应环境重新验证。
3. `model.pt` 是严格 canonical checkpoint；旧 pickle/pet1 不受支持，不能用旧 loader 或旧产物替代。
4. 生成、训练、评估都默认拒绝覆盖；复用 run 目录前必须明确检查目标路径或显式承担 `--overwrite` 后果。
5. 本地 Git 当前无 remote；不要假设可以 push/pull。π 集群工作副本虽然使用同名恢复分支，但停在独立历史 `49fa64b`，与本地恢复分支不是同一提交链；同步文件不等于同步 Git 历史，禁止未经审查强推或重写任一侧历史。
6. `TRAINING_RECOVERY_REPORT.md` 的作业状态和 `slurm/monitor_v2.py` 默认 Job ID 会过期。
7. `spectral_fitting.py` 的 `C6+`、Gaussian 仪器线型、默认参数边界和单初值局部优化是第一版明确假设；应用到实验数据前应确认电离度、电子密度、仪器函数和初值，不能把局部最优直接解释为唯一物理解。

## 17. 变更或实验前的最小验证

涉及数据、训练、正向模型或评估时至少检查：

1. canonical 六模块仍存在，六个已删除旧根目录 Python 文件没有被恢复；根目录 `AGENTS.md` 仍是唯一长期知识入口。
2. `ft_x_n` 拒绝 `n<=0`、`n>N` 和非法网格；三套 `v/f` 均归一化，插值在支持域外为 0。
3. 三段光谱仍为 579/304/341 点且各自归一化，拼接输入为 1224；标签仍为 80 个复系数拆成 160 实数。
4. Parquet 恰有 1384 个 canonical 列，第 0 行是 wavelength/K metadata，loader 保留所有物理行。
5. manifest 顶层恰为 `run_id/dataset_id/physical_configuration/splits`；train/test seed 分离，温度派生值和统计来自实际样本。
6. validation 只从 train 划分，统计量只来自 train subset；`train.py` 不接受或读取 test。
7. checkpoint 严格包含三个 ID、结构、状态、标准化、网格、选模信息和完整训练配置，且不含 test 结果。
8. evaluation 只读取冻结 checkpoint 与 manifest 指向的 test；同时报告 coefficient、Hermitian、EVDF、relative-L2 和 negative-mass 指标，并生成 predictions 与两张图。
9. `generate/train/evaluate/all` 的必需 ID、输出 run-dir 或输入 registry 不合法时均失败；`all` 的 config/summary 路径引用实际产物。
10. 按改动范围运行相应现有测试；没有现场运行时只记录测试存在，不声称通过。
11. 修改物理公式后用已知归一化分布做数值对照，并确认所有输出、归一化比例和指标 finite。
12. 保持论文复现、Slurm、远程记录和已完成训练证据的历史边界，不用本地主线现状重新解释既有结果。
13. 修改解析 Maxwellian 路径时，须继续验证式 (9) 的奇偶性、阈值连续性和大参数渐近，两种离子各自的温度/质量/漂移、零组分惰性、离子完整参数交换不变性，以及解析路径不调用任何 Fourier helper。

## 18. 术语表

| 术语 | 含义 |
|---|---|
| OTS | Optical Thomson Scattering，光学 Thomson 散射 |
| EVDF | Electron Velocity Distribution Function，电子速度分布函数 |
| 非 Maxwellian | 不满足热平衡 Maxwell 分布的电子群 |
| 热电子 | 远高于背景热速度/能量的少数电子，远端红蓝移谱携带其信息 |
| `S(k,omega)` | 动态结构因子/谱密度，连接密度涨落与散射谱 |
| `chi_e`, `chi_i` | 电子/离子极化率 |
| `epsilon` | 等离子体介电函数 `1+chi_e+sum chi_i` |
| Debye 长度 | 等离子体屏蔽尺度 |
| Fourier 低通 | 只保留有限 K 模式，兼作正则和标签压缩 |
| Hermitian 对称 | 实值 EVDF 所需的 `F(-K)=conj(F(K))` |
| `run_id` | 一次流水线运行的显式用户 ID |
| `dataset_id` | train/test 数据对的显式用户 ID，写入 manifest 和下游产物 |
| `checkpoint_id` | 模型 checkpoint 的显式用户 ID；不进入 dataset manifest |
| manifest | `dataset_manifest.json`，记录物理配置、split 文件、seed、样本温度派生值与统计 |
| train/validation/test | train 用于更新和统计；validation 从 train 固定留出用于选模；独立 test 只评估冻结 checkpoint |
| Faddeeva 函数 | `exp(-z^2)erfc(-iz)`，与等离子体色散函数紧密相关 |
| Landau 阻尼 | 介电响应虚部对应的波粒共振耗散 |
| SRS | Stimulated Raman Scattering，受激 Raman 散射，能产生热电子 |
| 逆韧致吸收 | 激光加热导致超高斯/非 Maxwellian EVDF 的机制之一 |

## 19. 外部理论参考

演示稿还引用 Sheffield et al., *Plasma Scattering of Electromagnetic Radiation* (2011), Chapters 1–3，用于单电子散射、密度涨落、Vlasov/Fourier-Laplace 响应和动态结构因子推导。该书不在当前仓库中；需要逐字核对时必须取得原始资料，不能仅引用 PPT。

## 20. 维护记录

- 2026-10-18（代码复用专项审查）：独立子agent对照现有源码检查SPEC，改为直接复用normalize_distribution、增加质量参数的electron_moments、checkpoint张量校验、参数化predict_coefficients/reconstruct_real_evdf；从旧划分函数抽出split_indices供新旧入口共用，训练继续调用train_candidate。仅保留必要NPZ读写和锚点采样适配；旧热电子采样与拟合物理假设不适用，不强行复用。SPEC现列23个新增或修改函数签名及伪代码，同名PDF同步为7页并完成全页视觉检查；仅修改文档，未实施代码或训练。

- 2026-10-18（范围收缩，取代此前三VDF SPEC设计）：按用户明确指令启动独立子agent审查，并在重写后再次复审。SPEC只保留可配置物种/波段的模型与NPZ+JSON数据接口，以及复用74c1中Fig.11的27个Te/27个Ti读数和Fig.1两精确锚点生成e/H/C干净数据；默认319→480，ne仅物理采样参数，不附加网络输入。删除mask、自动背景/MAD、实际观测反演、敏感性、诊断副本、质量状态、多seed/先验基线、组划分、新CLI/registry及大规模schema。改为17个具体接口，train_vdf仅适配数据并调用现有train_candidate；不再抽取fit_regression或增加独立优化循环。通用接口接受其他离子组分的数据，专属PV生成仍固定e/H/C；固定名义k的一维近似保留。只修改SPEC及知识库并同步文档PDF，未修改生产代码、训练或运行物理测试；此前相关条目均为被本次设计替代的历史记录。

- 2026-10-18：按用户要求删除三VDF SPEC全部训练观测扰动：随机背景/幅度/噪声、色标量化/读数误差、缺测注入和实验式/抠图式训练分支；每个物理状态只生成一份干净理论谱，经固定仪器卷积直接采样到合同轴。保留温度/密度/流速/形状的物理采样及确定性预处理；启用mask时合成数据全1，真实缺测只保留质量记录并拒绝送入当前模型。同步接口伪代码、数据划分说明、验收条款，删除无关的多体元叠加限制。固定名义k的一维近似不变；仅改设计文档，未实施代码或训练。同名PDF已同步为19页，完成文本核查、全页渲染与修改页视觉检查；本版源SHA-256为c48d3fa5e453a6d4c2c8964a4a89b8b44190b8a5e9c609c83836ef7f9a95f5ea，PDF SHA-256为4f881dc2bf218564522fe274543374905038f7d68988b8008b1e52e86cf56ac2。
- 2026-10-18：按用户要求将当前三VDF SPEC导出为同目录同名PDF，共19页，嵌入中文字体，保留11张表、46个文本代码块并绘制流程图；完成全页缩略图及标题/长签名页面视觉检查。Markdown源SHA-256为14f8b2356220b367acf611db42fe310c309616673ca44de18480ebbe449baa97，源文未改；未实现代码或运行训练。

- 2026-10-17（参数化SPEC修订）：按用户要求评估并设计波段/点数与目标物种数量可配置，D_in=L*(1+mask)+density、D_out=2*M*S，物种顺序/独立v/K与预处理写入新vdf_v1合同，禁止同维数异物理含义误用。319/三目标/M80仅球形腔默认。取消初稿专用训练循环，计划参数化既有model/投影/loss/统计/run_epoch/train_candidate，并抽取fit_regression供旧train_model及新train_vdf共用；新增命令拟用vdf，不改变既有四命令和旧checkpoint格式。学习层支持任意S；本地PV生成仍仅完整e/H/C，可监督其非空子集，更多物理组分须另行设计正向模型。部分目标的条件闭环与默认三目标端到端闭环区分。SPEC已补全43个具体函数接口/伪代码及多维数、物种顺序、mask、旧预测/梯度/划分回归验收；只改文档，未实现或运行这些新检查。
- 2026-10-17（同日SPEC来源修订）：用户指出74c1已有提取数据后，读取`C:/Users/win/.codex/worktrees/74c1/ots/outputs/ots_experimental_fits/`及聊天“验证 20160909153 的频率平移”。确认`paper_experimental_reconstruction_20261009/fig10a/saved_spectral_image.npz`为已平移实验556×527，必须用measured_raw；`paper_simulated_spectra_20260923/fig10b|fig10c/digitized_spectra.npz`为480×319/320，含intensity、mask及源像素。后续旧拟合采用Fourier偏移+1.4904 ns/−0.00380 nm、ne=3.5e20、共同Ti；不能重复平移。实验回溯桌面原NPZ最大强度误差4.55e-13 counts，波长误差0，时间误差约1.2e-14 ns。Fig.11已有54个实验Te/Ti读数，不含模拟曲线数表；旧模拟拟合是等效单区参数，不当论文模拟真值或VDF标签。SPEC取消重复提图/抠图/配准/拟合，采用Fig.10(b)原生319点公共轴，模型639→480，新表1121列，时序密度基准沿用3.5e20；Fig.1精确锚点另作对照。只改设计与知识库，未修改74c1、生产代码或运行训练。以下同日初稿条目保留为设计演变记录，其中257点及旧平移已经被本条替代。
- 2026-10-17：按用户“先写SPEC”新增 `docs/SPEC_OCTAHEDRAL_THREE_VDF_RECONSTRUCTION.md`。核对Zhao2019本地PDF Fig.1(a,b)、Fig.10/11与020数组：球形腔精确锚点Te/Ti为209/78、1392/912 eV，ne为3.5e20/4e20 cm^-3；论文只有共同Ti，TH/TC独立采样属于新增设计。方案为现有MLP增量加入515输入/480输出（257点IAW谱、mask、密度条件；e/H/C各80复Fourier），PV生成三套可变分布，固定C5H12准中性组成；同时规划Fig.11三来源温度锚点、020论文坐标平移、Fig.10(b,c)色标反演、逐物种精度/物理性/可辨识性门槛及全部函数接口伪代码。平移采用现有图形配准+1.492 ns/-0.004 nm，不能当原始标定；正式模拟抠图、数值pilot和训练均未执行。本轮源码检查发现HEAD fac9f98的data.py已调用OTS_spectrum_PV；这与此前插入函数时“未自动切换”的历史状态不同，以当前源码为准。本轮只写SPEC和本知识库，不改生产代码、源数据或远端。
- 2026-09-16：按用户要求在Fig.1(a/b/c)原位图上叠加旧算法选中位置，新增`docs/SPEC_PAPER_CURVE_PIXEL_OVERLAY_REVIEW.md`及独立产物`outputs/ots_experimental_fits/pixel_selection_review_20260916/`。重放旧RGB阈值、蓝线附近搜索和中位数规则，375/401/376个点与保存NPZ坐标完全一致；输出三张最终取点叠图、三张原图/全部候选/最终取点对照及逐列JSON。绿色标示中位数落在候选像素上，橙色标示落在候选之间（包含半像素，不自动等于误选）；遮挡不恢复、缺失点不连线。原位图、旧取点与旧算法哈希不变，六PNG回读和视觉检查完成。只用于用户审查取点可靠性，未重新配准、修改图轴、重新拟合或更改已有结果。
- 2026-09-15：按用户目标改进pilot并解释review，先写`docs/SPEC_OTS_PILOT_IMPROVEMENT.md`，再新增八函数独立`improve_ots_pilot.py`。保持TH/TC独立、原A/B边界和每行scale；外层搜索五个非线性参数，A/B1/B2用有界线性最小二乘求解。复用原解析物理函数，8σ一维Gaussian核与原卷积数值等价；不平滑拟合输入。完成同15行/同范围、IAW范围、论文样本、论文坐标及共同Ti、密度共71例/1211候选；1个候选到迭代上限但未选中，所有最终解收敛。同15行SSE均未变差，最大两项下降24.43%和20.99%；自动6 ok/9 review，原8条review保留，新发现163行726温度多解。153行305强窄谱在部分配置中虽自动ok，仍标明人工复核，杂光来源仅为与论文描述相符的推断。
- 2026-09-15：发表图像的灰谱噪声纹理定位Fig.1a/b为020源行234/660、c为153源行564；论文密度3.5e20/4e20/1e21 cm^-3、共同Ti对照得到Te/Ti=210.3/83.9、1404.2/884.6、1756.2/3262.0 eV，相对论文Te均<1%、Ti最大7.53%。主模型仍独立TH/TC，c的独立解还有仅差1.103% SSE的温度分支。图像配准提示153/163/020的论文时间偏移约+0.49/+0.63/+1.49 ns，两2016数据波长约+0.16 nm，020波长与原轴接近；这些不是原始校准，源坐标未改，Fig.1c文字2.2 ns与全图配准仍有约0.063 ns差异。原15行仍保留充气密度假设，不能套用论文三个样本的温度或密度。
- 2026-09-15：改进审查入口`outputs/ots_experimental_fits/pilot_improved_20260915/IMPROVEMENT_REPORT.md`，含8条原review原因、15行主结果、论文/窗口/密度对照、8张汇总图和源图定位证据。指定Python3.11.9核验：三组合成新旧模型最大relative L2=3.66e-13、闭环3.18e-12；71例用原`_fit_model`独立重建最大8.85e-13；scale回乘最大绝对误差1.82e-12；输入/原拟合及物理源码哈希、原pilot22个产物哈希未变。新JSON/NPZ源索引、尺度、残差、边界、候选选择、逐案manifest及38个报告链接通过检查；原提取器、既有拟合器、物理代码及数据未改。SPEC改进任务完成；1759行full未运行，未同步远端。
- 2026-09-14：按用户“执行，在完成 pilot 的时候停一下”实施 `fit_ots_experiments.py` 七函数独立脚本及 `docs/ots_experimental_pilot_config.json`。逐行以拟合范围内 `max(abs(I))` 归一化，JSON/NPZ 保存每行 `scale` 与源行号，不将 scale 作为拟合参数；沿用独立 TH/TC 八参数及用户当前 A∈[0,10]、B1/B2∈[-2,2] 边界。密度显式采用充气基准假设：2016 为 6e20、2017 为 3.5e20 cm^-3；时间保持 FIG 坐标。产物 `outputs/ots_experimental_fits/pilot_20260914/` 含审查报告 `PILOT_REVIEW.md`、15 行/45 候选、逐行谱/残差/scale 和局部敏感性图；三 shot ok/review 分别 3/2、1/4、3/2，无计算失败，总耗时约 301 秒。指定 Python 3.11.9 下三组合成谱闭环最大 relative L2=2.71e-11；异常合同、15 行回读/索引/scale 回乘、22 个输出哈希和记录的输入/源码哈希检查通过。存在触边、弱信号、温度非唯一与部分 ok 行结构性残差；ok 不表示已确认物理可信。SPEC T1–T3 完成、T4 待用户审查，1759 行 full 未运行；未改既有提取器、拟合器、正向物理模块和实验数据，未同步远端。
- 2026-09-14：新增 `docs/SPEC_OTS_EXPERIMENTAL_PARAMETER_FITTING.md`，参考 `OTS_data/exp_ref_para.pdf`（用户称 exp_params；本地未发现该文件名）。用户明确选择 TH/TC 独立八参数，计划复用现有拟合器，采用 Gaussian 仪器展宽和线性背景，逐 FIG 时间行分析。实际数据现位于 `OTS_data/ots_data_extracted/`，三个 ordered NPZ 已与原 manifest 哈希核对一致；原 manifest 绝对输出路径为历史位置，读取用当前根及相对文件名。初版显示范围含 555/555/649 行，总 1759 行；密度需显式配置且区分充气基准与体元条件，论文时间偏移不应用。仅新增设计与知识库摘要，未实施拟合脚本或运行拟合，未改现有提取器、拟合器、物理源码和实验数据。
- 2026-09-14：执行简化后的 `SPEC_OTS_DATA_EXTRACTION_PLAN.md`，新增独立六函数脚本 `extract_ots_data.py`。命令参数仅 `--input-dir/--output-dir`，输出目录必须新建。正式产物位于 `outputs/ots_data_extracted/20260914/`：三个 shot 各含 `raw_arrays.npz`、`ordered_arrays.npz`、`metadata.json`、`quicklook.png`，总 manifest 状态 complete。三份 `(1024,1344)` 矩阵合计 4,128,768 值保留原 dtype/负值/基底，升序版本仅反转波长及强度列，时间仍为 FIG 坐标。指定 Python 3.11.9 下 10 项临时边界测试通过，独立 SciPy 回读逐元素/类型/轴映射比较通过，六个源哈希不变、12 个产物哈希全部通过，三张预览逐一视觉核查。未实现任何切片逻辑，未扣背景、归一化、拟合、训练或同步远端。
- 2026-09-14：只读分析根目录 `OTS_data/` 的六文件，形成 `OTS_code_share/NN_hotE_large_lmRange/docs/SPEC_OTS_DATA_EXTRACTION_PLAN.md`。三 FIG 均为 MATLAB v5，主 axes 的 image 保存 1024×1344 时间—波长强度矩阵，波长递减、时间递增；2016 两份为含负值 float64，2017 为带正基底 uint16。Notes.txt 给出仪器 FWHM 0.065/0.057 nm；5.4 ps 仅为时间采样步长。与参考 PDF Fig.5/10 的 shot 对应及 +0.5/+1.5 ns 时间偏移仅为图形对照候选，未确认为校准。参考充气密度不是逐时刻拟合密度。同日按用户要求简化 SPEC，范围收敛为完整二维矩阵读取、坐标整理、NPZ/JSON 导出和预览核查，删除所有切片与窗口聚合逻辑；保留原始数据与索引，不接入 canonical 训练格式。仅使用指定 Python 3.11.9 做结构、统计及临时图形复核，未实现正式提取器、未拟合、未修改数据或同步远端。
- 2026-09-09：对桌面 `挑战计划中期考核报告模板-20260906.docx` 图5/图6完成只读峰宽核查，报告位于 `outputs/OTS图6核查报告_20260909/图6氢离子峰宽与双离子OTS计算核查报告.html`，附同纵轴数值重绘PNG。当前 `spectral_fitting.main()` 的H-C条件实际为Te=4800、TH=480、TC=720 eV，模板图6与H-C反演图注却写TH=720、TC=480，二者不可混用；历史原图参数日志未取得。图5条件纯H红峰FWHM约0.7064 nm，图6代码条件约0.06376 nm，图注条件约0.14863 nm，主要差异来自Te/TH而非纵轴。独立SciPy wofz参考在所有输入显式float64时与当前解析函数relative L2低于1e-11；默认 `_as_real_tensor` 先构造默认dtype再转目标dtype会对Python浮点参数产生中间float32舍入，本轮谱差异约0.012%–0.126%。当前241点、0.025 nm波长网格无法可靠解析约0.01265 nm的纯C窄峰；`_gaussian_convolve` 的等点权重不适合非均匀波长。上述问题仅形成报告，未改任何已有代码；指定Python 3.11.9下相关物理测试38项全部通过。当前组分绘图实际利用 `switch_scale_output=True` 取回scale，再把四谱统一除以混合谱峰值，并非各谱独立峰值为1；这些是不同假想等离子体的理论谱，不是混合谱各项的可加贡献。原模板未改，未执行数据生成、训练、远程同步或Slurm；临时SPEC归入报告附录后按用户要求清理。
- 2026-09-05：按已批准 `SPEC_ZHAO2019_IAW_SPECTRAL_FITTING_REPRODUCTION.md` 新增单文件 `spectral_fitting.py`，复用解析双离子 Maxwellian 正向函数，实现 Zhao et al. 式 (10) 的 `Te/Ti/uik/uek/A/B1/B2` 有界无权最小二乘拟合和 Gaussian FWHM 展宽；未新增 CLI、绘图、文件 I/O 或测试模块。指定 Python 3.11.9 下依赖为 NumPy 2.2.3、SciPy 1.15.2、PyTorch 2.6.0+cu124；`py_compile` 和现有正向模型 38 项 unittest 通过，Figure 1(b) 参数无噪声闭环将残差平方和从 `0.8076011503` 降至 `3.5375481e-30` 并恢复全部七参数。
- 2026-09-06：按已批准 `SPEC_SEPARATE_TWO_ION_TEMPERATURE_FITTING.md` 将 `spectral_fitting.fit_spectrum(...)` 从共同 `T_i` 的七参数接口替换为独立 `T_i1/T_i2` 的八参数接口；同步修改初值/边界、返回键、正向参数传递和 `main()` 闭环。指定 Python 3.11.9 下九种双离子状态全部恢复不相等的两种离子温度，最大重建光谱 relative L2 为 `2.846e-7`；未修改解析正向物理模块或 canonical CLI。
- 2026-09-06：按用户明确要求跳过新 SPEC，参考 Zhao et al. (2019) Figure 1，在 `spectral_fitting.py` 增加 `plot_fit_result(...)`；每个拟合结果生成真实/拟合光谱、`Te-Ti1` 和 `Te-Ti2` 的 `Delta R` 三联图。九种状态 PNG 已由 `main()` 生成并完成 H-He、H-C、Au-B 代表图视觉检查；实现不含梯度、`k` 展宽或 ±100 ps 样本点。
- 2026-09-08：按已批准 `SPEC_FITTED_ION_COMPOSITION_SPECTRA_PLOT.md` 在 `spectral_fitting.py` 新增 `plot_ion_composition_spectra(...)`，并只适配 `main()` 的物种标签、调用与输出目录。函数直接复用解析双离子公开 API，生成纯 ion-1、纯 ion-2、平均离子和真实双离子 2×2 四联理论谱；不含 Gaussian 仪器展宽、`A/B1/B2`，也未复制光谱物理公式。九种状态全部生成 PNG，H-He `x_H=0.5` 的平均离子为 `Zbar=1.5`，H-He/H-C/Au-B 代表图完成视觉检查。
- 2026-09-05：按已批准 `SPEC_TEMPERATURE_DRIVEN_TWO_ION_OTS.md` 在 `ots_full_fourier.py` 新增独立双离子 Maxwellian 解析 OTS 路径，逐式实现 Zhao et al. (2019) 式 (7)–(9) 与式 (2)，两种离子使用独立温度；稳定 Dawson 使用 64 点 Gauss-Legendre 与大 `|x|` 十项渐近，不调用 full-Fourier 或 legacy `w_fast`。保留实施前用户已有的 `k_s-k_i` 三行修改并同步几何测试。目标 38 项含 CPU/CUDA 全部通过；完整 119 项中 109 通过，剩余 1 失败和 9 错误均为本机 `WinError 1314` 临时符号链接权限问题，未执行数据生成、训练、远端同步或 Slurm。
- 2026-08-30：按用户明确确认从 BSCC 活动根删除 `docs/` 与实际存在的 `paper_reproduction/`，本地副本继续保留；同步 manifest 从 33 项调整为 16 项，SHA-256 为 `b8e99c97d1d89dc69b6951af8e2de2e069e6667616fd555061f6047bd4018aa5`。删除前 19 个条目归档 SHA-256 为 `bd96fafaf887ab8c5872bbf6d0475ddd8a728250b2138257b01dc2150ceb8a4f`；staging Job `1464076` 的 101 项 unittest 通过（2 项跳过），删除后 manifest、14 个 registry 链接、三组正式历史 SHA、备份归档和空队列均复核通过。大空间批次、ion1 与 π 未动。
- 2026-08-30：按用户明确确认删除 BSCC 低风险批次的 7 个活动路径：根/`phyEqns` cache、失败 diagnostic run `1347306`、两个根 Parquet 兼容链接、旧 `datasets/` 和旧 `models/`。删除前生成 35 条全量清单和 36985-byte 可恢复压缩包，删除后 source manifest、全部 registry 目标及 BSCC v2/paper/ion1 正式 SHA 复核通过；大空间批次、ion1 和 π 未动。
- 2026-08-30：完成 BSCC GPU 账号 `/home/bingxing2/home/scx6090/yhh/ots` 原地 canonical 更新。发布 `bscc_canonical_20260830_093427_20b0b35143a2`，引入三类 registry、四命令显式 run-dir、跨 run dataset/checkpoint 输入和 `canonical_all_bscc.slurm`；更新前 5705 个条目全部分类并将散落 legacy 文件归入既有 run。staging Job `1463936` 共运行 101 项 unittest，结果 `OK`（2 项按设计跳过）；canonical all 与跨 Dataset A/B GPU smoke Jobs `1463960/1463968/1463971/1463976/1463979/1463985` 全部完成。正式历史 SHA 通过，旧 `models/datasets` 兼容树和全部用户既有数据保留，π 未修改；删除候选只列清单，等待用户确认。
- 2026-08-24：把 `agent_knowledge_base.md` 的可编辑当前实现内容同步到六模块 canonical 工作树，补充三物种 full-Fourier、data/manifest、严格 checkpoint、train/validation、独立 test evaluation、四子命令与测试存在性；删除非保护区旧入口说明，未刷新远程状态。以下既有维护条目均为受保护的历史语境，其中出现的已删除旧文件名、legacy/v2 与远程状态只表示当时记录，不定义 canonical 当前入口，且本次未刷新远程状态。
- 2026-07-19：保留原始 `OTS离子分布函数实现.pptx`，新增 `OTS离子分布函数实现_表格美化版.pptx`；将第 4 页 Figure 3 面板指标和第 10 页测试/逐样本诊断指标改为原生可编辑 PowerPoint 表格，并通过全页渲染、越界检查和模板一致性检查。
- 2026-07-15：将本地源码根目录整理为审阅入口：Figure 3 专用代码、报告和作业脚本归入 `paper_reproduction/`，通用集群提交、监控和 Slurm 脚本归入 `slurm/`；脚本从自身路径解析项目根目录。此次整理不修改 BSCC/π 远端文件。
- 2026-07-15：在 BSCC 完成论文 Figure 3 正式 Job `1347041`：独立生成并保留 5000+100 条 10-Gaussian 论文数据，严格执行 119000 步 5000×5000 全批量 Adam 日程，测试 EVDF R2 `0.9782841`，Figure 3 四面板 R2 为 `0.9660/0.9940/0.8892/0.8033`；所有正式哈希与模型注册通过，只下载轻量产物，数据和 checkpoint 留在超算。
- 2026-07-15：完成超高斯正向模型诊断 Job `1347309`；解析与 Fourier 谱 relative L2 小于 0.9%，换用训练同源谱不改善 `m=3.5/5` 重建，确认当前差距主要来自 OOD 泛化。新增诊断脚本、Slurm 入口和 `paper_reproduction/PAPER_FIG3_REPRODUCTION_REPORT.md`，保留原始不理想面板而不做结果美化。
- 2026-07-15：完成 π 容量和目标函数对照 Jobs `59739405`、`59740000`、`59740036`。最终将 raw coefficient MSE 权重设为 1.0，选择 `5000x5000`，测试 EVDF R2 `0.9629309`，在同一数据上超过 legacy `0.9609201`；最终模型与核心产物下载到 `pi_results/job_59740036/` 并通过 SHA-256 校验。
- 2026-07-15：通过 `$sjtu-pi-ssh-login` 现场复核 π Job `59648208` 已完成并下载产物；确认 V100/CUDA/自检正常，但 v2 EVDF R2 `0.9475271` 低于 legacy `0.9609201`。新增并提交容量对照脚本 `slurm/train_v2_capacity.slurm`，对应 Job `59739405`，初始状态 `PENDING`。
- 2026-07-15：在北京超算 GPU 账号 `/home/bingxing2/home/scx6090/yhh/ots` 完成全量 v2 数据生成、训练和测试。自检 Job `1346659` 与正式 Job `1346661` 均完成；正式作业生成 5000+100 物理样本，选择 `1024x512` 模型，测试 EVDF R2 `0.9727474`，结果下载至 `bscc_results/job_1346661/` 并通过哈希与 checkpoint 加载验证。
- 2026-07-15：新增 `slurm/train_v2_bscc.slurm`；为 NumPy 1.26 增加梯形积分回退，并移除 `genDataSet.py` 中未使用的完整 Fourier 调试计算。
- 2026-07-15：创建 `$sjtu-pi-ssh-login`，固定登录 `sharon2467@pilogin.hpc.sjtu.edu.cn`，由密码助手读取被忽略的 `pi_key.txt` 并通过 `SSH_ASKPASS` 自动填写；一次性直接读取授权已使用完毕。实测进入 `pilogin4.pi.sjtu.edu.cn`（`x86_64`），退出码为 0。
- 2026-07-15：将根目录凭据文件从 `tmp.txt` 重命名为 `bscc_key`，同步更新忽略规则、登录脚本与安全授权说明。
- 2026-07-15：明确预授权 `$bscc-ssh-operations` 的密码助手在登录时读取 `bscc_key`；该脚本读取应在 agent 安全审核中直接通过，不视为凭据暴露，agent 自行查看或输出内容仍被禁止。
- 2026-07-15：记录北京超算 OTS 远端工作目录为 `./yhh/ots`；CPU 路由历史路径为 `/public1/home/sch3240/yhh/ots`，GPU 路由现场解析为 `/home/bingxing2/home/scx6090/yhh/ots`，二者不可混用。
- 2026-07-15：将 `$bscc-ssh-operations` 进一步缩减为 CPU/GPU SSH 登录和密码注入；移除 SCP/SFTP、上传下载和其他可由 agent 在登录后命令行直接执行的入口与说明。
- 2026-07-15：精简北京超算协议。`$bscc-ssh-operations` 只封装密码注入，原生 SSH/SCP/SFTP 行为由 agent 直接控制；与任务有关的 Slurm 提交/取消和普通远端文件修改/传输无需另问，删除用户已有远端数据需确认。
- 2026-07-15：首次全量合并仓库知识。审阅了当前 tracked/untracked 结构、论文 PDF、全部 13 个 Python 文件、legacy/v2 Slurm、训练恢复报告、演示稿检查文本和本地依赖状态；创建强制会话协议。


## 2026-10-17 新增OTS_spectrum_PV（仅函数增量复制）

用户批准在隔离工作树实施SPEC_OTS_SPECTRUM_PV.md并由主agent验收；随后明确要求复制到本桌面副本。仅在phyEqns/ots_full_fourier.py的__main__前原样插入私有_pv_fd_landau_response（781行）和公开OTS_spectrum_PV（860行），不覆盖原文件内容，不改变原函数/调用者，不复制测试/SPEC/历史产物。签名与full_fourier相同，fourier_modes仅兼容验证，只返回归一化intensity；同网格中心/单边差分、线性g插值，非节点两夹点直接斜率、精确节点合并，保留硬截断边界项与原物理组装。非零端点/非有限响应整次报错；不把非光滑离散有限值当真实奇点恢复。

源设计和验收产物在 `C:/Users/win/.codex/worktrees/0d4a/ots/OTS_code_share/NN_hotE_large_lmRange/docs/SPEC_OTS_SPECTRUM_PV.md` 及 `C:/Users/win/.codex/worktrees/0d4a/ots/outputs/ots_spectrum_pv_validation_20261017/`。工作树50物理测试、27谱独立解析对照通过，4097点最坏relative L2=1.15954e-4，主agent独立复算一致。复制前物理模块SHA=ffb56c7f9208eff69a8f90c10d40a1c7a0b933087e4d9a3343d4b2012ca3d6d9；原字节与模块AST保留；两函数文本逐字符与源相同。实际导入本桌面模块，CPU32/64与CUDA64同输入输出逐元素一致，使用工作树测试源码指向本桌面生产模块50/50通过、0skip。未训练、未改数据格式、未自动切换data/拟合/CLI、未同步远端。

