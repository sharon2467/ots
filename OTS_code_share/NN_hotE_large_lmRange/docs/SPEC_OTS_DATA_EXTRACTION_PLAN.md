# OTS_data 完整二维数据提取计划

日期：2026-09-14。状态：已实施并完成三份 FIG 的完整数据导出与验收。

## 1. Requirements（需求）

从 `OTS_data/` 的三个 MATLAB FIG 读取完整波长轴、时间轴和二维强度矩阵，导出可追溯的数据文件与预览图，供后续分析直接使用。

用户需要取得每个 shot 的全部数值，明确数组方向、单位和来源，并能核对导出数据与 FIG 一致。本任务只包含读取、坐标顺序整理、完整导出和核查；原始文件保持不变，不做背景扣除、归一化、插值、论文时间对齐或物理拟合。

### 1.1 文件结构

2026-09-14 现场清点：共 6 个文件，无子目录，合计约 23.26 MiB。

```text
OTS_data/
├── 20160909153.fig   7,344,758 bytes
├── 20160919163.fig   7,340,212 bytes
├── 20170811020.fig   5,049,297 bytes
├── exp_ref_para.pdf 4,649,079 bytes
├── Notes.txt              177 bytes
└── para.txt             2,912 bytes
```

- `.fig`：MATLAB v5 图形文件，已用 SciPy 成功读取，包含完整数值数据。
- `Notes.txt`：提供 XData/YData/CData 读取提示及仪器光谱 FWHM。
- `para.txt`：实验条件的中文整理，不是逐 shot 校准日志。
- `exp_ref_para.pdf`：Zhao et al. (2019) OTS 综述，共 11 页，用于解释实验背景。

### 1.2 FIG 内部结构与数组

```text
hgS_070000 → figure
├── axes → image.properties
│          ├── XData：波长，1344 点，nm
│          ├── YData：时间，1024 点，ns
│          └── CData：强度，1024 × 1344
│   └── text：轴标签与 shot 标题
├── scribe.colorbar：包含另一份 image，必须排除
└── scribe.scribeaxes：2017 文件的注释覆盖层
```

只读取 `hgS_070000`，不使用 `hgM_070000` 或 `__function_workspace__`。主图应通过对象类型、标签和数组维度识别，不依赖固定 children 序号，不执行图形回调。

| shot | 原始波长范围及方向 / nm | 原始时间范围 / ns | 强度 dtype | 仪器 FWHM / nm |
|---|---|---|---|---:|
| 20160909153 | 266.28080 → 260.47904 | -1.3986 → 4.1256 | float64 | 0.065 |
| 20160919163 | 265.87904 → 260.07728 | -1.4256 → 4.0986 | float64 | 0.065 |
| 20170811020 | 265.57240 → 260.46900 | -0.7668 → 4.7574 | uint16 | 0.057 |

三份矩阵均为 `(1024, 1344)`，共 4,128,768 个强度值；本次检查全部为有限值。波长递减、时间递增且近似均匀。2016 波长步长为 0.00432 nm，2017 为 0.00380 nm，时间步长均为 0.0054 ns。

必须保留的约定：

1. `CData[i,j]` 对应 `YData[i]` 和 `XData[j]`，第一维为时间，第二维为波长。
2. `YDir=reverse` 仅控制时间向下增大的显示方式，不要求翻转数据行。
3. XLim/YLim/CLim 是显示属性，不能用来裁剪或截断导出数据。
4. 2016 数据含负值，2017 数据有正基底；历史背景处理未知，保留全部原值和 dtype。强度单位记为 `fig_native`，不解释为绝对功率。
5. 5.4 ps 是时间采样间隔，不是已核实的仪器时间分辨率；仪器光谱 FWHM 来源为 `Notes.txt`。
6. FIG 时间零点与论文图可能不同，本任务保持 FIG 时间。论文中的充气基准密度不是逐时刻实测密度。

## 2. Design（设计）

### 2.1 实现与数据流

使用单文件 `OTS_code_share/NN_hotE_large_lmRange/extract_ots_data.py`，采用普通函数、dict 和 NumPy 数组。依赖为 SciPy、NumPy、Matplotlib 及 Python 标准库，执行时使用仓库指定的 Python 3.11.9 绝对路径。

```mermaid
flowchart LR
    A[清点源文件并计算哈希] --> B[读取唯一实验主图]
    B --> C[检查维度与坐标]
    C --> D[保留原始数组并整理升序坐标]
    D --> E[导出完整数组与预览]
    E --> F[回读核查并写清单]
```

加载方式为 `loadmat(path, variable_names=['hgS_070000'], struct_as_record=False, squeeze_me=True)`。当前三文件均为 MAT v5；其他格式明确报错。提取程序无需解析 PDF，也无需实验物理参数配置。

### 2.2 输出合同

```text
outputs/ots_data_extracted/<本次提取标识>/
├── manifest.json
└── <shot_id>/
    ├── raw_arrays.npz
    ├── ordered_arrays.npz
    ├── metadata.json
    └── quicklook.png
```

| 文件 | 内容 |
|---|---|
| `raw_arrays.npz` | `wavelength_nm_raw`、`time_fig_ns_raw`、`intensity_raw`，形状分别为 `(1344,)`、`(1024,)`、`(1024,1344)`；与 FIG 原值及 dtype 一致 |
| `ordered_arrays.npz` | 严格递增的 `wavelength_nm`、`time_fig_ns`，同步重排的 `intensity`，以及零基 `source_row_index`、`source_col_index` |
| `metadata.json` | shot_id、源相对路径/哈希、图形对象路径、标题/轴标签、单位、shape/dtype、原图显示属性、FWHM 及来源、坐标重排记录、强度 min/max/负值数；历史强度处理记为 unknown |
| `quicklook.png` | 完整时间—波长强度图，标明单位与 shot，时间向下增大；保留全部数据范围 |
| `manifest.json` | schema_version、执行环境、输入/输出目录、六个源文件的大小/哈希、各 shot 输出相对路径/哈希、运行状态及错误信息 |

整理后的强度保持原 dtype，满足：

```text
intensity[i,j] == intensity_raw[source_row_index[i], source_col_index[j]]
```

当前三个 shot 仅需反转波长轴及强度列，时间行保持原序。重复或非单调坐标报错。输出目录必须是新目录；全部回读验证成功后状态才为 `complete`，失败则记录错误并返回非零退出码。

这些文件只有约 260–266 nm 的 IAW 数据，不具备 canonical 三波段输入及 EVDF 标签，使用独立输出格式，不写入训练数据或 registry。

### 2.3 函数接口与伪代码

以下六个函数是首版完整实现范围，均已实现。

**F1. `inventory_sources(root: Path) -> list[dict]`**

```text
确认 root 是目录；按相对路径排序枚举普通文件，不跟随链接
逐文件分块计算 SHA-256，记录路径、大小和后缀
返回全部源文件清单
```

**F2. `find_spectral_images(node: object, path: str = 'hgS_070000') -> list[dict]`**

```text
仅递归遍历图形节点的 children，兼容单对象和对象数组
在 type=axes 下检查 image 及同轴 text 标签
要求 CData 二维、XData/YData 一维，轴长与矩阵维度一致，标签表达 nm/ns
记录合格 image 的属性、父轴显示属性和对象路径
排除 colorbar 与注释层；返回候选列表
```

**F3. `read_fig(path: Path) -> dict`**

```text
验证 MAT v5 格式，只加载 hgS_070000
调用 find_spectral_images，要求候选恰好一个
核对文件 stem 与图标题中的 shot_id
检查数组数值类型、非空、finite、维度和坐标严格单调性
返回原始数组副本、shot_id、对象路径和显示属性，保留 dtype
```

**F4. `canonicalize_axes(raw: dict) -> dict`**

```text
为两个坐标轴生成原始索引
递增轴保持原序；递减轴反转轴及对应索引；其他情况报错
按两组索引同步重排强度，不转置或插值
返回升序轴、完整强度矩阵及源行列索引
```

**F5. `write_shot_bundle(output_dir: Path, raw: dict, ordered: dict, source: dict) -> dict`**

```text
创建尚不存在的 shot 输出目录
检查 ordered 与 raw 的索引对应关系逐元素成立
保存 raw_arrays.npz 和 ordered_arrays.npz，不使用 object/pickle 数组
写 UTF-8 metadata.json（allow_nan=False），包含基本统计与来源
用 pcolormesh 绘制完整二维图，时间向下，标注 shot/单位/强度色标，保存并关闭图
重新 np.load(allow_pickle=False)，核对全部数组、dtype 与索引映射
计算输出文件哈希，返回输出清单
```

**F6. `main(argv: list[str] | None = None) -> int`**

```text
argparse 接收 --input-dir 与 --output-dir，要求输出为新目录
inventory_sources 获取源清单，创建状态为 running 的 manifest
按文件名顺序对三个 FIG 调用 read_fig → canonicalize_axes → write_shot_bundle
FWHM 按 Notes.txt 的已核实 shot 对应关系写入元数据；未知 shot 不猜测
重新检查六个源文件哈希未变化
成功则写完整 manifest，状态 complete，退出 0
异常则在 manifest 记录错误，状态 failed，退出非零
```

## 3. Tasks（任务拆解）

| 编号 | 任务 | 依赖 | 验证结果 | 状态 |
|---|---|---|---|---|
| T0 | 检查目录、FIG 数组及说明文件 | 无 | 本文文件结构和数组合同 | 已完成 |
| T1 | 实现 F1–F3 | T0 | 三个 FIG 均读取唯一实验主图，排除色条 | 已完成 |
| T2 | 实现 F4–F6 | T1 | 完整 NPZ、元数据、预览与 manifest | 已完成 |
| T3 | 对三 shot 执行导出及回读核查 | T2 | 数组无损、方向正确、源文件未变 | 已完成 |

## 4. Acceptance Criteria / Tests（验收标准）

1. 六个源文件均有清单及哈希；三 FIG 各导出一个完整 `(1024,1344)` 矩阵，共 4,128,768 个强度值。
2. 原始 NPZ 的值及 dtype 与 FIG 完全一致；升序数组通过源行列索引逐元素对应原始数组。
3. 色条和注释层被排除；多候选、轴长不匹配、非有限值、重复或非单调坐标明确报错。
4. 负值、正基底、显示范围外数据和超过 CLim 的强度全部保留；时间仍为 FIG 原坐标。
5. NPZ 无 pickle 对象，JSON 可正常解析；三张完整二维预览逐一核对坐标方向和谱形。
6. 运行前后源文件哈希一致，已有输出不被覆盖；全部回读检查通过才标记 complete。

### 实施与验收记录

输出位于 `outputs/ots_data_extracted/20260914/`，包含一个 manifest 和三个 shot 各四个文件，共 13 个文件。manifest 状态为 `complete`，三个 shot 均为 `(1024,1344)`，共 4,128,768 个强度值。

- 指定 Python 3.11.9 下执行成功；NumPy 2.2.3、SciPy 1.15.2、Matplotlib 3.10.1，具体环境写入 manifest。
- 10 项临时 unittest 全部通过，覆盖主图选择、色条排除、多候选、维度错误、非有限值、非单调/重复坐标、标题不匹配、dtype 与双轴反转、回读与防覆盖、失败清单及输入目录隔离。
- 独立于提取函数，直接用 SciPy 再次读取三份 FIG 的已核实主图对象，逐元素比较原始导出数组及 dtype，并验证升序波长、时间和完整矩阵对应关系，全部通过。
- 六个源文件哈希未变化，12 个 shot 产物哈希全部验证；三张完整二维预览已逐张核对。
- 预览使用 FIG 原有 CLim 以便查看谱形，颜色范围外的强度仍完整保存在两个 NPZ 中。

从仓库根目录执行的命令如下；重跑需换一个尚不存在的输出目录：

```powershell
& '<LOCAL_PROJECT_PATH>' 'OTS_code_share/NN_hotE_large_lmRange/extract_ots_data.py' --input-dir 'OTS_data' --output-dir 'outputs/ots_data_extracted/20260914'
```
