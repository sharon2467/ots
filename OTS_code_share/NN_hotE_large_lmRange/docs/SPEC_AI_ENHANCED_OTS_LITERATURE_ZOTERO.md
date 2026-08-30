# SPEC：AI Enhanced OTS 文献脉络、实验参数与传统反演基线的 Zotero 整理

## 0. 状态与执行门槛

- 状态：**已确认并完成（用户于 2026-08-24 回复“执行”；同日完成验收）**。
- 已完成联网检索、交叉核验、Zotero 集合/条目写入、三份中文导读和最终回读验收。
- 本任务不修改 OTS canonical 源码、物理公式、训练数据或模型。
- 研究对象中的 OTS 专指 Optical Thomson Scattering（光学 Thomson 散射）。

## 1. Requirements（需求）

### 1.1 功能目标

围绕当前仓库的 “AI Enhanced OTS / 从 OTS 光谱反演非 Maxwellian EVDF” 方向，建立一套由浅入深、能够直接约束代码物理配置和传统基线设计的文献知识库，并整理进 Zotero 的“等离子体物理”门类下一个独立集合。

默认知识结构：

```text
等离子体物理/
└── AI Enhanced OTS｜实验参数·传统反演·AI
    ├── 00_导读与核心教材
    ├── 01_散射理论与谱形
    ├── 02_实验系统与标定
    ├── 03_实验应用与参数区间
    ├── 04_非Maxwellian与EVDF
    ├── 05_传统反演与不确定度
    ├── 06_AI与机器学习反演
    └── 07_当前项目配置审计
```

若 Zotero 不支持一次性创建全部子集合，则保留一个独立父集合，并以同名标签和三份总览笔记实现等价分类；不改变知识结构。

实际落地采用 `等离子体物理/OTS` 单独集合，并以 `route::*`、`OTS::*`、`evidence::*` 标签和三份中文导读实现上述等价结构。复用了用户已经存在的 `OTS` 集合，没有创建近义重复集合。

### 1.2 用户故事

1. 作为刚从 Sheffield 等教材进入 OTS 的研究者，我可以按“基本散射—集体/非集体谱—仪器—应用—反演—AI”的顺序阅读，而不需要自己判断先后关系。
2. 作为当前 OTS 代码维护者，我可以看到真实实验中的探针、散射几何、密度、温度、时间/空间分辨率、谱宽和分布函数类型的证据区间，并据此识别当前固定配置是否过窄或脱离实验。
3. 作为反演算法研究者，我可以找到不依赖神经网络的传统基线，从简单的 Maxwellian 参数拟合逐步到非参数/贝叶斯 EVDF 反演，并了解每种基线的可辨识量、先验、仪器卷积与不确定度输出。
4. 作为 Zotero 用户，我能在“等离子体物理”下直接看到已去重的核心文献、稳定 DOI/出版方链接、分类标签和结构化中文笔记。

### 1.3 研究问题

必须回答以下问题：

1. OTS 的单粒子散射、动态结构因子、介电响应、集体性参数 `alpha = 1/(k lambda_D)` 与电子/离子特征之间是什么关系？
2. OTS 常用于哪些等离子体：磁约束聚变、惯性约束/高能量密度、激光产生等离子体、低温/放电等离子体、天体实验室类比等；每类诊断目的是什么？
3. 实验常见或已报道的参数区间是什么：`n_e`、`T_e`、`T_i`、流速、组分/电离度、探针波长、脉宽/能量、散射角、`alpha`、谱仪分辨率、采样体积、时间和空间分辨率？
4. 典型预期 EVDF/IVDF 包括哪些形态：Maxwellian、漂移 Maxwellian、双温/多组分、超高斯、kappa/长尾、束流/平台/空穴、各向异性、热电子尾、碰撞或波粒相互作用导致的非 Maxwellian？哪些形态已被 OTS 实验直接辨识，哪些主要是理论或合成数据假设？
5. 传统反演如何从光谱得到参数或速度分布：前向模型的非线性最小二乘/最大似然、多 Maxwellian/超高斯等参数化拟合、正则化速度网格或基函数反演、最大熵、贝叶斯/MCMC/嵌套采样与不确定度量化？
6. AI/ML 在 OTS 中已经解决了哪些任务：去噪、快速拟合、等离子体参数估计、异常检测、谱到 EVDF 的反演；与传统方法相比证据强弱和适用范围是什么？
7. 当前项目的 `351 nm, 123°`, `n_e=1.81e26 m^-3`, `T_e≈1370 eV`, `T_i≈171.3 eV`, H/C 混合、三段独立归一化、`[-0.8c,0.8c]` EVDF 支持域和合成 Gaussian/热尾先验，分别落在什么实验语境内？哪些假设需要随机化、仪器化或设置域外测试？

### 1.4 文献覆盖与数量

采用“核心集而非无界堆积”的默认规模：

- 1 本用户已有教材作为锚点，不重复制造版本；
- 约 8–12 篇基础理论、综述与实验系统文献；
- 约 12–20 篇覆盖主要等离子体类型的代表性实验文献；
- 约 8–15 篇非 Maxwellian / EVDF 与传统反演文献；
- 约 5–12 篇 AI/ML OTS 或高度相邻的光谱反演文献；
- 预计最终核心条目约 35–55 条。若某分支证据不足，宁缺毋滥并在笔记中标记证据缺口。

### 1.5 纳入与排除规则

纳入：

- 同行评议论文、权威教材/专著、正式会议论文或机构技术报告；
- 能直接支持理论、仪器、应用参数、分布函数、传统反演或 AI 方法中的至少一个研究问题；
- 有可核验的 DOI、出版方页面、arXiv/机构永久链接或完整书目信息；
- 参数范围优先采自原始实验论文的正文、表格或补充材料。

排除：

- 仅标题含 “Thomson scattering” 但属于非等离子体、非光学诊断或无法支持本任务的条目；
- 二手网页、营销材料和无法追溯原始数据的概述；
- 重复预印本与正式版并存时，默认保留正式版并在 extra/note 中记录预印本链接；
- 泛化的机器学习光谱论文，除非其方法可清楚映射到 OTS 正向模型、噪声或不确定度问题。

### 1.6 输出物

1. Zotero 独立集合及子集合/等价标签结构。
2. 去重后的核心文献条目，包含标题、作者、年份、期刊/书籍、DOI 或稳定链接；尽量保留摘要，是否自动下载 PDF 取决于合法可用性和 Zotero 能力，不以 PDF 齐全作为验收条件。
3. Zotero 总览笔记《AI Enhanced OTS：由浅入深阅读路线与知识框架》。
4. Zotero 总览笔记《OTS 实验应用参数证据表与当前代码配置审计》。
5. Zotero 总览笔记《传统 OTS 反演基线：从参数拟合到非参数贝叶斯反演》。
6. 工作区研究清单与证据表（Markdown/CSV，作为 Zotero 写入和验收的可审计中间产物；具体路径在执行前固定）。
7. 最终简报：集合路径、条目数、重复/缺失情况、最重要的配置风险和建议的传统基线优先级。

## 2. Design（设计）

### 2.1 架构决策

采用四层证据架构：

```text
现有 Zotero 库盘点
       ↓
检索矩阵（理论 / 实验体系 / EVDF / 传统反演 / AI）
       ↓
原始来源核验 + 参数/方法结构化抽取 + 去重
       ↓
Zotero 集合、标签、条目与三份总览笔记
```

关键决策：

- 先查本地 Zotero，复用现有教材和条目，避免重复。
- 搜索结果只是候选；关键结论必须回到 DOI、出版方、期刊或原始论文页面核验。
- 参数范围不做“所有文献最小值到最大值”的无语境合并，必须按等离子体类别和 `alpha`/集体性分组。
- “文献明确报告的值”“由几何和参数计算得到的值”“本任务综合建议”三类信息严格分栏。
- 传统基线按可实现难度分级，优先提供能够与当前 AI 模型公平比较的同一前向模型、同一仪器响应和同一测试集方案。
- Zotero 写入在候选清单审校和去重后进行；集合创建、导入和笔记写入均属于用户已明确授权的范围。

### 2.2 数据模型

#### `LiteratureRecord`

| 字段 | 含义 |
|---|---|
| `canonical_id` | 优先 DOI；否则 ISBN、arXiv ID 或规范化标题哈希 |
| `title/creators/year/venue` | 规范化书目信息 |
| `url/doi` | 稳定来源 |
| `document_type` | book/review/theory/experiment/method/AI |
| `plasma_regime` | MCF/ICF-HEDP/laser-produced/low-temperature/other |
| `knowledge_level` | 入门/进阶/专题/方法 |
| `evidence_roles` | 支持的研究问题列表 |
| `zotero_item_key` | 导入或复用后的 Zotero item key |
| `source_status` | metadata-only/abstract-checked/fulltext-checked |
| `notes` | 纳入理由、证据限制和阅读顺序 |

#### `ExperimentalParameterEvidence`

| 字段 | 含义 |
|---|---|
| `paper_id` | 对应 `LiteratureRecord.canonical_id` |
| `plasma_regime/facility/target` | 实验语境 |
| `probe_wavelength_nm` | 探针波长 |
| `probe_pulse_duration/energy` | 探针脉宽与能量（若报告） |
| `scattering_angle_deg` | 散射角 |
| `n_e_m3` | 电子密度 |
| `T_e_eV/T_i_eV` | 温度 |
| `flow_m_s` | 体速度/漂移 |
| `composition/ionization` | 物种与平均电离度 |
| `alpha` | 文献报告值或标明公式的推导值 |
| `spectral/spatial/temporal_resolution` | 仪器分辨率 |
| `measured_feature` | EPW/IAW/电子特征/离子特征/非集体翼等 |
| `distribution_claim` | Maxwellian/双温/超高斯/束流等及证据强度 |
| `locator` | 页码、图表、补充材料或段落定位 |

#### `InversionMethodEvidence`

| 字段 | 含义 |
|---|---|
| `paper_id` | 来源 |
| `method_family` | NLLS/MLE/MaxEnt/Tikhonov/Bayesian/AI 等 |
| `unknowns` | `n_e,T_e,T_i,u,Z,f(v)` 等 |
| `forward_model` | Maxwellian dielectric/arbitrary VDF/instrument-convolved 等 |
| `parameterization` | 单/多 Maxwellian、super-Gaussian、spline、Fourier、velocity-grid |
| `constraints` | positivity/normalization/smoothness/Hermitian/priors |
| `noise_model` | Gaussian/Poisson/empirical covariance 等 |
| `optimizer/inference` | LM/L-BFGS/MCMC/nested sampling 等 |
| `uncertainty_output` | covariance/profile likelihood/posterior/none |
| `validation` | synthetic/experiment/cross-diagnostic |
| `baseline_relevance` | 与当前代码可比性的说明 |

### 2.3 检索矩阵

搜索词组按以下轴组合，并使用同义词扩展：

1. 基础：`optical Thomson scattering plasma dynamic structure factor collective noncollective`。
2. 仪器：`Thomson scattering spectrometer calibration instrument function stray light spatial temporal resolution`。
3. 应用：分别叠加 `tokamak`, `stellarator`, `inertial confinement fusion`, `high energy density`, `laser produced plasma`, `warm dense matter`, `low temperature plasma`, `gas discharge`。
4. EVDF：叠加 `electron velocity distribution`, `non-Maxwellian`, `super-Gaussian`, `kappa`, `hot electron`, `beam`, `anisotropic`, `inverse bremsstrahlung`, `SRS`。
5. 传统反演：叠加 `spectral fitting`, `inverse problem`, `regularization`, `maximum entropy`, `Bayesian`, `MCMC`, `uncertainty`, `distribution function reconstruction`。
6. AI：叠加 `machine learning`, `deep learning`, `neural network`, `surrogate`, `physics-informed`, `spectral inversion`。

### 2.4 Zotero 分类与标签

每条文献至少有：

- 一个主题标签：`OTS::theory`, `OTS::instrument`, `OTS::experiment`, `OTS::EVDF`, `OTS::inverse`, `OTS::AI`；
- 一个等离子体语境标签（适用时）：`plasma::MCF`, `plasma::ICF-HEDP`, `plasma::laser-produced`, `plasma::low-temperature`；
- 一个阅读层级标签：`route::01-foundation` 至 `route::06-frontier`；
- 证据不足或仅相邻方法的条目标记 `evidence::adjacent`，避免与直接 OTS 证据混同。

### 2.5 知识框架

最终导读按以下依赖关系组织：

```text
单电子散射与实验几何
  → 密度涨落、S(k,ω)、介电函数
  → α 与集体/非集体谱区
  → 电子特征、离子特征及仪器卷积
  → 从谱峰拟合 ne/Te/Ti/u/Z
  → 非 Maxwellian 对 χ 和 f(ω/k) 的影响
  → 参数化传统反演
  → 正则化/贝叶斯非参数 EVDF 反演
  → AI surrogate / amortized inverse
  → 域覆盖、校准、噪声、OOD 与不确定度
```

### 2.6 当前项目配置审计方法

对每个代码参数给出四列：

1. 当前值/分布；
2. 文献中的已报道区间（分实验语境）；
3. 覆盖判断：典型、边缘、缺少证据或明显过宽/过窄；
4. 建议：保留固定、扩大为随机变量、加入仪器模型、添加 OOD split 或只作为特定实验 case。

特别检查：

- `alpha` 是否覆盖 collective 与 non-collective 边界；
- 当前 100–600 nm 观测窗对 351 nm 探针与 `[-0.8c,0.8c]` 的物理映射、探测器/光学可实现性；
- 三波段独立归一化丢失的相对强度信息；
- 固定 H/C、固定 `T_i`、零漂移与真实实验参数退化之间的冲突；
- Gaussian 混合先验是否覆盖 super-Gaussian、beam、anisotropy、hot-tail、kappa 等实验形态；
- 是否缺少仪器函数、杂散光、光子噪声、标定误差与背景辐射。

### 2.7 传统基线分级

建议至少形成四级基线：

1. **B0：参数化 Maxwellian NLLS**——同一正向模型和仪器卷积，拟合 `n_e,T_e,T_i,u` 与必要的幅度/背景；作为最容易复现的实验常规基线。
2. **B1：扩展参数化 NLLS/MLE**——双温、多 Maxwellian、super-Gaussian 或热尾分量；用 AIC/BIC、profile likelihood 或 bootstrap 控制额外自由度。
3. **B2：正则化 EVDF 反演**——velocity-grid、spline 或 Fourier 系数，施加非负、归一、平滑/Hermitian 约束，交叉验证或 L-curve 选择正则强度。
4. **B3：贝叶斯反演**——对 B0/B1/B2 的参数或低维系数求后验，显式输出退化与可信区间；用于给 AI 置信度和 OOD 检测提供参照。

公平比较要求：同一 synthetic/experimental spectrum、相同 spectral window、相同仪器响应与噪声模型；AI 与传统方法同时报告点估计误差、物理约束违背、计算时间和不确定度/失败率。

## 3. Workflow Function Interfaces 与伪代码

以下是研究工作流的函数合同，不要求为仓库新增可执行代码；执行记录应能按这些接口追溯。

### 3.1 `probe_zotero`

```text
probe_zotero() -> ZoteroStatus
```

伪代码：

```text
status = run zotero helper "status --json"
if local API disabled:
    enable and restart Zotero using approved helper
assert connector/local API is reachable
return status including library route and write capability
```

### 3.2 `resolve_destination_collection`

```text
resolve_destination_collection(parent_title: str,
                               project_title: str,
                               child_titles: list[str]) -> CollectionPlan
```

伪代码：

```text
collections = list Zotero collections
find exact parent_title "等离子体物理"
if absent:
    stop and ask user; do not silently create a different top-level taxonomy
find or create exact project_title under parent
for each child_title:
    reuse exact child or create it under project
return collection keys and any UI/API fallback needed
```

### 3.3 `inventory_existing_ots`

```text
inventory_existing_ots(scope: CollectionPlan,
                       queries: list[str]) -> ExistingInventory
```

伪代码：

```text
search local library for OTS aliases, key authors, DOI and title variants
collect item key, BibTeX key, DOI, ISBN, title and current collections
identify the Sheffield book and preserve its existing record
return exact duplicates, near duplicates and reusable records
```

### 3.4 `build_search_matrix`

```text
build_search_matrix(research_questions: list[str],
                    plasma_regimes: list[str]) -> list[SearchQuery]
```

伪代码：

```text
for each knowledge branch:
    combine OTS core terms with method/regime synonyms
    assign expected evidence fields and stop condition
deduplicate equivalent queries
prioritize primary experiments and original inversion methods
return bounded query list
```

### 3.5 `search_and_verify_candidate`

```text
search_and_verify_candidate(query: SearchQuery) -> list[LiteratureRecord]
```

伪代码：

```text
search scholarly web and publisher/DOI sources
for each candidate:
    verify title, authors, year, venue, DOI/stable URL
    confirm relevance from abstract or full text when available
    classify evidence role and source_status
    reject ambiguous Thomson-scattering meanings
return verified candidates with source links
```

### 3.6 `extract_experimental_parameters`

```text
extract_experimental_parameters(record: LiteratureRecord) 
    -> list[ExperimentalParameterEvidence]
```

伪代码：

```text
read abstract, methods, tables, figures and supplements as available
extract only explicitly reported quantities with units and locators
if alpha is derived:
    store inputs, formula and mark as derived
normalize units without discarding original text
separate diagnostic capability from the specific shot value
return one evidence row per experimental configuration
```

### 3.7 `extract_inversion_method`

```text
extract_inversion_method(record: LiteratureRecord) 
    -> InversionMethodEvidence | None
```

伪代码：

```text
identify unknowns, forward model and instrument convolution
identify parameterization, constraints, noise model and optimizer
record uncertainty method and validation data
map method to B0/B1/B2/B3 or AI category
state whether the method reconstructs parameters or an actual VDF
return structured evidence or None
```

### 3.8 `deduplicate_and_rank`

```text
deduplicate_and_rank(existing: ExistingInventory,
                     candidates: list[LiteratureRecord])
    -> tuple[list[LiteratureRecord], DuplicateReport]
```

伪代码：

```text
match exact DOI/ISBN/arXiv IDs
then match normalized title + first author + year
prefer published version over preprint unless metadata is poorer
reuse existing Zotero item instead of importing duplicate
score directness, evidence completeness, representativeness and reading value
retain the bounded core set and log excluded near-duplicates
return canonical records and duplicate report
```

### 3.9 `audit_current_configuration`

```text
audit_current_configuration(current_config: dict,
                            evidence: list[ExperimentalParameterEvidence])
    -> ConfigurationAudit
```

伪代码：

```text
group evidence by plasma regime and collective/noncollective condition
for each current parameter:
    compare only against compatible evidence groups
    mark typical/edge/outside/insufficient-evidence
    distinguish reported ranges from synthesis recommendations
identify coupled parameters and instrument feasibility constraints
produce prioritized changes without modifying source code
```

### 3.10 `import_or_reuse_zotero_items`

```text
import_or_reuse_zotero_items(records: list[LiteratureRecord],
                             destination: CollectionPlan) -> ImportManifest
```

伪代码：

```text
for each canonical record:
    if exact existing Zotero item:
        add/reclassify it in destination without duplication
    else:
        import verified BibTeX/RIS/connector metadata
    apply category, regime and reading-route tags
    record Zotero item key and destination collection
return manifest of reused/imported/failed items
```

### 3.11 `write_synthesis_notes`

```text
write_synthesis_notes(destination: CollectionPlan,
                      roadmap: KnowledgeRoadmap,
                      parameter_audit: ConfigurationAudit,
                      baselines: BaselinePlan) -> list[ZoteroNoteRef]
```

伪代码：

```text
render three Chinese notes with citations linked to Zotero items
label every range as reported/derived/synthesized
include reading order, evidence gaps and current-project implications
create notes in destination collection via local API or controlled UI fallback
return note keys/titles
```

### 3.12 `verify_zotero_delivery`

```text
verify_zotero_delivery(destination: CollectionPlan,
                       manifest: ImportManifest,
                       expected_notes: list[str]) -> AcceptanceReport
```

伪代码：

```text
re-list destination collections and items
assert no duplicate DOI/ISBN/title identities inside project set
assert each item has stable metadata, a topic tag and reading level
assert all expected synthesis notes exist
sample-check evidence locators and source links
report counts, failures, unresolved metadata and evidence gaps
```

## 4. Tasks（任务拆解）

- **T0 [无依赖]：确认本 SPEC。** 用户确认默认集合名称、35–55 条核心规模及三份中文总览笔记。
- **T1 [依赖 T0]：Zotero 就绪与现有库盘点。** 探测本地 API，定位“等离子体物理”，查找 Sheffield 教材和已有 OTS 条目。
- **T2 [依赖 T1]：建立检索矩阵与候选清单。** 分理论、仪器、应用、EVDF、传统反演、AI 六支检索。
- **T3A [依赖 T2]：实验应用/参数证据抽取。** 覆盖主要等离子体语境，整理参数和预期分布。
- **T3B [依赖 T2]：传统反演与 AI 方法证据抽取。** 区分参数拟合与真实 VDF 反演，建立 B0–B3 基线层级。
- **T3C [依赖 T2]：基础理论/阅读路线抽取。** 把教材章节与原始论文放入学习依赖图。
- **T4 [依赖 T3A/T3B/T3C]：交叉核验、去重与核心集裁剪。** 核验 DOI/元数据/出处，记录排除原因。
- **T5 [依赖 T4]：当前项目配置审计。** 将文献证据与仓库当前配置逐项对照，不修改代码。
- **T6 [依赖 T4]：Zotero 写入。** 创建/复用集合，导入或复用条目，设置分类和标签。
- **T7 [依赖 T5/T6]：写三份 Zotero 总览笔记。** 导读、参数/配置审计、传统基线。
- **T8 [依赖 T7]：验收。** 重新读取 Zotero 集合，检查条目数、重复、标签、笔记和失败项，提交最终简报。

子代理分工（遵守共享工作区单写者原则）：

- 子代理 A：只读研究实验应用与参数证据，返回结构化建议，不写 Zotero/文件。
- 子代理 B：只读研究传统反演与 AI 方法谱系，返回结构化建议，不写 Zotero/文件。
- 主代理：基础路线、交叉核验、全部本地文件和 Zotero 写入、最终验收。

## 5. Acceptance Criteria / Tests（验收标准）

### 5.1 SPEC 验收

- 用户明确确认执行范围，或提出修改后形成已确认版本。
- 集合名称、默认文献规模、笔记语言和是否包含 PDF 自动获取没有歧义。

### 5.2 文献质量验收

- 最终核心集预计 35–55 条；若少于 35 条，必须说明哪个分支缺少可靠原始文献；若超过 55 条，必须说明额外条目的必要性。
- 每条文献至少有标题、作者、年份和 DOI/ISBN/稳定链接之一。
- 所有关键实验参数范围至少关联到一篇原始实验来源，并标明语境与 locator；不得只引用搜索摘要拼接范围。
- “直接 OTS 证据”和“相邻光谱/ML 方法”明确区分。
- 预印本/正式版、书籍版本和已有 Zotero 条目完成去重。

### 5.3 知识框架验收

- 导读覆盖理论、集体性、仪器、应用、EVDF、传统反演、AI 与 OOD/不确定度。
- 至少覆盖四类等离子体语境，并说明 OTS 在每类中通常诊断的量和可见谱特征。
- 明确区分“从光谱拟合参数”和“从光谱重建 VDF”两类问题。
- B0–B3 每级至少有一个文献依据、输入/输出、约束和公平比较方案。

### 5.4 参数与项目审计验收

- 对当前 `351 nm, 123°, n_e, T_e, T_i, H/C, velocity support, spectral windows, EVDF prior, normalization` 均有逐项结论。
- 对仪器函数、杂散光、噪声、背景、标定和绝对/相对强度信息给出是否缺失的判断。
- 建议按“立即应补的合成基线/中期扩域/需特定实验约束”分级，不把跨装置的总范围直接变成训练采样范围。

### 5.5 Zotero 验收

- “等离子体物理”下存在独立项目集合，或存在经过说明的等价单集合+标签结构。
- 三份指定中文总览笔记可见。
- Zotero 复查无已知 DOI/ISBN/规范化标题重复项。
- 导入/复用/失败/缺元数据数量有清单；失败项不被静默忽略。
- Sheffield 教材若已存在则复用，不创建重复版本。

### 5.6 非目标

- 本任务不实现传统反演代码，不更改 AI 模型，不重新生成训练数据。
- 不承诺为所有文献自动下载受版权或访问控制限制的 PDF。
- 不把横跨所有 OTS 装置的极端参数并集直接当作一个合理训练分布。

### 5.7 最终验收记录（2026-08-24）

- Zotero `等离子体物理/OTS`：83 个含 child 的 item、73 个 top-level item。
- 本任务核心参考文献：69 条，其中新导入 65 条、复用 4 条；另有 3 份中文导读及各 1 个 child note。
- 两份 BibTeX 合计 64 个唯一 DOI，回读缺失 0、DOI 重复 0；另有 1 条无 DOI 的历史反演记录。
- 65 条项目导入文献缺题名/作者/年份/稳定标识均为 0，缺路线/主题/证据标签均为 0。
- 规范化题名有 1 组跨既有记录碰撞：Sheffield 教材与其同名书评；二者 item type、出版信息和 DOI 不同，记录为有意保留而非重复导入。
- 最终规模超过默认 55 条上限，是为了补齐 MCF、低温三种散射制度、HEDP/pulsed-power、多视线与机制型非 Maxwellian 的直接实验锚点；理由和逐项清单见 `literature/ai_enhanced_ots/zotero_import_manifest.md`。

## 6. 待用户确认的默认选择

若用户确认本 SPEC 而不另作修改，则采用：

1. 集合名：`AI Enhanced OTS｜实验参数·传统反演·AI`；
2. 核心规模：35–55 条；
3. 笔记：中文，保留英文题名与术语；
4. Zotero 中合法可获取的 PDF 可保留，但不为了 PDF 齐全阻塞任务；
5. 最终给出面向当前代码的配置审计，但不修改代码；
6. 传统基线先设计到 B0–B3，后续另立 SPEC 再实现。
