# Zotero 导入与验收清单

## 1. 目标位置与结构

- 文库：`我的文库`
- 父分类：`等离子体物理`，collection key `2F59K6XV`
- 独立子分类：`OTS`，collection key `ASFEIZST`
- 采用“单独集合 + 路线/主题/证据标签 + 三份中文导读”的等价结构；没有再创建八层子分类。
- 验收时间：2026-08-24（Asia/Shanghai）

## 2. 本轮写入与复用

| 类别 | 数量 | 说明 |
|---|---:|---|
| 从 `ots_core_references.bib` 新导入 | 43 | 42 条 DOI 记录 + 1 条无 DOI 的 Zhuravlev 1974 历史反演记录 |
| 从 `ots_experiment_expansion.bib` 新导入 | 22 | 全部有 DOI；补充 MCF、低温/推进、HEDP、pulsed-power 与多视线实验 |
| 新导入文献合计 | 65 | 均带 `project::AI-Enhanced-OTS`、`route::*`、`OTS::*`、`evidence::*` 标签 |
| 复用既有相关记录 | 4 | Sheffield 教材、Froula 2012 书评、Zhao 2019、Turnbull 2020；未重复导入 |
| 新建中文导读父记录 | 3 | 每条各有 1 个 child note |
| 本任务核心参考记录 | 69 | 65 条新导入 + 4 条复用；三份导读不计入文献数 |

三份导读记录：

1. `HH7SHNFG`：AI Enhanced OTS：由浅入深阅读路线与知识框架；
2. `8QUH2A3C`：OTS 实验应用参数证据表与当前代码配置审计；
3. `PE34WPET`：传统 OTS 反演基线：从参数拟合到非参数贝叶斯反演。

复用记录：

- `5UWHVEIZ`：用户已有 Sheffield 教材；
- `PNDUB3R4`：DOI `10.13182/FST12-A13342`，同名书评；
- `QNLJH8RR`：DOI `10.1063/1.5090971`，ICF gas-filled hohlraum 综述；
- `64BGWYC6`：DOI `10.1038/s41567-019-0725-z`，Langdon effect/CBET；本轮只把既有条目加入 `OTS`，保留其原分类。

## 3. 本地 API 回读验收

对 collection `ASFEIZST` 的最终回读结果：

| 检查 | 结果 |
|---|---:|
| 全部 item（含 notes/attachments） | 83 |
| top-level item | 73 |
| child item | 10 |
| 带项目标签的新导入文献 | 65 |
| 三份导读父记录/child note | 3 / 3 |
| 两份 BibTeX 中预期 DOI | 64 |
| 在 OTS 集合中找到的预期 DOI | 64 |
| 缺失 DOI | 0 |
| 新项目 DOI 重复 | 0 |
| 新项目规范化题名重复 | 0 |
| 缺题名/作者/年份/稳定标识 | 0 / 0 / 0 / 0 |
| 缺 `route::*`/`OTS::*`/`evidence::*` 标签 | 0 / 0 / 0 |

`OTS` 中还有 1 条本轮开始前已经存在、与当前整理无关的 Raman instability 条目，因此 `73 top-level` 不等于 `69 文献 + 3 导读` 的简单相加。

## 4. 去重边界与已知碰撞

- 64 个本轮 BibTeX DOI 彼此唯一；与 OTS 集合回读结果完全匹配。
- Sheffield 教材 `5UWHVEIZ` 与 Fusion Science and Technology 的两页书评 `PNDUB3R4` 规范化题名相同，但 item type、出版信息和 DOI 不同，属于“书与书评”，不是重复导入。因二者均为用户既有记录，予以保留并明确记录。
- Turnbull 2020 已存在于文库，故从扩展 BibTeX 中排除，再把原 item 加入 OTS；没有制造第二条 DOI 记录。
- 未删除、覆盖或移动用户原有记录；只给 Turnbull 既有 item 增加了一个 collection membership。

## 5. 导入状态与缺口

- 导入失败：0。
- 缺失稳定元数据：0。
- Zotero connector 的两个大批量请求在客户端等待窗口内返回 timeout，但本地 API 随后逐条回读确认全部 65 条落库；timeout 是响应等待问题，不是导入失败。
- 没有为 PDF 完整度阻塞任务；受版权或访问控制限制的全文未强制下载。
- 详细参数 locator、证据等级与代码审计保存在同目录 Markdown 文件中；Zotero 的三份中文记录是可检索的执行摘要，不替代逐项证据表。

## 6. 对 35–55 条默认规模的偏离

最终核心参考数为 69，超过 SPEC 的默认上限 14 条。保留扩展项的理由是：原始核心集虽覆盖理论、反演和 AI，但不足以支撑用户特别要求的“常见实验参数区间、等离子体类型和预期分布函数”。新增 22 条实验扩展用于分别覆盖：

- MCF 常规系统与历史非 Maxwellian 观测；
- 低密度 noncollective、 atmospheric jet 和 collective metal-vapor arc 三种低温制度；
- 直接 EVDF/EEDF 形变、probe 扰动与多视线可辨识性；
- HEDP 的 super-Gaussian、heat-flux、counterstreaming、hot-electron 与非热波场景；
- pulsed-power/reconnection 的流速、离子组分和时空积分约束。

这些条目不是为凑数量，而是为避免把跨制度的少量锚点误写成统一“常见范围”。
