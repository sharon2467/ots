# OTS 实验应用参数证据表

更新时间：2026-08-24

## 0. 读表规则

- 表中数值优先采用原始论文/正式系统设计明确报告的量；单位统一到 SI，同时保留实验常用单位。
- “代表性包络”是本核心文献集的数量级摘要，**不是 OTS 的全局硬上下限，也不应原样当作一个网络的统一采样范围**。
- `alpha = 1/(k lambda_D)`；部分文献使用其倒数或用其他符号，比较时必须核对定义。
- “分布”均指沿散射波矢 `k` 的一维投影，除非论文使用多个散射角/矢量做额外约束。
- 证据标记：`[报]` 论文直接报告；`[扫]` 同一系统的扫描/设计范围；`[图读]` 从图轴读取；`[推]` 由报告参数计算；`[综]` 综述汇总；`NR` 为不可可靠提取。

## 1. 磁约束聚变（tokamak/stellarator）

| 证据 | 明确报告的范围/配置 | 测量目标与分布假设 | 证据性质 |
|---|---|---|---|
| Bretz et al., PLT multichannel TS, 1977/1978 | `n_e≈10^13–10^14 cm^-3 = 10^19–10^20 m^-3`；`T_e≈0.05–3 keV`；约 1.2 cm 空间元 | 多点 `T_e/n_e` 剖面；常规 Maxwellian 参数反演 | 已运行系统/技术报告与仪器论文 |
| Bilkova et al., COMPASS, DOI `10.1016/j.nima.2010.03.121` | 两台 Nd:YAG，各 1.5 J、30 Hz，可合成 60 Hz；core 24 点约 10 mm；edge 32 点约 3–5 mm | core/edge `T_e/n_e` profile | 系统设计 |
| ITER core TS design | `T_e=0.5–40 keV`；`n_e=3×10^19–3×10^20 m^-3`；1064 nm；约 66 mm（可优化到边缘约 20 mm） | 反应堆 core `T_e/n_e`，目标精度约 10%/5% | 设计量程，不是单次实测范围 |
| HL-2M vertical edge TS | 文献给出 TS 通用/系统目标量程 `T_e≈1–10^4 eV`、`n_e≈10^18–10^21 m^-3`；1064 nm、0.6–2.0 J、30 Hz、15 ns | edge profile；Raman/相对标定 | 系统开发与首结果 |
| Lee et al., KSTAR, DOI `10.1063/1.4961079` | NN 输出 `T_e`，与 χ²/传统 teacher 对照，约 20×分析加速 | 快速参数回归；不恢复 EVDF | 直接光学 TS 的 AI 参数估计 |
| van Lammeren et al., TORTUR, DOI `10.1088/0029-5515/32/4/I10` | 两个观察方向；30 个脉冲平均 `[报]`；非热凹陷约为 Gaussian 峰值的 7% `[报]` | 径向+切向谱联合构造 `f(v_perp,v_parallel)`；空间丝化可与真实 EVDF 退化 | 早期直接非 Maxwellian 实验 |
| Zhuang et al., TCV, DOI `10.1088/0741-3335/47/9/011` | ECCD 1.35 MW `[报]`；不同谱段推得的 `T_e` 相差 25–30% `[报]` | CQL3D/三参数 bi-Maxwellian；预期高能尾与低能 bulk 压平 | 直接实验分析 |
| Beausang et al., JET HRTS, DOI `10.1063/1.3567785` | 1064 nm、90°、5 J、20 Hz；论文重点 `T_e=1.5–6 keV`，示例 `n_e=2×10^19 m^-3`；63 点、约 15 mm `[报]` | 未发现可分辨 Lorentzian 热尾或 bulk flattening；约 5% 形状差异已接近系统灵敏度 | 重要阴性结果；`[推] alpha≈9.3×10^-4–1.9×10^-3` |
| Pasch et al., W7-X, DOI `10.1063/1.4962248` | 1064.14 nm、0.7–1.5 J、约 10 ns、10 Hz；`T_e=20 eV–10 keV`、`n_e=2×10^18–5×10^20 m^-3` `[扫]`；10 个体积、25–37 mm | 标准 Maxwellian/Bayesian `n_e/T_e` profile | 已运行系统；量程是优化包络，不是单发范围 |

代表性工程包络：

```text
ne: 10^18–3×10^20 m^-3
Te: ~1 eV（边缘/偏滤器）到 40 keV（ITER core 设计）
probe: Nd:YAG 1064 nm 最常见，也有 ruby/倍频系统
energy/rate: ~0.6–2 J，10–60 Hz（系统例）
spatial: few mm edge 到 few cm core
regime: 通常 alpha << 1，以 noncollective electron feature/polychromator 参数反演为主
```

预期分布：核心热电子常近 Maxwellian；ECRH/ECCD、NBI、逃逸电子、强电流和输运可产生 bi-Maxwellian、各向异性、bulk flattening 或弱非热尾。常规少通道 polychromator 对完整 EVDF 的可辨识性有限；“存在非 Maxwellian”与“系统能重建它”必须分开。单一观察方向还可能对沿磁场的尾部近乎盲视。

## 2. 低温、放电、基础和推进等离子体

| 证据 | 明确报告的范围/配置 | 测量目标与分布 | 证据性质 |
|---|---|---|---|
| Ghazaryan et al., LAPD, DOI `10.1063/5.0099172` | `n_e=4.0×10^12–2.8×10^13 cm^-3 = 4.0×10^18–2.8×10^19 m^-3`；`T_e=1.2–6.8 eV`；532 nm、90°；整束积分体积约 33 mm × 0.74 mm；单 fiber 亚毫米；1.5 nm notch | 基础磁化等离子体 `n_e/T_e`；noncollective | 已运行系统，Raman 标定 |
| Friss & Yalin, DOI `10.1364/OL.43.005343` | 1064.5 nm cavity-enhanced OTS；Xe/BaO hollow-cathode 例 `n_e=2.8×10^18 m^-3`、`T_e=2.7 eV`、SNR≈1100 | 提高低密度 TS 灵敏度 | 直接实测点；作者的检测极限预测不与实测混写 |
| Huang et al., DOI `10.1016/S0584-8547(00)00244-5` | 大气压 He microwave plasma，100/350 W；测得 EEDF 覆盖 0.1–6.6 eV | 邻波道强度差分直接求各向同性 EEDF；明显非 Maxwellian | 直接实验 EEDF 反演，适用前提严格 |
| Carbone et al., DOI `10.1088/0022-3727/45/47/475202` | 中等压力 Ar surface-wave discharge；TS 与 OES/CR model 对照 | 沿等离子体柱出现尾部耗尽的非 Maxwellian EEDF | 直接实验 + Boltzmann solver |
| Carbone & Nijdam review, DOI `10.1088/0741-3335/57/1/014026` | 一个具体 532 nm 系统的 ±4 nm 窗对应约 0.1–7 eV；讨论光泵浦、多光子电离、stray light 与激光扰动 | 低温非平衡 EEDF 常偏离 Maxwellian | 综述/仪器实践，不是统一量程 |
| Washeleski et al., DOI `10.1063/1.4821980` | 低密度 Hall-thruster/稀薄 TS，photon-counting | Poisson/count likelihood 下的 `T_e/n_e` MLE | 直接实验统计基线 |
| Vincent et al., DOI `10.1088/1361-6595/aabd13` | Xe hollow cathode；532 nm、90°、430 mJ、6 ns；`n_e=(9.41±0.75)×10^17 m^-3, T_e=2.27±0.18 eV`，最低实测 `1.8×10^16 m^-3, 0.46 eV`；6000 shots/约 10 min `[报]` | Maxwellian；另给“平滑—求导—速度变换”的历史直接 EEDF 反演 | `[推] alpha=5.2×10^-3/1.59×10^-3`；0.018/0.09 nm 分辨率 |
| Yamamoto & Yalin, DOI `10.1063/5.0180534` | pulsed Ar；532 nm、50 mJ、8 ns；`n_e≈1.7×10^19 m^-3, T_e=1.54 eV`，最低实测 `7.9×10^17 m^-3`；每点 `5×2000` shots `[报]` | Maxwellian；中心 `±0.5 nm` 和发射线显式 mask | 0.15 nm IRF；10 ns 是光学 gate，实际相位分辨为 µs |
| Law et al., DOI `10.1088/1361-6595/ac2be4` | bipolar HiPIMS；532 nm、240 mJ；`n_e~10^18 m^-3, T_e≈0.4–3 eV`；600–2400 shots `[报/图读]` | Gaussian/Maxwellian；二次电子热组分在现有 SNR 下不可稳健区分 | 0.071 nm；7.5 ns gate、约 5 µs 相位分辨；近靶 stray light 不可达 |
| Vincent et al., DOI `10.1088/1361-6595/ab6c42` | Hall thruster；532 nm、约 90°；`n_e=2.6×10^16–1.3×10^17 m^-3`、`T_e=3.7–58 eV`；漂移 `93 km/s` 到径向约 `±1000 km/s` `[报]` | drifting Maxwellian；近出口有弱非 Gaussian 偏差但 SNR 不足以稳定非参数反演 | `[推] alpha≈2×10^-4–1.5×10^-3`；强自发光可高于 TS 十倍 |
| Koiso et al., DOI `10.1016/j.vacuum.2023.112760` | microwave Xe cathode；532 nm；期刊正文中几何/能量/分辨率的可靠数值为 NR；会议给典型 `n_e~10^18 m^-3, T_e=2–4 eV` | 低流量下直接见非 Maxwellian 投影 EVDF；流量影响强于阳极电流 | 高 probe 功率可光电离亚稳态并人为增加 `n_e` |
| Slikboer & Walsh, DOI `10.1038/s41598-021-97185-8` | Ar plasma–liquid jet；532 nm、90°；代表 `n_e=1.4×10^21 m^-3, T_e=2.8 eV`；1200–6000 shots `[报]` | Gaussian/Maxwellian | 约 0.141 nm、50 µm；`[推] alpha≈0.14–0.29`，弱 noncollective/intermediate |
| Tomita et al., DOI `10.1541/ieejfms.133.458` | atmospheric W/W–Cu arc；532 nm、90°、90 mJ、10 ns；`n_e=0.8–2.1×10^23 m^-3, T_e=1–2.2 eV` `[报]` | Maxwellian/equilibrium；离子特征 `<0.1 nm`，落在遮挡/IRF 内，实际只拟合电子特征 | 0.55 nm、0.52 mm、5 shots；`[推] alpha≈1.5–3.6`，明确 collective |
| Kono & Funahashi, DOI `10.1063/1.1493659` | 25 mTorr C4F8/Ar、CF4/Ar ICP；精确光学参数在当前可核正文中 NR | 约 5 eV 处的非 Maxwellian 弯曲；振动能损与非局域电子动力学 | 直接实验 EEDF 证据 |
| Devin et al., DOI `10.1063/5.0310529` | partially magnetized Ar/Kr；`n_e≈2–3.5×10^17 m^-3, T_e≈3–3.5 eV`；532 nm、90°、180 mJ、7 ns、1000 shots `[报]` | LTS 近 Maxwellian；低 B 有弱各向异性高能尾；与 Druyvesteyn/Langmuir 对照 | 磁化条件下 probe 二阶导数会严重扭曲低能 EEPF |

本核心集应分三层读，而不是压成单一低温参数箱：

```text
稀薄 noncollective: ne≈1.8×10^16–10^19 m^-3, Te≈0.46–58 eV, alpha≈2×10^-4–5×10^-3
大气压 plasma jet: ne~10^21 m^-3, Te~1–3.5 eV, alpha~0.1–0.3
大气压 metal-vapor arc: ne≈0.8–2.1×10^23 m^-3, Te≈1–2.2 eV, alpha≈1.5–3.6
probe/geometry: 532 nm、近 90°最常见；cavity-enhanced 例为 1064 nm
instrument: low-density 例约 0.018–0.15 nm；常积累 10^3–6×10^3 shots
mask: 中心通常丢失约 ±0.3–0.5 nm；必须作为观测 mask 进入训练与反演
limitation: stray light、emission、notch、光泵浦/光电离、probe heating 与时间平均常支配误差
```

预期分布：最常见仍是 Maxwellian 或 drifting Maxwellian。较可靠的非 Maxwellian 场景包括低流量微波阴极羽流、含氟 ICP 的振动能损/非局域动力学、Hall 推进器近出口，以及部分磁化等离子体的各向异性高能尾。只有在 noncollective、近各向同性、仪器响应与中心遮挡得到控制时，谱形才较直接对应一维投影 EEDF；`T_i` 在这组低温实验中几乎从未由离子特征可靠取得。

## 3. ICF/HEDP、激光产生和脉冲功率等离子体

| 证据 | 明确报告的范围/配置 | 测量目标与分布 | 证据性质 |
|---|---|---|---|
| Ross et al., DOI `10.1103/PhysRevLett.104.105001` | N2 gas jet；`n_e=3×10^18–7×10^19 cm^-3 = 3×10^24–7×10^25 m^-3`；`T_e=85–700 eV`；相速度 `0.03–0.12c` | 同时测 electron/ion features；显示高相速度下相对论修正 | 直接集体 TS 实验 |
| Froula et al., DOI `10.1063/1.2203232` | OMEGA CH/SiO2 hohlraum；约 263 nm、101°、约 200 J/1 ns/60 µm；`T_e=2–4 keV`，4 keV 时 `n_e=6×10^26 m^-3, alpha=1.4` `[报]` | H/C 双离子 IAW mode，可独立约束 `T_e/T_i` | 约 `60×110×80 µm^3`、50 ps、0.05 nm |
| Ross et al., DOI `10.1103/PhysRevLett.110.145005` | OMEGA counterstreaming CH2；527 nm、116.8°；示例 `n_e=4×10^24 m^-3, T_e=300 eV, T_i=100 eV, U=0.005c` `[报]` | 两个 Doppler-shifted IAW 组分显示 counterstream；不能用单个加宽 Maxwellian 代替 | 直接实验；电子/离子通道各自时空分辨 |
| Henchen et al., DOI `10.1103/PhysRevLett.121.125001` | OMEGA Al blowoff；526.5 nm、60°、40 J/2 ns；`T_e=1.27→1.12 keV`、`n_e≈1.1–1.3×10^26 m^-3` `[报/图读]` | 红/蓝 EPW 幅度不对称测奇对称 heat-flux perturbation；热流约比 Spitzer–Härm 小 2 倍 | 20 ps、0.5 nm、约 `50^3 µm^3` |
| Milder et al., DOI `10.1103/PhysRevLett.124.025001` | Ar gas jet；527 nm、约 100°、60 ps；58 ps 时 SG 拟合 `T_e=406 eV, n_e=2.04×10^25 m^-3, m=3.1`；`k lambda_D≈0.6` `[报]` | 加热期 `m≈3.1–3.4` super-Gaussian；关断后碰撞电离继续改变 EDF | 2.2 ps、1.27 nm、`60×60×40 µm^3` |
| Turnbull et al., DOI `10.1038/s41567-019-0725-z` | OMEGA CBET；时间/空间分辨 EPW+IAW | Langdon `m` 改变电子特征，也移动 IAW 频率并改变 CBET | 直接实验；已存在 Zotero 条目复用 |
| Banasek et al., DOI `10.1109/TPS.2018.2850278` | Al jet `n_e≈5×10^18 cm^-3 = 5×10^24 m^-3`；初始 `T_e≈25 eV`，约 2 ns 内被 probe 加热到 80–100 eV；526.5 nm，最高 10 J；sub-ns streak | jet 温度演化和探针加热 | 直接实验；也是 probe-perturbation 警告 |
| Zhao et al., DOI `10.1063/1.5090971` | 中国 10 kJ 与 100 kJ 级激光设施的 gas-filled hohlraum；时分 OTS | hohlraum 内 `T_e/n_e`、冲击与非局域热传导 | 综述 + 代表实验；Zotero 已有 |
| Milder et al., DOI `10.1103/PhysRevLett.127.015001` | 角分辨 TS；代表条件 `T_e≈1.16 keV`、`Z≈25`；bulk 拟合 `m≈3.9` | bulk `v/v_th<3` 为 super-Gaussian，tail `v/v_th>3` 近 Maxwellian；吸收约降低 40% | 直接实验 EVDF 证据 |
| Katz et al., DOI `10.1063/5.0219182` | 连续约 120° 角范围、约 1° 角分辨、约 0.8 nm 光谱分辨、约 `50 µm^3` 散射体积 | 多 `k` 方向约束角向 VDF | 直接仪器论文 |
| Zhang et al., DOI `10.1038/s41567-023-01972-1` | laser-driven anti-parallel reconnection；time-resolved burst | IAW burst 后出现 EAW burst；平衡 Maxwellian form factor 不能解释，提示 electron outflow/two-stream | OOD/拒识案例，不应强制映射成 EDF |
| Suttle et al., DOI `10.1063/5.0041118` | MAGPIE 综述/实例：532 nm、3 J、8 ns；`n_e=10^23–10^25 m^-3`、`T_e=10 eV–数 keV`、`alpha≈0.5–3` `[综]` | shock/jet/wire-array/reconnection；IAW 联合给 `T_e,T_i,u,Z` | 约 100 µm、约 4 ns gate；有限接受角 3° 或 7–9° |
| Eisenbach et al., DOI `10.1088/1361-6463/ad89d6` | Phoenix high-repetition platform；532 nm、90°；>1 Hz 场景与二维 raster data | ML/transfer 回归 `n_e/T_e`；传统标签来自 PlasmaPy NLLS | 直接实验 AI 参数估计 |

本核心集的代表性包络：

```text
ne: ~10^23–10^27 m^-3
Te: ~10 eV 到 few keV（具体平台可更宽）
Ti: ~10 eV 到 keV
bulk flow: 常见约 10^5–10^6 m/s
probe: 263/351/526.5/532 nm 均有设施实例
time: ps–ns probe；streak 或 gated detector 提供 ps/sub-ns 到 ns 级演化
regime: alpha ~ O(1) 到 >1 很常见；EPW/IAW、组分和流速联合影响谱形
```

预期分布：逆韧致吸收通常给 `m≈2–5` 的 super-Gaussian bulk，而 Milder 2021 直接显示 `v<3v_th` 可为 super-Gaussian、`v>3v_th` 回到近 Maxwellian；温度梯度给奇对称 heat-flux perturbation；SRS/PIC 可给 0.1–1% 热电子尾；碰撞流、return current 与 reconnection 可给 beam/multistream 或 IAW/EAW burst。离子端常为多组分、多流和非共同温度。高相速度/高能尾必须做相对论一致性检查。

## 4. 高重复率和实验室天体物理

| 证据 | 明确报告的范围/配置 | 测量目标 | 证据性质 |
|---|---|---|---|
| Kaloyan et al., raster TS, DOI `10.1063/5.0059244` | 约 2 cm laser plasma；heater 5–10 J、1 Hz；代表值 `n_e≈4×10^16 cm^-3`、`T_e≈7 eV`，`alpha` 接近 1 | 二维扫描的前身：逐点 raster `n_e/T_e` | 已运行系统 |
| Zhang et al., DOI `10.3390/instruments7030025` | 二维 TS 光谱/空间扫描平台 | 2D plasma characterization | 方法/系统论文 |
| Pilgram et al., DOI `10.1063/5.0201112` | 二维 TS 测得错位 `n_e/T_e` 梯度及 Biermann battery 相关场 | 实验室天体/磁场产生 | 直接实验 |
| Pokornik et al., DOI `10.1063/5.0201148` | collective IAW，CNN 与 Bayesian/优化比较 | 多离子速度、比例、`T_i/T_e` 快速反演 | AI 参数估计；Bayesian 在噪声/OOD 更稳 |

该场景的参数会随平台跨越低温 LAPD 到 HEDP blast wave，因此不应设置一个统一密度/温度箱。真正共同的工程特征是：shot 数量大、二维/时序数据量大、自动 QC 与 sim-to-real 重要。

## 5. 分布函数—谱形—实验语境对照

| 分布形态 | 物理语境 | OTS 可见性 | 当前项目应如何用 |
|---|---|---|---|
| Maxwellian / drifting Maxwellian | 碰撞充分、常规参数诊断 | 标准 B0；峰宽/位置与介电响应 | 必须作为 sanity 与标定基线 |
| super-Gaussian / Druyvesteyn | 逆韧致吸收；某些低温放电尾部耗尽 | HEDP 有直接角分辨证据；低温 TS/OES 有间接/联合证据 | 加入训练族与 `m=3.5/5` 盲 OOD |
| bi-/multi-Maxwellian | 冷/热电子群、加热、多个流体组分 | 可通过宽翼/多尺度改变谱，但易与 IRF/背景退化 | B1 模型选择；混合权重非负且排序 |
| κ/长尾 | 弱碰撞、波粒作用 | 对谱翼最敏感，也最易受背景和动态范围限制 | 作为物理 OOD；不得仅凭合成拟合声称实验已辨识 |
| weak hot tail/beam | SRS/PIC 热电子、激光/脉冲功率 | 需足够波长覆盖与动态范围；0.1–1% 量级极具挑战 | 当前热 Gaussian 先验要与 counts/stray/IRF 联合测试 |
| heat-flux skew/return current | 温度梯度、非局域传输 | 多角度/正负 k 方向更有辨识力 | 加入不对称 OOD；单视角结果只称 projected EVDF |
| plateau/hole/multistream | 两流/波粒相互作用、shock/reconnection | collective spectrum 可受影响，但非唯一反演 | 作为前沿压力测试；需 B2/B3 resolution analysis |
| 3D anisotropy | 磁化、各向异性加热 | 单 `k` 不可恢复；必须多角度 | 不应塞进单视角网络标签 |

## 6. 对当前代码的定量映射

当前 `351 nm, 123°, n_e=1.81×10^26 m^-3, T_e=1370 eV` 给出 `alpha≈1.55`，明显落在 HEDP/ICF 风格的弱到中等 collective 区，而非低温 noncollective 或常规 tokamak polychromator 范式。

与代表实验对照：

- `n_e` 比 Ross 2010 gas-jet 上限高约 2.6 倍，仍属于 HEDP 数量级；
- `T_e=1.37 keV` 接近 Milder 非 Maxwellian 激光等离子体条件，而远高于 LAPD/放电；
- H/C 固定混合与 `T_i=171.3 eV` 是特定物理假设，不是上述应用的通用设置；
- 351 nm 可用于特定 HEDP/NIF 类探针，但文献同样常见 263、526.5/527、532 nm；训练时应把波长视为域标签或条件量；
- `[-0.8c,0.8c]` 远超大部分 bulk thermal support，并进入相对论能量区；详见 `CURRENT_CONFIG_AUDIT.md`。

实验可实施性上还需明确十项红线：

1. 电子蓝侧窗口下探到 `100.1144 nm`，已进入 VUV；调研中成熟 HEDP OTS 的短波实例常见约 263 nm，约 210–211 nm 已需要深紫外专项设计。100 nm 端不能直接当作普通光学谱仪可测波段。
2. 351 nm 在 ICF 中常同时是驱动波长，作为 probe 会面临强驱动散射和自发光背景；OMEGA 常改用 263/266 或 526/527 nm 避开该问题。
3. 100 nm 端的 `[推] v_phase≈0.606c`；Ross 2010 在 `0.12c` 已看到显著相对论 form-factor 效应，所以当前非相对论正向在短波端不是小修正。
4. 覆盖约为 `+0.606c→+0.0187c` 与 `-0.0381c→-0.2947c`，而标签是 `[-0.8c,0.8c]`；未观测尾部和正负不对称区域只能由训练先验猜测。
5. 三段独立归一化删除绝对功率、red/blue EPW 幅度比和 electron/ion feature 相对强度；Henchen 2018 的核心信息恰来自红蓝幅度不对称。
6. 训练先验应按机制分族：Maxwellian、`m=2–5` bulk + Maxwellian tail、bi-Maxwellian/ECCD tail、odd heat-flux perturbation、beam+bulk、counterstream 与 anisotropy；不能只靠任意 Gaussian 混合。
7. 单视角只恢复 `f(v·k_hat)`；恢复三维/gyrotropic EVDF 至少需要多个不共线 `k`，PHASMA 的 45°/90°/135°多视线就是直接证据。
8. 若 fraction 是离子数分数，CH 应为 H/C=`0.5/0.5`、CH2 为 `0.667/0.333`；当前 `0.55/0.45` 必须注明定义并把 `Z_C,T_i,composition,drift` 作为 nuisance/条件量。
9. 观测算子需要加入 IRF、有限接受角、散射体积/时间积分、notch mask、throughput、背景、Poisson+read noise、标定误差和 probe perturbation；文献分辨率跨 `0.01–1.27 nm`，shots 跨 `5–6000`。
10. IAW/EAW burst、SBS-driven waves、return-current instability 等非热涨落必须触发 OOD/拒识，不能强制解释成某种平衡 EDF。

## 7. 最小参数记录模板

每个未来实验或合成数据集至少保存：

```yaml
probe:
  wavelength_nm: ...
  pulse_energy_J: ...
  pulse_duration_s: ...
  polarization: ...
geometry:
  scattering_angle_deg: ...
  collection_angle_distribution: ...
  scattering_volume_m3: ...
plasma:
  ne_m3: ...
  Te_eV: ...
  Ti_eV: ...
  species_Z_fraction: ...
  drift_m_s: ...
derived:
  k_m1: ...
  lambda_D_m: ...
  alpha: ...
  phase_velocity_coverage_m_s: ...
instrument:
  wavelength_grid_nm: ...
  IRF: ...
  notch_mask: ...
  throughput: ...
  background_model: ...
  detector_likelihood: ...
  covariance: ...
provenance:
  source_DOI: ...
  value_type: measured | fitted | design | scan | review-envelope
```
