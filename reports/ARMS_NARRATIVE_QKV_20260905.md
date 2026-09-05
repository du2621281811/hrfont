# HR-Font 全对照组、Joint 叙事与 Δ-RSI QKV 决策稿

**日期：** 2026-09-05  
**仓库基线：** `6616836`  
**用途：** ICLR 2027 方法与实验设计决策；本文件不改变代码、配置或既有台账。  
**裁决原则：** 状态冲突时，以 2026-09-05 最新的 `REVIEW_STAGE_A_IMPL_20260905.md` 与 `DESIGN_E2E3_FUSION_QKV_20260905.md` 为准；旧执行规格用于还原历史 arm 的原始 estimand。新拓扑尚属设计建议，须经 PI 决策后才能成为执行真源。

## 0. 决策摘要

1. 新主线应是 **F0 → F1/F2/F3**：F1 vs F2 隔离 Δ source，F2 vs F3 隔离 exemplar support；旧 E1 保留为同数据域 FT anchor，official P1 zero-shot 保留为外部 anchor。
2. Joint 不再讲“先学哪里、再学什么”的阶段故事；应讲两种互补条件在单阶段端到端系统中共同约束生成：Δ 是跨字体聚合的 residual structure prior，support 是实例级局部证据。联合优化是方法特征，不是对旧两阶段的优越性结论。
3. QKV 创新是**可选项，不是论文成立的必要条件**。必要的是 identity-safe RSI 初始化和严格 matched controls。建议只用 F0 做 20k/seed3407 的冻结 val16 预注册筛选；role-swap 与 additive K/V 至多一个胜者进入主方法，且一旦更换，F1/F2/F3 必须同构重跑。

## 1. 全对照组总表

### 1.1 统一口径

- `✅已完成` 只表示训练/历史实验已完成，不自动表示可作当前主线 matched 因果证据。
- `🔄进行中` 的旧 E2/E2b 是 9/5 快照状态；最新实现审查建议在安全 checkpoint 停止，且因 source-drop 不一致只能作 pilot/诊断。
- `⚠️被取代` 表示保留历史身份和结果，但不再投入新主线预算。
- `全文必需` 指 9/25 前支撑核心方法归因所需；`附录` 可延期，不应阻塞主结果冻结。
- 所有 F1/F2/F3 的正式 matched 结论要求相同 F0、identity-safe RSI、QKV 拓扑、数据/过滤表、80k endpoint、seed、batch/order、drop 与随机流；3407 先行，主张稳健性时补 3408/3409。

### 1.2 权威总表

| 编号 | 名称/ID | 初始化 | 结构源 | 训练内容 | 预算 | 回答什么问题 | matched 关系 | 状态（及依赖/9-25优先级） |
|---|---|---|---|---|---|---|---|---|
| B0 | official P1 zero-shot | 官方 P1 原权重 | 官方 `Ec(ref)` RSI | 无训练；A/test16、统一 ref/noise 下评测 | 0 train；test16×295×生成 seeds | 未经 A/train228 适配的外部性能锚点 | 非 matched；与所有方法共享评测协议，不作单变量因果对 | 📋待评测；依赖 E12 评测器过门；**全文必需** |
| L0 | `FT-CNSTYLE-25K` / legacy `ft_cnstyle@25k` | official P1；旧42字体/旧渲染域 | 官方 RSI | 历史 FT | 25k | 旧域适配能到什么水平 | 仅 `legacy-domain` 单列；不得初始化新实验、不得参与 matched 显著性 | ✅已完成（`retro_partial`）；附录/主表脚注参考，非9/25核心 |
| L1 | `A-MVP-CONTROL` / `A-MVP-DELTA` | legacy FT@25k | control vs Δ，旧1-shot“永” | 旧 Stage-A 早筛 | 各10k | 早期 Δ 信号是否足够大 | 旧 matched pair；结果仅 `INCONCLUSIVE`，不可外推新 A/p260 拓扑 | ✅已完成；附录历史 |
| E1 | `E1-FTV2-A-S3407`（FT-v2） | official P1 | 官方 RSI | A/train228 全训 UNet+Ec+Es，SCR off | 100k，seed3407 | 单纯同域 FT 的收益；“不是换数据”的 anchor | 与 P1 非严格结构 matched，但同模型起点；与 F0/F1/F2共享 A/train228，支持 data-domain 解释，不能替代 F1–F2 | ✅已完成（best@98k，final@100k）；**全文必需** |
| E1c | `E1C-FT-CONTINUE-S3407` | E1@100k | 1-shot official `Ec(R[0])` | 冻 Es/Ec；原拟80k官方 RSI arm | 80k | 旧拓扑下 frozen-encoder 1-shot official-RSI continuation | 非 E2 matched；原与 E2b 比 style protocol，与 E2 比整包 | ❌取消（`superseded-before-launch`）；最新 `DESIGN_E2E3...` D-F8 覆盖早期 planned/blocked 口径；不属于9/25 |
| E2b-old | `E2B-FT-CONTINUE-S3407`（名称已不准确） | E1@100k | n-shot official `Ec(R[0])` | 冻 Es/Ec，训 UNet+offset；旧拓扑 pilot | 80k计划；快照27.8k | 旧拓扑 Δ 的 official-source control | 原拟与 E2 只差 RSI source；实际 official source 无 `.25` drop，**不 matched** | 🔄进行中（审查建议安全停止并降级诊断）；⚠️将被 F1 取代；非9/25正文证据 |
| E2-old | `E2-STAGE-A-S3407` | E1@100k | top-10 Δ → RSI | 冻 Es/Ec，训 UNet+offset；旧拓扑 pilot | 80k计划；快照9.2k | 旧拓扑 Δ-RSI 是否优于 official RSI | 原拟配 E2b；实际仅 E2 有 `.25` source-drop，**不 matched** | 🔄进行中（最新实现审查建议停止，保存 provenance）；⚠️将被 F2 取代；非正文证据 |
| E2c | P1 cold-system appendix | official P1 + **P1** cache | Δ → RSI | 旧拓扑 cold start | 80k（若执行） | Δ 是否依赖 E1-adapted encoder/初始化 | 非 matched；不可与 E2 混 cache | 附录 / 📋待跑；依赖独立 P1 cache；非9/25必需 |
| E2d | SCR matched ablation | 对应 E2 或 E2b 起点 | 与母臂相同 | 仅增 `.01 SCR`，其余 matched | 原计划每组80k×3 seeds | SCR loss 是否改变 Δ/official 结果 | 与各自母臂成对，只允许 SCR 开关变化 | ⚠️旧拓扑被新主线取代；若保留仅附录，依赖有效母臂；非9/25必需 |
| E3 | 主对比评测 | 已训练的 anchors/arms | 各臂自身 | 无训练；test16×295，paired noise，三轴指标 | 评测预算 | 谁总体更好 | 同一 test manifest/noise 的 paired evaluation；新主表应评 P1/E1/F0/F1/F2/F3 | 📋待跑；依赖 E12 过门及 F0–F3；**全文必需** |
| E4 | gap 统计 | E3 生成结果 + 冻结 gap | 无 | mixed-effects、Spearman、font-cluster bootstrap | 统计预算 | 方法收益是否随 RS-gap 系统变化 | F1↔F2、F2↔F3 的 paired outcomes 进入 gap×method | 📋待跑；依赖 E0/gap 冻结、E3；**全文必需** |
| E5 | 旧 Stage B SupportAdapter | E2 best | Δ；另加 support | Stage A 全冻，仅训 SupportAdapter | 25k×seeds | 冻结 Δ 系统后 support 的纯增量 | E2 vs E5；只有 adapter 可学习，旧归因最干净 | ⚠️被 F3 取代；可做单seed 25k staged sanity，附录候选 |
| E5b | 旧 Stage B + SCR | E2d best | Δ；support | Stage A 全冻，仅 adapter，保留 SCR | 25k×seeds | SCR 路径上的 support 增量 | E2d vs E5b | ⚠️被 F3 取代；附录，低优先级 |
| E6 | gap 机制（旧定义） | E2/E5 或 E2d/E5b 输出 | 无 | 无训练；gap×method统计 | 统计预算 | support 是否主要补高-gap/ref盲区 | 旧 E2↔E5、E2d↔E5b；新拓扑应改为 F2↔F3 | 📋待跑（按新 F2/F3 重定义）；依赖 E0、E3/F3；**全文必需** |
| E7 | support 消融 | F3（旧文为E5） | Δ + support variants | random-q、whole-glyph kNN、wrong-font、no-neutral、oracle 等；必要时重训 | 推理消融或25k训练臂 | 哪种 support 信息真正有效 | 每项对 F3，保持 checkpoint/noise/retrieval；可学习改动须 matched retrain | 📋待跑；依赖 F3；关键 `no-support` 已由 F2覆盖，其余附录 |
| E8 | ref-set 干预 | F1/F2（旧文为E2b/E2） | official或Δ | 无训练；ref8 / +口日 / +随机2 | 小型推理评测 | 参考覆盖变化是否影响闭合字 | 同 checkpoint、paired noise、只改 ref set；F1/F2均做避免 source 混杂 | 📋待跑；依赖 F1/F2；附录优先，可不阻塞9/25 |
| E9 | Δ 接入/破坏/定位 | F0或F2共同起点 | 正确Δ与RMS-matched破坏Δ | 接入位置短跑；shuffle/channel-mean/magnitude/wrong-char/wrong-style；定位诊断 | 20k筛选；胜者才正式80k×seeds | 增益是否来自 Δ 的空间/字符/字体信息及接入位置 | 每个破坏对正确Δ；逐层RMS匹配；学习型接入必须共同起点重训 | 📋待跑；依赖 F2；**正文最低集：wrong-char或wrong-style + spatial-shuffle +定位**，全量附录 |
| E10 | CN→CN | P1/E1/F2/F3 | 各自 | held-out中文128字，无训练 | 推理评测 | 跨脚本机制是否损害同脚本生成 | 同 ref/noise 的 non-inferiority 对比 | 📋待跑；依赖 F2/F3；附录，非9/25必需 |
| E11 | 人评 | E3/E6 主结果 | 无 | 240个2AFC + 120 MOS，3–5评审 | 人力预算 | 自动指标结论是否符合设计感知 | P1/E1/F2/F3 随机盲评；GT仅正控 | 📋待跑；依赖主生成结果；**全文强烈建议**，若截止受限至少完成预注册核心子集 |
| E12 | 独立评测器（a训练/b只读） | 外部字体；与方法编码器隔离 | 无 | φ_s2 + ID-CLS；T1–T4；再只读评分 | 50 epochs×3 seeds + 评分 | 避免用 Es 自证；验证 style/identity 指标 | 非生成 arm；所有模型共享同一冻结 evaluator | 📋待跑；无 F0依赖，可立即并行；**全文必需且是 E3/E4 前置门** |
| F0 | `F0-RSIFREE-FT-A-S3407` / FT-noRSI | official P1 的可匹配非RSI权重 | 无 RSI source；保留 style attention，旁路 offset+DCN | A/train228 FT；构建 shape-compatible RSI-free topology | 100k cap；预注册60k early-stop候选 | 新拓扑本身及去 RSI 的 baseline | 与 E1比较去RSI拓扑影响；是 F1/F2/F3共同唯一初始化 | 📋待跑；依赖 PI 批准 D-F3/D-F5与实现；**全文必需，最高优先级** |
| F1 | `F1-OFFRSI-S{seed}` | F0 | official `Ec(R[0])` | 新增所有臂共用 identity-safe RSI；UNet+RSI训练，support off | 80k；3407先行，后补3408/09 | 加回 RSI 容量/official source 的收益；Δ 的直接 control | **F1 vs F2只改 source**；同构QKV、drop、RNG、预算 | 📋待跑；依赖 F0、cache、QKV选择冻结；**全文必需** |
| F2 | `F2-DELTARSI-S{seed}` | F0 | top-10 Δ | 与F1同，support off | 80k；同F1 | Δ source 是否优于 official source | **F1↔F2** 隔离Δ；**F2↔F3** 隔离support | 📋待跑；依赖 F0、cache、QKV选择冻结；**全文必需** |
| F3 | `F3-JOINT-DS-S{seed}` | F0 | top-10 Δ + exemplar support | UNet、identity-safe RSI、SupportAdapter 单阶段端到端联合训练；support_drop=.20 | 80k，必须与F2同 endpoint | 最终系统与 support 联合条件的整体处理效应 | **F2 vs F3只改 support treatment**；所有随机draw仍同序消费 | 📋待跑；依赖 F0/F2接口与support实现；**全文必需** |
| Q0 | inherited QKV 20k screen control | F0 | Q=source，K/V=skip | identity-safe RSI；support off | 20k，seed3407 | 继承 QKV 的验证基线 | 与Q1/Q2同 F0、数据、预算、参数量尽量对齐 | 📋待跑；依赖 F0 checkpoint；**9/25前仅在PI批准筛选时必需** |
| Q1 | role-swap screen | F0 | Q=skip，K/V=source | support off；处理输出网格/通道 | 20k，seed3407 | 被形变位置主动查询Δ是否更合适 | Q0↔Q1；除QKV角色外 matched | 📋待跑；依赖 F0；预注册筛选，不进 test16 |
| Q2 | additive K/V screen | F0 | inherited主路 + `K'=K+P_k(Δ), V'=V+P_v(Δ)` | zero-init投影，support off | 20k，seed3407 | Δ直接调制被检索内容是否有额外收益 | Q0↔Q2；需参数容量对照/同构 official arm设计 | 📋待跑；依赖 F0；预注册筛选，不进 test16 |

**状态冲突说明。** `PI_DECISIONS_20260905.md`、执行规格与 registry 把 E1c/E2/E2b 记为 planned/running 的旧正式三臂；`REVIEW_STAGE_A_IMPL_20260905.md` 随后发现 source-drop 等阻断，判定当前 E2/E2b 不能作为论文 matched 证据并建议停止；最新 `DESIGN_E2E3_FUSION_QKV_20260905.md` 又用 F0–F3 替代该拓扑，明确建议 E1c `superseded-before-launch`、旧 E2/E2b 保存后停止。因此本表保留现场状态“进行中”，但把论文地位标为“被取代/非正文证据”。

## 2. Joint 之后的论文叙事

### 2.1 新框架：两种互补条件，而非两个训练阶段

旧叙事把 Stage A 称为“Δ 决定哪里改”，Stage B 称为“support 决定改成什么、补 ref 盲区”。合并后应删除阶段因果，把方法定义为一个**双条件、单阶段、端到端生成器**：

- **Δ residual structure prior**：对目标字符，在冻结表征空间中用目标字体参考集检索相似训练字体，将同字 `Ec` 特征相对中性字体 `Ec(Content)` 的残差加权聚合。它提供目标字符对中性结构应如何偏移的、跨字体统计先验；它是有空间结构的 residual prior，但不保证只含“where”。
- **Exemplar support**：从可见实例中提供字体特有、局部且非参数化的证据，补充 top-10 残差凸组合和 ref8 无法唯一确定的细节，例如衬线形态、端点、曲率或局部笔势。它不是第二阶段的修补器，而是与 Δ 同时可用的证据通道。
- **互补性**：Δ 的优势是 target-character aligned、可泛化的结构偏移先验；support 的优势是 exemplar-specific、能表达近邻平均之外的局部风格。前者降低搜索空间，后者减少由有限参考和库近邻引起的歧义。
- **设计师类比**：字体设计师会同时使用“这个目标字相对中性骨架应怎样变”的结构草图，以及同一字体其他字形中的笔端、曲率、粗细等实例证据；不会先完成整套结构再冻结，才看实例。F3 模拟的是这种同步协调。
- **Joint 是特征**：F3 从共同 F0 起点，在一次训练中让 denoiser、identity-safe RSI 与 support adapter 协同适配；无 staged freezing、无中间 checkpoint 选择、无串行误差传递。可主张“single-stage/end-to-end”，不可主张“优于 staged training”，除非补直接消融。

### 2.2 精确归因链与论文表格放置

| 对比 | 唯一可接受的解释 | 放置 |
|---|---|---|
| official P1 zero-shot → E1 | A/train228 域内 FT 能带来的收益 | 主结果表，external anchor 与 FT anchor |
| E1 → F0 | RSI-free 拓扑/重新 FT 的影响；不是 Δ 效应 | 主结果表中的 baseline 行，正文简述 |
| F0 → F1 | 从 RSI-free baseline 加回 identity-safe official-source RSI package 的整体收益 | 主结果表或紧邻主表的结构控制 |
| **F1 → F2** | **只把 official source 换成 Δ source 的平均处理效应** | **主表核心 contrast；正文 Δ claim** |
| **F2 → F3** | **允许 exemplar support 参与联合优化的整体处理效应** | **主表核心 contrast；正文 support claim** |
| E1 vs F1/F2 | 二者都使用 A/train228，说明收益不能归因于“获得更多/不同训练字体”；但训练预算与拓扑不同，不能写成严格单变量对比 | 主表 anchor + ablation文字；不要给单因素因果措辞 |
| F3 staged sanity vs F3 joint | staged freezing 与 joint 的选择 | 仅新增时放附录；当前没有结果，不进入贡献主张 |

主表建议固定六行：`official P1 / E1 / F0 / F1 / F2 / F3`，并在表内或表下注明两个预注册 paired contrasts：F1–F2（Δ）和 F2–F3（support）。QKV role-swap/additive、Δ破坏、ref干预、support细项进入 ablation/appendix；不要让架构搜索淹没 conditioning-source 的主贡献。

### 2.3 可直接入论文的 claim 句子草稿

**(a) 方法贡献**

> “We introduce a single-stage, end-to-end cross-script font generator conditioned jointly on a character-aligned residual structure prior and exemplar support.”

**(b) 为什么是 Δ**

> “The residual prior aggregates same-character feature differences from style-retrieved training fonts relative to a neutral glyph, providing a target-character-aligned structural bias rather than reusing the structure of a mismatched reference character.”

**(c) 为什么需要 support**

> “Exemplar support supplies font-specific local evidence that cannot be uniquely recovered from a finite reference set or a convex mixture of retrieved residuals, complementing the population-level structure prior.”

**(d) 为什么 joint**

> “Joint optimization lets the denoiser, identity-safe residual structure interaction, and support pathway co-adapt in one training stage, without staged freezing or intermediate checkpoint selection; matched comparisons isolate the effects of the residual source and exemplar support.”

更保守的总括句可写：

> “Our results attribute gains separately to the residual structure source (F1 vs. F2) and to exemplar support under joint optimization (F2 vs. F3), with an official zero-shot model and same-domain fine-tuning as external anchors.”

### 2.4 叙事禁区与 reviewer traps

1. **不得写“Δ 更纯”或“Δ 只包含 where”。** Δ 是冻结编码器特征差，可能混合结构、风格、字符和编码器偏差；只有分解 probe、破坏实验和定位指标支持后，才能限定性描述其携带的信息。
2. **不得把 GT 称为唯一正确答案或天花板。** GT 只作 positive control、评测器校准和定位参照；字体生成存在多种可接受实现。
3. **不得声称 two-stage 不如 joint。** 当前 F2 vs F3 只识别 support 在 joint treatment 下的总体效应，没有 staged-vs-joint 对照。
4. **不得把 F2 vs F3 解释为“只有 adapter 参数的纯增量”。** joint 会令 UNet/RSI 沿不同梯度轨迹共同变化；可说“support-enabled joint training 的整体处理效应”。
5. **不得用 E1 vs F2 直接证明 Δ。** 它同时改变拓扑、初始化路径、训练预算和结构源；Δ 的因果证据只能来自 F1 vs F2。
6. **不得把 QKV 搜索胜者与旧 QKV control 跨轨比较。** 任一可学习 QKV 更改进入主线后，F1/F2/F3须同构、同起点重跑。
7. **不得把 identity-safe 初始化包装成 Δ 独有能力。** 它是所有 RSI arms 共用的优化与可比性契约。

### 2.5 是否值得补 staged-vs-joint 25k

**建议：不作为9/25前必需；有空闲GPU时做一个 seed3407 附录 sanity。** 设计为从同一已完成 F2 checkpoint 分叉：`S-stage` 冻结 F2 的 UNet/RSI、只训 zero-init SupportAdapter 25k；`S-joint` 从同一 F2 checkpoint 联合更新 UNet/RSI/adapter 25k，二者共享 support、batch/order、lr schedule、drop、noise和预算。它回答“相同25k增量预算下 staged freezing vs continued joint adaptation”，但仍不同于主 F3 从 F0 起始的全程 joint。若没有 paired 的 S-joint 臂，单跑旧式25k staged 只能作 sanity，不能支持 joint superiority。考虑截止日与新增两条25k轨的机会成本，推荐延期到附录。

### 2.6 Identity-safe RSI 的方法表述

> “We instantiate RSI as an identity-safe residual deformation of the U-Net skip pathway: newly introduced residual/deformation parameters are initialized so that the RSI-enabled model is numerically equivalent to the RSI-free F0 model at step 0. The same parameterization and parity gate are used for both official-source and Δ-source arms.”

实现层面应是显式 residual/bypass（如 `skip + g·(DCN(skip,offset)-skip)` 且 residual 路径零初始化）并通过逐层及端到端 fp32 parity；仅 zero-offset 不足以保证 DCN 等于 raw skip。

### 2.7 9/25 时间线

**必须在全文结果冻结前完成：**

1. PI 冻结 F0 定义、identity-safe 契约、80k endpoint 与 QKV筛选规则；实现和 parity gate。
2. F0（60k预注册早停/100k cap）及其 Es/Ec cache；E12a/T1–T4可并行，必须先于正式主指标。
3. Q0/Q1/Q2 20k 筛选（仅若 PI 选择做）；最迟在 F1/F2/F3 开跑前冻结胜者，禁止中途改主方法。
4. F1/F2/F3 seed3407 完整80k；随后优先补 F1/F2/F3 的3408/3409。若无法补齐，论文须明确单seed探索性并收缩统计主张。
5. E3主评测、F1–F2和F2–F3 paired statistics、E4/E6 gap统计；E9最小破坏+定位；E11核心人评；E12b只读评分。

**可放附录/截止后补：** E2c、E2d、旧E5/E5b、staged-vs-joint 25k、E7全量、E8扩展、E9全接入位置与全破坏、E10、K=3/10、额外尺度、坐标编码、K/V语义分拆。旧 E2/E2b 只保存为非 matched 工程诊断，不应占用主线 GPU。

## 3. Δ-RSI QKV 优化评估

### 3.1 当前 QKV 是什么、计算什么

官方 `OffsetRefStrucInter` 先分别对结构特征和 UNet down-path skip 做 GroupNorm、1×1 Conv、LayerNorm，然后调用 cross-attention：

- `Q = W_q(structure)`：官方为参考字 `Ec(ref)`；HR-Font 为目标字符的 Δ residual structure prior；
- `K = W_k(skip)`，`V = W_v(skip)`：K/V 来自同一个 UNet skip context；
- 计算 `A = softmax(QK^T / sqrt(d))`，再得 `A V`，经输出投影、FFN和 `proj_out` 生成每位置18通道的3×3 DCN offset，最后形变 skip。

需要精确纠正一个容易误写的点：K/V **共享输入源（同一 skip）而不是共享线性投影权重**；代码中 `to_k` 与 `to_v` 是两个独立 `Linear`。attention 输出 token 网格跟 Q 对齐，因此 inherited 设计让 Δ 网格提出查询，再从 noisy denoising skip 中读取用于 offset 的内容。

### 3.2 是否必须在 QKV 上创新

**不必须，且“不创新 QKV”本身不是审稿风险。** 本文核心新意是 conditioning source：把跨脚本不匹配的单个参考字结构，替换为 target-character-aligned 的检索残差先验，并与 exemplar support 联合条件化。继承经过验证的 RSI attention 能让 F1 vs F2 成为更干净的 source 对比；若同时发明注意力，反而容易让 reviewer 质疑收益来自新容量或优化技巧。

真正风险不是“QKV 不新”，而是三点：① inherited 排列可能不适合 Δ，且未经验证；② identity 起点不成立；③主张超出证据。用预注册短筛、matched F1/F2、attention/offset诊断和Δ破坏即可化解。若 inherited QKV 在 val16 不劣，保留它会让论文贡献边界更清晰。

### 3.3 候选分类

| 候选 | 分类 | 9/25处理 | 理由/进入主线条件 |
|---|---|---|---|
| identity-safe residual/bypass | **主线必须** | F1/F2/F3前完成 | 解决 step0 等价硬问题；所有 RSI arms 共用，不是QKV创新 |
| inherited `Q=source, K/V=skip` | 主线默认/control | Q0 20k基准；无胜者则保留 | 最干净地隔离 Δ source |
| role-swap `Q=skip, K/V=source` | **预注册20k筛选** | Q1，seed3407 | 输出位置与被形变skip自然对齐；若显著胜出，成为唯一QKV变更并重跑F1/F2/F3 |
| additive K/V modulation | **预注册20k筛选** | Q2，seed3407 | `K/V` 增加zero-init Δ投影；只能与role-swap二选一进入主线 |
| extra LN | **skip（默认）** | 只记录诊断 | 现有结构与skip均已有GN+Conv+LN；先看RMS/logit std/entropy。只有预注册异常门触发才考虑RMS floor/temperature，启用需matched重训 |
| 更多 RSI scales | 附录消融 | 9/25不进主线 | 官方两尺度24²/48²足够首版；先做offset定位/无训练诊断，再决定20k |
| coordinate encoding / relative bias | 附录消融 | 9/25跳过 | 改变位置先验，可能帮助也可能压制跨位置匹配；启用须matched重训 |
| K/V semantic split | **skip至截止后研究** | 不进9/25 | 只有layout/content来自不同语义特征才非平凡；需新cache/contract，成本和混杂均高 |

### 3.4 Q0/Q1/Q2 预注册筛选协议

**目的。** 只决定 F1/F2/F3 共用哪一个 RSI attention 拓扑；不形成 test claim，不触碰 test16。

**arms。** 从同一冻结 F0 checkpoint、同一 cache 和 identity-safe RSI 初始化出发，均用 **Δ source、support off**：

- Q0：inherited `Q=Δ, K/V=skip`；
- Q1：role-swap `Q=skip, K/V=Δ`；
- Q2：inherited attention + zero-init `P_k/P_v` additive modulation。

三臂固定20k、seed3407、effective batch=8×1、相同 train episode manifest/order、lr/scheduler/warmup、source/CFG drop、noise/timestep与checkpoint规则；尽量参数量匹配，并报告参数差。训练只使用 train228，模型选择只使用 val16；**不得生成、查看或用 test16 调参**。

**评测时点。** 固定比较20k endpoint，不各挑 best；可报告5k间隔轨迹作诊断。每臂用同一 val16×295 ordered-R、paired noise，DPM-Solver++ 20 steps、CFG7.5。E12若尚未过门，筛选暂停，不以内部 Es 分数替代独立风格指标。

**主选择指标。** 不用 val16 L1 单独选，因为像素误差会偏向平均字形且不能代表跨脚本风格。预注册主指标为冻结独立评测器的 **style `φ_s2` composite**（建议 `SC-Gap`，方向统一为越大越好），以 font 为聚类单位计算 Q候选相对Q0的 paired bootstrap 95% CI。质量/身份护栏为 coverage/LPIPS 与 ID-CLS/OCR；val L1仅作训练稳定性和退化诊断。

**胜者规则。** 推荐用最小实用差异而非“最高点即赢”：

1. Q1或Q2仅当其 `ΔSC-Gap` 相对Q0的 font-cluster bootstrap 95% CI 下界 `> 0`，且点估计至少达到 calib16 上预先冻结的最小实用差异 `δ_style`；
2. 同时 ID-CLS/OCR 与 coverage/LPIPS 均不得越过预注册 non-inferiority margin（建议各指标为 calib16 SD×0.1，方向分别处理），且无 NaN、offset爆炸或identity parity失败；
3. 若两候选都过门，选 `ΔSC-Gap` 更高者；若差值的95% CI包含0，按简约原则优先 Q1（改动更小）；若仍无法区分，保留Q0；
4. 若没有候选过门，保留 inherited Q0，绝不因“需要创新”而切换；
5. 胜者一旦冻结，F1/F2/F3全部使用该拓扑从共同F0重新训练。Q1/Q2的20k权重不得续训成单独方法臂，也不得与不同QKV的F1作matched比较。

`δ_style` 必须在运行前由 calib16/历史 evaluator variance 冻结；若当前无法给出可信数值，采用更保守规则“CI下界>0且所有护栏non-inferior”，但不得看完结果后补阈值。

### 3.5 最终建议与 D-F14 映射

结论是：**QKV innovation 是 option，不是 need。** D-F14 的9/25范围应首先保证 F0、identity-safe RSI、F1/F2/F3 80k matched链和独立评测。QKV最多占一个短筛窗口；筛选无明确胜者就锁定 inherited QKV。额外LN、更多尺度、坐标编码和K/V split均不得延迟主 matched runs。

## 4. 必须 PI 决策的新增点

以下 D-N 决策用于把建议变成可执行预注册；它们不替代既有 D-F1–D-F14。

### D-N1｜是否执行QKV 20k三臂筛选

- A. **执行 Q0/Q1/Q2 三臂，按本报告规则冻结唯一拓扑（推荐）**。代价可控，并直接回答 inherited QKV 是否适配 Δ。
- B. 只做 Q0/Q1，additive移附录。更省预算，但未验证连续调制候选。
- C. 不筛选，直接沿用Q0。科学上可接受，贡献更干净；适用于F0实现或GPU日程已落后。

### D-N2｜QKV主选择指标与最小实用差异

- A. **独立 `φ_s2 SC-Gap` + 身份/质量non-inferiority（推荐）**；PI须在开跑前批准 `δ_style` 数值或批准“CI下界>0”的保守替代规则。
- B. val16 L1最小。省事但不对应字体风格主张，不推荐。
- C. 多指标事后投票。自由度过高，拒绝。

### D-N3｜staged-vs-joint附录预算

- A. **9/25后再做 paired 25k `S-stage/S-joint`（推荐）**。
- B. 9/25前插队做两臂。只有当F1/F2/F3与E12已锁定且有空闲GPU才可选。
- C. 永不做，但正文完全不比较 staged/joint superiority。可接受。

### D-N4｜旧 E2/E2b 的运行处置落档

- A. **最近安全checkpoint停止两臂，标 `stopped_topology_superseded/nonmatched_pilot`（推荐）**。
- B. 仅让E2b跑完作appendix，E2停止；仅当F0实现预计阻塞超过2天。
- C. 两臂跑完并补seed。与新主线竞争GPU且不能修复已有混杂，不推荐。

### D-N5｜F0早停规则的数值冻结

- A. **60k可早停、100k cap，但在开跑前用val composite冻结“50k与60k连续平台”的最小改善阈值（推荐）**。
- B. 固定100k。最稳但占时间。
- C. 固定60k。缺少收敛依据，不推荐。

