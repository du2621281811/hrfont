# HR-Font 论文完成实验执行计划（PI Review Draft）

日期：2026-09-12
适用版本：以 Git `fba024d1` 为修订起点；论文方法为 Set-Delta + Graphics-Ref 设计候选
证据边界：F0/F1/F2/F3/F3b 是旧实现；F2-P/F3b-P 只提供 R1 接口与诊断，不是新方法结果。

## 1. 先确定论文要证明什么

最小可投稿闭环只保留两个主张：

1. **Set-Delta：** 训练字体库提供同字符、锚点相对的几何变化候选；canonicalized geometry package 负责限制 donor appearance，保留候选轴到局部聚合负责保持合法结构模式。
2. **Graphics-Ref：** 目标字体参考图提供可观察的外观证据；graphics primitive 只负责 correspondence key，Es12 learned feature 只负责 appearance value。

Support/own-font 不进入论文方法。Es、Ec 在所有主实验中冻结并使用版本化 cache；不与生成器联合训练。首轮不增加额外字重、不做几何增强，避免把表示改进与数据规模混在一起。

## 2. 三层实验预算

### A. 论文最低闭环（必须完成）

| 模块 | Arms | 直接回答 | 训练策略 |
|---|---|---|---|
| Delta 主筛查 | D0 / D1 / D2 / D3，统一 R1 | Delta 总效应、geometry package、保留候选轴 | seed 3407；40k horizon；20k 冻结中检 |
| Ref 主筛查 | R1 / R2 / R3，固定 D* | local value 与 graphics key 是否有效 | R1 复用；只新跑 R2/R3 |
| Delta 必要控制 | D4、D6 | residual 是否必要；是否真用同字符 | 先 20k；对应主张保留才续 40k |
| Ref 必要控制 | R4、等预算 FSFont-style K/V | primitive key 是否有语义，而非容量/普通 local attention | 先 20k；R3 保留才续 40k |
| 正式主比较 | 完整方法 + 最强必要基线 | 论文 headline 结果 | 固定 seed 3407，与所有 matched arms 对齐 |

D2−D1 识别的是从 legacy feature Delta 到 canonicalized geometry Delta 的整体表示替换，不拆称为单独的 normalization 或 carrier 效应。只有 D2 在主筛查中胜出、且论文确实需要拆分收益来源时，才启动条件控制 D1n：保持相同 donor、检索权重、空间配准和 raster normalization，但使用 legacy feature carrier。

首版坚持 **warp-only**。只有 warp-only 无法改善孔洞、连断或端点题本且 value path 的干预信号正确时，才启动 D7；否则不为“完整”而增加双路径和额外消融。

### B. 主表完整性（必须审计，能跑则完成）

- FontDiffuser：同数据协议下的 diffusion 基础线；优先级最高。
- FTransGAN、FCAGAN：直接跨语系 few-shot 对手；优先完成代码、数据方向和 reference budget 审计。
- CF-Font：最接近 bank-supported corresponding-glyph fusion，必须尽量纳入。
- FSFont：最接近 local style correspondence；若原生跨语系适配不成立，保留为协议标注的外部结果和仓内等预算 K/V 控制，二者不能混称。
- DRA-font：先确认可执行实现与完整训练协议；在此之前只进入 Related Work，不预占主表结果位。
- VQ-Font：作为可选补充；若不能在冻结协议下忠实复现，不用弱改写版本凑主表。

外部方法按各自原生输入协议报告；另设同资源 matched control 表。不能把额外 bank、额外 reference 或不同预训练数据隐藏在同一排名里。

### C. 增强项（核心结论成立后）

- `n = 1/2/4/8`：固定最终 checkpoint，只改变 ref mask，不重训。
- `K = 5/10/20`：检验 proposal prior 稳定性，不作为调参搜索。
- D5、D8、D9：仅当正文要声称 alpha 排序、局部延迟聚合或 attention 形式本身必要。
- R5 local style loss：只作为 R2/R3 优化失败后的诊断，不与首轮表示比较混跑。
- 多样生成：只作定性展示；主表固定 deterministic soft fusion。

## 3. 按依赖关系执行

### Wave 0 — 合同与零训练诊断（0.5–1 天）

1. 为 p260 数据发布唯一 manifest ID/hash；确认 228/16/16 按 font family 隔离，避免同家族不同字重跨 split。
2. 固定 F0@100k artifact hash、ref8 顺序、neutral B0、renderer 和 geometry cache v1。
3. 冻结 val 题本：identity、hole/component、endpoint、terminal、contrast、dry-brush；标出 ref-observable / bank-only / unsupported。
4. 对 F2@80k 做 true/zero/wrong Delta 与 donor/font/style leakage probe。
5. 统计 retrieval 的 top-K overlap、$\alpha$ entropy 和 effective donor count，决定 learned ranking 是否值得保留为正文主张。
6. 审计 F2-P 的 per-ref mask、同步 CFG drop、正式 `sample.py` 和 run provenance；只读取同 step 的 matched probe。

**出口条件：** cache 可复现；mask 与条件清零干预通过；未通过时不启动批量训练。

### Wave 1 — 新 variant 与 smoke（编码 2–3 天）

在 `code/variants/hrfont_setdelta_graphicsref/` 中从 F0 派生：

- R1 per-ref global tokens + padding mask；
- geometry cache、D0–D3 与 warp-only Set-Delta；
- 训练和正式采样共同使用同一 conditioning API；
- 单测覆盖 donor permutation、wrong-char、all-mask、CFG drop、zero/true condition、checkpoint resume；
- 500-step smoke 记录梯度、参数更新、输出范数、显存和 p50/p95 step time。

**出口条件：** 输出层首步有梯度，上游在若干步后更新；置换不改变集合输出；清零分支回到匹配基线；resume 后数据/CFG draw 连续。

### Wave 2 — Round D（4 arms 到 20k）

统一 R1、同 parent、同 seed/data order/CFG draw：D0、D1、D2、D3。20k 只做冻结中检，不重启、不看图改超参。

选择顺序：先过 identity/readability/artifact 非劣门，再看预注册的 style/geometry endpoint；低于最小相关效应则选更简单方法。

- D2 不胜 D1：删除“canonicalized geometry package 更优”的主张。
- D3 不胜 D2：正式方法降为 geometry mean-Delta。
- D2 胜 D1 且需要拆分 normalization/carrier 归因：启动 D1n；否则不增加该 arm。
- D3 胜 D2：D2+D3 成对续 40k，并启动 D4/D6。
- D0 始终保留到 40k，给完整方法总效应提供共同底座。

### Wave 3 — Round R（2 个新 arms 到 20k）

固定 D*；R1 直接复用 Wave 2 checkpoint，只新跑 R2、R3。

- R2 不胜 R1：先做 Es layer/分辨率 probe；不立即加 loss。
- R3 不胜 R2：删除 graphics-key 主张，保留 learned local attention。
- R3 胜 R2：R2+R3 成对续 40k，并启动 R4 与等预算 FSFont-style K/V。

### Wave 4 — 必要对照与联合确认（40k）

正文每保留一句机制主张，就必须把它和直接控制成对做到 40k：

- canonicalized geometry package 主张：D1 + D2 + leakage probe；
- Set 主张：D2 + D3；
- residual 主张：D3 + D4；
- target-character 主张：D3 + D6；
- local evidence 主张：R1 + R2；
- graphics key 主张：R2 + R3 + R4 + 等预算 FSFont-style K/V；
- 完整方法总效应：D0 + final。

单模块增量成立后才做联合 final；若联合退化，做 Delta-only、Ref-only、joint 三格确认，而不是直接调更多 loss。

### Wave 5 — 正式评测、外部基线与盲评

1. 冻结 code SHA、resolved config、checkpoint parent、数据 hash、评测脚本和 Go/No-Go protocol。
2. 完整方法、最强必要基线和机制消融统一使用 seed 3407，并保持 paired data/CFG draws。
3. 外部基线完成 protocol card；测试集只在配置冻结后打开一次。
4. 自动指标对 16 个 test fonts 做 font-cluster bootstrap；方法比较保持样本配对。
5. 正式盲评建议覆盖全部 16 test fonts × 47 targets（752 对/方法对），每对至少 3 个独立判断；允许 tie，方法顺序随机，置信区间按 font 聚类。

## 4. 20k 冻结中检如何做

20k 不以单一 L1 排名。每个直接 comparison 生成同一组：

- 全量 val16 自动指标；
- 16 fonts × 12 预注册字符的盲化 paired board；
- identity/readability/artifact gate；
- ref-observable 的 terminal/contrast/dry-brush 分项；
- bank-supported geometry 的 Chamfer/SDF、hole/component、endpoint/radius；
- condition intervention 与 leakage probe；
- 参数量、FLOPs、显存、p50/p95 step time。

具体非劣 margin 与最小相关效应值不能由当前文档凭空填写。应先用冻结 val pilot 的 paired bootstrap 与人工重复标注方差估计，再在查看任何 40k 结果之前写入独立 protocol 并锁定。

## 5. 算力与时间管理

当前 Git 只确认 3090 侧具备完整数据/权重；V100 缺数据、Es/Ec/ckpt，且两机不能直连。因此排期先按 **一个可执行队列**，V100 只有在迁移 manifest 全部通过后才计入容量。

核心筛查是 10 个 seed-3407 arms 的最多 20k 初筛：D0–D4、D6、R2–R4、FSFont-style（R1 复用）。实际 40k 总量由保留主张决定；final 与最强基线直接复用同一 seed 的正式 run。D1n、D7、D5/D8/D9、R5 不预占主线预算。

不要沿用旧 run 的 1.15 s/step 直接报工期。500-step smoke 后用：

`ETA = remaining_steps × measured_p50_step_time / available_verified_GPUs`

同时保留 20% 的采样、评测、失败恢复和 baseline 适配余量。

## 6. 结果不及预期时，论文怎样收缩而不换故事

| 结果 | 方法收缩 | 仍可保留的叙事 |
|---|---|---|
| D3 ≈ D2，但 D2 > D1 | 用 geometry mean-Delta | bank 是结构 proposal；去 donor appearance 有效 |
| D2 ≈ D1 | Delta 降为检索先验/诊断 | 重点转向 Ref 的 observed appearance，不保留 geometry-package 增益主张 |
| R3 ≈ R2，但 R2 > R1 | 用 learned local attention | ref 的局部外观证据有效；不声称 graphics correspondence |
| R2 ≈ R1 | 只留 per-ref global | 先查 Es 层级和输入信息缺失；不靠额外 loss 掩盖表示失败 |
| Delta 与 Ref 各自有效、joint 退化 | 单分支主方法 + 另一分支附录 | 两类证据的边界仍成立，但联合优化尚未解决 |
| 全部视觉增益弱 | 转为分析性论文风险较高 | 需要增加输入信息或数据覆盖，不能声称效果保证 |

## 7. 数据与编码器决策

- **额外字重：** 不加入首轮 matched experiments。若后续加，必须按 family group 切分、整组只落在一个 split，并作为独立 data-scale robustness，不回写主结果。
- **数据增强：** 仅允许不改变字体风格语义的渲染噪声，并须所有 arms 同步；旋转、膨胀/腐蚀、随机笔画宽度会直接改变要评估的 slant/weight/radius，不进入首轮。
- **Es/Ec：** 冻结，不与新模块一起训练。它们满足“稳定 content/style cache”的工程需求，但不能保证包含飞白、端点等局部证据；这正由 R1/R2 layer probe 和 graphics/ref ablation 验证。

## 8. 需要 PI 现在拍板

1. 是否同意首版 final 固定 warp-only，D7 只在拓扑失败时触发。
2. 是否同意 graphics 主张的最低证据包含 R2/R3/R4 + 等预算 FSFont-style K/V。
3. 外部 baselines 的最低集合是否定为 FontDiffuser + FTransGAN + FCAGAN + CF-Font；FSFont 作为 local-correspondence 近邻，DRA-font/VQ-Font 按忠实复现审计决定。
4. 20k 专家盲筛是否采用 16 val fonts × 12 chars × 3 judgments/direct comparison。
5. 正式盲评是否采用 16 test fonts × 47 chars × 至少 3 judgments/method pair。

这些选择不影响立即执行 Wave 0、cache、接口和 smoke；但必须在批量 40k 与正式测试前冻结。

## 9. 新 Idea 的融合入口

新 Idea 不按“模块数量”判断，而按它增加的证据类型进入现有矩阵：

| Idea 类型 | 融合位置 | 首个比较 | 进入主线的条件 |
|---|---|---|---|
| 更好的结构/图形学先验 | D-side，替换或增强 $\phi$、key、warp | 与 D2 或 D3 做等预算单变量比较 | geometry endpoint 改善且 identity/readability 非劣 |
| 更好的 reference 局部特征 | R-side，替换 Es12 value | 与 R2 做等 token/参数比较 | ref-observable stroke/style 属性改善 |
| 更好的 correspondence | R-side，替换 graphics key | 与 R3/R4/FSFont-style K/V 比较 | correct key 胜 shuffled/learned key |
| 新 loss 或联合训练 Es/Ec | optimization wave，不进入首轮表示筛查 | 在最佳表示上做有/无该训练策略 | 表示路径已有干预信号但学习不足 |
| 新数据、字重或增强 | data robustness wave | 所有保留 arms 同步扩展 | 不改变字体风格语义，且按 family 隔离 split |
| 推理期可控性或多样性 | final checkpoint 后评测 | 固定 checkpoint 的 matched sampling | 不牺牲主结果，且对应明确用户控制 |

任何新 Idea 首先写成一句可证伪问题，指定直接 control、复用的 checkpoint 和停止条件。能补强“bank 提供结构空间、reference 选择可观察外观”之一即可融合；若同时改变数据、表示和 loss，则先拆成最小单变量 probe。
