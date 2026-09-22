# Experiments 写作方案 v3（含对照组与人评展示）

日期：2026-09-23（v3，取代 2026-09-22 的 v2）
依据：PI 决策（K6-B 参数 / V3 数据 / 外部复现已有 / 人评后补 / 砍过细实验，2026-09-22）；对照组四层与「防御性写作判据 = 句子主语」（2026-09-23）；评测器归属与防御性判据参照 `references/paper-method-audit.md` §7b/§7c。
目标读者：PI。**本文只给方案与结构；完整 LaTeX 见 `EXPERIMENTS_DRAFT_LATEX_20260923.md`；main.tex 未改。**

## 0. 决策与影响

| PI 决策 | 落地 |
|---|---|
| ① 训练参数以 K6-B 为主 | §3 按 K6-B run 的 config 权威值写（旧的「10,000 updates」说法作废） |
| ② 数据写 V3，只讲不泄露 | §3：V3 = v2 + v0921（315 字体、41,663 pairs，train-only、**不进 bank**）；泄露只写一句设定 |
| ③ 外部复现结果已有、效果低于我们 | §4 主表外部行按已有复现结果填（数字来源待指路） |
| ④ 人评结果后到 | §4／§6：结构先写，三处落点定好，数值留占位 |
| ⑤ 砍掉过细实验 | §7 清单：主文对照压到 4 行；逐层/逐 token/初始化/精度类消融、donor 审计、第二坐标消融、逐字体分解表下沉或删除 |
| ⑥ 主实验之外需其他对照组 | §2 对照四层：外部方法／条件消融／设计替代／设置与指标对照 |
| ⑦ 人评如何展示 | §6：主表一列 + 人评小节与独立表 + 附录协议；人评同时充当评测器的校验 |
| ⑧ 防御性写作判据修正 | §5：以**指标/任务**为主语的度量有效性研究（去相关、人评一致性、正负对照）计入评测贡献；以**我方成绩**为主语的解释才禁止 |

---

## 1. 节结构（7 小节，主文约 2.5–3 页）

| 小节 | 功能 | 篇幅 |
|---|---|---|
| 5.1 Setup and protocol | 数据 V3、协议 A、参考预算、训练配置、评测轴与统计口径 | ~0.8 页 |
| 5.2 Cross-script consistency evaluator | 构造一句 + 验证门一句 + 用途一句（结果占位） | ~0.3 页 |
| 5.3 Main comparison | 主表：外部方法 + CRP (fixed weights) + ours | ~0.6 页 |
| 5.4 Control studies | 4 行对照（2 removal + 2 substitution） | ~0.4 页 |
| 5.5 What the metrics measure | point vs set 指标轴：去相关 + 人评一致性 + 正负对照 | ~0.4 页 |
| 5.6 Human evaluation | 人评小节 + 独立表（均分、胜率、n、p、α） | ~0.3 页 |
| 5.7 Reference budgets and qualitative analysis | 1/2/4/8 预算曲线（含外部方法同预算）+ 三档定性 | ~0.4 页 |

## 2. 对照四层（主实验之外）

| 层 | 回答什么问题 | 对照形态 | 落点 |
|---|---|---|---|
| ① 外部方法（主实验） | 我们比已有方法好吗 | FontDiffuser / FTransGAN / CF-Font / FSFont + CRP (fixed weights) | 主表 |
| ② 条件消融（必要性） | 结构条件、外观条件各自是否必要 | 去掉 structural offset；去掉 appearance condition | 对照表 Removal 块 |
| ③ 设计替代（形式选择） | 为什么是学习路由而非固定权重；为什么是相对中性的偏移而非直接寄 donor 图 | routed vs fixed prior weights；neutral-referenced offset vs direct donor rendering | 对照表 Substitution 块 |
| ④ 设置与指标对照 | 优势是否只靠更多参考；分数是否真的量到任务 | 预算图含外部方法同预算对比；point vs set 指标轴（含人评一致性） | 5.7 图、5.5 表 |

互补性不单独设行：②的两行各自去掉一个条件后与完整模型的对比即给出互补证据。
其余候选对照（单 donor 结构、第二坐标、donor swap、同库同字绝对聚合）按 §7 进附录。

## 3. 数据与训练配置（权威值）

**数据（V3）**
- v2 主体：260 字体 = 228 train / 16 val / 16 test；295 目标字符、338 中文参考；协议 A（96×96 原生、逐字体统一字号、无 resize）。
- 补充集 v0921：315 字体、41,663 训练 pairs，`role = train-only`；bank 冻结在 v2 的 donor 列表。
- 泄露只写一句设定：补充集只用于训练，且不加入检索 bank（bank 保持冻结在 v2 donor 列表）。

**训练（K6-B run 权威配置）**

| 项 | 值 |
|---|---|
| 结构 | 双支路 Ec/Es + RSI；目标 = endpoint-generation only，无 offset 正则、无 pure-noise 辅助 |
| 初始化 | F0 checkpoint @10k（K6-B：`G0b-F0-V0913-BS256-A-S3407/global_step_10000`；K7-B：`F0-CLEAN-V0913-A-S3407/global_step_10000`） |
| 并行 | 8 GPU × micro-batch 8 × accum 1 = global batch 64 |
| 步数 | 20,000 updates |
| 日程 | warmup 500 / 平台至 5,000 / cosine 衰减至 10% |
| lr | 峰值 2e-5（denoiser 与结构模块）、1e-4（reader 与 Es 支路读出头）；AdamW betas (.9,.999)、wd .01 |
| 精度 / EMA | fp16（logits 与 loss 走 FP32）；EMA `min(.999, 1-1/(u+1))` |
| seed / 采样 | 3407；attempt-keyed 全局采样 |
| 损失 | perceptual .01；外观项 .01（ramp `β=.8·min(u/1000,1)`）；细节项 .05；offset 0 |
| 条件 dropout | joint CFG .02；source drop .05 |

## 4. 表格（四张）

**主表**：`Method | Identity ↑ | Style pref. ↑ | Family ↑ | LPIPS ↓`
行：外部方法（已有复现结果，效果低于我们）／CRP (fixed weights)／ours。
外部方法一行一句交代参考预算与适配方式；训练数据等细节进附录。
像素指标按数字照报，caption 与正文均不加"diagnostic"类价值标注，也不写为结果辩护的句子。

**对照表（4 行）**：Removal = 去掉结构偏移／去掉外观条件；Substitution = routed vs fixed prior weights／neutral-referenced offset vs direct donor rendering。列：`Study | Control | Identity | Family`。行名即基线定义；正文不为对照逐行声明"保留了哪些条件"。

**指标表（5.5）**：`Measure | Conditioned on (point/set) | Agreement with pixel measures | Agreement with human ratings`。含 L1/SSIM/LPIPS、风格特征距离、笔触与骨架相似度、compatibility score、recognition accuracy。

**人评表（5.6）**：`Comparison | Style pref. mean | Win rate (%) | n | p (Holm)`，正文另报 Krippendorff's α；附录给抽样规则与逐维度分布。

## 5. 防御性写作判据（修正版）

- **允许**：以指标或任务为主语的度量陈述——某度量与任务量纲的关系、度量间的排序一致性、与人评的一致性、正负对照下各度量的响应。这类内容归评测贡献（5.2／5.5），与我们的成绩无关，reviewer 读作评测有效性研究。
- **禁止**：以我方成绩为主语的解释——"我们的方法在 L1 上更低是因为…"、trades pixel fidelity for style、"该指标不适配本任务"的辩护段、为我们的像素数字专设的解释性对照。
- **必要的设定陈述只写一次**（补充集 train-only、bank 冻结），不替读者补结论（不写"因此不会有泄露"），不写 `the same X, the same Y, the same Z` 排比式受控条件清单。
- 数字照报，不藏（不从表里删像素列），也不解释。

## 6. 人评展示（三处落点）

1. **主表一列**：`Style pref.` 放人评均分（5 点，mean ± CI），跨行可比；正文另给 ours vs 每行的成对胜率。
2. **人评小节 + 独立表**：成对胜率、n、Holm 校正 p、评委间一致性 α；同时把人评作为评测器的校验（人评 vs 评测器一致性 ρ 放在 5.5），一份数据两个用途。
3. **附录**：完整协议（界面、抽样规则、评委构成、每图 3 评）、两维度分布、逐 script 分解。
- 正文两句话：人评是 cross-script consistency 的主要证据；消融臂只用自动轴评估（成本所限）。

## 7. 不做 / 下沉附录

**不进主文**
- 逐 token / 逐层 / 初始化方式 / 精度细节类消融。
- donor swap 敏感性审计；第二坐标（Es 中间层读取）独立消融。
- 单 donor 结构对照、逐字体 / 逐字符分解表、每个基线的训练细节枚举。
- 「同库同字绝对聚合」对照（bank 守卫；放附录）。
- **以我方成绩为主语**的像素解释（判据见 §5）；「像素评估不适配本任务」的前提属 intro / related work 的任务设定（main.tex:61、:132 已写），实验节不重述。

**下沉附录**：完整训练超参表、评测器细节与门结果、参考预算逐档数字、定性面板字样清单、人评协议全文、F0 初始化臂说明。

## 8. 待确认（需你拍板）

| # | 事项 | 推荐 |
|---|---|---|
| D-E1 | 主表用哪条 run 的权重：K6-B（v2 数据）还是 K7-B（V3 数据） | 按 PI 指示先不管（参数按 K6-B、数据写 V3）；出表时再定 |
| D-E2 | 外部复现结果的数字来源 | 你指路后接入主表 |
| D-E3 | F0 初始化臂差异写正文还是附录 | 附录一句 |
| D-E4 | 对照压到 4 行（§2 的②③），其余下沉附录 | 按 §7 执行 |
| D-E5 | 人评抽样规则与 n（决定胜率列分母口径） | 你定 |
| D-E6 | 5.5 的「正负对照」是否保留在主文（weight siblings / 异族同像素距离） | 保留（是指标特殊性最直接的证据，零训练） |

## 9. 未核实项

主表各行数字（K6-B `DONE.json` 的 `inference_complete=false`，需另跑固定协议推理）、人评采集状态、评测器门结果、K7-B 训练完成度、5.5 各项 ρ 与正负对照所需的变体/外部字体池产物。
