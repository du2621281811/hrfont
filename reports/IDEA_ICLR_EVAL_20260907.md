# HR-Font ICLR 2027 Idea 评估（Reviewer + Area Chair 视角）

**日期：** 2026-09-07  
**审查基线：** Git `513cbb224dbc8eb4eaebafc75e9f7a891919f379`  
**证据快照：** F0 已完成 100k；F3 约 13.5k/80k；F2 暂停于 6573；F1 暂停于约 300；E12-v3 的 T2 为 `0.732±0.057 < 0.90`。本文评价的是当前 idea 与实验设计，不把进行中训练、训练 loss 或 PI 目测当论文结果。

## 1. 总评

**Verdict：Reject（当前状态）；若核心结果与评测闭环按下文最低集补齐，可提升至 Borderline。**

一句话理由：问题真实、因果消融拓扑也比一般字体论文严谨，但核心技术目前更像“检索增强的、同目标字对齐的 RSI 条件源替换”，技术新颖性偏中低，而 SOTA、变化空间表征、可靠控制和“设计合理性”四条高层主张均无合格结果支撑。

从 AC 角度，当前不是“idea 不值得做”，而是**论文承诺显著大于 9/7 已有证据**。最有希望的论文不是把四个故事都做大，而是收缩为：

> 跨脚本时，参考字符的绝对空间结构未必是合适的形变条件；我们用检索得到的、同目标字符相对中性字形的残差先验替代它，并用严格 matched controls 与一个经人类验证的跨脚本三轴协议评估。

**9/25 前最重要的一件事：完成并冻结 F1/F2/F3 的同端点生成与独立可信评测，拿到 F1↔F2、F2↔F3 的效应量、按字体聚类置信区间和失败案例。**没有这条证据链，其他 controllability demo、叙事和外部 baseline 都不能救稿。

---

## 2. Novelty

### 2.1 新意究竟在哪里

FontDiffuser 已经提出 RSI：用参考字符的 `Ec` 空间特征作为 Query，与 UNet skip 的 Key/Value 做 cross-attention，预测 DCN offset。它本来就允许参考字与目标字不同，并展示过中文到韩文的跨语言定性结果。因此，论文不能把“异字参考”或“跨语言时存在空间错位”写成首次发现，也不能说 FD 假设参考字与目标字同字。

DG-Font 已有目标字符相关低层特征驱动 deformable skip 的思想；CF-Font 已有通过字体/部件原型或训练库先验重组未见字形的思想；few-shot 字体生成普遍已有多参考聚合。相邻通用生成工作如 InstantStyle、AnyText 更适合作为条件注入/文字生成的邻接背景，而不是直接同协议 competitor：它们若不接收“少量同字体跨脚本参考并补全整套 glyph”，不能替代字体专用 baseline。

本工作的可辨识新增点是以下**组合**：

1. 以参考集的冻结风格特征检索训练字体；
2. 读取检索字体中与目标字符相同的 `Ec(B_s,c)`；
3. 减去中性字体 `Ec(B_0,c)` 得到目标字符对齐的残差先验；
4. 保持 RSI 主体不变，把该先验作为结构条件；
5. 再加入 exemplar support，并用 matched arms 分别估计两个 treatment。

这比“只换一张输入图”更有方法含义，但仍属于**条件构造/检索先验创新**，不是新生成范式，也不是新的可学习表示目标。若没有强结果与机制 probe，“replace RSI source”本身通常不足以支撑 ICLR 高分。

### 2.2 分组件新颖性评分

采用 1–5 分：1=已有常规做法，3=有辨识度的组合创新，5=明显开辟新方法方向。

| 组件 | 新颖性 | Reviewer 判断 |
|---|---:|---|
| 同目标字残差 Δ：`ΣαEc(B_s,c)-Ec(B0,c)` | **3/5** | 最集中的技术新意；“same-character + neutral residual + style-retrieved bank + RSI source”组合有辨识度，但每个原语均接近既有检索、原型与残差条件思想。必须证明 residual、same-char、retrieval 三者各自必要。 |
| 直接替换 FontDiffuser RSI structure source | **2/5** | 工程上简洁、因果上易对照；本身是局部改造。继承 QKV/RSI 是优点，不应伪装成架构创新。 |
| multi-support | **1.5/5** | 多样例风格条件在字体生成/个性化中常见；当前 own-font 8 字更接近增强 conditioning，而非新的跨语言语义机制。 |
| identity-safe RSI 起点 | **1.5/5** | 很好的实验控制与优化设计，不是主要科学贡献。 |
| rationality-centric 三轴评测 + 设计师验证 | **3/5（若验证成功）/1.5（当前）** | 有潜在独立价值，但必须有可靠 evaluator、清晰 construct、公开协议/题本和足量人评；当前 T2 失败使其尚未成为贡献。 |
| bank 策展的 inference-time steering | **2/5** | “改检索库即改先验”几乎由机制定义直接推出；新意取决于能否证明方向性、单调性、质量不退化与跨字体稳定性。 |
| “Δ 是跨语言变化空间表征” | **潜力 3.5/5，当前证据 1/5** | 如果出现稳定线性结构、可组合方向和跨 bank 泛化，会提升方法层次；当前只是命名，尚无表征学习或几何证据。 |

### 2.3 是否足够 ICLR

单独的“library-prior same-char residual 替换 RSI source”是**偏弱但可发表的技术点**，是否够 ICLR 完全依赖三件事：显著且稳定的跨脚本收益；能排除检索增强/额外信息量的替代解释；证明 Δ 的结构确实被模型按预期使用。若只能展示若干视觉例和像素指标小幅提升，更像专门领域会议/期刊增量工作。若评测协议成为可信资产、matched contrasts 清晰、Δ probe 有机制发现，则组合贡献可达到 ICLR Borderline/Accept 水平。

---

## 3. Significance

### 3.1 问题重要性

few-shot cross-script font completion 是有实际价值的：字体家族国际化成本高，中文、拉丁、假名、注音的结构统计差异大，少量参考下既要保持 glyph identity，又要产生“属于同一字体家族”的合理设计。它同时连接：

- 字体生成、字体补全与计算字体设计；
- reference-based image generation / image editing；
- personalization 与 retrieval-augmented generation；
- 生成模型评测、人机协作设计。

最可能引用本文的是字体生成和计算设计社区；图像编辑/个性化社区只有在 Δ 被证明为可泛化的条件表示或 bank intervention 原理时才会广泛引用。纯粹在一个 96×96、16 个测试字体的数据集上刷表，跨社区影响有限。

### 3.2 对 ICLR 是否足够重要

**题目足够，当前外部有效性还不够。**主测试为 test16×295、训练 bank 228 字体、单 seed，足以做严谨受控实验，但不足以自然支持广泛的“cross-lingual representation”主张。至少应报告按字体而非按 glyph 的不确定性，并在一个额外外部字体集或真实设计师任务上验证。否则 4720 glyph 会制造虚假的大样本观感，而独立统计单位实质更接近 16 个字体。

### 3.3 评测贡献会加分还是稀释方法论文

若做到“清晰构念 → 独立 evaluator 自测 → 与设计师判断相关 → 冻结后评主方法”，它会显著加分，并可成为第二主贡献。若 E12 不过门却仍塞入多个自定义 style 分数，再附小规模主观问卷，它会稀释论文并引出 metric hacking 质疑。

建议把它定位为**支持方法结论的验证协议**，而不是在方法结果未稳时并列声称“新 benchmark”。只有满足外部池、预注册、人类一致性、开放题本/manifest 后，才升级为独立 benchmark contribution。

---

## 4. Soundness：声明—证据审计

### 4.1 因果设计中真正扎实的部分

- `F1 ↔ F2`：若二者最终保持同 F0、同 identity-safe RSI、同预算、同数据顺序/RNG/drop、只换 official source 与 Δ source，则可识别 **Δ 条件源的平均处理效应**。
- `F2 ↔ F3`：若同 endpoint 和其余契约一致，可识别 **允许 own-font support 参与联合优化的整体处理效应**；不能说只测了 adapter 参数，也不能说 joint 优于 staged。
- `official P1 → E1`：表明数据域内 FT 带来的收益，但不是 Δ 的单变量证据。
- `F0 → F1`：测的是加回整个 identity-safe official-RSI package 的效应。
- identity-safe parity：保证新增 RSI 在 step 0 与 F0 等价，是控制优化起点的证据，不是 Δ 能力证据。

### 4.2 当前最弱的三条链

1. **“Δ 是变化空间表征”**：目前只有一种手工构造的条件张量和未来 probe 计划；没有线性、组合、插值、方向或跨 bank 泛化证据。Reviewer 会说它只是 retrieval-conditioned feature average。
2. **“合理性评测可信”**：φ_s2 的核心 family/style discrimination gate T2 只有 0.732，低于 0.90；用它评主结果会变成未校准 ruler 测方法。
3. **“few-shot cross-lingual SOTA”**：主 arms 未完成、推理路径尚需 parity、无主表、外部 baselines 未在统一协议下跑完；该声明目前为零证据。

### 4.3 Claim–evidence gap 表

补救成本按 9/25 前的相对工作量估计：低=数小时至 1 天，中=1–3 天，高=多臂训练/人评/外部复现。

| Claim | 当前证据状态 | 主要差距 | 最小补救 | 成本 |
|---|---|---|---|---:|
| 跨脚本时 official RSI structure source 不如目标字对齐先验 | 机制动机合理；FD 本来处理异字错位 | 尚无完成的 F1/F2 结果，不能把假设写成事实 | 完成 F1/F2 80k；统一生成；font-cluster paired CI；按脚本分层 | 高 |
| Δ 优于 official source | matched 设计已建立 | F1≈300、F2=6573，结果缺失 | F1/F2 同 endpoint、同推理与相同 noise；预注册主指标 | 高 |
| 收益来自 Δ 而非训练域/容量 | F0/F1/F2 拓扑可分解 | 需要全部结果；F0 与 E1 不是单变量 | 主表保留 P1/E1/F0/F1/F2；核心归因只用 F1↔F2 | 中-高 |
| support 提供额外跨语言风格语义 | F2/F3 matched treatment 设计 | support 实际为目标字体 own-font 8 字；“跨语言”与“额外可见真值”边界不清 | 明确 ref/support 可见集契约；F2/F3 同 endpoint；support shuffle/drop/数量消融 | 高 |
| F3 是 single-stage joint 系统 | 架构/训练事实可核验 | 无性能含义；不能推出 joint 优于 staged | 只作描述；不声称 superiority | 低 |
| Δ 去除了字符 identity、仅表示 change | 只有减 neutral 的设计直觉 | `Ec` 特征差仍可能含 identity、style、encoder bias | wrong-char、neutral-only、shuffle、RMS-matched corruption；identity probe | 中 |
| Δ 是“跨语言 style change-space representation” | 无直接证据 | 无 learned objective、空间几何或可迁移 probe | bank interpolation、direction consistency、linear probe、层级 Δ/output 对齐 | 中-高 |
| bank 策展可定制偏好 | 由 top-k 机制可推导“输入会变” | 方向性、稳定性、质量代价、训练/推理 shift 未测 | full/随机同规模/serif/sans 等固定 checkpoint probe；F2/F3 双跑；独立指标+人评 | 中 |
| bank 越大越符合设计师通用原则 | 仅 PI 视觉观察 | 无定义、无随机规模控制、无理论 | 多随机嵌套 bank + 设计师预注册判断；最好 matched retrain | 高；建议删 claim |
| Es-α 能检索真实风格邻居 | V1/V2 正式结果缺失；Es 只作排序 | 无语义邻居标注；AUC 不能转成污染率 | 报真实 cosine、top-k turnover、人工邻居审计；α 替代消融 | 中 |
| 外部 library 信息不是泄漏 | LOO、train228 prior、Es 不入生成器是设计事实 | `α×Ec(library target char)` 明确带入不可由 ref 唯一确定的先验 | 清楚披露 transductive/closed-bank prior；test 字体排除；random/wrong bank controls | 低-中 |
| 自动指标能测跨脚本合理性 | 三轴框架合理，T1/T3/T4 过 | T2 0.732；外部训练只有 5 字型；尚无人类效度 | 扩到 ≥25 个独立字型并按字型切分；T2 过门；独立人评相关性 | 高 |
| 26-typeface pool 足以训练 φ_s2 | 目前表述与 v3 实际训练池不一致 | “26 字体文件”不等于 26 独立字型；当前只有 5 字型参与训练 | 以独立设计家族/字型为单位审计并扩池；报告 license、weight grouping | 中-高 |
| rationality 是可测 construct | 目前主要是总括词 | 容易与 style similarity、legibility、aesthetics 混淆 | 拆成 style-family consistency、glyph identity/legibility、design preference 三个问题 | 中 |
| 自动指标优于 pixel metrics | 有合理批评与拟议人评验证 | 尚无 metric-human correlation 或 predictive AUC | 预注册比较 LPIPS/SSIM/SC 系列预测 2AFC 的能力 | 中 |
| few-shot cross-script SOTA | 无完成主表 | baseline 范围、协议公平性、统计均未闭环 | 统一 test16×295 跑强专用 baselines；报告成本与可见信息 | 高 |
| 方法泛化到 image editing/personalization | 仅概念邻接 | 没有字体外任务或通用表征证据 | 降为 future work；不要作为主 significance 证据 | 低 |

---

## 5. 方法 critique

### 5.1 Δ：变化空间表征，还是 retrieval-augmented RSI？

**当前更准确的名字是“character-aligned retrieved residual prior”，不是“change-space representation”。**

原因如下：

- Δ 没有通过一个专门的表征目标学习；它是冻结 `Ec` 特征的凸组合减法。
- α 来源于 `Es` 的相似度排序，Δ 的几何受两个预训练 encoder 与 bank 组成共同决定。
- 每个目标字符都有自己的 Δ，尚未证明不同字符的方向可比、同一风格方向跨字符一致，或线性运算有语义。
- 模型可能仅把 Δ 当作有用的 retrieval feature，甚至学习忽略它；最终输出改善不自动证明“捕获了变化空间”。

要让 representation claim 可信，至少需要以下四类 probe 中的三类：

1. **跨字符方向一致性**：同一 bank/style intervention 在多个字符和脚本上的 Δ 方向，与输出变化方向具有稳定 cosine/CCA 对齐；以字体为统计单位。
2. **插值与单调性**：在两个预注册 bank/style prototype 间对 `Δ(t)=(1-t)Δ_A+tΔ_B` 插值，独立 style axis 随 t 单调变化，同时 identity/quality 不退化。
3. **可组合性**：如 weight/serif/slant 等已标注方向，验证向量差可跨字体或字符迁移；需要 held-out style labels，不能用 Es 自证。
4. **解码/线性 probe**：从 Δ 预测独立设计属性或 held-out family，比较 absolute `Ec`、未减 neutral、随机/均匀 α、top-1、官方 source。

还应做最小必要性消融：`retrieved same-char absolute Ec`、`retrieved Δ`、`uniform top-10 Δ`、`wrong-char Δ`、`random-bank Δ`。否则 reviewer 无法判断真正有效的是 residual、same-character alignment，还是单纯给模型看到了训练库目标字。

### 5.2 support 从 cross-font retrieval 改成 own-font 8 chars 的影响

这会明显改变论文语义。own-font support 的优点是它确实提供目标字体的局部风格证据，能补充 bank 平均；缺点是：

- 它不再证明“从跨字体 support 检索出跨语言语义”，而是**利用更多目标字体可见 exemplar**。
- 如果 ref8 与 support8 是不同的可见字符，F3 的信息预算高于 F2；F2↔F3 仍可测“增加 support treatment”的系统效应，但 SOTA 公平比较必须让 baseline 获得相同 16 个或明确总可见 glyph 数。
- 如果 support 与 ref8 相同，只是换 encoder/pathway，则应明确写“multi-path encoding of the same support set”，而非额外 few-shot 信息。
- “跨语言 style semantics”过强。support 给的是同字体跨字符局部证据，是否形成 script-invariant semantics 仍需 probe。

建议把它改写为 **target-font exemplar support**，并在协议第一页明确：ref/support 是否重叠、总可见 glyph 数、字符脚本、训练/测试如何固定。至少做 support count `{0,1,4,8}` 与 shuffled-font support；否则 F3 的贡献可能只是“看了更多目标字体图片”。

### 5.3 bank-prior / controllability：第三贡献还是 bolt-on？

目前是 bolt-on 风险大于贡献价值。它与 Δ 有逻辑关系：bank 决定候选邻域，因此策展 bank 会改变 Δ；但“输入可干预”不等于“可控生成”。在固定 checkpoint 上换 bank 是 OOD intervention，只能证明 sensitivity/steering，不能证明训练得到可靠控制器，也不能证明 curated bank 提高总体合理性。

9/25 的最佳处理是：

- 正文作为**机制分析或 emergent property**，最多一张 probe 图；
- 必须包括 full bank、多个同规模随机子集和策展子集；固定 ref/content/noise；
- 同时报告方向效应、identity、coverage/LPIPS 与失败率；
- 不写“designer principles”“free reliable control”或“trained controllability”。

若 P2 没有清晰单调方向或造成质量下降，移附录，不让它拖累主论文。

### 5.4 Es-based α 的四个 reviewer 问题

**循环论证。** Es 用于检索，若再用同一 Es 评价风格就是自证。独立 φ_s2 在概念上足以打破直接循环，但前提是它通过 T2、训练数据与方法 bank 隔离、且人类效度成立。当前 T2 失败，所以还不够。

**非参考风格污染。** 真正进入生成器的非参考信息是 `α×Ec(library font,target char)`。这不是实现 bug，而是方法定义的 library prior；错误邻居可能是 top-1，而不仅是低权重尾部。应报告 top-1 mass、effective-N、邻居人工审计和 wrong/random bank 敏感性。

**α 是否必要。** 需要比较 uniform top-10、top-1、random top-10、Es-ranked top-10，以及至少一个 independent retrieval embedding。否则无法区分“检索质量”与“库目标字平均就是强先验”。

**train228 是什么假设。** 必须明确披露为训练期可访问、包含目标字符渲染的字体库先验；测试字体严格不在库，train query 执行 LOO。可借鉴 CF-Font 对 basis/prototype library 的披露方式，但不能借其先例免除公平性问题。与不使用库或使用相同库的 baseline 比较时，要明确方法可访问的信息不同。

---

## 6. 评测 critique

### 6.1 “合理性”是好 hook，但必须可操作化

“合理”不是单一可观察变量。设计师可能同时考虑：字符读不读得出、是否像参考字体、字面比例是否协调、是否美观、是否忠于某个唯一 GT。这些判断会冲突。建议避免单一 rationality score，拆成三层：

1. **Glyph identity / legibility**：字符身份、可读性、关键部件不混淆；自动 ID-CLS/OCR + 人工 identity 题。
2. **Style-family consistency**：与参考字族在笔重、端点、曲率、对比度、宽高/重心等属性上是否一致；独立 embedding + 分属性 2AFC。
3. **Designer preference / plausibility**：在身份正确前提下，哪一个更像可信的同字族设计；盲法 2AFC 为主，MOS 为辅。

LPIPS/SSIM/L1 继续保留，但定位为 GT reconstruction fidelity，而不是合理性的充分指标。GT 只是 positive control 或一种可接受实现，不是唯一正确答案。

### 6.2 外部池与 T2 失败

“26-typeface external pool”只有在 26 个是**独立设计字型/家族**、而非少数字型的多个 weight/style 文件时才有意义。当前 v3 诊断表明实际训练只有 5 个字型；重字重 AUC 接近 1、轻/常规字重接近随机，说明 evaluator 主要抓住笔画粗细，没有学到足够的 family identity。

T2 `0.732±0.057 < 0.90` 的含义不是“差一点”，而是该 evaluator 目前**不具备承担主风格结论的判别效度**。继续用它选 checkpoint、选 QKV、报告 SC-Gap 或验证 bank direction 会产生评价者—方法共同偏差。必须先扩充独立字型、按字型 group split、训练完成后重新过 T1–T4；不能事后降低 0.90 gate。

时间上，E12 已成为 9/25 的关键路径。若 9/12–9/15 前仍不过门，应立即启用降级方案：自动风格指标只作探索性附录；正文以身份/质量客观轴 + 扩大且预注册的人评为主，避免虚构一个不可靠的 composite。

### 6.3 设计师人评建议

当前“240 个 2AFC + 120 MOS，3–5 名评审”可作 pilot，但若这些题由每位只看一部分，统计功效与一致性可能不足。最低可信方案：

- **评审人数：** 至少 5 名具有字体/平面/文字设计经验者；最好 8–12 名。另招普通读者评 legibility，避免设计师偏好代替可读性。
- **2AFC 为主：** 每个核心对比（F1/F2、F2/F3、最佳方法/最强外部 baseline）按字体与脚本分层；同一题随机左右、方法匿名、ref/support 同屏；允许“无差异”会降低强迫噪声，但需预注册统计模型。
- **MOS 为辅：** 分开问 style consistency 与 design plausibility，不问含混的“总体合理性”；给出锚点与训练题。
- **统计单位：** mixed-effects logistic/ordinal model，方法为固定效应，字体、字符、评审为随机效应；同时报告 bootstrap CI、评审间一致性。不能把所有单字判断当独立样本做简单 t-test。
- **预注册：** 在看主结果前冻结题本抽样、排除规则、主 comparison、最小效应、tie 处理、质量/身份失败过滤、分析式；记录评审背景与报酬。
- **避免 cherry-picking：** 题目由 manifest 随机分层抽样，不由 PI 挑图；失败案例必须进入抽样框。

### 6.4 评测贡献成立的最低门槛

1. φ_s2 在真正独立字型池上通过 T1–T4，尤其 T2≥0.90；
2. evaluator 训练/选择不访问 test16 或方法输出；
3. SC 指标对设计师 2AFC 的预测显著优于 LPIPS/SSIM，并报告 CI；
4. 题本、manifest、定义和评分代码可发布；
5. 不用一个 composite 掩盖 identity/style/quality trade-off。

---

## 7. SOTA framing

### 7.1 9/25 最小可信说法

在主表未完成前，摘要和标题都不应写 SOTA。若最低集完成，可写：

> “On our fixed Chinese-to-Latin/Japanese/Bopomofo test protocol with eight visible reference glyphs, the proposed system outperforms reproduced FontDiffuser and the strongest compatible baselines on independently validated style consistency while preserving glyph identity and quality.”

只有当强外部 baseline、统一可见信息、统一训练/推理分辨率与统计 CI 齐全时，才可改为：

> “We establish a new state of the art under the specified test16×295 protocol.”

这仍是**协议限定 SOTA**，不能泛化成所有 cross-lingual font generation 的普遍 SOTA。

### 7.2 主表最低六行

| 行 | 作用 | 必须公平控制 |
|---|---|---|
| FontDiffuser official P1 zero-shot | 官方外部 anchor | 官方权重、统一 ref 数/采样器；注明原训练域不同 |
| E1 | 同 train228 数据域 FT anchor | 与本法相同数据、字符集、分辨率 |
| F0 | RSI-free 直接底座 | 说明与 E1 的拓扑差异，不作 Δ 单变量结论 |
| F1 | identity-safe official-source RSI | 与 F2 同起点、预算、RNG、drop、推理 |
| F2 | Δ-RSI | 核心 `F2−F1` paired contrast |
| F3 | Δ+support | 核心 `F3−F2` contrast；披露 support 信息预算 |

每行至少报告：ID/legibility、独立 style consistency、quality/fidelity、采样时间/额外 bank 存储；按 Latin、平假名、片假名、注音分层；font-cluster 95% CI。单 seed 训练必须坦白，不能以 4720 张图伪装模型方差。

### 7.3 外部 baseline 的优先级

**必须：**

1. **DG-Font**：deformable skip 的直接机制前身；同数据重训，给相同 ref/support 信息。
2. **CF-Font**：library/prototype/basis 先验的最近概念对照；若官方实现不能自然跨脚本，需报告适配方式和失败，而不是省略。
3. **FontDiffuser**：official zero-shot + same-domain FT（E1）都要有；它已声称可扩展到 cross-lingual，因此是最直接 baseline。
4. **至少一个强 few-shot font completion baseline**：如 LF-Font/MX-Font 中可复现且协议最接近者；用于避免只打 diffusion 家族内部对照。

**条件性 baseline：**

- **FontStudio / UniGlyph 或其他明确支持 multi-script/few-shot 字体生成的最新工作**：一旦核实任务协议相近，必须列为强 competitor 并尽量复现；如果仅是 text-to-image 字体风格化，则作为 related work，不强行纳入像素主表。
- **InstantStyle**：可作为 reference-style image generation baseline，但需字符 identity 控制适配；若无法稳定输出精确 glyph，应报告适用性限制。
- **AnyText**：主要是场景文字生成/编辑，通常不是完整字体补全；可作为文字渲染与 identity 保持邻接工作，不足以替代字体 baseline。

基线公平性的第一原则是**相同目标字体可见 glyph 数**。如果 F3 实际使用 ref8+额外 own-font support8，而 baseline 只看 ref8，SOTA 结论不成立；要么让所有方法看同一 16 张，要么让 support 从同一 ref8 派生。

---

## 8. Related-work positioning 与检索风险

### 8.1 建议的定位轴

不要按“GAN / diffusion”简单列工作，应按解决的信息瓶颈组织：

1. **分解式 few-shot font generation：** LF-Font、MX-Font、DG-Font、CF-Font；比较 component/local feature、deformable alignment、prototype/basis bank。
2. **diffusion 字体生成：** FontDiffuser 及相近 font diffusion；比较 reference structure、style conditioning、跨语言展示与监督。
3. **多脚本/跨语言字体家族补全：** 按输入脚本、输出脚本、是否 unseen font、可见 glyph 数、是否有 paired multi-script GT 做表格。
4. **通用文字图像生成/参考风格迁移：** AnyText、InstantStyle 等；明确它们优化的是场景文字/图像风格，不必然满足 glyph-set consistency。
5. **计算字体设计评测：** 像素/感知指标、人评、家族一致性与可读性。

### 8.2 必须主动处理的近邻风险

- **FontDiffuser 已做 cross-lingual 定性展示。**本文贡献不能写“首次跨语言”，而应是“为跨脚本 RSI 条件错位提出 target-character-aligned prior，并建立受控评测”。
- **DG-Font 的 FDSC 已把形变与目标字符特征联系起来。**需清楚说明本文不是首次 target-character-conditioned deformation，而是把 style-retrieved same-character residual 注入 diffusion RSI。
- **CF-Font 的 basis/prototype/library precedent。**需比较 bank 角色：CF-Font 如何形成/组合 basis，本法如何用 Es 排序并把 library `Ec(target char)` 作为条件；最好有同 bank 预算对照。
- **FontStudio / UniGlyph / DreamFont3D / FontWorks 名称有歧义风险。**截至本次审查，在线检索因网络连接失败，仓库材料也不足以确认这些名称分别对应 2D glyph completion、multi-script font family、3D text asset 还是工业系统。投稿前必须用“精确标题+作者+年份+官方论文/代码”逐一消歧；不能凭名称声称已覆盖或不是 competitor。DreamFont3D 若主要做 3D 字体/文字资产，通常不是直接 baseline；若 FontStudio/UniGlyph 确实从少量参考生成多脚本 glyph，则属于必须引用并同协议比较的高风险近邻。
- **2025–2026 新工作缺口。**ICLR 2027 投稿必须再做一次系统检索，覆盖 arXiv、CVPR/ICCV/ECCV、SIGGRAPH/TOG、AAAI/IJCAI、ICLR/NeurIPS，并记录任务协议，而不是只沿用 FontDiffuser 2024 前后的文献表。

建议 related-work comparison table 至少包含：`unseen font?`、`input scripts`、`output scripts`、`#reference glyphs`、`paired target training?`、`external font library at inference?`、`target-character prior?`、`human designer evaluation?`。

---

## 9. 最可能被攻击的 5 个点与答辩预案

### 攻击 1：这只是把 RSI 输入换成检索平均，技术增量太小

**有效答辩前提：** 不争辩“架构复杂度”，而展示问题特异性与机制证据。

答辩预案：明确承认继承 RSI，贡献在 conditioning source；用 F1↔F2 证明只换 source 的稳定效应；用 absolute-vs-residual、same-vs-wrong-char、ranked-vs-uniform/random 消融证明三个设计元素必要；再用 offset localization 或 Δ corruption 证明模型确实使用该先验。没有这些结果时，此攻击无法靠文字化解。

### 攻击 2：“change space representation”是过度包装

答辩预案：主文先降为 `retrieved residual structure prior`。只有当跨字符方向一致性、插值单调性和独立属性 probe 均成立，才在分析节使用“the residuals exhibit a structured change space”，而非先验定义式地宣布 representation。

### 攻击 3：自定义 evaluator 不可靠，T2 都没过

答辩预案：扩充至 ≥25 个真正独立字型，严格 group split，冻结 evaluator 后再看 test；报告 T1–T4、与设计师 2AFC 的相关/预测 AUC，并让 LPIPS/SSIM 参加同一预测比较。若 T2 仍不过，撤下该指标，主结论改以预注册设计师 2AFC 和客观 identity/quality 轴支持。

### 攻击 4：F3 只是多看了 8 张目标字体图，不是新的跨语言语义

答辩预案：第一，明确总可见信息并让所有 baselines 使用同一 support budget；第二，做 same-ref8 adapter、额外-support8、wrong-font support 三组对照；第三，把 claim 改为 target-font exemplar evidence，不声称已证明 script-invariant semantics。

### 攻击 5：SOTA 不公平/不完整，且单 seed、test 字体仅 16 个

答辩预案：限定为固定协议 SOTA；纳入 FontDiffuser official+FT、DG-Font、CF-Font、至少一个强 few-shot baseline，以及经核实的最近 multi-script competitor；以字体聚类 CI 和每字体胜率报告，不把 glyph 当独立样本。单 seed 明确列为局限；若来得及，优先给 F1/F2 核心 contrast 补第二 seed，而非增加边缘 baseline 或更多 controllability 图。

---

## 10. 9/25 前必须完成的最小 credible 集合

### P0：决定论文能否成立

1. **锁定并完成 F1/F2/F3 80k。**不再改 QKV、support 定义或 endpoint；保存完整 provenance。
2. **推理 parity gate。**训练与推理的 full-bank indices、weights、逐尺度 Δ、support、n-shot style condition 一致；固定 ref/content/noise/采样器生成 test16×295。
3. **先得到核心因果表。**F1−F2、F2−F3 的 effect size、font-cluster bootstrap CI、每字体胜率、按脚本分层；identity 与 quality non-inferiority 必须同时看。
4. **E12 作出 go/no-go。**扩真正独立字型并重训；T2≥0.90 才进入主指标。设硬截止，失败即启用“自动 style 指标降级 + 人评主导”方案。

### P1：使主张可信

5. **最小机制消融。**至少 `absolute retrieved Ec` vs `retrieved Δ`、wrong-char/spatial-shuffle、uniform/random α；记录 offset magnitude 与真实变化区域定位。若时间只能做两个，优先 wrong-char 与 uniform/random α。
6. **最小人评闭环。**预注册、盲法、随机分层题本；核心三对比；≥5 名设计师；2AFC 为主；mixed-effects 或 font-cluster CI；公开题本 manifest。
7. **统一信息预算的 baselines。**至少 FontDiffuser official/E1、DG-Font、CF-Font、一个强 few-shot completion baseline；核实 FontStudio/UniGlyph 后决定是否强制加入。所有方法看到相同 ref/support。

### P2：有结果才保留

8. **bank steering 仅作 probe。**full、多个同规模 random、serif/sans 等 curated；F2/F3 双跑；方向、identity、quality 三轴。失败或非单调则移附录。
9. **representation probe。**优先做跨字符 direction consistency + interpolation；没有阳性结果就删“change-space representation”。
10. **写作与可复现。**明确 train228 prior、LOO、单 seed、test16 局限、失败案例；冻结主表后才写 SOTA。

**建议砍掉：** 9/25 前的 matched curated-bank retraining、staged-vs-joint superiority、额外 QKV 搜索、更多 RSI scale、泛化到通用 image editing。它们会争抢核心证据链资源。

---

## 11. 给 PI 的三条最重要叙事调整

1. **把“表示学习”降为“目标字符对齐的检索残差先验”。**先用结果证明它比 official RSI source 有效，再用 probe 决定能否升级成 structured change space；不要从一个减法公式直接推出表示几何。
2. **把论文主线压缩为两个可归因贡献。**贡献一是 Δ source（F1↔F2），贡献二是 rationality-aware evaluation；support 是最终系统增强（F2↔F3），bank steering 是分析性质。四条并列主贡献会显得 scope creep。
3. **把“设计师合理性”拆成可读性、字族风格一致性、设计偏好。**不再声称 bank 越大自动符合“通用原则”；让独立指标和盲法设计师判断共同定义证据，而不是由 PI 目测先定义结论。

---

## 12. AC 式最终判断

这是一个**有清晰失败模式、有聪明而简洁的条件先验、也有较好因果实验意识**的项目；潜在亮点是把跨脚本问题从“生成更像 GT”改写为“在保持身份与质量时生成设计上可信的同字族 glyph”。但在当前证据快照下，技术贡献尚不足以抵消主结果、评测效度和强 baseline 的缺失，故为 Reject。

若 9/25 前只完成一件事，应是让 `F1 → F2 → F3` 在**同一可信评价尺**上形成不可争辩的 paired evidence。若这条链效应小或不稳定，应诚实收缩为 Δ-RSI 方法分析论文；若效应大且人评一致，再把 benchmark 与 steering 升级为附加贡献。
