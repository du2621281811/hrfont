# Delta 减中性 Content 与 RSI 接入的设计理由

## 结论

应使用

\[
\Delta_c=\sum_s\tilde\alpha_s E_c(A(B_s,c))-E_c(A(B_0,c))
\]

作为 RSI 的结构源。这里必须是**同字符、特征空间内加权后相减**：先分别编码各个 `A(B_s,c)`，再按 alpha 混合多尺度特征；不能把像素先混合再编码。Delta 不是第二份内容输入，而是相对固定中性底 B0 的、与目标字符空间对齐的变化条件。

## 问题一：为什么 Delta 要减去中性 content？

### 1. 移除身份/内容公共分量

绝对特征 `Ec(A(B_s,c))` 的主导因素仍是字符 `c` 的身份和几何结构，而 MCA 已从 `Ec(C)` 获得同一个 `c` 的身份条件。若 RSI 再接收绝对内容特征，就会重复输入“这个字长什么样”，并把身份信息混入“这个字需要怎样改变”。减去同字符的中性 `Ec(A(B0,c))`，会抵消两者共享的大部分内容分量，避免 RSI 与 MCA 争夺身份建模职责。

### 2. 把信号集中到相对外观变化

差分回答的是：在 alpha 选出的风格近邻中，同一个 `c` 相对中性底偏离了什么。它把能量集中到笔画宽度、曲率、端点、字腔和局部比例等外观差异上；当近邻与中性底一致时，Delta 自然趋近 0。固定 B0 也使不同字体、字符和 run 的变化都处在共同坐标系中。

### 3. 与 CFG/unconditional 语义一致

Delta-drop 时将该条件置零，语义是“不提供 Δ 信息”；offset 头复用 E1 权重、无新增参数（PI 2026-09-04 决策不加 zero-init gate，接受头对 Δ 统计的轻度过渡期），单通路归因较干净。但必须区分**条件为零**与**形变为零**：E1 offset head 的 GroupNorm affine 和多层 projection 均有非零 bias，因此 `Delta=0` 仍可能产生非零 offset，不能称为严格的 no-deformation 路径。

### 反事实：不做减法会怎样？

若直接把 `sum alpha Ec(A(B_s,c))` 送入 RSI，其数值与空间结构会主要表现为 `c` 的内容特征。RSI 便退化成官方 `Ec(S)` 接线的同类问题：用另一份绝对内容骨架驱动 offset，只是把汉字 `S` 换成库中的 `c`；alpha 的风格近邻信息容易被强内容公共分量淹没。此时 drop 分支也不再对应自然的“无变化”原点，RSI 与 MCA 重复，无法清楚归因收益来自相对风格变化。

## 问题二：这样的 Delta 用 RSI 嵌入合适吗？

### 为什么架构上匹配

RSI 的 `OffsetRefStrucInter + DCN` 不是通用风格编码器；它以参考字 `Ec(S)` 为 Query、UNet skip 为 Key/Value，经 cross-attention 预测 deformable-convolution offset。官方训练**本来就随机选择与目标不同的参考字**，并明确用 cross-attention 处理二者的空间错位，所以不能把“异字”或“错位”本身写成实现错误。我们的动机应限定为待验证假设：同语系异字仍共享较多笔画规律，而汉字参考与拉丁目标的局部结构可迁移性可能更弱，使官方绝对参考结构不是最合适的 RSI 条件。

Delta 则由**同一个目标字符 c** 的近邻字体特征减去中性特征得到。它把 RSI Query 改到目标字符坐标，并旨在突出“相对中性应怎样变化”；用户参考 R 的真实风格仍独立走 `Es(R)` 风格交叉注意力。该接法在架构上更接近 DG-Font 以目标字符低层特征为形变对象的几何动机，但是否优于官方参考结构必须由 E2/E2b 及受控破坏实验验证。

### 三处必须诚实保留的张力

1. **Delta 并非纯 where。** 它是同字符特征差，offset 的方向和幅度本身已经包含粗粒度的“怎么变”。论文只能表述为“架构主要将 Delta 用作定位/对齐信号”，不能宣称它完全不含 what；应由 E9 接入与破坏实验验证。
2. **尺度与分布改变。** Delta 是有符号残差，而 E1 offset 头原先接收绝对 `Ec(S)`。GroupNorm 会缓解尺度差，却不会保证语义分布匹配。offset 复用 E1 头（无新增参数、无 zero-init gate）；Delta-drop 仅表示不给 Δ 条件，不保证零 offset。任何破坏实验都须逐层匹配正确 Delta 的 RMS，避免能量差成为混杂因素。
3. **Ec 的西文字形质量未知。** 官方 Ec 在拉丁字符上的局部对应是否可靠尚未实证。E0/E9 的 wrong-char Delta、wrong-style Delta、spatial shuffle 与 magnitude-only 消融，以及 offset 定位指标，必须承担这项验证，不能仅凭架构直觉下结论。

### “Delta 适合 RSI”的预注册判定条件

只有 E9 同时出现以下模式，才支持较强结论：

- spatial shuffle 在严格 per-layer RMS 对齐后显著伤害主要指标，说明空间布局确实被使用；
- magnitude-only 保留大部分正确 Delta 的收益，说明 RSI 对变化强度/粗粒度方向有效，而不是仅记忆库字符身份；
- offset 对 `|GT-B0|` 轮廓变化区的定位 AUROC 高，并有一致的 AUPRC/IoU/correlation 证据。

若 spatial shuffle 不伤害，Delta 更可能只是全局调制；若 magnitude-only 丢失大部分收益，则需要承认细粒度空间 what 也是关键；若定位 AUROC 不高，则不能把 offset 解释为变化区域对齐。

## 论文可直接采用的表述

“We construct a character-aligned residual condition by subtracting the neutral-font content features from an alpha-weighted mixture of same-character neighbor features. This is intended to suppress the identity component already supplied by MCA and to represent changes relative to a shared neutral reference.”

“Although the original RSI uses cross-attention to accommodate different reference and target characters, we hypothesize that the transferability of absolute reference structure weakens across scripts. We therefore replace `Ec(R)` with this same-character residual while retaining `Es(R)` for style cross-attention, and test the hypothesis with matched controls, RMS-controlled interventions, and offset localization.”

## Q&A 增补（2026-09-04）

### Q1. Δ 的机制五连问

#### (a) “找邻居”是字体 kNN，还是 CF-Font 式聚类基底？

默认是**在完整 train228 字体池上按风格相似度找邻居并加权**，属于 kNN/soft-neighborhood 路线，不先做 K-Means，也不把字体压成少数聚类中心；当前实现直接把 query 与 `proto_feats` 做 cosine，并提供 `soft/topk/threshold` 三种稀疏化方式（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`131`](../scripts/hrfont_delta_v2.py#L131)），执行规格则冻结库为 train228、目标字体 leave-one-out（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`25`](../.cursor/rules/hrfont-execution-spec.mdc#L25)）。这更符合本方法的 insight：CF-Font 需要小 spanning basis 来做内容插值，而我们只需从风格相近的库字体取**同一个目标字符**的 Ec 特征并加权；先聚类会把 228 套字体量化到 medoid，可能丢掉恰好最接近 query 的非 medoid，而没有带来本方法必需的表示优势（[`reports/ICLR2027_HRFONT.md:120`](./ICLR2027_HRFONT.md#L120)–[`135`](./ICLR2027_HRFONT.md#L135)）。计算上，228 个已缓存原型只做一次向量 cosine，Ec 只处理筛出的邻居；原型构建本身也是固定 ref8 的离线循环并可落 manifest（[`scripts/hrfont_delta_v2.py:210`](../scripts/hrfont_delta_v2.py#L210)–[`242`](../scripts/hrfont_delta_v2.py#L242)），因此没有必须靠聚类压库的算力瓶颈。CF-Font 式变体仍作为“先 K-Means、每簇取 medoid、再在 medoid 上算 α”的消融保留在 P2-12；旧实验表把它写作 `kNN vs 12 medoid` 的 E7（[`reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:310`](./hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md#L310)–[`317`](./hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md#L317)、[`reports/ICLR2027_HRFONT.md:383`](./ICLR2027_HRFONT.md#L383)）。

#### (b) B₀ 是不是直接使用 Content 图？为什么保留两个中性字体？

不是：`Content` 是 **Noto Sans CJK Regular** 上目标字符 $c$ 的 A 渲染，送入 $E_c(C)$ 作为 MCA/Identity 条件；`B_0` 现已统一为 **Noto ContentImage**（合作者 2026-09-04 决策）：Δ 减数与 Content/Identity 输入是同一张 Noto 同字渲染，并统一 RS-gap 与 Support 的结构坐标（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`23`](../.cursor/rules/hrfont-execution-spec.mdc#L23)–[`26`](../.cursor/rules/hrfont-execution-spec.mdc#L26)）。两者现已统一为同一张 Noto 同字渲染：B₀=Noto ContentImage（合作者 2026-09-04 决策），Δ 减数 = Ec(Content)，与 MCA/Identity 输入共享同一张图，不再存在第二个原点；Δ、gap、support 的坐标系随之统一到 Content。

#### (c) 风格邻居究竟按什么找？Es 不准怎么办？

没有额外手工风格特征或另训检索器：α 只用冻结 $E_s$ 的 ref8 编码；具体实现把目标字体八个已归一化向量 `[8,D]` 与各库字体的同字 per-char 向量逐字做 cosine、再对字平均（per-char cosine 平均，合作者对齐），随后 softmax/τ 全池加权 + ε/K_max 截断（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`95`](../scripts/hrfont_delta_v2.py#L95)、[`225`](../scripts/hrfont_delta_v2.py#L225)–[`239`](../scripts/hrfont_delta_v2.py#L239)）。这里的 Es 是 official P1 初始化后、经 E1 在 A/train228 上跨语系联合适配并最终冻结的 style encoder；E1 训练 Ec/Es/UNet，E2 再冻结 Ec/Es（[`.cursor/rules/hrfont-execution-spec.mdc:47`](../.cursor/rules/hrfont-execution-spec.mdc#L47)–[`49`](../.cursor/rules/hrfont-execution-spec.mdc#L49)），所以 α 是方法内部的、与生成模型共享表征的检索机制。用 Es 选 α 不构成评测上的循环论证，因为最终风格结论由隔离于方案编码器的 $φ_{s2}$ 与 T1–T4 门控承担（[`.cursor/rules/hrfont-execution-spec.mdc:56`](../.cursor/rules/hrfont-execution-spec.mdc#L56)–[`60`](../.cursor/rules/hrfont-execution-spec.mdc#L60)）；只有拿 Es 自己给最终方法打分才会循环。Es 是否“够准”由验证电池实证回答：V1 报检索 rank/Recall 与 pairwise AUC，V2 报跨语系同字体对异字体 AUC，V6 报 leave-one-out 最近邻 cosine 与 top-1 α mass（[`scripts/hrfont_validate_e1_encoders.py:109`](../scripts/hrfont_validate_e1_encoders.py#L109)–[`147`](../scripts/hrfont_validate_e1_encoders.py#L147)、[`202`](../scripts/hrfont_validate_e1_encoders.py#L202)–[`214`](../scripts/hrfont_validate_e1_encoders.py#L214)）。因此论文只需主张 Es 提供了**经验证可用的风格相似度量**；向量不具有人可解释轴并不削弱该机制，但 V1/V2/V6 不过门时不能把 α 当作已验证可靠。

#### (d) 到底要不要减 B₀？不减能不能跑？

主方法减 B₀ 的第一层动机是**削弱身份公共量**：绝对 `Ec(B_s,c)` 的主导公共量仍是字符 $c$，而身份已经由 Content→MCA 提供；减同字 B₀ 将 RSI 条件改写为“相对中性底怎样变化”，旨在减少与 Identity 支路的重复（实际 feature-mix 见 [`scripts/hrfont_feature_cache.py`](../scripts/hrfont_feature_cache.py)）。第二层是**条件零点语义**：邻居特征等于 B₀ 时 Δ 精确为零，Delta-drop 也可把输入条件置零；但由于 offset head 含非零 affine/bias，这不等于 offset 或 DCN 形变为零。第三层是**数值条件**：固定 B₀ 旨在消去大的同字 common-mode，使输入成为较小的相对量；这不是“必然更稳定”或“身份已被完全去除”的定理，因此仍须记录逐层 RMS 并做消融。不减当然能前向运行，但得到的是另一份绝对同字内容特征，只能作为 `no-neutral-subtraction` 反事实。

#### (e) 为什么不用 RSI 原注意力“自己学偏移”？

这个前提需要纠正：我们**就是使用 RSI 原来的注意力和偏移机制，块不拆**，只把结构源从 $E_c(S)$ 换成 Δ；官方 model 先以 `self.style_encoder(style_images)` 产生风格条件，又把同一 `style_images` 送入 `self.content_encoder` 产生 RSI 结构源（[`code/official/FontDiffuser/src/model.py:34`](../code/official/FontDiffuser/src/model.py#L34)–[`47`](../code/official/FontDiffuser/src/model.py#L47)）。官方 `OffsetRefStrucInter` 的关键原句是 `hidden_states = self.cross_attention(style_content_hidden_states, context=res_hidden_states)`，随后投影为 offset（[`code/official/FontDiffuser/src/modules/attention.py:288`](../code/official/FontDiffuser/src/modules/attention.py#L288)–[`330`](../code/official/FontDiffuser/src/modules/attention.py#L330)）；`StyleRSIUpBlock2D` 原封不动实例化 `DeformConv2d(...)`（[`code/official/FontDiffuser/src/modules/unet_blocks.py:460`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L460)–[`476`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L476)），并执行 `offset = sc_inter_offset(...)`、`res_hidden_states = dcn_deform(res_hidden_states, offset)`（[`553`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L553)–[`561`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L561)）。不能指望直接喂原始 $E_c(S)$ 后让同一注意力自行补救，因为跨语系 $S$ 是汉字而 skip 是目标西文字 $c$，空间 query/context 没有同字对应；直接喂原始 $E_c(B_s,c)$ 虽同字，却又要求 offset 模块同时从强身份公共量中分离“改什么”并完成“在哪里改”，且失去 Δ=0 的 no-deformation 语义（[`reports/ICLR2027_HRFONT.md:246`](./ICLR2027_HRFONT.md#L246)–[`259`](./ICLR2027_HRFONT.md#L259)）。E9 正是对此的诚实检验：比较 RSI-offset、普通 cross-attn、MCA 与组合接入，并做 RMS 对齐的 wrong-char/wrong-style/spatial 等破坏；另有“关 DCN、只靠交叉注意力”的明确对照（[`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:117`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L117)–[`122`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L122)、[`reports/ICLR2027_HRFONT.md:294`](./ICLR2027_HRFONT.md#L294)）。只有正确 Δ 的空间布局、attention 和 DCN 均带来可重复收益，才能说“变化引导的偏移机制”而不只是多加了一路条件。

### Q2. α 的粒度

#### (a) 相似度用一张还是八张风格参考？

训练用与 style 同一组 **随机 n 张**（n∈[1,8]）；评测用固定八张 `永和书风骨韵天地`。聚合始终是「已归一化 Es 与库字体同字向量逐字 cosine、对字平均」（[`scripts/hrfont_delta_v2.py:88`](../scripts/hrfont_delta_v2.py#L88)–[`101`](../scripts/hrfont_delta_v2.py#L101)），然后 **top-10 softmax**。单字方差大是用 n>1 的动机，不是已证实结果。

#### (b) 每个库字体都要另外提供八张 ref8 渲染吗？

每个库字体和训练目标字体都必须能提供这八张图，但**不需要额外渲染**：A 数据集的每字体 `StyleImage/<font>/` 已对应完整 style338，而 ref8 是其中的固定子集；数据契约定义 style338 与固定 ref8（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`23`](../.cursor/rules/hrfont-execution-spec.mdc#L23)–[`26`](../.cursor/rules/hrfont-execution-spec.mdc#L26)），构建器直接按字符码从既有 `StyleImage` 查这八个 PNG（[`scripts/hrfont_delta_v2.py:200`](../scripts/hrfont_delta_v2.py#L200)–[`207`](../scripts/hrfont_delta_v2.py#L207)、[`225`](../scripts/hrfont_delta_v2.py#L225)–[`231`](../scripts/hrfont_delta_v2.py#L231)）。必须区分两次采样：生成模型的 style 条件输入从该字体目录用 `random.choice(images_related_style)` 在 338 池随机取一张（[`code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py:59`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L59)–[`69`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L69)、[`90`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L90)–[`92`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L92)），而 α 训练时读取该 episode 采样的 R 字集、评测时读取固定 ref8，两者均满足 per-char `[n,D]` 契约（n-shot 协议，PI 2026-09-04）；聚合口径一致（同字 cosine 平均）。训练时当前字体若在 train228 中仍须 leave-one-out，评测字体则只作 query、绝不进库（[`reports/E2_IMPLEMENTATION_READINESS.md:55`](./E2_IMPLEMENTATION_READINESS.md#L55)–[`60`](./E2_IMPLEMENTATION_READINESS.md#L60)）。

### 术语速查

| 术语 | 是什么字体/图 | 进入哪条支路或计算 | 训练采样规则 | 评测规则 |
|---|---|---|---|---|
| Content | Noto Sans CJK Regular 的目标字符 $c$，A 协议 96×96 RGB PNG | $E_c(C)$ → MCA/Identity；**同时是 Δ 减数** | 随 target 字符确定，不从字体池随机 | 同一固定 Content 字体与目标字符集 |
| B₀ | **Noto ContentImage**（合作者 2026-09-04 决策） | Δ/RS-gap/Support 共用的中性结构坐标，与 Content/Identity 输入同一张图 | 固定同字，无随机采样 | 同一 B₀ 与同字规则 |
| Style | 当前目标字体在 `StyleImage/<font>/` 中的汉字图 | $E_s$ 的空间图 `style_emd` 3×3 → 9 token（n 张逐元素平均）；官方还将同图送 $E_c$ 给 RSI，E2 取消后者改用 Δ | E2/E2b：338 池抽 n∈[1,8]；E1c：抽 1 张 | E2/E2b 固定 ref8；E1c 仍 1-shot |
| R / ref8 | 与 style 同一组参考字；评测默认 `永和书风骨韵天地` | α：同字 cosine 平均后 **top-10 softmax**；不作为八份 RSI 结构图 | 训练随机 n；目标字体 leave-one-out | 固定八字；val/test 只作 query |
| Δ 减数 | $E_c(A(Content,c))$，与 Identity 输入共享的同一张 Noto 同字渲染 | 从每层 `Σ_s α̃_s Ec(A(B_s,c))` 中减去，结果 Δ → RSI | 随 $c$ 确定；不随机 | 同一 Content、同一 $c$、同一 Ec/checkpoint 契约 |

### 实施一致性提醒

PI 2026-09-05 已冻结：α **必取 top-10** 再 `softmax(s/0.07)`（`mode=topk, k_top=10`）；`eps_alpha` 只作 `≤1e-6` 数值地板，**不得**把邻域截空。K=3 仅附录消融。相似度聚合为同字 cosine 再平均（[`scripts/hrfont_delta_v2.py`](../scripts/hrfont_delta_v2.py)）。Stage-A 已实现 top-10、9-token 与 cache-only，详细审查见 [`RSI_CORRECTNESS_REVIEW_20260905.md`](./RSI_CORRECTNESS_REVIEW_20260905.md)。
