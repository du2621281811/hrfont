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

Delta-drop 时将该分支置零，模型应退化到“没有额外形变引导”的路径；zero-init 的新 offset 接入使 step 0 也近似该基线。这一零点语义清晰：`Delta=0` 表示不提供相对中性底的变化，而不是提供某个仍带字符身份的绝对结构。

### 反事实：不做减法会怎样？

若直接把 `sum alpha Ec(A(B_s,c))` 送入 RSI，其数值与空间结构会主要表现为 `c` 的内容特征。RSI 便退化成官方 `Ec(S)` 接线的同类问题：用另一份绝对内容骨架驱动 offset，只是把汉字 `S` 换成库中的 `c`；alpha 的风格近邻信息容易被强内容公共分量淹没。此时 drop 分支也不再对应自然的“无变化”原点，RSI 与 MCA 重复，无法清楚归因收益来自相对风格变化。

## 问题二：这样的 Delta 用 RSI 嵌入合适吗？

### 为什么架构上匹配

RSI 的 `OffsetRefStrucInter + DCN` 不是通用风格编码器；它要求结构源能与内容 skip 的空间位置建立有意义的对应，再据此预测 deformable-convolution offset。官方同语系场景用 `Ec(S)`，成立的隐含前提是 S 与 C 是同一个字符。跨语系时 R 是汉字、C 是拉丁字母，`Ec(R)` 的汉字骨架与拉丁 skip 错位；交叉注意力据此产生的 offset 没有正确的位置语义，这是官方接线跨语失败的直接机制。

Delta 则由**同一个目标字符 c** 的近邻字体特征减去中性特征得到。它的多尺度空间网格与 identity skip 指向相同字符部位，RSI 的交叉注意力可以在正确位置上问“这里相对中性应怎样移动”，再由 DCN 形变 skip。因此 Delta 是 RSI 结构源的合适替换；用户参考 R 的真实风格仍独立走 `Es(R)` 风格交叉注意力，不由 Delta 取代。

### 三处必须诚实保留的张力

1. **Delta 并非纯 where。** 它是同字符特征差，offset 的方向和幅度本身已经包含粗粒度的“怎么变”。论文只能表述为“架构主要将 Delta 用作定位/对齐信号”，不能宣称它完全不含 what；应由 E9 接入与破坏实验验证。
2. **尺度分布改变。** Delta 能量通常比绝对 `Ec(S)` 小，且各层分布不同。offset 接入/末端必须 zero-init，并允许经预注册的 scale adapter 校准；Delta-drop 保证无 Delta 时可退化。任何破坏实验都须逐层匹配正确 Delta 的 RMS，避免能量差成为混杂因素。
3. **Ec 的西文字形质量未知。** 官方 Ec 在拉丁字符上的局部对应是否可靠尚未实证。E0/E9 的 wrong-char Delta、wrong-style Delta、spatial shuffle 与 magnitude-only 消融，以及 offset 定位指标，必须承担这项验证，不能仅凭架构直觉下结论。

### “Delta 适合 RSI”的预注册判定条件

只有 E9 同时出现以下模式，才支持较强结论：

- spatial shuffle 在严格 per-layer RMS 对齐后显著伤害主要指标，说明空间布局确实被使用；
- magnitude-only 保留大部分正确 Delta 的收益，说明 RSI 对变化强度/粗粒度方向有效，而不是仅记忆库字符身份；
- offset 对 `|GT-B0|` 轮廓变化区的定位 AUROC 高，并有一致的 AUPRC/IoU/correlation 证据。

若 spatial shuffle 不伤害，Delta 更可能只是全局调制；若 magnitude-only 丢失大部分收益，则需要承认细粒度空间 what 也是关键；若定位 AUROC 不高，则不能把 offset 解释为变化区域对齐。

## 论文可直接采用的表述

“We construct a character-aligned residual condition by subtracting the neutral-font content features from an alpha-weighted mixture of same-character neighbor features. This removes the dominant identity component already supplied by MCA and gives zero a natural no-deformation semantics.”

“We replace the cross-script-misaligned `Ec(R)` structure source of RSI with this same-character residual, while retaining `Es(R)` for style cross-attention. We describe Delta as primarily an alignment cue—not a pure where-only signal—and test that interpretation with RMS-controlled interventions and offset localization.”

## Q&A 增补（2026-09-04）

### Q1. Δ 的机制五连问

#### (a) “找邻居”是字体 kNN，还是 CF-Font 式聚类基底？

默认是**在完整 train228 字体池上按风格相似度找邻居并加权**，属于 kNN/soft-neighborhood 路线，不先做 K-Means，也不把字体压成少数聚类中心；当前实现直接把 query 与 `proto_feats` 做 cosine，并提供 `soft/topk/threshold` 三种稀疏化方式（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`131`](../scripts/hrfont_delta_v2.py#L131)），执行规格则冻结库为 train228、目标字体 leave-one-out（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`25`](../.cursor/rules/hrfont-execution-spec.mdc#L25)）。这更符合本方法的 insight：CF-Font 需要小 spanning basis 来做内容插值，而我们只需从风格相近的库字体取**同一个目标字符**的 Ec 特征并加权；先聚类会把 228 套字体量化到 medoid，可能丢掉恰好最接近 query 的非 medoid，而没有带来本方法必需的表示优势（[`reports/ICLR2027_HRFONT.md:120`](./ICLR2027_HRFONT.md#L120)–[`135`](./ICLR2027_HRFONT.md#L135)）。计算上，228 个已缓存原型只做一次向量 cosine，Ec 只处理筛出的邻居；原型构建本身也是固定 ref8 的离线循环并可落 manifest（[`scripts/hrfont_delta_v2.py:210`](../scripts/hrfont_delta_v2.py#L210)–[`242`](../scripts/hrfont_delta_v2.py#L242)），因此没有必须靠聚类压库的算力瓶颈。CF-Font 式变体仍作为“先 K-Means、每簇取 medoid、再在 medoid 上算 α”的消融保留在 P2-12；旧实验表把它写作 `kNN vs 12 medoid` 的 E7（[`reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:310`](./hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md#L310)–[`317`](./hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md#L317)、[`reports/ICLR2027_HRFONT.md:383`](./ICLR2027_HRFONT.md#L383)）。

#### (b) B₀ 是不是直接使用 Content 图？为什么保留两个中性字体？

不是：`Content` 是 **Noto Sans CJK Regular** 上目标字符 $c$ 的 A 渲染，送入 $E_c(C)$ 作为 MCA/Identity 条件；`B_0` 现已统一为 **Noto ContentImage**（合作者 2026-09-04 决策）：Δ 减数与 Content/Identity 输入是同一张 Noto 同字渲染，并统一 RS-gap 与 Support 的结构坐标（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`23`](../.cursor/rules/hrfont-execution-spec.mdc#L23)–[`26`](../.cursor/rules/hrfont-execution-spec.mdc#L26)）。两者都是“中性同字”，但职责不同：Content 沿 official/FT-v2 的 Noto 数据约定以保持身份输入和基线可比；B₀ 沿已冻结的 FZKTJW 结构域，使 Δ、gap、support 共用一个不随目标字体变化的原点（[`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:26`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L26)–[`30`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L30)）。概念上可以令 `B₀=Content`，减法与零模式仍成立，并不存在理论禁忌；但这会同时更换 Δ 原点、RS-gap/support 坐标和现有 cache/SHA，不能在主实验中无痕替换，否则破坏 matched 比较。若要统一，应作为独立消融重建全链路产物，而不是把两者在论文符号或数据加载中混称为同一张图。

#### (c) 风格邻居究竟按什么找？Es 不准怎么办？

没有额外手工风格特征或另训检索器：α 只用冻结 $E_s$ 的 ref8 编码；具体实现把目标字体八个已归一化向量 `[8,D]` 先均值池化并再归一化，然后与各库字体同样均值池化、归一化的 prototype 做 cosine（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`95`](../scripts/hrfont_delta_v2.py#L95)、[`225`](../scripts/hrfont_delta_v2.py#L225)–[`239`](../scripts/hrfont_delta_v2.py#L239)）。这里的 Es 是 official P1 初始化后、经 E1 在 A/train228 上跨语系联合适配并最终冻结的 style encoder；E1 训练 Ec/Es/UNet，E2 再冻结 Ec/Es（[`.cursor/rules/hrfont-execution-spec.mdc:47`](../.cursor/rules/hrfont-execution-spec.mdc#L47)–[`49`](../.cursor/rules/hrfont-execution-spec.mdc#L49)），所以 α 是方法内部的、与生成模型共享表征的检索机制。用 Es 选 α 不构成评测上的循环论证，因为最终风格结论由隔离于方案编码器的 $φ_{s2}$ 与 T1–T4 门控承担（[`.cursor/rules/hrfont-execution-spec.mdc:56`](../.cursor/rules/hrfont-execution-spec.mdc#L56)–[`60`](../.cursor/rules/hrfont-execution-spec.mdc#L60)）；只有拿 Es 自己给最终方法打分才会循环。Es 是否“够准”由验证电池实证回答：V1 报检索 rank/Recall 与 pairwise AUC，V2 报跨语系同字体对异字体 AUC，V6 报 leave-one-out 最近邻 cosine 与 top-1 α mass（[`scripts/hrfont_validate_e1_encoders.py:109`](../scripts/hrfont_validate_e1_encoders.py#L109)–[`147`](../scripts/hrfont_validate_e1_encoders.py#L147)、[`202`](../scripts/hrfont_validate_e1_encoders.py#L202)–[`214`](../scripts/hrfont_validate_e1_encoders.py#L214)）。因此论文只需主张 Es 提供了**经验证可用的风格相似度量**；向量不具有人可解释轴并不削弱该机制，但 V1/V2/V6 不过门时不能把 α 当作已验证可靠。

#### (d) 到底要不要减 B₀？不减能不能跑？

主方法必须减：第一层是**身份剥离/change-only 语义**，绝对 `Ec(B_s,c)` 的主导公共量仍是字符 $c$，而身份已经由 Content→MCA 提供；减同字 B₀ 才把 RSI 条件改写为“相对中性底应怎样变”，避免与 Identity 支路重复（本文件第 15–21 行；实际 feature-mix 在 [`scripts/hrfont_delta_v2.py:151`](../scripts/hrfont_delta_v2.py#L151)–[`183`](../scripts/hrfont_delta_v2.py#L183)）。第二层是**零模式/CFG 语义**：邻居等于 B₀ 时 Δ 精确为零，Delta-drop 也可把整条分支置零并解释为“无额外形变”；当前纯函数以空权重返回 `None`（[`scripts/hrfont_delta_v2.py:159`](../scripts/hrfont_delta_v2.py#L159)–[`166`](../scripts/hrfont_delta_v2.py#L166)），E2 设计要求独立 `.25` Delta-drop 与 step-0 zero-init（[`reports/E2_IMPLEMENTATION_READINESS.md:68`](./E2_IMPLEMENTATION_READINESS.md#L68)–[`78`](./E2_IMPLEMENTATION_READINESS.md#L78)）。第三层是**数值条件**：固定 B₀ 消掉大的同字 common-mode，使 offset 头围绕共同零点学习较小的相对量，跨字体/字符的尺度更容易校准；这不是“必然更稳定”的定理，因此仍须记录逐层 RMS，并允许 zero-init/scale adapter（本文件第 41–43 行）。不减当然能前向运行，但 RSI 得到的是另一份绝对同字内容特征，既与 Identity 重复，又没有干净的 null 原点；本文件第 27–29 行已经把它定义为反事实，因此它只能作为 `no-neutral-subtraction` 消融，不能与 Δ 主方法同名。

#### (e) 为什么不用 RSI 原注意力“自己学偏移”？

这个前提需要纠正：我们**就是使用 RSI 原来的注意力和偏移机制，块不拆**，只把结构源从 $E_c(S)$ 换成 Δ；官方 model 先以 `self.style_encoder(style_images)` 产生风格条件，又把同一 `style_images` 送入 `self.content_encoder` 产生 RSI 结构源（[`code/official/FontDiffuser/src/model.py:34`](../code/official/FontDiffuser/src/model.py#L34)–[`47`](../code/official/FontDiffuser/src/model.py#L47)）。官方 `OffsetRefStrucInter` 的关键原句是 `hidden_states = self.cross_attention(style_content_hidden_states, context=res_hidden_states)`，随后投影为 offset（[`code/official/FontDiffuser/src/modules/attention.py:288`](../code/official/FontDiffuser/src/modules/attention.py#L288)–[`330`](../code/official/FontDiffuser/src/modules/attention.py#L330)）；`StyleRSIUpBlock2D` 原封不动实例化 `DeformConv2d(...)`（[`code/official/FontDiffuser/src/modules/unet_blocks.py:460`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L460)–[`476`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L476)），并执行 `offset = sc_inter_offset(...)`、`res_hidden_states = dcn_deform(res_hidden_states, offset)`（[`553`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L553)–[`561`](../code/official/FontDiffuser/src/modules/unet_blocks.py#L561)）。不能指望直接喂原始 $E_c(S)$ 后让同一注意力自行补救，因为跨语系 $S$ 是汉字而 skip 是目标西文字 $c$，空间 query/context 没有同字对应；直接喂原始 $E_c(B_s,c)$ 虽同字，却又要求 offset 模块同时从强身份公共量中分离“改什么”并完成“在哪里改”，且失去 Δ=0 的 no-deformation 语义（[`reports/ICLR2027_HRFONT.md:246`](./ICLR2027_HRFONT.md#L246)–[`259`](./ICLR2027_HRFONT.md#L259)）。E9 正是对此的诚实检验：比较 RSI-offset、普通 cross-attn、MCA 与组合接入，并做 RMS 对齐的 wrong-char/wrong-style/spatial 等破坏；另有“关 DCN、只靠交叉注意力”的明确对照（[`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:117`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L117)–[`122`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L122)、[`reports/ICLR2027_HRFONT.md:294`](./ICLR2027_HRFONT.md#L294)）。只有正确 Δ 的空间布局、attention 和 DCN 均带来可重复收益，才能说“变化引导的偏移机制”而不只是多加了一路条件。

### Q2. α 的粒度

#### (a) 相似度用一张还是八张风格参考？

用**八张**固定参考 `永和书风骨韵天地`，不是单张：manifest 明确把它列为 `style_ref8_subset`（[`manifests/charset_cn2west_v2_planned.json:23`](../manifests/charset_cn2west_v2_planned.json#L23)–[`25`](../manifests/charset_cn2west_v2_planned.json#L25)）。准确的当前实现语义是“八个已归一化 Es 向量先 mean-pool、再归一化、再与库 prototype 做一次 cosine”，而不是只取某一字，也不是八个未经说明的图像像素平均（[`scripts/hrfont_delta_v2.py:88`](../scripts/hrfont_delta_v2.py#L88)–[`95`](../scripts/hrfont_delta_v2.py#L95)）。原因是单字会受该字特有的笔画/拓扑影响而有较大估计方差，八字覆盖不同笔画与结构后，字体级 prototype 更稳；这是设计动机，不应冒充已证实结果。稳定性与 α 质量最终由 V6 的 leave-one-out neighbor cosine/top-1 mass，并结合 V1/V2 的检索和跨语系 AUC 验证（[`scripts/hrfont_validate_e1_encoders.py:202`](../scripts/hrfont_validate_e1_encoders.py#L202)–[`214`](../scripts/hrfont_validate_e1_encoders.py#L214)）。

#### (b) 每个库字体都要另外提供八张 ref8 渲染吗？

每个库字体和训练目标字体都必须能提供这八张图，但**不需要额外渲染**：A 数据集的每字体 `StyleImage/<font>/` 已对应完整 style338，而 ref8 是其中的固定子集；数据契约定义 style338 与固定 ref8（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`23`](../.cursor/rules/hrfont-execution-spec.mdc#L23)–[`26`](../.cursor/rules/hrfont-execution-spec.mdc#L26)），构建器直接按字符码从既有 `StyleImage` 查这八个 PNG（[`scripts/hrfont_delta_v2.py:200`](../scripts/hrfont_delta_v2.py#L200)–[`207`](../scripts/hrfont_delta_v2.py#L207)、[`225`](../scripts/hrfont_delta_v2.py#L225)–[`231`](../scripts/hrfont_delta_v2.py#L231)）。必须区分两次采样：生成模型的 style 条件输入从该字体目录用 `random.choice(images_related_style)` 在 338 池随机取一张（[`code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py:59`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L59)–[`69`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L69)、[`90`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L90)–[`92`](../code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py#L92)），而 α 始终读取固定八字并满足 `ref8_feats=[8,D]` 契约（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`90`](../scripts/hrfont_delta_v2.py#L90)）。训练 α 与评测 α 使用同一 ref8 字集，因而 α 的输入定义训评一致；训练时当前字体若在 train228 中仍须 leave-one-out，评测字体则只作 query、绝不进库（[`reports/E2_IMPLEMENTATION_READINESS.md:55`](./E2_IMPLEMENTATION_READINESS.md#L55)–[`60`](./E2_IMPLEMENTATION_READINESS.md#L60)）。

### 术语速查

| 术语 | 是什么字体/图 | 进入哪条支路或计算 | 训练采样规则 | 评测规则 |
|---|---|---|---|---|
| Content | Noto Sans CJK Regular 的目标字符 $c$，A 协议 96×96 RGB PNG | $E_c(C)$ → MCA/Identity；不是 Δ 减数 | 随 target 字符确定，不从字体池随机 | 同一固定 Content 字体与目标字符集 |
| B₀ | FZKTJW/FZFXKTJW 的同一目标字符 $c$，A 协议渲染 | Δ/RS-gap/Support 共用的中性结构坐标；不替代 Content 身份输入 | 固定字体、固定同字，无随机采样 | 同一 B₀ 与同字规则 |
| Style | 当前目标字体在 `StyleImage/<font>/` 中的汉字图 | $E_s(S/R)$ → UNet 风格 cross-attention；official 还将同图送 $E_c$ 给 RSI，我们的方法取消后者并改用 Δ | 从该字体 style338 目录随机取一张；与 α 的 ref8 读取独立 | 使用冻结的参考协议，不参与 Δ 库；ref8 规则见下一行 |
| ref8 | 当前目标字体及每个 train228 库字体的 `永和书风骨韵天地` 八张既有 StyleImage | 冻结 Es → `[8,D]` → 字体 prototype/query → α；不作为八份 RSI 结构图 | 固定八字；目标训练字体作 query 时 leave-one-out | 同一固定八字；val/test 只作 query |
| Δ 减数 | $E_c(A(B_0,c))$，即 B₀ 上与当前目标完全相同的字符 $c$ 的多尺度特征 | 从每层 `Σ_s α̃_s Ec(A(B_s,c))` 中减去，结果 Δ → RSI | 随 $c$ 确定；不随机、不取 Content 图 | 同一 B₀、同一 $c$、同一 Ec/checkpoint 契约 |

### 实施一致性提醒

方法冻结文本目前写的是 top-3 α̃（[`.cursor/rules/hrfont-execution-spec.mdc:49`](../.cursor/rules/hrfont-execution-spec.mdc#L49)），但新工具的无参默认是 `mode="soft", eps_alpha=.01, k_max=10`，`k_top=3` 只在 `mode="topk"` 时生效（[`scripts/hrfont_delta_v2.py:30`](../scripts/hrfont_delta_v2.py#L30)–[`42`](../scripts/hrfont_delta_v2.py#L42)、[`111`](../scripts/hrfont_delta_v2.py#L111)–[`123`](../scripts/hrfont_delta_v2.py#L123)）。正式实验前必须由 resolved config 明确锁定究竟是 top-3 还是稀疏 soft（最多 10），并同步公式、cache manifest 与消融命名；否则“kNN 还是 soft neighborhood”的实现会与论文口径漂移。另一个较小的文档差异是旧 §4.1 写“逐字 cosine 再平均”，而当前代码是“八向量 mean-pool 后 cosine”（[`reports/ICLR2027_HRFONT.md:60`](./ICLR2027_HRFONT.md#L60)–[`67`](./ICLR2027_HRFONT.md#L67) 对比 [`scripts/hrfont_delta_v2.py:88`](../scripts/hrfont_delta_v2.py#L88)–[`95`](../scripts/hrfont_delta_v2.py#L95)）；论文应以最终锁定实现为准。
