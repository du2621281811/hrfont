# E12 训评一体与 Δ 编码空间：方法设计审查

> 日期：2026-09-07；审查基线：`6de1336ce3c73cf8a7ce014cf77ba2c262acf79b`。  
> 范围：仅回答 PI 的两个设计问题；不改变现有冻结方法。文中“推荐”是后续决策建议，不自动覆盖执行规格。  
> 一句话结论：**主线继续用冻结 Es 做 α、用 Ec 构造 Δ、坚持相对 top-10 无硬门控；E12-α 只做 20k matched 消融。评估端复用现有 membership verifier，但不要把多维“合理性”压成一个总分。**

---

## Q1：是否把 E12 同时用于训练期 α 与评估期“合理性”判定

### 1. 当前非循环性由什么保证；提案会破坏哪一条

当前设计刻意分开了三个角色：

1. **方法内 routing/weighting：冻结 Es。** 参考集 `R` 与 train228 库字体的同字 Es 向量逐字算 cosine、再平均，取相对 top-10，并以 `softmax(s/0.07)` 加权。Es 只决定“从库里取谁、各取多少”，库 Es 特征本身不进入生成器；真正进入 RSI 的是被选字体的同目标字符 Ec 特征。[执行规格 §1.6–1.7](../.cursor/rules/hrfont-execution-spec.mdc#L32)；[训练实现](../code/variants/cn2west_f123_rsi/FontDiffuser/train.py#L154)；[bank 审查](./REVIEW_BANK_STYLE_20260906.md#L105)。
2. **方法内 style condition：同一冻结 Es 的空间图。** n 张 `[1024,3,3]` `style_emd` 逐元素均值形成 9-token 条件；这是生成条件而不是最终裁判。[训练实现](../code/variants/cn2west_f123_rsi/FontDiffuser/train.py#L128)；[执行规格 §1.6](../.cursor/rules/hrfont-execution-spec.mdc#L32)。
3. **方法外最终判断：独立 φ_s2 + 人评。** φ_s2 是 ResNet18+GAP+L2、用外部字体和对称 InfoNCE 训练，并须先过 T1–T4；设计师 2AFC/MOS 是感知终审。[EVAL_FRAMEWORK §3–4](./EVAL_FRAMEWORK.md#L75)；[STORY_IDEA §4](./STORY_IDEA_20260907.md#L29)。

因此，“Es 用于检索不循环”的完整理由不是“Es 冻结”，而是：**内部选择器与最终风格裁判在训练数据、参数和表征上隔离，且有人类感知验证。** `EVAL_FRAMEWORK` 的 P2 明文禁止用方案的 α 编码器给最终风格结论打分。[EVAL_FRAMEWORK P2](./EVAL_FRAMEWORK.md#L23)

若把 α 改为 φ_s2/E12，同时又以 φ_s2、或以冻结 φ_s2 后接的 membership head，作为风格/合理性证据，就破坏了上述第 3 项的**表征隔离**：同一个空间既选择 bank 邻居，又判断输出是否接近 reference。没有被破坏的是数据无泄漏和梯度隔离——只要 E12 固定、库限于 train228、test 仅作 query，仍没有 test glyph 泄漏。

### 2. 这是实质循环，还是表面循环

#### 2.1 必须分开的两种“独立”

- **优化独立性成立。** 若 E12 在 F1/F2/F3 前已预训练并冻结，α 从 cache 读取，生成损失不反传到 E12，那么 α 是固定的预训练检索函数。它不会为了让自己的评估分数变好而端到端适配，也不存在 evaluator gradient hacking。
- **测量独立性不成立。** 方法先在 φ_s2 空间最大化 `ref ↔ library` 的近邻性，再用同一 φ_s2 空间测 `ref ↔ output`。这不是“同一个量被恒等复制”：输出还经过 `Ec(library,c) → Δ → RSI/DCN → diffusion`，所以高检索分不数学保证高输出分；身份、质量和生成误差也会使两者分离。它是**软耦合（metric-aligned selection bias）**，不是代数恒等或标签泄漏。

软耦合仍然是真实审稿风险。方法获得了沿 evaluator 认为重要的方向挑选先验的机会，因此 φ_s2 分数的改善可能部分来自“按尺子选材料”，而不是一般人类风格合理性改善。尤其当同一 backbone 后接 membership head 时，head 虽非 α 的同一个 cosine 函数，仍继承同一 embedding 的归纳偏置；这只能减弱“完全相同打分函数”的指控，不能恢复独立性。

#### 2.2 与 CLIP-retrieval + CLIP-score 先例的边界

检索增强图像生成已有相似先例：RDM/KNN-Diffusion 与 Re-Imagen 用预训练语义表征检索外部图像，并同时报告与 CLIP 相关的自动指标。此类设计可以作为“耦合代理仍可报告”的先例，却不是免疫证书。合理防线通常是：

- 明示 retrieval 与 metric 共享/相关表征，CLIP 类分数只作 coupled proxy；
- 主结论同时由不共享表征的质量/身份指标支撑；
- 以盲化人评回答最终语义或偏好问题，而不是把共享 proxy 当唯一主指标。

可核验原论文入口：[Retrieval-Augmented Diffusion Models (RDM/KNN-Diffusion)](https://arxiv.org/abs/2204.11824)；[Re-Imagen](https://arxiv.org/abs/2209.14491)。这里的类比仅支持评审策略，不能证明 HR-Font 的 φ_s2 已有人类效度。

**审稿判断：**如果设计师 2AFC 是预注册的 co-primary endpoint，问题、抽样、盲化、评审数和统计检验均完整；φ_s2 明示为 coupled proxy；同时报告独立 ID/quality 指标，则 **E12-α 可以防守，但明显弱于现有 Es-α 主线**。若 φ_s2/membership 是唯一或主要“合理性”证据，尖锐 reviewer 的 circular metric objection 成立。更重要的是，当前 T2=`0.732±0.057<0.90`，φ_s2 尚不具备主裁判资格，更不应先升级成主线 routing。[E12 证据快照](./IDEA_ICLR_EVAL_20260907.md#L1)；[T2 判定](./IDEA_ICLR_EVAL_20260907.md#L197)

### 3. E12 替换 Es 后还会改变什么

#### 3.1 T1–T4 从评估准入门变成训练准入门

当前 gate 的语义是“尺子不合格则不得进入主结论”；生成训练仍可继续。E12-α 后，尺子同时成为方法部件：

- T1/T2/T4 不过，邻居排序本身就缺乏已验证的跨脚本/字族效度；
- T3 实际验证 ID-CLS，不直接验证 φ_s2 routing，但若论文把整个 E12 套件称为统一“合理性模块”，T3 仍是其输出可用性的准入条件；
- 因而至少 **T1、T2、T4 必须在 cache build 和 F2/F3 开训之前通过**，并锁定 checkpoint/SHA；完整“E12 系统”若作联合声称则 T1–T4 全过。当前 T2 失败，所以现状下 E12-α 是 hard blocker，而不是可边训边补的评估事项。[self-test gates](../scripts/eval_framework/configs/self_tests.yaml#L1)；[E12-v3 审查建议](./E12_SELFTEST_V3_REVIEW_20260906.md#L81)

#### 3.2 外部字体到 train228 的 ranking domain gap 尚未验证

φ_s2 在外部字体上学到“跨脚本同字族靠近”，不自动推出它能在 A-library 的 228 个字体间给出对 Δ 有用的细粒度排序。T2 是 same-vs-different family AUC，既不是 top-K neighbor precision，也不是生成效用。仓库现有证据没有验证 φ_s2 对 train228 的邻居质量；当前 T2 失败还说明基础 family discrimination 未过门。

新增 **V-E12R**（只用 calib16/train228，禁止 val16/test16 调参）：

1. **排序一致性，不当真值：** E12 vs Es 的 top-K Jaccard、top-1 agreement、Kendall/Spearman、turnover、权重 JS divergence/effective-N，按 serif/sans/calligraphic 和字符数 n 分层。低一致性只说明两者选择不同，不能判谁正确。
2. **独立邻居质量：** 先按 D-B3 冻结人工字体类别或第三种独立标签；若暂用 φ_s2 定义“好邻居”，就不能再把该项叫独立验证 E12，只能叫 self-consistency。推荐由设计师对 reference–neighbor 做小规模盲化 2AFC/相关性排序，并报告 precision@K、wrong-style mass、top-1 wrong、tail-5 wrong mass。[D-B3](./REVIEW_BANK_STYLE_20260906.md#L147)
3. **下游 matched probe：** 固定 F0、数据顺序、noise、预算与所有超参，仅替换 α cache/selector，跑 Es-α vs E12-α 20k；以设计师 2AFC 和不共享 φ_s2 的 ID/quality 为主要选择依据，φ_s2 仅作耦合诊断。

#### 3.3 cache/build 契约扩大

E12-α 不是换一行 cosine。E12 对 train228×style338（并覆盖所有训练 query split 的 refs）必须预构建 pooled feature cache；正式训练仍绝对 cache-only。cache manifest 至少锁定：E12 checkpoint SHA、代码/config SHA、外部字体训练数据 manifest/SHA、A-render 协议、split/font/char 顺序、dtype/shape、complete progress 和 payload SHA。启动时 fail-closed；resume 也须核对 selector/cache SHA。现有 `_LibraryEs` 明确假定 `[228,338,D]` Es 表，故需新的显式 cache type/contract，而不能把 E12 数组伪装成 Es cache。[现有 cache contract](../code/variants/cn2west_f123_rsi/FontDiffuser/train.py#L78)；[`_LibraryEs`](../code/variants/cn2west_f123_rsi/FontDiffuser/train.py#L93)

#### 3.4 Es 的角色与归因

替换后 Es 只保留为 9-token style conditioning；E12 承担 retrieval+evaluation。表面上可说“一个 encoder 负责生成风格条件，一个负责检索/评估”，但后半句恰好暴露 coupled selection/evaluation。它也不比当前故事更简单：需要两套预训练域、两套 cache、E12 gates 前移，并使 F2 相对 F0/F1 的变化同时包含“引入 Δ”与“换 retrieval metric”的新归因负担。若只在 matched 20k 消融中替换，则这个差异反而成为清楚的 robustness question。

### 4. 评估端的“合理性 head”应怎样定义

E12 已有 PI 所描述组件：`MembershipVerifier` 冻结 φ_s2，将 reference set 的 embedding 做 mean/max DeepSets-style 聚合，再融合 `q, r, |q-r|, q⊙r` 输出 logit；验证集做 temperature scaling，并报告 AUC、PR-AUC、Brier、ECE。[模型实现](../scripts/eval_framework/models.py#L29)；[训练与校准](../scripts/eval_framework/train_membership.py#L18)

它回答的是狭义问题：**“生成字形在学习到的 family-membership 意义下是否属于参考集？”** 新增另一个 encoder+binary head，若标签定义仍是同/异字体族，只会重复 membership；若直接用方法输出标“合理/不合理”，则必须先回答谁标、按什么 rubric、是否盲化、是否跨字体/字符泛化，否则只是把主观概念装进一个不可解释标量。

“合理性”至少包含三个不能互相替代的构念：

| 轴 | 问题 | 自动/人工输出 |
|---|---|---|
| Legibility / identity | 是否正确、可辨认为目标字符？ | ID-CLS/OCR accuracy、confusion；人工可读性 |
| Family consistency | 是否像参考字体家族？ | φ_s2 SC-R/SC-Gap、calibrated membership probability；设计师 family-match 2AFC |
| Design preference / usability | 是否是设计师愿意采用的干净、协调字形？ | coverage/伪影诊断；设计师 preference 2AFC/MOS |

建议输出**三轴向量 + 置信区间/校准信息**，不要输出单一 `reasonableness score`。若产品层必须给 verdict，可在 calib16 上预注册 conjunction，例如 `legible AND family-consistent AND no severe artifact`，并同时展示各轴及 abstain/uncertain；不得用 φ_s2 训练出来的单阈值替代设计偏好。现有故事也已经把 E12、三轴指标和设计师终审分开，而非声称一个网络测完“合理性”。[STORY_IDEA §4](./STORY_IDEA_20260907.md#L29)

### 5. 推荐阶梯

#### A. 保留 Es-α 主线；增加 20k E12-α matched 消融（**推荐**）

- 不改冻结主方法，不让当前 T2 阻塞 F1/F2/F3；保住 evaluator–method representation separation。
- 待 E12 T1/T2/T4 通过、V-E12R 完成后，从同一 F0 起点做 20k head-to-head；所有差异白名单仅 selector/cache。
- 若两种 α 得出一致结论，可主张 Δ 对 retrieval embedding 具有稳健性；若 E12-α 更好，也先作为消融/后续版本，不在当前证据下追改主线。

#### B. E12-α 作为主线（有条件可防守，不推荐当前采用）

前置条件：T1/T2/T4 全过；V-E12R 证明 train228 排序可用；cache 合同完成；设计师 2AFC 为 co-primary；φ_s2/membership 明示为 coupled proxy；另有独立 ID/quality。代价是失去当前最干净的反循环论证，并改变冻结方法与训练准入顺序。

#### C. 将 E12 backbone 纳入方法并随 F2/F3 fine-tune（**拒绝**）

这不仅取消评测独立性，还允许生成目标改变 selector；原 E12 gates、外部训练语义、cache-only/SHA 合同全部失效，必须重建 evaluator、重训 matched arms，并重新论证稳定性。它是新方法，不是“训评一体”的小修改。

**Q1 决断：选 A。**

---

## Q2：“Es 门控筛选 top-K style，再用 Es 编码同 content 得 Δ”是否合理

### 1. Es 排序与 Ec 编码必须消歧

#### 1.1 当前冻结方法就是“Es 选，Ec 编”

准确流程为：

\[
s_j=\frac1{|R|}\sum_{r\in R}\cos(E_s(r),E_s(B_j,r)),\quad
\tilde\alpha=\operatorname{softmax}(\operatorname{TopK}_{10}(s)/0.07),
\]

\[
\Delta_c^{(\ell)}=\sum_{j\in\operatorname{TopK}_{10}}\tilde\alpha_j E_c^{(\ell)}(B_j,c)-E_c^{(\ell)}(Content,c).
\]

也就是：**Es 只做相对排序/权重；同目标字符由 Ec 编码并在每个被消费尺度混合、减 Ec(Content)。** `_structure_features` 先 `compute_alpha`，再读 neighbor/neutral 的 Ec cache 并 `mix_cached_delta`，实现正是此式。[训练实现](../code/variants/cn2west_f123_rsi/FontDiffuser/train.py#L174)；[Δ rationale](./DELTA_RSI_DESIGN_RATIONALE.md#L1)

#### 1.2 若 PI 的“用 Es 编码 Δ”是字面意思，会变成另一种方法

RSI 的 `OffsetRefStrucInter` 接收与 Ec structure pyramid 对齐的空间 feature；up block 按 `structure_feature_begin * 2 / upblock_index` 固定通道，并从对应尺度取 `structure_features[-upblock_index-2]`，用其作为 query 与 UNet skip cross-attend 后预测 DCN offset。[attention](../code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/attention.py#L266)；[unet block](../code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py#L475)；[forward](../code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py#L549)

当前实际消费尺度是约 `128×24×24` 与 `64×48×48` 的 Ec 空间图；它们保留目标字符局部坐标，适合询问“相对 Content 在哪里、怎样形变”。[Δ rationale](./DELTA_RSI_DESIGN_RATIONALE.md#L22)

把同字 glyph 改由 Es 编码会产生 style-space residual。直接塞进 RSI 至少破坏：

- **shape contract：** Es 的主条件是 `[1024,3,3]` 9-token map，不等于 RSI 期待的两级 Ec feature pyramid；会在 channel/spatial shape 上失败，除非新增 projection/upsampling adapter；
- **semantic contract：** 3×3 style embedding 丢失目标字符的 24²/48²局部坐标，offset head 得到的是“像什么风格”，不是“目标字符何处相对中性改变”；
- **归因 contract：** 新 adapter、接入点和训练分布都改变，不能再称“只替换 RSI structure source”。

合法的 Es-Δ 方案应把 `Σα Es(B_j,c)-Es(Content,c)` 作为**style conditioning residual**，例如与 9-token `Es(R)` 拼接/相加后进入 style cross-attention；这需要 matched 新臂、维度投影与独立 drop。它更接近 CF-Font 式 reference/basis style blending，而不是 character-aligned structural residual，不回答当前 Δ-RSI 的跨脚本结构错位问题。

因此，若 PI 只是把“Es 筛选”和“Ec 编同字”口头合并，答案是：**当前管线已经正是所希望的两阶段，只需统一措辞；不要改代码。**

### 2. “门控过滤”与冻结 top-K 语义冲突

#### 2.1 硬 gate 会重新引入什么

当前冻结 D-P4 是相对 top-K：候选只在 bank 内相对竞争，必取 K=10，无绝对阈值；D-A6 已作废；`ε` 仅数值地板，不能截空；空邻域不是方法语义。[执行规格 §1.5–1.7](../.cursor/rules/hrfont-execution-spec.mdc#L20)

在 top-K 前/后加 `s>θ` 的硬 gate 会重新引入：

1. 空邻域与 `<K` 邻域，以及“回退到 Content/zero/top-1/放宽 θ”的新方法分支；
2. θ 在 calib16 的校准目标、跨 script/font/n 的漂移和 test 冻结问题；
3. matched arms/cache/parity 的新契约与 retraining 要求；post-hoc inference gate 只是 OOD intervention，不能冒充主方法；
4. bank-prior 叙事改变：原设计承认所有输出由相对邻域构成、弱匹配也可能被采用；硬拒识则把“bank 定义 prior support”改成“bank + rejection/fallback 定义 prior”。

还应修正一句过强叙事：soft tail 是确定性凸组合，不是显式随机变量；只能称为“prior mixture/tail mass”，不能把它直接宣称为有益 `prior variance`。[bank 审查](./REVIEW_BANK_STYLE_20260906.md#L114)

#### 2.2 gate 真正能解决和不能解决的问题

硬 gate 的目标是拒绝低绝对相似度的 wrong-style contamination，特别是 reference 在 bank 中没有近邻时。但它只可靠清掉低分尾部；若 Es 把错误风格排成 top-1，或正确/错误字体分数都高，gate 无法修复排序错误。现有审查还表明 τ=.07 未必 one-hot：top-1 质量可能只有约 14%–47%，错误质量既可能在 tail，也可能在 top-1；目前没有真实语义污染界。[BANK_STYLE_NARRATIVE](./BANK_STYLE_NARRATIVE_20260906.md#L39)；[REVIEW_BANK_STYLE](./REVIEW_BANK_STYLE_20260906.md#L112)

#### 2.3 保持冻结语义的软替代

- **先测 D-B3，而非先设阈值。** 用独立人工类别或通过 gate 的 φ_s2 距离冻结 wrong-style 定义，报告 `wrong-style mass / top-1 wrong / tail-5 wrong mass`；再做 top1-only、top1-removed、tail-only 等 RMS-matched 干预。这能判定问题究竟是 tail 还是 rank-1。[D-B3 与 probe](./REVIEW_BANK_STYLE_20260906.md#L118)
- **τ sharpening 是软 gate，但仍须消融/重训。** 在保持必取 top-K、无空邻域的条件下减小 τ，可连续压低尾部；它不修复错误 top-1，也不能在看 test 后调。主线 τ=.07 已冻结，故只可预注册 τ sweep/附录 matched 消融，不能静默改主线。
- **若动机是已观察到的 style preference，应归入 bank prior。** 相对 top-K 的输出本来就由 bank 候选竞争决定；策展 bank 导致偏好是 prior/steering 假设，不是 gate failure。先做固定 checkpoint 的 P1/P2 sensitivity probe，并以 φ_s2（过门后）+ ID/quality + 2AFC 判断方向与代价。[BANK_STYLE_NARRATIVE §2–4](./BANK_STYLE_NARRATIVE_20260906.md#L13)

### 3. Q2 判定

1. **编码：保持 Es-routing + Ec-Δ。** 这是当前设计，架构与叙事一致。所有文稿统一写“Es ranks/weights; Ec encodes the same target character and constructs the residual”。
2. **门控：保持 no-gate relative top-K。** “筛选”只能指 `TopK` 相对选择，不能写成 absolute eligibility gate；不得引入阈值、空邻域或 fallback。
3. **重开 gate 的证据门槛：** 先完成 D-B3；只有当跨 seed/font/char/n 的独立定义显示显著 wrong-style mass 主要稳定集中于低分 tail，且 RMS-matched `tail removed/correct-only` 改善设计师 2AFC/家族一致性并不损害 identity/quality，才值得预注册 threshold-vs-τ matched retrain。若错误主要为 top-1，应该修 selector/ranking，而不是加 gate。

---

## 决策清单（供 PI 拍板）

### D-E1｜E12 是否替换主线 α

- A. 保持 Es-α 主线，增加 20k E12-α matched 消融。**推荐。**
- B. E12-α 主线；须先过 T1/T2/T4、V-E12R、cache contract，并以设计师 2AFC co-primary。
- C. E12 随生成训练 fine-tune。**拒绝。**

### D-E2｜E12-α 消融的准入与判据

- A. `T1/T2/T4 pass → V-E12R → same-F0 20k matched`；主判据为设计师 2AFC + 独立 ID/quality，φ_s2 作 coupled diagnostic。**推荐。**
- B. 以当前 T2=.732 的 E12 直接建 cache/开训。不推荐，检索效度未成立。

### D-E3｜评估端“合理性”输出

- A. 复用 calibrated MembershipVerifier 表示 family consistency；与 ID/legibility、quality/design preference 并列输出三轴，不合成单分。**推荐。**
- B. 新训单一 reasonable/unreasonable head。仅在独立专家标签、rubric、校准及外部效度成立后可研究；当前不推荐。

### D-E4｜Δ 的编码空间

- A. **Es 排序/加权，Ec 编码同目标字符并构造多尺度 Δ。推荐且保持冻结。**
- B. Es 编码 Δ 并注入 RSI。拒绝：shape/semantic mismatch，是另一种方法。
- C. Es-style residual 注入 style stream。只可作未来独立消融，不称 Δ-RSI。

### D-E5｜“门控筛选”措辞与语义

- A. 改称“relative top-K selection/routing”，明确无绝对门、必取 K=10、τ=.07。**推荐。**
- B. 加 absolute threshold/fallback。当前拒绝，覆盖 D-P4/D-A6/P2，需 PI 新决策与 matched retrain。

### D-E6｜何时重议 gate

- A. 先做 D-B3 wrong-style mass + top/tail RMS-matched interventions；仅当错误稳定集中在低分 tail 且移除有下游收益时重议。**推荐。**
- B. 因目测 style preference 直接加 gate。不推荐；该现象应先按 bank-conditioned prior/steering 解释和验证。

---

## 论文措辞建议（英文 claim 句）

### 推荐主线（Es-α + 独立 E12）

> “A frozen method-side style encoder ranks the training-font bank using reference-to-bank similarity, while a separately trained cross-script encoder is used only for evaluation; the latter is never used for retrieval, generation training, or model selection unless explicitly stated.”

> “We select a fixed number of neighbors by relative similarity (K=10) and normalize their scores with a temperature of 0.07. We do not apply an absolute rejection threshold; consequently, every prediction is conditioned on the available bank neighborhood, including potentially weak matches.”

> “The style encoder performs routing and weighting only. The residual supplied to RSI is constructed in the content-encoder feature space from same-target-character glyphs: Δc = Σj αj Ec(Bj,c) − Ec(Content,c).”

> “We report reasonableness as a vector of legibility, reference-family consistency, and design quality/preference, rather than as a single latent score; blinded designer 2AFC serves as a co-primary perceptual endpoint.”

### E12-α 仅作消融

> “As a robustness ablation, we replace the method-side retrieval embedding with the independently pretrained evaluation embedding while keeping the generator, bank, data order, and training budget fixed. Because retrieval and SC-R then share an embedding space, SC-R is reported as a coupled diagnostic rather than independent confirmatory evidence.”

### 若未来选择 E12-α 主线，必须披露

> “Retrieval and automated family-consistency evaluation share a frozen pretrained embedding space. This creates metric-aligned selection but no gradient path from the generator to the evaluator; we therefore treat the automated score as a coupled proxy and base confirmatory conclusions jointly on blinded designer judgments and representation-independent identity and quality measures.”

### 不应使用的强 claim

- 不写 “E12 makes training and evaluation objectively unified.”
- 不写 “The reasonableness head directly measures whether a glyph is reasonable.”
- 不写 “The gate removes incorrect styles.”（主线无 gate，且错误者可为 top-1。）
- 不写 “Δ is encoded by Es.” 或 “Δ is a pure style vector.”
- 不写 “The soft tail represents beneficial prior variance.”（当前只是确定性 mixture，利弊待测。）
