# HR-Font ICLR 初稿 Review — 操作指引

- **日期**：2026-09-12
- **来源**：Hermes 主模型人工 review（deai skill + hrfont-project skill + 全文 426 行精读）；基于 commit `70f6f811`（已含 14 处 de-AI 定点修改）
- **目标读者**：写作 agent。本文是操作清单：每条 = 位置 + 问题 + 修改指令 + 优先级。决策权在 PI，本文只给指令与理由。
- **铁律**：只做定点修改；不动 `\cite/\ref/\label` 与数学模式；不碰 TBD 占位符（magenta）；保持匿名（不解除 `\iclrfinalcopy`）；改动后附对照表（原文→改为→理由），单独 commit。

---

## Part A：方法学与结构问题（优先级排序）

### A1（高）D2 vs D1 一处对比混了两个变量
- **位置**：§Experiments 表 `tab:ablation` 行「Remove donor appearance & D2 geometry mean & D1 old feature mean」+ 附录矩阵 D1/D2 定义。
- **问题**：D2−D1 同时改变「载体」（Ec 特征 → SDF 几何）和「归一化」（clip/scale）。审稿人必问增益归属。
- **指令**：二选一——(a) 矩阵补 **D1.5 = 同 clip/scale 归一化的 Ec-feature mean-Delta**，让 D2−D1.5 只差载体；(b) 在表注明确标「D2−D1 = 载体+归一化组合效应」。推荐 (a)。

### A2（高）跨语系竞品文献缺失
- **位置**：§Related Work 与 `references.bib`。
- **问题**：跨语系论文未 cite 任何跨语系对手。必补三篇：**FTransGAN**（cross-lingual few-shot font style transfer, GAN）、**DRA-font**（IEICE TransInf 2026, cross-lingual diffusion, DOI 10.1587/transinf.2026EDP7009）、**CFFont**（cross-lingual full-domain convolutional attention）。核实：CFFont 与 CF-Font（CVPR 2023 内容融合）是两篇，不要混引。
- **指令**：各加一句定位（它们做什么、与我们差异在哪），bib 条目附 arXiv/DOI。

### A3（中）L143-144 自曝放错位置
- **位置**：§Method Set-Delta 段末句「Whether this flexibility helps is an empirical question tested by D2 versus D3」。
- **问题**：Method 段自我削弱。
- **指令**：Method 段删该句（保留操作性区别句）；把「D3 vs D2 是否有效」作为预注册问题写进 §Experiments 的 Round D 说明。

### A4（中）基线适配口径
- **位置**：§Baselines。
- **问题**：FSFont/VQ-Font 原生 CN→CN；只写「documented input protocols」不够。
- **指令**：补两句——跨语系设定下外部基线的输入协议（ref 用中文、target 换拉丁字形；同库同 shot）；bank-enabled matched controls 的口径（与 HR-Font 共享 bank 预算、单独标注）。

### A5（中）novelty 差异句加硬
- **位置**：§Related Work Font-bank content priors 段。
- **问题**：Set-Delta 会被读成「CF-Font 内容融合 + 注意力」。
- **指令**：加一句硬差异：「We aggregate anchor-relative residuals, not fused absolute glyph features」——把 D4（absolute vs residual）的对立前置到 related work。

### A6（低）摘要与图注
- **位置**：abstract L117 相关句（§Method 内 α 句）与 figure captions。
- **问题**：α 近均匀实证下「attention-logit bias」作用微弱，别让摘要暗示 α 是 routing 核心；图注普遍 5 行长（L60 等），ICLR 页面预算紧。
- **指令**：α 句保持事实级（「α ranks the top-K donors」），不要「supplies a bias」类功能承诺；图注各砍到 2-3 行。

---

## Part B：防御性写作清单（全部指出，按严重度）

> 口径：防御性写作 = 预先辩解设计选择、事后解释为什么不做某件事、软化失败表述、用对比句（rather than）给指标降级。与「合法披露」（信息预算/公平口径/matched 契约）区分：合法披露是审稿要求的事实声明，不删。

| ID | 位置 | 原文要点 | 问题 | 操作指令 |
|---|---|---|---|---|
| D1 | §Method Set-Delta 段末 | "Whether this flexibility helps is an empirical question tested by D2 versus D3" | 方法段内自我削弱（=A3） | 删除；挪到实验段当预注册问题 |
| D2 | §Method Set-Delta | "A residual-value injection is isolated as an ablation because it can directly alter ink amount and reintroduce donor appearance" | 「because...」事后辩护消融设计 | 删 because 从句；value injection 只在 D7 消融表出现，不加理由 |
| D3 | §Metrics | "LPIPS, SSIM, and L1 are pixel diagnostics rather than style claims" | 「rather than」防御性降级对比 | 改事实陈述："Pixel metrics (LPIPS, SSIM, L1) serve as diagnostics." |
| D4 | §Metrics 分层句 | "...observed-style transfer, bank-supported completion, and graceful fallback" | 「graceful fallback」为失败层预先软化 | 改事实陈述："...and cases where neither source applies." |
| D5 | §legacy 动机段 | "The strongest current signal sits in Delta, and local appearance has substantial headroom" | 「substantial headroom」=弱点的委婉语 | 直接写观察："Local appearance contributes little in the legacy system." |
| D6 | §Method Set-Delta（anchor 句） | "...donor-identity and style probes in Section~... measure this separation directly" | 预先防御：在方法段预告「我们有探针证明」 | 删第二分句；探针证据留给 §Diagnostics 自己说话 |
| D7 | §Method training objective | "The first local-reference experiment uses no additional local style loss so that R1--R3 isolate the conditioning representation" | 「so that...isolate」设计选择的事后辩护 | 改协议事实："R1--R3 add no local style loss."（理由进实验计划） |
| D8 | §Scope 第一段 | "The current graphics primitives provide an efficient raster interface; vector control points and learned stroke graphs offer promising upgrades..." | 光栅选择的预先防御 + 未来工作找补 | 砍第一分句的「provide an efficient raster interface」；升级句保留但只做 future work 陈述 |

**保留项（合法披露，写作 agent 不要动）**：
- 「The full method operates from the requested reference set and requires no additional target-font glyph collection」——信息预算披露（review 要求）。
- 「with the same coefficients in every matched arm」——matched 契约。
- 「Formal test data are opened once after the configuration and validation thresholds are frozen」——预注册声明。
- 「both values are frozen before formal evaluation」——预注册声明。
- 「HR-Font targets compatible completion rather than reproduction of a single hidden design」——项目口径定位（多兼容解），保留。

**可选微调（低优先）**：全文「directly」出现 4 次（L35/128/140/155），保留 ≤2 次；「Together, ...」出现 2 次（摘要 L37、结论 L328），保留 1 次。

---

## Part C：已完成的 de-AI 修改（commit `70f6f811`，写作 agent 无需重复处理）

14 处已改：双重 that 从句（摘要）、"creates an opportunity"、 "are useful but leave"、"turns into"×4、"transforms into"、"brings X to Y"、"advances this direction"、"These works establish the value of"、"Thus" 过渡、"reveal/expose/motivate" 三连动词、"grows naturally/creating a direct path"、破折号（"principle---use"）。对照表见 commit message 与对话记录；git log 可查。

---

## Part D：写作 agent 操作规则

1. 每条修改 = 一个 commit；commit message 注明 A#/D# 编号与日期。
2. 修改后必须附改动对照表（原文 → 改为 → 理由）追加到本文「Part E：变更记录」或随 commit message 交付。
3. 不新增段落级重写；句子级定点替换优先。
4. 科学内容零改动：不动公式、实验矩阵 ID（D0-D9/R0-R5）、指标定义、TBD 占位符。
5. 匿名性：任何修改不得引入作者/机构标识。
6. 与 PI 确认后才允许结构性改动（如 A1 加 D1.5 行、A2 加文献）。

---

## Part E：变更记录（由写作 agent 追加）

（空）
