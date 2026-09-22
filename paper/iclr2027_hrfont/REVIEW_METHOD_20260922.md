# Method 章节复查：内容/逻辑 + 防御性写作 + AI 痕迹 — 2026-09-22

- 日期：2026-09-22
- 对象：`paper/iclr2027_hrfont/main.tex` §Method（116–247 行）；`\title` 与摘要见另案
- 使用的 skill：`deai`（detect 模式：禁用词 / 禁用句式 / LaTeX 规则）、`hrfont-project`
- 防御性写作判据：仓库 `REVIEW_OPERATIONAL_20260912.md` Part B（D1–D8 口径 + 合法披露保留项）
- 依据（实现侧）：K6-B（`hrfont_k6_20260920_r3`，run `K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2`）、`scripts/hrfont_i.py`、`scripts/k_components.py`
- 计数：内容/逻辑 9 项（高 2 / 中 6 / 低 1）、防御性 4 项（中 3 / 低 1）、AI 痕迹 9 处（P0 0 / P1 5 / P2 4）

---

## Part A：内容与逻辑

### A1（高）符号在三处撞车

| 位置 | 冲突 | 说明 |
|---|---|---|
| 式 (2) 135 行 `s_j(\Rset_f)`；146 行 "For each content scale $s$" | $s$ = 相似度 vs 尺度 | 同一节内同一字母两种含义；式 (3) 的 $E_c^s$ 紧跟 $s_j$ 出现，读者无法区分 |
| 121／178 行 `$E_\ell$`（局部编码器）；式 (4) 157–161 行 `$q_{\ell,t,p}$`、$\gamma_\ell$、$K_\ell$ | $\ell$ = 编码器名 vs 模块层索引 | 式 (4) 里 $\ell$ 是层，第 178 行的 $E_\ell$ 是网络 |
| 143 行 top-$K$（检索宽度）；式 (4) 160 行 $K_\ell(\cdot)$（key 投影） | $K$ 两种含义 | 式 (4) 与式 (2) 同段，$K{=}10$ 与 $K_\ell$ 相邻 |

**指令**：尺度改用 $\rho$（或 $r$）；局部编码器改名 $E_{\mathrm{loc}}$；key 投影改 $\Pi_\ell$ 或 $\kappa_\ell$。三处一起改，别只改一处。

### A2（高）第二支路、门控、适配器在方法里不存在

- **位置**：120 行（$E_s$ 只列"reference tokens and bank-retrieval descriptors"）、122 行、式 (1) 124–128 行、222 行、226 行、231 行。
- **事实**：K6-B 的结构条件是 `o = o^geo + tanh(g_ℓ)·o^sty`，style 支路读 $E_s$ 前两个 block 组的 48×48 / 24×24 中间图，经 identity 初始化的 1×1 适配器与独立 router 产生 offset（`k5_runtime.py:12-39, 64-68, 113-122`）。
- **后果**：方法章描述的不是论文的主方法；式 (1) 缺一个条件项，损失与"训练内容"清单也少一路可训练参数。
- **指令**：按 `METHOD_DRAFT_K6B_20260922.md` Part 2 的 A1/A2 段落替换。

### A3（中）$U_c$ 的来源没有定义

- **位置**：186 行 "The neutral content grid $U_c\in\mathbb R^{144\times256}$ defines one query per target location."
- **问题**：$U_c$ 与式 (1) 的 $E_c(x_c^0)$ 关系未写；144 与 12×12 的对应在 129 行只对 $A$ 说过。读者无法判断 $U_c$ 是编码器哪一层/哪个尺度。
- **指令**：补一句：$U_c$ 取自 $E_c$ 的 $12\times12$ 尺度特征（与式 (1) 的 content 条件同源）。

### A4（中）router 的 content 项尺度未限定

- **位置**：157–158 行 $C_\ell(E_c^s(x_c^0))_p$。
- **问题**：$s$ 未限定，而每个结构模块只在自己那个尺度上取特征（`hrfont_i.py:64-71` 断言 `hidden.shape[-2:] == (h,w)`）。
- **指令**：写成"该模块对应尺度"（如 $C_\ell(E_c^{s(\ell)}(x_c^0))_p$），并在正文说明。

### A5（中）目标函数那段有语法错误

- **位置**：229–230 行。
- **原文**：`$\mathcal L_{\mathrm{diff}}$ is the epsilon-prediction mean squared error; The perceptual term averages three VGG feature losses; $\mathcal L_{\mathrm{off}}$ is the raw offset regularizer returned by the backbone.`
- **问题**：分号后接大写 `The`（分号不能连接独立句），三个定义挤在一句。
- **指令**：拆成三句；同时按决策 3 删掉 `$\mathcal L_{\mathrm{off}}$` 那一句与式 (7) 的 `+0.25\mathcal L_{\mathrm{off}}`。

### A6（中）结构条件的 dropout 位置没有声明

- **位置**：236–237 行。
- **事实**：实现里 dropout 作用在**路由之后**（`offset = original(hidden, mixed) * active`，`hrfont_i.py:78-80`），注释写明目的是避免被丢弃的条件仍留下 affine offset。
- **问题**：现文 "removes ... structural conditions from the denoiser" 可被读成"结构 offset 置零"，与实现不等价。
- **指令**：补半句"条件的置零发生在权重归一化之后"。

### A7（中）donor 排除规则不完整

- **位置**：143 行。
- **原文**：`$\mathcal J_c$ contains the top-$K$ eligible training fonts after removing the episode font and script-invalid donors.`
- **事实**：还排除该字体的**显式 weight 变体**（run config `donor_policy = exclude_self_and_explicit_weight_variants_before_alpha`），且排除发生在先验归一化之前。
- **指令**：补"及其显式字重变体；排除在 $\alpha$ 归一化之前"。

### A8（低）实现数值宜移附录

- **位置**：144 行 "The implementation uses $K=10$ and $\tau=0.07$."
- **指令**：移至附录表（页面预算）；正文只需说明 $K$/$\tau$ 是超参数。

### A9（中）结构条件的尺度数与实现不符

- **位置**：146 行 "For each content scale $s$, we retain a set of anchor-relative candidates:"
- **事实**：结构条件只挂在**两个**尺度上（48×48 与 24×24）；其余尺度取零（`k5_runtime.py:113-121` 的 `if s not in (1,2)`），逐层 offset 头也只在对应尺度上工作（`hrfont_i.py:67` 的 shape 断言）。
- **问题**：现文读起来像所有内容尺度都带结构候选；式 (4) 又以模块层 $\ell$ 索引，读者会以为每一层都有。
- **指令**：写明"在两个结构尺度（$48\times48$、$24\times24$）上"。

---

## Part B：防御性写作（4 项）

| ID | 位置 | 原文 | 问题（按 D1–D8 口径） | 指令 |
|---|---|---|---|---|
| DM1 | 219 行 | "...share $M_{f,c}$, **so both objectives shape** the online reference encoder and its aggregation." | 事后辩护设计共享（D7 型：`so that…` 类） | 改事实句："The readout and the synthesis tokens are both produced from $M_{f,c}$." |
| DM2 | 217 行 | "A learned gain $\eta$, initialized at $0.01$, and a 1,000-update ramp $r(u)$ **introduce the new appearance path gradually**." | 目的状语合理化 ramp；"new" 是相对旧实现的措辞 | 改纯事实："The gain is initialized at $0.01$ and ramped over 1,000 updates." |
| DM3 | 202 行 | "Retaining the spatial grid supervises where target-specific appearance **should be** expressed." | 软化/目的化措辞（`should`） | 改事实："$F^*_{f,c}$ keeps the $12\times12$ grid." |
| DM4 | 197 行 | "The target queries **can select** different local evidence even when only one reference glyph is available." | 功能承诺（与 9-12 review A6 对 $\alpha$ 的批评同类） | 删除该句（构造性事实已在 186–187 行给出） |

---

## Part C：AI 写作痕迹（9 处）

### P1（明显，5 处）

| 位置 | 原文 | 问题 | 改法 |
|---|---|---|---|
| 123 行 | "The denoiser **thus** receives" | deai 禁用过渡词 | "The denoiser receives" |
| 183 行 | "**Thus,** observations remain available individually until the target-conditioned readout." | 过渡词 + 解释句 | "Observations stay separate until the target-conditioned readout." |
| 229–230 行 | 分号连接三个独立句、分号后大写 | 句式（同 A5） | 拆三句 |
| 165 行 | "**The resulting** residual drives deformable structural interaction…" | 分词型引导（AI 高频） | "The combined residual drives deformable structural interaction…" |
| 170 行 | "…, **providing a direct comparison for learned combination**." | 分词尾 + 承诺（预告消融） | 删该分词短语，保留 "Mean-Delta is recovered by fixing $w_j=\alpha_j$." |

### P2（风格，4 处）

| 位置 | 原文 | 问题 | 改法 |
|---|---|---|---|
| 175 行 | "It **first** extracts evidence at multiple scales, **then** arranges…" | 机械 `first…then` | 拆两句：一句讲多尺度证据，一句讲目标条件记忆 |
| 187 行 | "Learned positional slots $P$ **complement this content signal**:" | "complement" 空泛 | "Learned positional slots $P$ accompany the content queries:" |
| 218 行 | "The global tokens summarize **family context**, while…" | "family context" 模糊 | "$G$ carries the family-level summary, while the spatial memory supplies target-local evidence." |
| 144 行 | "The implementation uses $K=10$ and $\tau=0.07$." | 方法段塞实现细节（同 A8） | 移附录 |

**deai 检查通过项**：无 Tier-1 禁用词（delve / leverage / robust / comprehensive / seamless / showcase / underscore / harness 等均未出现）；无 "It is worth noting"、"Moreover / Furthermore / In conclusion"；无三连形容词堆叠；破折号 0 处（238 行的 `content--reference` 是连接号，不计）；未触碰 `\cite/\ref/\label` 与数学模式。

---

## Part D：保留项（合法披露，不要删）

1. 143 行的 donor 资格与排除规则（信息预算披露）——按 A7 补全，不要删。
2. 237 行 "The completion loss retains its observed inputs; an additional source-drop event removes structural offsets."（dropout 语义的可复现披露）。
3. 232 行 "one to eight Chinese references and a valid target glyph from the same font"（参考预算）。
4. 169 行 "This separates neighborhood retrieval from cross-script realization."（设计分工陈述；需与实验节的支路/router 对照保持一致）。
5. 236 行的联合条件 dropout 列表（CFG 契约）。

---

## Part E：优先级

1. A1 符号撞车（改名字级操作，一次改净）。
2. A2 第二支路缺失（方法章主体，按 `METHOD_DRAFT_K6B_20260922.md` 替换）。
3. A5 + 决策 3（删 offset 项与其句子）。
4. DM1–DM4（防御性）。
5. P1 五处（AI 痕迹）。
6. A3/A4/A6/A7/A9 补定义与规则；A8、P2 四处。
7. 辅助监督整节删除（决策 5）：`Within-script auxiliary supervision`（240–246 行）及其在 267、274、315、334、416、435–439 行的落点。

---

## Part F：与实现对齐的缺口（另案）

方法章的**内容缺口**（K6-B 双支路、detail 项、采样配额、donor 策略、预算数字）在 `METHOD_REVISION_PLAN_20260922.md` Part 1 的对照表里逐条列出，本条不重复计数；本文件只负责"现有文字本身对不对、有没有防御性、有没有 AI 痕迹"。

---

## Part G：未核实

- 式 (1) 的 $G$ 是否为 9 个 token：与 `hrfont_i.py:113, 122`（`style` 经 `flatten(2).transpose(1,2)` 后按 token 维做 dropout）一致，判为 9 ✓；如需我在服务器上打一次 shape 打印可再确认。
- 143 行的 "script-invalid donors" 是否涵盖 weight variants 之外的全部排除项（如 family guard 的分组规则），需对照 `k4_runtime` 的 family policy 代码确认。
- `V_f\in\mathbb R^{720n\times256}$` 的 $n$ 是否在训练时随 episode 变化（实现为变长 + mask），论文正文未写；属内容补齐项（低）。
