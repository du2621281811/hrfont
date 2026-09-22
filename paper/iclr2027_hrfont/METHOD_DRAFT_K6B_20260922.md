# Method 章节定稿草案（K6-B 主方法）— 2026-09-22

- 日期：2026-09-22
- 依据：`METHOD_REVISION_PLAN_20260922.md`（对照与证据）；实现 `hrfont_k6_20260920_r3`，run `K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2`
- 使用的 skill：`deai`（去 AI 味 + 禁令清单）、`tech-paper-template`（骨架与四项一致性检查）；防御性写作清单取自仓库 `REVIEW_OPERATIONAL_20260912.md` Part B（D1–D8）
- PI 决策（2026-09-22，本文件据此撰写）：
  1. **K6-B 为主方法**（单臂叙述，正文不出现"我们试验/我们比较"）
  2. **双支路是 K6-B 的组成部分**，写进正文
  3. **offset 惩罚不进 loss**（不写"去掉"，也不解释）
  4. **detail 监督一笔带过**（实现细节，附录给分项定义）
  5. **不写任何辅助监督**（辅助实验不存在）
  6. 双支路口径由本文件给出，等 PI review
- 目标读者：PI（口径审核）、写作 agent（执行）

---

## Part 1：待审核的口径（T1–T8）

| ID | 选择 | 采用的写法 | 备选 | 影响面 |
|---|---|---|---|---|
| T1 | 支路命名 | **content-coordinate branch / style-coordinate branch**（只声明差异来自哪个编码器的坐标，不声明它编码了"几何"或"风格"） | (a) geometric / style branch（K5 内部命名）；(b) 第一/第二支路 | 摘要中"two complementary conditions"叙事不变：双支路在 Dynamic Delta 内部，paper 级仍是"结构条件 + 外观条件"两件 |
| T2 | 符号 | 上标 `\mathrm{geo}` / `\mathrm{sty}`（$o^{\mathrm{geo}}$、$\gamma^{\mathrm{sty}}_\ell$），尺度仍用 $s$，模块层用 $\ell$ | 单字母 $b\in\{\mathrm{c},\mathrm{s}\}$（会与字符 $c$、尺度 $s$ 冲突） | 新符号需在首次出现处定义 |
| T3 | 门控句 | 只陈述事实：per-layer scalar，initialized at zero | 加一句"使其逐步进入"（解释性，删） | 公式 + 半句 |
| T4 | 第二支路适配器 | 一句话：bias-free $1\times1$ adapter，identity initialization | 省略（但读者会问为什么残差尺度不变） | 一个从句 |
| T5 | 第二支路存在的理由 | **不解释**（正文只给定义与接口） | 加一句设计意图（如"两套坐标各自路由"） | 若审稿人问，靠 §Contribution studies 的支路对照回答 |
| T6 | 与 TC 的边界 | 明确 TC 读的是**参考像素**（$E_\ell$），两支路读的是**库内 donor 渲染** | — | 避免"风格从结构通路进入"的误读 |
| T7 | loss 里的 detail | 一项 + 一句 gloss + 附录指向（weight 0.05） | 完全不进公式（隐藏度更高，但训练目标不完整） | 公式多一项 |
| T8 | 预算数字 | 正文写选定 checkpoint 的实际预算（K6-B = 20,000 updates），其他臂的 10k 里程碑在实验节说明 | 统一写 10k（与实现不符） | §Training settings + 附录 |

---

## Part 2：Method 章节英文草案（替换 `main.tex` 116–247 行）

标注：`[改]` 本次改动；`[留]` 原文保留；`[删]` 删除。

### 2.1 Overview（118–129 行）`[改]`

```latex
\subsection{Overview}
We build on a conditional diffusion generator with a frozen content encoder $E_c$ and a frozen style encoder $E_s$.
The two encoders serve separate roles.
$E_c$ supplies multi-scale features of the neutral target.
$E_s$ supplies the nine global reference tokens $G$, the pooled descriptors that drive donor retrieval, and the two intermediate maps read by the second structural branch.
A trainable local encoder $E_\ell$, initialized from the first three blocks of $E_s$, reads the reference pixels online.
Dynamic Delta conditions structural interaction from two donor-side residuals, and TC turns local evidence into target-aligned appearance tokens.
The denoiser receives
\begin{equation}
 \hat\epsilon=\epsilon_\theta\bigl(x_t,t,E_c(x_c^0),\Delta^{\mathrm{geo}}_{c,t}(\Rset_f),\Delta^{\mathrm{sty}}_{c,t}(\Rset_f),
                       G(\Rset_f),A(\Rset_f,c)\bigr),
 \label{eq:denoiser}
\end{equation}
where $x_t$ is the noisy target image, $G$ contains nine global reference tokens, and $A$ contains 144 appearance tokens arranged around the target's $12\times12$ content grid.
```

改动点：`frozen global style encoder` → `frozen style encoder`；补 $E_s$ 的第三个用途；`Δ` 拆为两支；`thus` 删除（deai：过渡词）。

### 2.2 Dynamic Delta（131–170 行）

保留：bank 叙述、检索公式（eq. 2）、几何候选公式（eq. 3）、末段"权重非负和为 1 / Mean-Delta 由 $w_j=\alpha_j$ 恢复"。

```latex
\subsection{Dynamic Delta: a structural variation space}
The bank supplies examples of how the requested character is realized in different fonts.
To prioritize a relevant neighborhood, we compare the target references with bank renderings of the same reference characters:
[式 2，原文保留]
$\mathcal J_c$ contains the top-$K$ eligible training fonts after removing the episode font and its explicit weight variants.
The implementation uses $K=10$ and $\tau=0.07$.

For each content scale $s$, we retain a set of anchor-relative candidates:
[式 3，原文保留，正文首次出现处称其为 content-coordinate candidates]
Every donor renders the same requested character.
Subtracting a common anchor expresses the bank evidence in coordinates relative to the content already available to the generator.
These candidates define a bank-supported space of possible changes to the neutral structure.
```

`[改]` 句末补 donor 排除（weight variants）。`[删]` 原句 "Retrieval supplies an initial distribution $\alpha$, and a learned router adapts that distribution to the current synthesis state." 被下面的双支路段落取代。

**新增段落（插在式 3 之后）**：

```latex
\paragraph{A second coordinate for the same donors.}
The retrieved donors are also read in the coordinates of the style encoder.
For donor $f_j$ and requested character $c$ we take the first two intermediate maps of the frozen style encoder, at $48\times48$ and $24\times24$, and form
\begin{equation}
 e_{j,l}=E_s^{l}(x_{f_j,c})-E_s^{l}(x_c^0),\qquad l\in\{1,2\},
 \label{eq:style_delta}
\end{equation}
where $x_{f_j,c}$ is the donor's rendering of the requested character and $x_c^0$ is the neutral rendering.
A bias-free $1\times1$ adapter maps $e_{j,l}$ to the branch width and is initialized as the identity, so the branch acts on the residual itself.
Both branches read the same neighborhood $\mathcal J_c$ and the same prior $\alpha$.
```

**router 与合成（替换原式 4，line 155–165）**：

```latex
Each branch routes and predicts its own offsets:
\begin{align}
 q^{b}_{\ell,t,p} &= Q^{b}_\ell(h_{\ell,t})_p+C^{b}_\ell(E_c^s(x_c^0))_p
                 +R^{b}_\ell(\bar G)+T^{b}_\ell(t),\\
 w^{b}_{j,\ell,t,p} &= \operatorname{softmax}_{j}\!\left[
       \log\alpha_j+\gamma^{b}_\ell\cos\left(q^{b}_{\ell,t,p},K^{b}_\ell(d^{b}_{j,c,p})\right)\right],\\
 o^{b}_{\ell,t,p} &= \operatorname{Offset}^{b}_\ell\Bigl(h_{\ell,t},\;
       \textstyle\sum_{j\in\mathcal J_c}w^{b}_{j,\ell,t,p}\,d^{b}_{j,c,p}\Bigr),
       \qquad b\in\{\mathrm{geo},\mathrm{sty}\},
 \label{eq:dynamic_delta}
\end{align}
$d^{\mathrm{geo}}_{j,c}$ is the content-encoder difference of \eqref{eq:delta} and $d^{\mathrm{sty}}_{j,c}$ the style-encoder difference of \eqref{eq:style_delta}.
The two offsets are combined through a learned per-layer scalar,
\begin{equation}
 o_{\ell,t,p}=o^{\mathrm{geo}}_{\ell,t,p}+\tanh(g_\ell)\,o^{\mathrm{sty}}_{\ell,t,p},
 \label{eq:gate}
\end{equation}
with $g_\ell$ initialized at zero.
The combined offset drives deformable structural interaction with the neutral content, and condition dropout is applied after routing.
```

`[留]` 末段原文：weights nonnegative and sum to one / adaptation selects within the available residual variations at each location / Mean-Delta 由 $w_j=\alpha_j$ 恢复。

**删掉的原文句**（属防御性/预告类）：无新增；原 "This separates neighborhood retrieval from cross-script realization." 保留（事实性区分）。FP32 合成、门控梯度细节移附录。

### 2.3 Target-character appearance completion（172–219 行）`[留]`

与实现逐行一致（`hrfont_i.py:10-48, 88, 96, 119-121`）。仅两处措辞：

- `initialized from the early blocks` → `initialized from the first three blocks`；
- 保留 `Z=\operatorname{CA}(h,G)+\eta\,r(u)\operatorname{CA}(h,A_{f,c})`（$\eta$ 初值 0.01，$r(u)$ 1,000 步 ramp），与实现一致。

### 2.4 Training and generation（221–238 行）`[改]`

```latex
\subsection{Training and generation}
We jointly optimize the local reader, both Delta branches, and the denoiser from a pretrained generation checkpoint:
\begin{equation}
 \mathcal L=\mathcal L_{\mathrm{diff}}
      +0.01\mathcal L_{\mathrm{perc}}
      +r(u)\bigl(0.01\mathcal L_{\mathrm{comp}}+0.05\mathcal L_{\mathrm{detail}}\bigr).
 \label{eq:objective}
\end{equation}
$\mathcal L_{\mathrm{diff}}$ is the epsilon-prediction mean squared error.
The perceptual term averages three VGG feature losses.
$\mathcal L_{\mathrm{comp}}$ supervises the appearance readout of \eqref{eq:completion}, and $\mathcal L_{\mathrm{detail}}$ is a region-weighted endpoint term over the ink added and removed by the design, with the mask definitions and the $\bar\alpha_t$ weighting listed in Appendix~\ref{app:repro}.
The content, style, and VGG encoders stay frozen; the local reference encoder is trainable.
Each episode uses one to eight Chinese references and a valid target glyph from the same font.
An update draws 64 targets under fixed script quotas with an oversampling of characters whose family design departs from the neutral rendering; donor retrieval excludes the episode font and its explicit weight variants.
$r(u)$ ramps over the first 1,000 updates.
```

`[留]` 其后原文保留（TC teacher 只用于监督损失；denoiser 始终接收预测的外观记忆；joint condition dropout；completion loss 保留观测输入；source-drop 移除结构 offset；推理时 TC 每对算一次、Delta 权重逐模块逐步更新）。

`[删]` 原句 "$0.25\mathcal L_{\mathrm{off}}$"、"$\mathcal L_{\mathrm{off}}$ is the raw offset regularizer returned by the backbone."（决策 3：不写、不解释）。

### 2.5 删除整节 `[删]`

`\paragraph{Within-script auxiliary supervision.}`（240–246 行）整段删除（决策 5）。

---

## Part 3：连带改动（实验节与附录）

| 位置 | 现状 | 改为 |
|---|---|---|
| 262–269 行 Training settings | "The auxiliary experiment adds 32 Chinese examples per update…"；lr 只分两档 | 删辅助句；lr 改为 "the local reader, the routing modules, and the second branch use $10^{-4}$"；预算按 T8 写实际更新数（K6-B = 20,000） |
| 274 行 | "…and auxiliary supervision using shared initial weights…" | 删 "and auxiliary supervision" |
| 302–317 行 Contribution studies | "The Chinese auxiliary study then measures…" | 换成第二坐标支路的对照句（content-coordinate only / style-coordinate only / both） |
| 334 行 ablation 行 | "Auxiliary supervision / Full model with versus without Chinese completion" | 换成 "Second coordinate / Content-coordinate only versus content plus style-coordinate branch" |
| 415 行 | "perceptual 0.01; raw offset 0.25; completion 0.01" | "perceptual 0.01; completion 0.01; detail 0.05" |
| 416 行 | "Auxiliary Chinese study batch 32; weight 0.5" | 删除，替换为 "Second branch / identity-initialized $1\times1$ adapter; per-layer gate initialized at 0; FP32 branch sum" |
| 417 行 | "Reference / source dropout 0.10 / 0.25" | "0.02 / 0.05" |
| 新增行 | — | "Detail masks / ink, near-ink, background; add, remove, high-frequency"；"Batch composition / 32, 24, 8 script quotas; 19, 14, 5 confirmed-detail"；"Added parameters / <服务器计数：`sum(p.numel() for n,p in model.named_parameters() if 'es_branch' in n)`>" |
| 435–439 行 Run correspondence | I0 / I1 / I2 | K 线：K1（content-coordinate only）→ K5-A（style-coordinate only）→ K5-B（both）→ K6-B（both, offset penalty removed, 20k） |
| 251–254 行 数据 | 260 / 228-16-16 / 56,429 pairs | 与 V2（`manifests/v0917/*`）核对后定稿 |

---

## Part 4：去防御性写作自查

对照仓库 D1–D8 清单，本草案的处理：

| 模式 | 清单条目 | 本草案 |
|---|---|---|
| 方法段自我削弱 | D1 | 不出现"是否为经验问题"类句子 |
| `because` 事后辩护 | D2 | 门控、适配器、第二支路均只写事实，不写 because |
| `rather than` 降级对比 | D3 | loss 段落无 rather than 结构 |
| 失败预软化 | D4 | 无 graceful fallback / headroom 类措辞 |
| 预告有探针证明 | D6 | 无"Section X measures this"类预告 |
| `so that … isolate` | D7 | 删除式协议句，不写设计理由 |
| 删掉的原文句子 | — | "$0.25\mathcal L_{\mathrm{off}}$" 与其解释句（不写"去掉"，避免事后解释）；辅助监督整节 |
| 保留（合法披露） | 清单保留项 | 参考预算、bank 排除规则、matched 契约、预注册声明一律保留 |

deai 禁令核查：无 Tier-1 禁用词；无 "It is worth noting"、"Moreover/Furthermore"、破折号（每千字 ≤1）；未出现三连形容词堆叠；未改 `\cite/\ref/\label` 与数学模式。

---

## Part 5：待 PI 确认与未补

1. T1 支路命名（content-coordinate / style-coordinate）是否采用；若不采用，需要 PI 给定命名。
2. T8 预算：正文写 20,000 updates 还是保留 10,000（影响 K5-A/K5-B 的 matched 说明）。
3. 第二支路的参数量与显存开销数字（服务器一条命令可出，附录表要用）。
4. 数据数字（V2 的字体数与 pair 数）核对后填 251–254 行。
5. 摘要 / 标题仍是旧版（`Completing Font Families across Scripts`、`cross-script completion benchmark`），与已定口径冲突，等 PI 一次性写入。
