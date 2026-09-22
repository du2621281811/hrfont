# Method 章节「像不像论文」改写方案 — 2026-09-22

- 日期：2026-09-22
- 对象：`paper/iclr2027_hrfont/main.tex` §Method（116–264 行，K6-B 版，commit `52bfb6f94`）
- 依据的 skill：`tech-paper-template`（模块↔挑战↔贡献映射、四查）、`deai`（Tier-1 禁用词/句式/LaTeX 规则）、`hrfont-project` references/paper-method-audit.md §6（ICLR 9 页；实现细节移附录、正文只留机制与语义）
- 结论：**内容已经对了，register 错了。** 现在这一节是「实现说明书 + 公式清单」，不是论文的方法章。
- 本文取代 `METHOD_DRAFT_K6B_20260922.md` 的 Part 2 草案（那稿即本问题的来源）。

---

## Part 0：诊断（为什么读起来不像论文）

| # | 症状 | 证据（当前正文的句子） | 修法 |
|---|---|---|---|
| 1 | **开篇是清单，不是问题** | "We use a conditional diffusion generator with a frozen content encoder $E_c$ and a frozen style encoder $E_s$." 后面三句全在列编码器各提供什么 | 首段先给任务缺什么（目标字符在参考 script 里没有对应字形），再给设计原则（把结构变化当"可检索的空间"、把家族外观当"从参考推出来的条件"），编码器清单压成一句 |
| 2 | **公式代替论证** | 连续四段都是 "Given … we compute" + 公式，段落没有主张句 | 每个公式前一句说它回答什么问题，后一句说它买到什么（选择范围受限/权重和为 1/静态基线可恢复） |
| 3 | **实现细节进正文** | "initialized from the first three blocks"、"bias-free $1\times1$ adapter … initialized as the identity"、"with $g_\ell$ initialized at zero"、"$24\times24\times128$ … $12\times12\times256$"、"720 tokens"、"four heads … FP32 softmax"、"gain $\eta$ … 0.01"、"64 targets … script quotas" | 全部移附录表（audit §6）；正文只留「哪些量进入条件、作用在哪、语义是什么」 |
| 4 | **第二条支路没有动机** | 式 (5) 前只有一句 "The same donors are also read in the coordinates of the style encoder." | 先说明九 token 的 3×3 汇总装不下笔画级外观（可核验：3×3 vs 48×48），再引出第二坐标 |
| 5 | **贡献边界不清** | 全文没有一句说哪些是继承骨干（RSI/DCN/offset head、感知项），哪些是本方法 | 概览末句写明继承部分；两支路/TC 才是贡献面 |
| 6 | **术语三套并行** | "structure condition" / "structural offsets" / "Delta"；"branch" / "coordinate"；"local reader" / $E_{\mathrm{loc}}$ | 定义一次：Dynamic Delta = 两坐标残差空间；branch、coordinate 固定用词；$E_{\mathrm{loc}}$ 只出现一次并给别名 |
| 7 | **训练节是设置堆栈** | lr 分组、64 targets、script 配额、dropout 置零时机、ramp 步数挤在一段 | 训练节只留：目标函数（1 式）+ 两项监督各自的角色 + 训练/推理时的条件行为；数值进附录 |

公式预算：现在 10 式 → 目标 **6 式**（并入两支路候选定义、门控并入 router 块、注入式改行内）。

---

## Part 1：改写后的 Method（LaTeX，可直接替换 116–264 行）

段落数 ~11，预计 1.6 页正文 + 附录表。

```latex
\section{Method}
\label{sec:method}

\subsection{Overview}
Extending a typeface to a new script leaves one half of the problem without evidence.
The requested character must be built from strokes the reference script does not share, so its construction cannot be read off the Chinese references; the family's appearance, on the other hand, is only observable in those references and in no other font.
We treat the two as separate conditions.
Dynamic Delta forms the structural one from other fonts: their renderings of the requested character tell us how that character deforms across designs, which turns an absent correspondence into a retrieval problem over a known bank.
Target-character appearance completion (TC) forms the appearance one from the references themselves, so that the condition follows the family rather than a stored prototype.
Both conditions enter a conditional diffusion generator whose content and style encoders remain frozen:
\begin{equation}
 \hat\epsilon=\epsilon_\theta\bigl(x_t,t,E_c(x_c^0),\Delta^{\mathrm{ec}}_{c,t}(\Rset_f),\Delta^{\mathrm{es}}_{c,t}(\Rset_f),
                       G(\Rset_f),A(\Rset_f,c)\bigr).
 \label{eq:denoiser}
\end{equation}
The generator, its deformable structural interaction, and the perceptual recipe are inherited from a pretrained cross-script backbone; the two conditions, the routing that forms them, and the spatial readout that supervises TC are new.

\subsection{Dynamic Delta}
A character admits many designs and few of them are reachable by deforming the neutral rendering at hand.
The bank supplies the alternatives: for every training font it stores that font's rendering of the requested character, so we can read the change a design applies instead of its absolute shape,
\begin{equation}
 d^{\mathrm{ec}}_{j,c}=E_c^{\rho}(B_{j,c})-E_c^{\rho}(x_c^0),\qquad
 d^{\mathrm{es}}_{j,c}=E_s^{\rho}(x_{f_j,c})-E_s^{\rho}(x_c^0),
 \label{eq:delta}
\end{equation}
where $x_c^0$ is the neutral rendering, $B_{j,c}$ and $x_{f_j,c}$ are donor $j$'s renderings, and $\rho$ is the content scale read by the module that consumes the candidates.
Subtracting the neutral rendering removes the structure the generator already receives as content, and the bank then spans a space of deviations for the requested character: the bank fixes the directions, and only their combination is learned.

Retrieval narrows that space. The reference characters exist in the bank as well, so the family's Chinese glyphs can be compared with the same characters set in other fonts, giving a distribution over donors
\begin{equation}
 \alpha_j=\frac{\exp(s_j/\tau)}{\sum_{\ell\in\mathcal J_c}\exp(s_\ell/\tau)},\qquad
 s_j(\Rset_f)=\frac{1}{n}\sum_{i=1}^{n}\cos\!\left(\bar E_s(r_i),\bar E_s(B_{j,a_i})\right),
 \label{eq:retrieval}
\end{equation}
over the top-$K$ eligible fonts $\mathcal J_c$.
This distribution is a prior: it is fixed per episode and says nothing about the character being generated.

The prior is therefore refined during synthesis.
At each structural module, at scale $\rho$, the same donors are re-weighted from the current hidden state, the neutral content, the family summary, and the denoising time, with the prior as the base of the logits,
\begin{align}
 q^{b}_{\ell,t,p} &= Q^{b}_\ell(h_{\ell,t})_p+C^{b}_\ell(E_c^{\rho}(x_c^0))_p
                 +R^{b}_\ell(\bar G)+T^{b}_\ell(t),\\
 w^{b}_{j,\ell,t,p} &= \operatorname{softmax}_{j}\!\left[
       \log\alpha_j+\gamma^{b}_\ell\cos(q^{b}_{\ell,t,p},\Pi^{b}_\ell(d^{b}_{j,c})_p)\right],\\
 \Delta_{\ell,t,p} &= \Delta^{\mathrm{ec}}_{\ell,t,p}+\tanh(g_\ell)\,\Delta^{\mathrm{es}}_{\ell,t,p},
       \qquad b\in\{\mathrm{ec},\mathrm{es}\}.
 \label{eq:dynamic_delta}
\end{align}
The weights are nonnegative and sum to one, so the router selects within the space the bank defines instead of extrapolating beyond it.
Fixing $w_j=\alpha_j$ recovers the static Mean-Delta combination over the same donors and the same interface.

Two coordinates are kept because the family's appearance is otherwise summarized too coarsely.
Nine tokens pooled over a $3\times3$ grid carry the family's overall weight and proportion, but not the stroke-level evidence that separates one design from another at the size we generate.
Reading the same donors through intermediate maps of the style encoder makes that evidence available at the resolution of the target grid, with its own router and offset prediction.
The two branches share the retrieved neighborhood and the prior, and a learned per-module scalar combines their offsets, so the contribution of the second coordinate is measured in the same units as the first.

\subsection{Target-character appearance completion}
\label{sec:tc}
TC predicts how the requested character should look in the target family, using only the reference pixels.
The global tokens give the generator a summary of the family; TC adds a condition that is tied to the requested character, formed online from the references rather than read from a stored prototype.
It reads each reference at two scales and keeps the observations separate, so that every location of the requested character can attend to the evidence it needs:
\begin{align}
 H_{f,c} &= \operatorname{Attn}\!\left(W_qU_c+P,\;W_kV_f,\;W_vV_f\right),\\
 M_{f,c} &= H_{f,c}+\operatorname{MLP}\!\left(\operatorname{LN}(H_{f,c})\right),\qquad
 A_{f,c}=W_oM_{f,c},\qquad \hat F_{f,c}=W_aM_{f,c},
 \label{eq:tc}
\end{align}
where $U_c$ are the neutral content queries, $P$ their learned positional slots, and $V_f$ the reference evidence.
The readout $\hat F_{f,c}$ is spatial, so supervision ties the condition to where the design has to appear, while $A_{f,c}$ conditions synthesis.
At every up-path attention stage the generator reads the appearance memory on top of the global tokens, $Z=\operatorname{CA}(h,G)+\eta\,r(u)\operatorname{CA}(h,A_{f,c})$, with the gain $\eta$ ramped from the start of training.
The readout and the synthesis tokens are produced from the same memory, so the condition that is supervised is the condition that is used.

\subsection{Training and generation}
The content, style, and VGG encoders stay frozen; training updates the local reader, the routing of both branches, and the denoiser from a pretrained checkpoint,
\begin{equation}
 \mathcal L=\mathcal L_{\mathrm{diff}}
      +0.01\mathcal L_{\mathrm{perc}}
      +r(u)\bigl(0.01\mathcal L_{\mathrm{comp}}+0.05\mathcal L_{\mathrm{detail}}\bigr).
 \label{eq:objective}
\end{equation}
The first two terms are the backbone recipe.
$\mathcal L_{\mathrm{comp}}$ supervises the TC readout against a teacher pooled from the target image at the same spatial resolution, and $\mathcal L_{\mathrm{detail}}$ weights the regions where the design departs from the neutral rendering by the ink it adds and removes (Appendix~\ref{app:repro}).
Each episode draws one to eight references from the target family and a target character from the same font; donors exclude the episode font and its explicit weight variants.
Condition dropout follows the backbone protocol, and dropping the structural condition zeroes its offsets rather than its inputs.
At inference the appearance condition is computed once for each content--reference pair and reused across denoising steps, while the routing weights are re-estimated at every module and step.
```

---

## Part 2：正文删掉的量，落在附录哪一行

| 现在正文里的量 | 移到附录 |
|---|---|
| $E_{\mathrm{loc}}$ 来自前三个 block、$24\times24\times128$ / $12\times12\times256$、720 token、144 query、4 头、FP32 softmax、256/1024/128 宽度 | `TC local evidence` / `TC memory` / `TC output` 三行（已有），补齐 $E_{\mathrm{loc}}$ 初始化行 |
| adapter 无 bias + 单位阵初始化、门控 0 初始化、FP32 分支求和 | `Second branch` 行（已有） |
| $\eta=0.01$、1,000 步 ramp、lr 分组、warmup/plateau/cosine | `Reference injection` / `Joint learning rates` / `Schedule` 行（已有 $0.01$，补 ramp） |
| 64 targets、script 配额 32/24/8、38 个 detail 超额采样 | `Batch composition` 行（新增） |
| dropout $.02/.05$、结构 offset 置零时机 | `Reference / source dropout` 行（已有）+ 一句时机说明 |
| 新增参数量 / 显存 / 吞吐 | 新增行（数值待服务器取数） |

## Part 3：术语表（正文只用这套）

| 术语 | 含义 | 不再使用 |
|---|---|---|
| Dynamic Delta | 两坐标残差空间的统称 | structure condition / Delta 混用 |
| content-coordinate / style-coordinate residual | 两支路各自的候选与 offset | geometric branch / style branch |
| router | 逐模块按状态重估 donor 权重 | gate、routing module（混用） |
| TC / appearance memory $A_{f,c}$ / readout $\hat F_{f,c}$ | 外观条件与其监督读出 | memory、local condition 泛指 |
| $E_{\mathrm{loc}}$ | 在线读参考的局部编码器 | local reader（正文首次定义后可简称 reader） |

## Part 4：四查（tech-paper-template）

| 检查 | 结果 |
|---|---|
| 局限 → 关键思想 | pass（参考 script 无结构对应 → 用库内同字变化空间补结构） |
| 关键思想 → 挑战 | pass（挑战 1：库只给方向不给组合；挑战 2：九 token 装不下外观 → 两坐标 + TC） |
| 挑战 → 方法模块 | pass（router 对挑战 1；TC 与第二坐标对挑战 2） |
| 方法 → 贡献 | pass（双支路空间 + 外观条件；评测器在 §5，另行对齐） |

## Part 5：待 PI 拍板

| ID | 事项 | 推荐 |
|---|---|---|
| W1 | 是否按本方案改写（诊断 7 条 + 公式 10→6） | 是 |
| W2 | 第二坐标的动机构述（"九 token 3×3 装不下笔画级外观"） | 用；可核验且非防御性 |
| W3 | 贡献边界句（继承 RSI/DCN/感知项 vs 新增）是否保留在概览 | 保留 |
| W4 | 训练节是否只留目标函数与条件行为，其余数值全进附录 | 是 |

## Part 6：未核实

- 9 token 的 3×3 汇总：与 `hrfont_i.py` 的 `_style_conditions` 一致，但"3×3 装不下笔画级外观"是本方案的**机制论证**，若要写成更强的表征声明需等中层敏感度探针结果。
- 第二支路参数量/显存/吞吐数字未取（服务器命令：`sum(p.numel() for n,p in model.named_parameters() if 'es_branch' in n)`）。
