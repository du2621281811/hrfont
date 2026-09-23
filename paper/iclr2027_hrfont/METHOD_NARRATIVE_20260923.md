# Method 叙事修改方案 v2（2026-09-23）

**依据**：`main.tex` @ `403590826`（Method = L165–229，附录 L375–443）；实现锚点 = K7-B run `/root/projects/hrfont/runs/K7-B-V3-K0-S3407-20K-R2` + 代码树 `hrfont_k7_v3_20260922`。
**触发**：PI 七点（insight / 不写 bank / 不写 9 token / readout 措辞 / TC 公式 / loss 参与度 / 公式总量）。
**用到的 skill**：`hrfont-project`（`references/paper-method-audit.md` §7 论证体 register、§7b 防御性写作、§7d 库叙事）、`tech-paper-template`（四查）、`polish-english`（register/时态/无缩写）、`deai`（禁用词与句式）。
**边界**：本文件是方案，`main.tex` 未改。PI 点头后只动 §3.1–3.4。

---

## 0. 结论速览

| # | PI 关注点 | 处置 |
|---|---|---|
| 1 | Overview 要接「pixel 无法很好监督」的故事，CRP/TC 逻辑不够强 | §3.1 重写：**训练只给一个实现的渲染 ⇒ 它不决定「家族会选哪种实现」与「家族如何画局部细节」⇒ 我们把这二者作为条件给出**。CRP=实现空间+参考选择，TC=局部外观，两条件在同一句子里各自对应一个未定项 |
| 2 | CRP 不要提 bank | method 内「bank」清零，改 candidate families；符号 $B_{j,c}$ 一并换成 $x_{f_j,c}$（与 $x_c^0$、$r_i$ 同一族记法），从符号层也去掉 bank 口吻 |
| 3 | 不要写 9 token | 改「the global summary is pooled from a $3\times3$ map」，去掉 token 计数 |
| 4 | 「读出」不通 | 全面弃用 readout：量=**appearance map** $M$，模块=appearance head（只在附录出现），动词=supervise |
| 5 | TC 核心实现要有公式 | 新增 TC 读取式（两尺度 token 集 + 位置槽 + 两个头） |
| 6 | $L$ 写法会被质疑参与度/监督力 | 新增**条件注入式**并把「结构偏移经生成目标学习」写成机制句；把两个辅助项监督的对象写成式中的量（$M$、ink-weighted detail），不靠系数自证 |
| 7 | 公式太少 | 显示式 3 → **6**（offsets / prior / routing / TC / inhibition 注入 / objective），每式前有主张句、后有作用句；其余量仍下沉附录 |

---

## 1. 诊断表（PI 原话 → 症状 → 定位）

| PI 原话 | 症状（我写的原句为证） | 定位 |
|---|---|---|
| 「Overview … pixel 无法很好监督的故事？现在 CRP 和 TC 显得逻辑性没有那么强」 | Overview 只讲「不对称 + 拆两部分」，没有任何一句说明**为什么监督本身不够**，于是 CRP/TC 看起来是两个并列模块而不是同一问题的两个必答项 | `${main.tex}:169–179$` |
| 「CRP 不要直接提到 Bank，太像增强检索」 | 「CRP represents … with the renderings **stored in the bank**」「**The bank** consists of training fonts」「so **the bank** supplies the directions」 | `:174, :188, :189` |
| 「不要直接写 9token，3×3 池化…就足够了」 | 「The **nine global tokens** are pooled from a $3\times3$ map」 | `:205` |
| 「读出到底是什么意思」 | 「a supervised **readout**」「The **readout** is spatial」「supervises the TC **readout**」 | `:211, :212, :224` |
| 「TC 核心实现的公式还是提供一下」 | §3.3 全叙述、零公式 | `:209–213` |
| 「L 这个写法会不会被质疑 loss 参与度不够高」 | 目标函数只列 3 项系数；**Δ 与路由没有任何一句说明它们靠什么学**；读者只能看到 0.01/0.05 两个小数 | `:217–224` |
| 「公式太少了」 | 显示式只有 3 式，且 $s_j$、readout、条件注入三处机制无式 | `:183–224` |

---

## 2. 公式预算（3 → 6）

| 式 | 回答什么问题 | 买到什么 |
|---|---|---|
| `eq:delta` 形变候选（已有，去 $\rho$） | 「同一字符在不同设计里怎么变」 | 候选提供的是**变化方向**，不是字形 |
| `eq:prior` **新增** | 「哪一族的设计更接近参考」 | 先验由参考逐字比较而来 ⇒ 参考真正决定选择 |
| `eq:crp` 路由与合成（已有） | 「哪种实现、在哪一层、哪一步」 | 非负归一 ⇒ 只在候选张成的空间内组合；逐位置逐模块重估 |
| `eq:tc` **新增** | 「局部外观长什么样、往哪儿放」 | 两尺度证据 + 位置槽 ⇒ 空间化的外观图 $M$ 与合成 token $T$ |
| `eq:inject` **新增** | 「两个条件在哪里进入生成器」 | 条件在去噪器内部 ⇒ 结构偏移与路由**由生成目标训练**（PI 第 6 点的正面回答） |
| `eq:objective` 目标（已有） | 「还额外监督什么」 | 监督对象写成式中的量（$M$ 与 ink 加权细节），不靠系数说话 |

正文只留这 6 式；token 数 / 头数 / 通道 / 初始化 / FP32 / 配额 / ramp / dropout 全在附录。

---

## 3. 逐点修改方案

### M1 §3.1 Overview 重写（PI 第 1 点）

**现状**：三段里没有任何「监督为何不够」的主张句。

**改法**：把 Overview 写成一条 4 步链：① 源/目标的不对称；② **训练只给一个实现的渲染，它不决定「家族会选哪种实现」与「家族怎样在此字符上画细节」**；③ 因此这两项必须以条件形式给出（CRP/TC 各对应一项）；④ 两条件与内容/全局参考特征在去噪器汇合。
最后一句保留「Agreement with that rendering is therefore one of several valid outcomes rather than a specification of the design.」——它把「一个渲染 ≠ 设计规范」讲成任务性质，与 intro L61 的评测主张同源，**不写成「pixel 监督不合理」**（§10.1 红线）。

```latex
\subsection{Overview}
Extending a typeface to a new script is asymmetric.
The target glyph is built from strokes the source script does not share, so the references do not determine its geometry; the visual character of the family, in turn, is visible only in those references.
Training gives one rendering of one realization of the target character, and that rendering does not determine either property the task depends on: which realization the family would choose for this character, and how the family draws local detail on it.
Agreement with that rendering is therefore one of several valid outcomes rather than a specification of the design.

We supply both properties as conditions.
The \emph{Character Realization Prior} (CRP) supplies the space of realizations: the training fonts show how the requested character deforms across designs, and the references select within that space, which yields a style-conditioned structural offset.
\emph{Target-Character Appearance Completion} (TC) supplies the local appearance, predicting how contour treatment, stroke endings, texture, and decoration are drawn on the requested character from the target content and the source-script references.
The two conditions are combined with the content and global reference features in the diffusion generator: the denoiser receives the neutral content, the structural offset, the global reference tokens, and the appearance representation.
Figure~\ref{fig:overview} summarizes the model.
```

### M2 §3.2 CRP：去 bank、换符号、加先验式（PI 第 2 点）

- 「bank」→「candidate families / candidates」；$B_{j,c}$ → $x_{f_j,c}$（与 preliminaries 的 $x_c^0$、$r_i$ 同族记法）。
- 新增 $s_j$ 定义（逐参考字符比较后取平均）——机理，不是细节。
- 保留「candidate families exclude that font and its explicit weight variants」（属设定，非找补）。

```latex
\subsection{Character Realization Prior}
For a requested character $c$, CRP reads how each candidate family $f_j$ renders it and takes the difference with respect to the neutral realization $x_c^0$ through both encoders:
\begin{equation}
 d^{\mathrm{ec}}_{j,c}=E_c(x_{f_j,c})-E_c(x_c^0),\qquad
 d^{\mathrm{es}}_{j,c}=E_s(x_{f_j,c})-E_s(x_c^0).
 \label{eq:delta}
\end{equation}
The neutral realization carries the character; each difference carries how one design changes it, so the candidates span the directions along which the character can be realized.

The references determine which realization is appropriate.
Reference characters also occur in the candidate families, so we compare every reference with the same character set in each candidate, using the style encoder's pooled descriptor $g$:
\begin{equation}
 s_j=\frac{1}{n}\sum_{c'\in\Rset_f}\cos\!\bigl(g(f_*,c'),\,g(f_j,c')\bigr),\qquad
 \alpha_j\propto\exp(s_j/\tau)\ \ \text{over the top-}K\ \text{candidates},
 \label{eq:prior}
\end{equation}
where $f_*$ is the target family.
The prior is fixed per episode and depends only on the reference characters.

The structural offset is resolved during generation.
Every structural module re-estimates the weights from the current hidden state, the neutral content, the reference summary, and the denoising time, with the prior as the base of the logits:
\begin{equation}
 w^{b}_{j,\ell,t,p}=\operatorname{softmax}_{j}\!\left[\log\alpha_j+\gamma^{b}_\ell\cos(q^{b}_{\ell,t,p},\Pi^{b}_\ell(d^{b}_{j,c})_p)\right],\qquad
 \Delta_{\ell,t,p}=\Delta^{\mathrm{ec}}_{\ell,t,p}+\tanh(g_\ell)\,\Delta^{\mathrm{es}}_{\ell,t,p}.
 \label{eq:crp}
\end{equation}
The weights are nonnegative and normalized, so one generation combines many realizations jointly, and the combination is re-decided at every location, module, and step.
Each branch maps its combined offset into the corresponding level of the denoiser, where it is added to the up-path features (Eq.~\ref{eq:inject}).

CRP reads the same candidates through a second coordinate.
The global summary of a family is pooled from a $3\times3$ map and carries its overall weight and proportion; the stroke-level evidence that separates designs at the resolution we generate lives in intermediate maps of the style encoder, which is read with its own routing and offset prediction.
The two readings share the candidate neighborhood and the prior, and their offsets are combined by a learned per-module scalar, which forms the structural offset that CRP provides.
```

### M3 §3.3 TC：公式 + 去 readout（PI 第 4、5 点）

```latex
\subsection{Target-Character Appearance Completion}
TC predicts a target-aligned appearance representation from the target content and the source-script references, covering how local properties such as contour treatment, stroke endings, texture, and decoration should appear on the requested target character.
It reads each reference at two scales and keeps the observations separate, so that every location of the target character can attend to the evidence it needs:
\begin{equation}
 Z=\operatorname{Attn}\bigl(Q(x_c^0)+\Pi_{\mathrm{slot}},\,[K(S^{1});K(S^{2})],\,[V(S^{1});V(S^{2})]\bigr),\qquad
 M=\phi(Z),\ \ T=\eta(Z),
 \label{eq:tc}
\end{equation}
where $S^{1,2}$ are the tokens read from every reference at the two scales, $\Pi_{\mathrm{slot}}$ learned positional slots, $M$ the spatial appearance map, and $T$ the tokens the denoiser consumes.
The map is supervised at the position where the design should appear, and at every up-path attention stage the generator reads the appearance representation on top of the global reference features.
Dimensions, attention settings, and initialization are listed in Appendix~\ref{app:repro}.
```

### M4 §3.4 Training and Inference：条件注入式 + 目标（PI 第 6、7 点）

- **新增 `eq:inject`**：两个条件在去噪器内部以加性项进入上采样特征 → 紧接一句「结构偏移与路由因此经生成目标学习」。这是对「参与度不够」的正面回答（机制，不是辩护）。
- 目标函数保持原样（系数不改），但把两个辅助项**监督的对象**写成式中的量（$M$、ink 加权的细节区域）。

```latex
\subsection{Training and Inference}
The two conditions enter the denoiser as additive terms on the up-path features,
\begin{equation}
 \tilde h_\ell=h_\ell+\mathrm{Off}_\ell\bigl(h_\ell,\Delta_\ell\bigr)+\lambda_\ell\,r(u)\,A_\ell,
 \label{eq:inject}
\end{equation}
where $h_\ell$ is the feature at module $\ell$, $\mathrm{Off}_\ell$ maps the structural offset into that level, $A_\ell$ is the appearance term supplied by the reader, and $\lambda_\ell$ a learned per-module scale.
The structural offset and the routing are therefore learned through the generative objective.
Training updates the denoiser, the reader, and the structural modules, while the content, style, and VGG encoders stay frozen.
The objective adds two terms to the backbone recipe,
\begin{equation}
 \mathcal L=\mathcal L_{\mathrm{diff}}
      +0.01\mathcal L_{\mathrm{perc}}
      +r(u)\bigl(0.01\mathcal L_{\mathrm{comp}}+0.05\mathcal L_{\mathrm{detail}}\bigr),\qquad r(u)=\min(u/1000,1),
 \label{eq:objective}
\end{equation}
where the appearance term supervises the predicted appearance map $M$ against the statistics of the target rendering at the same spatial resolution, and the detail term weights the regions that deviate from the neutral realization by the ink a design adds and removes (Appendix~\ref{app:repro}).

Each episode draws one to eight references from the target family and one target character from the same font; candidate families exclude that font and its explicit weight variants.
Dropping the structural condition zeroes its offset.
At inference the appearance representation is computed once for each content--reference pair and reused across denoising steps, while the routing weights of CRP are re-estimated at every module and step.
```

---

## 4. 术语表（before → after）

| before | after | 说明 |
|---|---|---|
| bank / font bank | candidate families（method 内）；preliminaries 的 $\Bset$ 见 W4 | 去掉检索仓储口吻 |
| $B_{j,c}$ | $x_{f_j,c}$ | 与 $x_c^0$、$r_i$ 同一记法族 |
| nine global tokens | the global summary pooled from a $3\times3$ map | 去计数 |
| readout（3 处） | the spatial appearance map $M$ / the appearance head（附录） | readout 在表示学习文献里指对比学习的投影头，本处语义不通 |
| the TC readout | the predicted appearance map $M$ | — |
| supervise … against a teacher pooled from the target image | supervise the predicted appearance map $M$ against the statistics of the target rendering | 主语改成被监督的量 |
| deformable structural interaction | $\mathrm{Off}_\ell(h_\ell,\Delta_\ell)$（式）+ plain 描述 | 换掉无法核对的比喻 |

---

## 5. 正文删掉的量 → 附录落点（都已存在，无需新增段落）

| 从正文移出/不进正文 | 附录落点 |
|---|---|
| 两个尺度的层与形状 | 实现细节段（$64\times48\times48$ / $128\times24\times24$） |
| identity-init 1×1 投影、zero-init gate | 同段 |
| token 数、头数、通道、FP32 | 实现细节段（已列） |
| $\sqrt{\bar\alpha_t}$、有效样本掩码、$n\sim U\{1,8\}$ | 同段（已补） |
| top-10、$\tau=0.07$、routing 维度、lr 分组 | 超参表（已列） |

---

## 6. tech-paper-template 四查

1. **限制 → 主张**：第 1 点的「一个渲染不决定实现与细节」直接对应 CRP/TC 两个条件 → pass。
2. **主张 → 挑战**：两个挑战（哪种实现 / 细节画在哪）派生自同一主张，不是为凑模块 → pass。
3. **挑战 → 模块**：CRP↔哪种实现、TC↔细节 → 一对一 → pass。
4. **模块 → 贡献**：与 intro 贡献列表（CRP / TC / 结构–外观分解）一致 → pass。

---

## 7. 待拍板

| # | 问题 | 选项 | 推荐 |
|---|---|---|---|
| W1 | method 去 bank 后，摘要（L21）、图 caption（L74）、preliminaries（L149–150）、实验节（L242/255/287-288/309-310/345/350）仍写 bank | A 全文统一术语（只换词，不动论证，逐条 before→after 报备）／B 只改 method，其余留 | **A**（叙事统一是 PI 一贯要求；且 CF-Font 的「font bank」是竞品术语，我方同词易被读成同类机制） |
| W2 | Overview 是否加一句与评测的衔接（同一不确定性 → set 式评测） | A 不加（intro L61 已说）／B 加一句 | **A**（method 只讲生成，评测归实验节） |
| W3 | 公式预算 6 式 | A 6 式／B 压到 5 式（把 `eq:prior` 行内化、`eq:tc` 的 $M,T$ 合并叙述，合计省约 0.3–0.6 页）；D 版**实测**：正文 10 → 11 页（scratch 编译），ICLR 主文 9 页上限下需与实验节一起排 | **A**（PI 要求核心公式齐）；若排不出版面，先砍 `eq:prior`（改行内 $s_j$），不砍 `eq:inject`（第 6 点的正面回答） |
| W4 | preliminaries 的 $\Bset$ 定义句 | A「the training-font renderings $\Bset$」+ donor retrieval → candidate selection／B 不动 | **A**（与 W1 同批） |
| W5 | `eq:inject` 放 §3.4 还是 Overview 末 | A §3.4（与目标函数相邻，支撑参与度主张）／B Overview | **A** |

---

## 8. 未核实 / 待实现时检查

- `main.tex` 中除 `eq:delta/eq:crp/eq:objective` 外是否还有引用公式编号的 `\ref`（落盘前 grep `\ref{eq:`）。
- `figures/method_overview.tex` 内部硬编码文字含旧术语（`Dynamic Delta`、bank、appearance memory）——按边界只提醒，等 W1 一起决定。
- 附录超参表数值已与 K7-B 对齐（`403590826`），本方案不再改动附录。
- 人评 / 主表数字仍未提供，与本方案无关。
