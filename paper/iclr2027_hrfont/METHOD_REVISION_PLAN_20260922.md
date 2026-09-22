# Method 章节修订方案（对齐 K6-B 实现）— 2026-09-22

- 日期：2026-09-22
- 依据（实现侧）：`sitonholy:/root/projects/hrfont_k6_20260920_r3`
  - `K6_CODE_IDENTITY.json` commit = `k6-isolated-20260920-r3-rsi-no-offset-retry2`
  - 文件 SHA：`scripts/train_k6.py` 3e547de3…、`scripts/k6_runtime.py` a970cb9a…、`scripts/k6_objective.py` 754f4351…、`scripts/queue_k6_b_rsi_no_offset.py` 80dbb424…、`experiments/K6/AUTHORIZATION_RSI_NO_OFFSET.json` edab4825…
  - 运行：`runs/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2`，`DONE.json` = `completed / step 20000 / inference_complete=false`
  - 父权重：`G0b-F0-V0913-BS256-A-S3407/global_step_10000`（unet sha 4a418058…）
  - 共享模块：`scripts/hrfont_i.py`（LocalMemory / SetOffset）、`scripts/k_components.py`（KSampler / extra_losses）、`scripts/k5_runtime.py`（EsOffset / DualOffset / EsTargetFeatures）
- 依据（论文侧）：`paper/iclr2027_hrfont/main.tex`（Method 在 116–247 行；附录 392–502 行）
- 使用 skill：`hrfont-project`（项目口径与叙事边界）、`tech-paper-template`（论文类型定位 + 方法骨架 + 四项一致性检查）
- 目标读者：写作 agent / PI。本文件是**定点修改指令**，不是重写建议；每条给出位置、现状、改法、LaTeX 草案、证据。

---

## 0. 结论速览

方法章节当前描述的是**单支路（Ec 残差）+ 中文辅助**的旧形态；K6-B 实际是**双支路（Ec 几何残差 + Es 空间风格残差，逐层门控）+ 端点细节监督 + 脚本/细节均衡采样，无 offset 惩罚、无中文辅助**。

必须改的 7 处：A1 Overview 三角色 | A2 Dynamic Delta 增双支路与门控 | A3 采样与 donor 策略 | A4 训练目标（去 offset、加 detail） | A5 辅助监督替换 | A6 附录表格数值 | A7 Run correspondence。

必须 PI 拍板的 6 项：D-M1 主方法取哪个臂 | D-M2 双支路是否进主方法 | D-M3 offset 惩罚的处置说法 | D-M4 detail 监督写法 | D-M5 辅助监督用哪条 | D-M6 双支路叙事口径（评审攻击面）。

---

## 1. 论文现状 vs K6-B 实现：逐条对照

| # | 论文当前写法（位置） | K6-B 实现事实（证据） | 判定 |
|---|---|---|---|
| 1 | 冻结 `E_c` 供多尺度中性目标特征；冻结 `E_s` 只供 "reference tokens and bank-retrieval descriptors"（118–120 行） | `E_s` 另有一路用途：前两个 block 组的**中间空间图** `(64,48,48)`、`(128,24,24)` 作为结构残差源（`k5_runtime.py:64-68`, `:81`） | ✗ 需改（A1） |
| 2 | 结构条件只有一条：`Δ = Σ w_j d_j`，`d_j = E_c^s(B_{j,c}) − E_c^s(x_c^0)`（146–163 行） | 两支路：几何支路同上；风格支路 `e_{j,l} = E_s^l(x_{f_j,c}) − E_s^l(x_c^0)`，`l ∈ {48², 24²}`，各自 router + offset head，合成 `o = o_c + tanh(g_l)·o_s`，`g_l` 逐层、初值 0（`k5_runtime.py:12-39`, `:81-82`, `:113-122`） | ✗ 需改（A2） |
| 3 | 未提 Es 支路适配器/门控初始化 | 风格支路经**无 bias 的 1×1 适配器、单位阵初始化**（不改残差尺度）；门控 0 初始化、适配器非零（"no double-zero"）；分支合成在 FP32 做，避免近零门控下 FP16 梯度下溢（`k5_runtime.py:16-17`, `:27`, `:36-38`） | ✗ 需补（A2，细节可进附录） |
| 4 | donor 只声明"排除本 episode 字体"（101 行） | 实策略：`exclude_self_and_explicit_weight_variants_before_alpha`——在 α 归一化**之前**排除自身与显式 weight 变体（run `config.json` 的 `data_identity.donor_policy`；`k5_runtime.py:106`） | ✗ 需补（A3） |
| 5 | 训练目标含 `+0.25 L_off`（226、230 行） | K6-B：`offset_loss_weight = 0.0`，`loss = eps + .01·perc + (0.0)·offset + extra`；offset 只作诊断读数（`train_k6.py:198`、`AUTHORIZATION_RSI_NO_OFFSET.json`、run `config.json`） | ✗ 需改（A4） |
| 6 | 目标只写 `L_diff / L_perc / L_comp`（223–228 行） | 另有端点细节监督：`extra = ramp·(.01·comp + .05·detail)`，`detail = mean(c · √ᾱ_t · D_out)`，`D_out = D_region + D_change + .25·D_high`（`k_components.py:95-105`）；`D_region` 为 ink/near/distant（权重 .5/.4/.1）+ 半权重 Sobel 边缘误差，96 与 48 两尺度（`i56_components.py:8-48`）；`D_change/D_add/D_remove` 用相对中性渲染的增墨/减墨模板，`D_high` 为高频残差（`k_components.py:77-97`） | ✗ 需补（A4） |
| 7 | 未写批次构成 | 每步 global batch 64，脚本配额 `{western 32, kana 24, bopomofo 8}`，其中 confirmed-detail 过采样 `{19, 14, 5}`（≈60%），配对位置 `{4, 3, 1}`（`k_components.py:11-13, 44-63`） | ✗ 需补（A3） |
| 8 | "Within-script auxiliary supervision"：中文同族补全辅助，`L_main + 0.5 L_CN`，每步 32 个中文样本（240–246 行、267 行） | K 线 recipe **没有**中文辅助；K 线的辅助是 K6-A 的 GT-vs-neutral 排序项（`k6_objective.py:44-58`），K6-B 明确关闭（`pure_noise_auxiliary=false`, 每步记录 `k6_aux=False`） | ✗ 需改（A5、D-M5） |
| 9 | TC 章节：`E_ℓ` 初始化自 `E_s` 早期块；720 token/参考；144 查询；4 头，逐头 Q/K 归一化，FP32 softmax；`144×1024` / `144×128`；注入 `Z = CA(h,G) + η r(u) CA(h,A)`，η 初值 0.01（172–219 行） | 与实现一致：`LocalMemory` = `copy.deepcopy(encoder.blocks[:3])`、`720 = 24²+12²`、`slots(144,256)`、4 头 64 维、1/8 缩放、FP32、`out→1024`/`readout→128`（`hrfont_i.py:10-48`）；注入 `old + local_gain(0.01) × gate × local × active`（`:88, :96, :119-121`） | ✓ 保留（A6 仅精化为 "first three blocks"） |
| 10 | router 公式 `q = Q(h)+C(E_c)+R(Ḡ)+T(t)`、`w = softmax(log α + γ cos)`（155–164 行） | 一致：`q = query(hidden) + content(neutral) + [ref(pooled 1024→32) + MLP(t/1000)]`，`key = 3×3 conv(donor)`，`logits = cos × strength(learned, init .1) + log α`，α≤0 位置 mask 为 −inf（`hrfont_i.py:57-77`） | ✓ 保留（A2 把 `d` 扩为两支） |
| 11 | 未写 drop 在路由**之后**应用 | `offset = original(hidden, mixed) * active`，注释明确 "Drop AFTER routing, so source/CFG drop cannot leave affine-offset leakage"（`hrfont_i.py:78-80`） | 建议补一句（A2） |
| 12 | 附录：dropout `0.10 / 0.25`；loss 系数 "perceptual 0.01; raw offset 0.25; completion 0.01"；辅助 "batch 32; weight 0.5"（415–417 行） | 实际 `joint_cfg=0.02`、`source_drop=0.05`；`vgg=.01`、`detail=.05`、`comp=.01`、`offset=0.0`（`train_k6.py:135-136, 176, 198`、run `config.json`） | ✗ 需改（A6） |
| 13 | Run correspondence 写 I0/I1/I2（435–439 行） | 当前线是 K 线：K1（原始 recipe 基线）→ K5-A（Es-only）→ K5-B（dual）→ K6-B（RSI / no offset, 20k）→ K7-B（+V3 数据, 跑动中） | ✗ 需改（A7） |
| 14 | 数据：260 字体 / 228-16-16 / 56,429 对（251–254 行） | V2 = `manifests/v0917/*`（train/val/test、donor_train、style_pool、sample_weights、reference_compatibility）；K7-B 用 V3 = V2 + v0921 补充集（315 字体，只取可用部分，**不进 bank**） | ✗ 待 D-M1 决定后改（A7） |

---

## 2. 定点修改指令（A1–A7）

### A1. Overview（118–129 行）

**现状**：`E_s` 只被描述为供 token 与检索描述子；结构条件只有一条。

**改法**：把 `E_c` / `E_s` / `E_ℓ` 的**角色分工**写清（三角色），并让 denoiser 条件含两条残差。

**LaTeX 草案**：

```latex
\subsection{Overview}
We build on a conditional diffusion generator with a frozen content encoder $E_c$ and a frozen style encoder $E_s$.
The two encoders serve separate roles.
$E_c$ supplies multi-scale features of the neutral target.
$E_s$ supplies the nine global reference tokens $G$, the pooled descriptors that drive donor retrieval, and the intermediate spatial maps consumed by the style-residual branch of Dynamic Delta.
A trainable local encoder $E_\ell$, initialized from the first three blocks of $E_s$, reads the reference pixels online.
Dynamic Delta conditions structural interaction through two donor-side residuals, and TC turns local evidence into target-aligned appearance tokens.
The denoiser thus receives
\begin{equation}
 \hat\epsilon=\epsilon_\theta\bigl(x_t,t,E_c(x_c^0),\Delta^{g}_{c,t}(\Rset_f),\Delta^{s}_{c,t}(\Rset_f),
                       G(\Rset_f),A(\Rset_f,c)\bigr).
 \label{eq:denoiser}
\end{equation}
Here $x_t$ is the noisy target image, $G$ contains nine global reference tokens, and $A$ contains 144 appearance tokens arranged around the target's $12\times12$ content grid.
```

### A2. Dynamic Delta（131–170 行）

**改法**：保留检索（eq. 2）与几何残差（eq. 3，改名并标注为 geometric branch），新增一段双支路定义 + 门控合成，router 公式扩为 `b ∈ {g,s}`，补一句 drop-after-routing。

**LaTeX 草案（新增段落，插在 eq. 3 之后）**：

```latex
\paragraph{Geometric and style residuals.}
Both branches read the same retrieved neighborhood $\mathcal J_c(\Rset_f)$ and the same prior $\alpha$.
The geometric branch keeps \eqref{eq:delta}: differences of the frozen content encoder taken relative to the neutral rendering.
The style branch takes the first two intermediate maps of the frozen style encoder, at $48\times48$ and $24\times24$, and forms
\begin{equation}
 e_{j,l}=E_s^l(x_{f_j,c})-E_s^l(x_c^0),\qquad l\in\{1,2\},
 \label{eq:style_delta}
\end{equation}
where $x_{f_j,c}$ is donor $f_j$'s rendering of the requested character and the anchor $x_c^0$ is again the neutral rendering.
A bias-free $1\times1$ adapter initialized to the identity maps $e_{j,l}$ to the branch width without rescaling the difference, so the branch starts from the donor residual itself.
```

**LaTeX 草案（router 与合成，替换现 eq. 4 的后半）**：

```latex
\begin{align}
 q^{b}_{\ell,t,p} &= Q^b_\ell(h_{\ell,t})_p+C^b_\ell(E_c^s(x_c^0))_p+R^b_\ell(\bar G)+T^b_\ell(t),\\
 w^{b}_{j,\ell,t,p} &= \operatorname{softmax}_{j}\!\left[
       \log\alpha_j+\gamma^b_\ell\cos(q^{b}_{\ell,t,p},K^b_\ell(d^{b}_{j,c,p}))\right],\\
 o^{b}_{\ell,t,p} &= \operatorname{Offset}^b_\ell\Bigl(h_{\ell,t},\ \textstyle\sum_{j\in\mathcal J_c}w^{b}_{j,\ell,t,p}\,d^{b}_{j,c,p}\Bigr),
       \qquad b\in\{g,s\},\\
 o_{\ell,t,p} &= o^{g}_{\ell,t,p}+\tanh(g_\ell)\,o^{s}_{\ell,t,p}.
 \label{eq:dynamic_delta}
\end{align}
```

**配文草案（接在公式后）**：

```latex
$d^{g}_{j,c}$ denotes the content-encoder difference of \eqref{eq:delta} and $d^{s}_{j,c}$ the style-encoder difference of \eqref{eq:style_delta}.
Each branch has its own routing parameters and offset head, and $\gamma^b_\ell$ is learned.
The per-layer gate $g_\ell$ is initialized to zero, so the style branch enters gradually; the adapter is nonzero at initialization, and the branch combination is computed in FP32 so that a near-zero gate does not underflow its gradients.
Condition dropout is applied after routing, so a dropped structural condition cannot leave a residual offset behind.
The combined offset drives the same deformable structural interaction as before.
```

保留原有 "weights nonnegative and sum to one / adaptation selects within the available residual variations / Mean-Delta recovered by fixing $w_j=\alpha_j$" 段；把末句改为两支均成立。

### A3. 批次构成与 donor 策略（新增，插在 232 行后）

```latex
\paragraph{Batch composition.}
Each update draws a global batch of 64 target glyphs with fixed script quotas (32 Latin, 24 kana, 8 phonetic), and oversamples characters whose family-specific design is confirmed to depart from the neutral rendering (19, 14, and 5 per update for the three scripts).
Reference sets are drawn from that font's Chinese references, with one to eight observations per episode.
Donor retrieval excludes the episode font and its explicit weight variants before the prior is normalized.
```

### A4. Training and generation（221–238 行）

**改法**：目标函数去掉 `0.25 L_off`、加入 detail 项；新增 `Endpoint detail supervision` 段说明各项作用在预测端点 `x̂_0` 上、掩码与权重语义；offset 只作诊断。

```latex
\begin{equation}
 \mathcal L=\mathcal L_{\mathrm{diff}}
      +0.01\mathcal L_{\mathrm{perc}}
      +r(u)\bigl(0.01\mathcal L_{\mathrm{comp}}+0.05\mathcal L_{\mathrm{detail}}\bigr).
 \label{eq:objective}
\end{equation}
```

```latex
\paragraph{Endpoint detail supervision.}
The perceptual, completion, and detail terms act on the predicted endpoint $\hat x_0$ of each denoising step, so one forward pass supplies both the epsilon target and the image-space terms.
The detail term concentrates supervision on the regions where the requested design departs from the neutral rendering:
\begin{equation}
 \mathcal L_{\mathrm{detail}}=\mathbb E\Bigl[c\;\sqrt{\bar\alpha_t}\;\bigl(d_{\mathrm{region}}+d_{\mathrm{change}}+0.25\,d_{\mathrm{high}}\bigr)\Bigr],
 \label{eq:detail}
\end{equation}
$c$ marks samples that keep their structural condition, and $\sqrt{\bar\alpha_t}$ downweights steps whose endpoint estimate is unreliable.
$d_{\mathrm{region}}$ is a region-weighted pixel and edge error over disjoint ink, near-ink, and background masks, where near-ink includes the interiors of outlined glyphs.
$d_{\mathrm{change}}$ averages the error over ink added and ink removed relative to the neutral rendering, and $d_{\mathrm{high}}$ is the high-frequency residual over non-background regions.
The raw offset regularizer of the backbone is not retained; offset norms are logged as a diagnostic instead.
```

**注意**：最后一句的表述需 D-M3 拍板后再定稿（去掉惩罚是 K6-B 的臂定义，不是可随意声称的"更优设计"）。

### A5. 辅助监督（240–246 行）

**改法**：删除中文补全辅助段（I 线遗留，K 线不使用）；改为 K6-A 的排序辅助，并明确"主模型不启用"。

```latex
\paragraph{Auxiliary generation ranking.}
We also test an auxiliary objective on the same frozen features.
Every 16 updates, one same-character pair from two different families is rolled out from pure noise, and the resulting endpoint is compared with the target and with the neutral rendering:
\begin{equation}
 \max\Bigl(0,\;0.2\,d(\hat y,y_{\text{GT}})+d(\hat y,y_{\text{GT}})-d(\hat y,x^0)\Bigr),
\end{equation}
enabled only for pairs whose target departs from the neutral rendering, $d(y_{\text{GT}},x^0)>\tau_{\text{script}}$.
The term ranks the generated design above the neutral rendering relative to the target; it is disabled in the main model.
```

（实现证据：`k6_objective.py:44-58`；权重 0.05、1000 步 ramp、8 步 rollout、每 16 次更新一次，见 `train_k6.py:235` 与 `K6_C_DECISION_ARCHIVE_20260921.md`。）

### A6. 附录同步（392–460 行）

- 415 行 `Loss coefficients` → `perceptual 0.01; detail 0.05; completion 0.01; offset 0.0 (diagnostic only)`。
- 416 行 `Auxiliary Chinese study` → 删除，替换为 `Auxiliary ranking study: pure-noise rollout, 8 steps, every 16 updates, weight 0.05, ramp 1,000`。
- 417 行 `Reference / source dropout` → `0.02 / 0.05`。
- 新增行：`Es residual branch  identity-initialized 1x1 adapter; per-layer gate init 0; FP32 branch combination`；`Detail masks  ink / near-ink / background; add / remove / high-frequency`；`Batch composition  32 / 24 / 8 script quotas; 19 / 14 / 5 confirmed-detail`；`Added parameters  <use: sum(p.numel() for n,p in model.named_parameters() if 'es_branch' in n)>`。
- 429 行 `local encoder is a trainable copy of the first three global-style encoder blocks` ✓ 保留。

### A7. Run correspondence 与数据（251–267、435–439 行）

- 435–439 行改为 K 线：K1（原始 recipe 基线）→ K5-A（Es-only 结构残差）→ K5-B（双支路）→ K6-B（RSI / 无 offset 惩罚，20k）→ K7-B（V3 数据）。每臂写明初始化（K0/G0b step10000）、预算（10k / 20k）、seed 3407。
- 251–254 行的语料数字在 D-M1 拍板后统一（V2 与 V3 的字体数、可用 pair 数、split 大小），并与主表口径一致。
- 267 行"auxiliary adds 32 Chinese examples per update"→ 按 A5 替换。

---

## 3. ICLR 合规核对（venue 模板 `iclr2027_conference.tex`）

| 要求 | 出处 | 现状 | 动作 |
|---|---|---|---|
| 主文 **9 页**上限（camera-ready 10 页），引用不限页 | 模板 131 行 | Method 现约 2 页（116–247 行），改后预计 +0.5~0.7 页 | 实现细节（token 数、初始化、FP32、参数量）移附录；正文只留机制与语义 |
| 匿名 | 模板 16 行 | `\author{Anonymous authors}` ✓ | 保持；新增引用不得含身份信息 |
| Reproducibility statement（推荐，不计页） | 模板 428–445 行 | 已有段落，但内容指向 I 线与旧数值 | 按 A6/A7 同步，并把"训练语料/bank/参考预算"的映射写清 |
| Ethics statement（推荐） | 模板 414–426 行 | 已有 | 更新一次（如数据许可随 V3 补充集变化） |
| 方法自足性（评审可复现） | ICLR 审稿惯例 | 新符号未定义，双支路来源未声明 | 定义 `e_{j,l}`、`g_\ell`、`d_region / d_change / d_high`、`τ_script`；声明训练与推理使用同一 bank、donor 排除策略、Es 支路消费的是**训练字体**在目标字符上的渲染（非目标字体拉丁字形） |
| 不得把未测事实写成结论 | 项目铁律 | 方法段落目前无越界断言 | 去掉 offset 惩罚的理由需有诊断读数或明确标为臂定义（D-M3） |

---

## 4. 方法链一致性检查（tech-paper-template 四查）

论文类型定位：**Technique paper**（任务既存；贡献 = 条件源 + 评测方法 + 完整系统）。因此 Key Idea 必须承载叙事，模块必须由挑战导出，不能靠"新设定"立论。

| 检查 | 结论 | 说明 |
|---|---|---|
| Limitation → Key Idea | pass | 脚本内结构对应关系跨语系失效 → 把"结构变化空间"（bank 检索 + 自适应路由）与"外观预测"（TC）分开：前者定方向，后者定落点 |
| Key Idea → Challenges | pass（改后更实） | ① 结构先验必须可迁移 → 检索先验 + 学习路由 + **双坐标残差**（Ec 几何 / Es 空间）；② 外观必须落到目标位置 → 144 查询 + 空间监督；③ 跨语系差异集中在少数区域 → **端点细节监督 + 脚本/细节均衡采样** |
| Challenges → Modules | pass | 一一对应：Δ（含双支路与门控）/ TC / detail+sampler。无"没有挑战支撑"的模块 |
| Modules → Contributions | 需注意 | 主贡献仍为两项核心（Δ 条件源、评测）+ 一个完整系统（双支路、detail、采样属系统级增强）。**不要把 dual branch / detail 升级为并列第三贡献**（与 7.7 分 review 的建议一致，否则易掉到 Borderline） |

---

## 5. 待 PI 拍板（每项：选项 + 推荐）

| 编号 | 问题 | 选项 | 推荐 |
|---|---|---|---|
| D-M1 | 论文主方法取哪个臂？ | A) K6-B（V2, 20k，已完成；推理未跑）B) K7-B（K6-B 架构 + V3 数据，跑动中）C) K5-B（含 offset 惩罚） | **B**（若 K7-B 收敛且外部对照齐备）；论文数字一旦用 V3 就必须同步语料表。A 作后备 |
| D-M2 | 双支路进主方法还是降为消融？ | A) 进主方法（K5-A/K5-B 已有匹配对照）B) 只留 Ec 单支路，Es 支路进附录 | **A**，但正文必须给出机制理由（风格编码器坐标下的 donor 残差补充几何残差无法表达的字重/对比度变化），且报告新增参数量与吞吐 |
| D-M3 | offset 惩罚（0.25）怎么说？ | A) 写为设计决定（去掉，只留诊断）B) 写成 K6-B 的臂设定并放进消融表 | **B**：没有匹配对照证明"去掉更好"；按 A 写会变成未证声称。（K5-A/K5-B 都带 0.25，K6-B 去掉，本身就是一个可报告的对照） |
| D-M4 | detail 监督写法 | A) 正文完整定义（含 3 个分项）B) 正文一句话 + 附录给分项定义 | **B**：分项定义（ink/near/distant 权重、增删墨迹模板、高频残差）移附录，正文保留 `eq. detail` 与语义 |
| D-M5 | 辅助监督写哪条 | A) 中文补全辅助（I 线）B) GT-vs-neutral 排序辅助（K6-A/C）C) 两条都写 | **B**（K 线实际使用且 K6-C 有臂）；中文辅助如要保留，只能作为历史/附录记录，不能出现在 Method |
| D-M6 | 双支路叙事口径 | A) 第二几何视图 B) style-encoder 坐标下的 donor 残差 | **B**：避免"风格从结构通路进入"与 TC 角色冲突的质疑；配套写法=无 donor-ID 参数、单位阵适配器、门控 0 初始化、drop-after-routing |

---

## 6. 跨节一致性提醒（不属 Method，但同一批改动）

1. **Abstract / Title 尚未落地**：`main.tex` 里仍是旧版（`Completing Font Families across Scripts`、`cross-script completion benchmark`），与已定口径（不用补全框定、不称 benchmark）冲突，待写入。
2. **任务节标题** `Cross-Script Font Completion`（93 行）与标题决定同步。
3. **数据数字**（251–254 行）与 A7 联动。
4. **主表/消融表**：辅助监督行（334 行）需按 D-M5 改写；`Mean-Delta baseline` 保留 ✓。
5. **评测节**：K 线未跑推理（`DONE.json` 的 `inference_complete=false`）——主表数字仍空，属已知状态。

---

## 7. 未核实 / 待补

- V3 语料的最终可用字体数与 pair 数（v0921 补充集只取可用部分，需从构建脚本/清单确认）。
- K7-B 是否作为最终论文臂（跑动中；完成后需核对 checkpoint 与对照）。
- 双支路的新增参数量与显存/吞吐实测（命令已给出，需在服务器上跑一次）。
- 各臂的匹配对照清单（K1 / K5-A / K5-B / K6-B / K6-B+aux）与预算是否满足"一次只改一个变量"。
- Es 支路残差与其检索描述子同源（都来自冻结 `E_s`）在论文里的披露措辞（避免"同一编码器既检索又监督"的循环质疑）。
