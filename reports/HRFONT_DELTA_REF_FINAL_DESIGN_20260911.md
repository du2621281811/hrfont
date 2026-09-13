# HR-Font Delta / Ref 最终统一设计稿

> **2026-09-13 PI 更新：本文 Set-Delta 路线已弃用，不再排期。以下保留为历史设计，当前方法为 Mean-Delta；执行状态以 PROJECT.md 为准。**

日期：2026-09-11  
状态：最终设计候选；尚未实现、训练或验证  
唯一训练 parent：已入库 F0@100k；F2-P/F3b-P 训练代码已于 `1dba686e` 入库，仅作 per-ref token/mask 接口参考，不作为训练 parent或权重依赖。正式论文协议只允许目标字体 ref8，不读取额外 own-font CN 图像

## 0. 最终裁决

HR-Font 的主问题保持不变：给定一个未见字体的少量中文参考字，目标拉丁、假名或注音字符在参考中不可直接观察，生成器既需要目标字符的结构变化可能性，也需要从参考字中读取目标字体真实可见的风格证据。

最终方法只保留两类输入：

1. **Set-Delta Variation Prior（候选变化集合先验）**：训练字体库提供“目标字符相对共同中性字形可以怎样变化”的候选集合；检索 Alpha 只负责缩小集合和提供先验权重，不再把候选提前平均成一个方向。
2. **Graphics-Informed Local Reference Attention（图形先验引导的局部参考注意力）**：ref8 提供目标字体实际可观察到的全局与局部风格；目标字符的局部结构作为 Query，按笔画原语从中文 refs 中读取相符的风格 Value。

二者的分工固定为：

> **Delta 回答“bank 支持这个目标字符怎样变化”；Ref 回答“目标字体已经展示了怎样的视觉实现”。**

Support 不进入论文主方法。own-font support 只允许作为额外观测工程模式或 oracle upper bound，必须披露额外目标字体图像数量。

| 信息源 | 允许承担的角色 | 不允许据此声称 |
|---|---|---|
| Delta / bank | 同目标字符、bank-supported 的边界与拓扑变化候选 | 目标字体真实笔刷或唯一设计 |
| Ref8 | 目标字体中可观察到的全局与局部视觉证据 | 未观察属性的唯一真值 |
| Support | 额外观测工程模式或 oracle upper bound | 相同 ref8 预算下的科学增益 |

字重、斜势、宽窄和端点位于几何与风格的共享边界，不靠命名强行归因；用 wrong-donor × wrong-ref 的 2×2 干预、路径清零和泄漏 probe 判断实际由哪一路控制。

## 1. 主叙事与贡献口径

### 1.1 两句话版本

跨语系 few-shot 字体补全的困难不是目标字符内容未知，而是有限中文 refs 没有覆盖目标字符可能采用的结构变化。HR-Font 从训练字体库检索同一目标字符相对固定中性锚点的 residual candidates；refs 先条件化候选 proposal，再以独立的局部风格路径约束这些候选在目标字体中的最终视觉实现。

### 1.2 投稿贡献

1. **核心方法贡献：检索条件化的目标字符候选轴。** 与把若干 donor 特征预先平均成单一 source 不同，HR-Font 保留相对固定中性锚点的 donor residual candidate axis，并延迟到目标字符空间位置进行聚合。
2. **核心机制贡献：先验与证据的结构化分工。** Delta 通过规范化边界表示提供 bank-supported variation；Ref 通过图形描述符引导的 learned local values 提供目标字体证据，降低 donor texture/colour appearance 直接控制最终笔触的风险。
3. **评测贡献：独立、lineage-aware 的 family compatibility 评价与盲评。** E12 仍只称家族兼容性代理，不称开放式设计合理性的完整概率。

Ref local attention、SDF、skeleton、cross-attention 均不单独宣称首创。创新候选是它们在跨语系缺失目标字符问题中的**角色分解、候选轴延迟聚合和 reference-conditioned proposal**；只有 matched controls 阳性后才升级为实证贡献。

### 1.3 明确不声称

- 不声称 ref8 唯一确定目标字体作者对未见字符的真实设计。
- 不声称 Alpha 恢复了真实后验；它只是由 refs 条件化的候选 proposal prior。
- 不声称 deterministic soft fusion 保留了严格的离散多模态；它只把候选轴保留到空间条件聚合之前。
- 不声称 SDF 将 style 完全移除；它只降低 texture/colour appearance 泄漏，轮廓风格仍需 probe。
- 不声称 SDF、骨架、局部 attention 或多尺度 style 本身是首次提出。
- 不把 own-font extra-CN support 的收益写成相同 8-shot 预算下的科学增益。
- 不把 E12 membership 写成普适“美观/合理概率”。

## 2. Delta 最终路线：Set-Delta Variation Prior

### 2.1 当前实现要修正的问题

当前实现为：

\[
\bar\Delta_c^{(l)}=
\sum_{j\in J(R)}\alpha_j E_c^{(l)}(B_{j,c})
-E_c^{(l)}(B_{0,c}).
\]

这会在进入网络前把多个设计模式压成一个平均点。它存在三个问题：

1. Alpha 不再只是缩小搜索空间，而会直接决定最终方向；
2. 圆点/星点、实心/空心、连/断等模式会相消；
3. F0 的 Ec 不是严格 style-invariant，donor 字重、斜势、圆角和纹理可能混入 Delta。

因此当前实现应称 weighted residual direction，不应称 variation space。本文后续的“centered”均指**相对固定中性锚点 \(B_0\)**，不是 donor 集合的统计均值中心化；正式报告必须给出 \(B_0\) 来源和 anchor-sensitivity control。

### 2.2 规范化几何表示

正式 Delta 不直接把 donor RGB appearance 作为先验值。对统一 A96 渲染的字形定义固定几何表示：

\[
G(I)=
[D_r(I),\partial_xD_r(I),\partial_yD_r(I),M(I)],
\]

其中：

- \(D_r\)：截断并归一化的 signed distance field；建议截断半径 \(r=8\) px；
- \(\partial_xD_r,\partial_yD_r\)：边界法向/方向信息；
- \(M\)：抗锯齿阈值后的 soft ink mask；
- 可选 skeleton/radius 只用于诊断和检索，不在首版同时增加输入通道。

所有 donor、B0、Content 必须使用相同 native96 渲染、阈值、距离变换和下采样规则。禁止依据生成结果调整阈值。

该表示的目的不是使 Delta 完全无风格，而是主动去除颜色、飞白和大部分笔刷 appearance，使 Delta 更接近边界、连通和笔画占据的变化。字重、衬线、端点和斜势仍可能泄漏，必须用 donor/font-ID probe 量化。

首版 cache contract 固定为：灰度 \([0,1]\)，ink=1；soft mask保留渲染器 alpha，二值几何阈值预注册为0.5；SDF约定字内为负、字外为正，clip到 \([-8,8]\) 后除以8；梯度用固定3×3 Sobel；先在96×96计算全部几何量，再用 area downsample到48/24。primitive 的 radius来自二值 ink 内 distance transform，orientation来自 SDF gradient 的二倍角表示，endpoint/junction来自固定 thinning 实现和8邻域度数。manifest 必须绑定 renderer、字体文件 hash、B0 hash、预处理代码 SHA、dtype/layout；规则改变即生成新 cache version。

### 2.3 候选变化集合

对每个 donor 字体 \(j\)、目标字符 \(c\) 和尺度 \(l\) 定义：

\[
d_{j,c}^{(l)}=
G_l(B_{j,c})-G_l(B_{0,c}).
\]

检索仍使用目标字体 ref8 与 bank 字体相同 ref8 的 Es 相似度：

\[
\alpha_j=\operatorname{softmax}(s_j/\tau),\qquad
J_K(R)=\operatorname{TopK}_j(s_j).
\]

精确合同为：用当前 active-ref mask 聚合检索分数 \(S:[B,M]\)，缺少目标字符的 donor 在 Top-K 前置 \(-\infty\)；按 score 降序、bank_id 升序稳定打破并列；再对 Top-K 内部分数计算 \(\alpha_K=\operatorname{softmax}(S_K/\tau)\)。训练采样 n refs 时，Alpha 与 Ref 分支使用相同 active refs，不偷用被 mask 的其余 refs。

但不再计算 \(\sum_j\alpha_jd_j\) 作为网络输入。网络接收：

\[
\mathcal D_c(R)=
\{(d_{j,c}^{(l)},\log\alpha_j)\}_{j\in J_K(R)}.
\]

Alpha 的职责限制为：

- 在有限预算下提供 K 个候选；
- 作为 attention logit bias；
- 在推理采样模式中作为 donor proposal distribution。

若 K 从 5 增至 10/20，性能应当近似稳定或逐渐饱和，而不是设计模式突然改变。该稳定性是“Alpha 主要缩小空间”的直接可证伪预测。

### 2.4 空间局部的集合融合

首版只在当前 RSI 实际消费的 48×48 与24×24两尺度实现。对每个位置 \(x\)：

\[
q_x^{(l)}=P_q^{(l)}(E_c^{(l)}(B_{0,c})_x),
\]

\[
k_{j,x}^{(l)}=P_k^{(l)}(d_{j,c,x}^{(l)}),\qquad
v_{j,x}^{(l)}=P_v^{(l)}(d_{j,c,x}^{(l)}),
\]

\[
\beta_{j,x}^{(l)}=
\operatorname{softmax}_{j}
\left(\frac{q_x^{(l)\top}k_{j,x}^{(l)}}{\sqrt d}
+\lambda_\alpha\log(\alpha_j+\epsilon)\right),
\]

\[
\Delta_{set,x}^{(l)}=
\sum_j\beta_{j,x}^{(l)}v_{j,x}^{(l)}.
\]

与旧 mean-Delta 的关键区别是：候选轴不会在进入模型前被全局压扁，不同位置可以得到不同聚合权重。该公式仍是 soft mixture，不能单凭结构声称保持离散模式或全字 donor 一致性。

实现建议：

- K=10；内部维度32或64；逐尺度共享 donor 投影；
- donor mask 必须真实进入 softmax；
- \(\lambda_\alpha\) 首版固定1，不在 test 上搜索；
- 新分支初始化 RNG 独立；
- Alpha、bank 和 cache 全部 stop-gradient。

每个尺度的最小张量合同：

| 项 | Shape / 操作 |
|---|---|
| donor geometry | \(D_l:[B,K,4,H_l,W_l]\) |
| fused candidate | \(\Delta_l:[B,d_l,H_l,W_l]\) |
| offset | \(o_l=\operatorname{OffsetHead}_l([\operatorname{skip}_l,\Delta_l]):[B,18,H_l,W_l]\)，首版 3×3 DCN |
| warped feature | \(\operatorname{DCN}_l(\operatorname{skip}_l,o_l):[B,C_l,H_l,W_l]\) |
| projections | \(Z_w:C_l\to C_l,\; Z_v:d_l\to C_l\) |

48/24 每尺度只挂到一个预先指定的 RSI skip；两尺度 head 不共享参数。若同尺度未来扩展到多个 block，必须另开版本而不是隐式复用。

### 2.5 候选双路径注入：warp + value

只用 RSI offset 会迫使所有变化通过 warp skip 间接产生，不适合新增/消失孔洞或装饰。候选完整配置采用同一 \(\Delta_{set}\) 的双路径：

\[
\text{skip}'=\text{skip}
+Z_w(\operatorname{DCN}(\text{skip},o_\Delta)-\text{skip})
+Z_v(\Delta_{set}).
\]

- **warp path**：保留 RSI，负责宽窄、斜势、位置和连续形变；
- **value path**：一个轻量投影直接提供创建/消失和边界修正信号；
- \(Z_w,Z_v\) 为独立1×1输出投影，记录梯度、更新量和输出范数；
- 首版不增加新 attention block，不增加额外 target-font 图像。

当前尚未批准 value path 进入最终方法。工程默认先实现 warp-only；value 作为 D7 独立 treatment，只有在预注册的创建/消失题本上带来增量且不破坏 Identity/伪影非劣约束时，才升级为完整配置。

为保持与 F0 的可比起点，输出投影可以 zero-init，但其上游投影不能全部同时 zero-init；smoke 必须确认首步输出层有梯度、若干步后上游也开始更新。

### 2.6 推理模式

正式主结果使用 deterministic soft set fusion。附加多解展示允许：

\[
j\sim\alpha,\qquad \Delta=d_{j,c},
\]

或从局部 \(\beta\) 中采样，生成多个兼容候选。多解采样只作能力展示；主表必须固定随机种子和 deterministic 模式。

### 2.7 Delta 必要对照

| ID | 对照 | 回答的问题 |
|---|---|---|
| D0 | no-Delta，保留同架构空分支 | Delta 总体是否有用 |
| D1 | 旧 Ec mean-Delta | 当前基线 |
| D2 | geometry mean-Delta | 去 appearance 是否改善风格冲突 |
| D3 | geometry Set-Delta | 保留变化集合是否优于提前平均 |
| D4 | same-bank absolute set，不减 B0 | 相对固定锚点的 residual 是否必要 |
| D5 | D3 + uniform Alpha bias | Alpha 排序还是集合本身起作用 |
| D6 | D3 + wrong-char set | 是否利用目标字符对齐 |
| D7 | D3 warp-only / value-only / dual | 增益来自何种读取接口 |
| D8 | global soft / local soft / whole-glyph top1 / local hard | 提升来自延迟聚合还是普通容量/混合 |
| D9 | K candidates concat + 等容量 Conv | attention 是否只是更强融合器 |

最小论文必须有 D0/D1/D2/D3/D4/D6，并保证 D2/D3/D8/D9 参数量和 FLOPs 尽量匹配。D5用于 Alpha 主张；D7若保留 dual injection 必须进入正文。还需做 donor 顺序置换、donor/font-ID leakage probe，以及孔洞/连通分量/模式完整性和 donor assignment 一致性诊断。

### 2.8 Delta 停止条件

- D2 不优于 D1：不能声称 appearance canonicalization 必要，保留为诊断。
- D3 不优于 D2：删除 Set 融合复杂度，保留 geometry mean-Delta。
- D4 不劣于 D3：删除 residual 必要性，只称 same-character retrieved geometry prior。
- K=5/10/20结果高度不稳定：不得声称 Alpha 只负责缩小空间。
- wrong-char 与正确 Delta 无可辨差异：当前网络没有按预期利用目标字符变化先验。
- Value path 只改变墨量或造成伪影：删除 value path，不把非零响应包装成拓扑能力。

## 3. Ref 最终路线：Graphics-Informed Local Reference Attention

### 3.1 当前 P 方案的边界

逐 ref pooled h token 能保留 ref 轴差异，但 h 已经丢失该 ref 的空间位置。它适合全局风格，却不足以单独承担顿笔、端点、连断和飞白。

`1dba686e` 已把 P 的训练侧 per-ref token、padding mask、launch/queue和专用 probe wrapper 入库，说明 R1不再需要从零设计；但正式 `sample.py` 尚未接 `style_seq`，现有 F2@40k vs F2-P@25k probe只有图像、没有量化或人工裁决，不能视为 R1效果结论。

最终 Ref 路线保留 P 的 global per-ref tokens，同时新增局部 reference field；不再沿 ref 轴或空间轴提前平均。

### 3.2 Global Ref Tokens

对每个有效 ref \(R_i\)：

\[
g_i=P_g(\operatorname{norm}(h_i)),\qquad i=1,\ldots,n.
\]

- 训练 \(n\sim U\{1,\ldots,8\}\)，推理固定 ordered ref8；
- 真实 padding mask；
- global tokens 负责字重、整体斜势、圆方、宽窄和总体笔刷气质；
- 不使用 mean token 替代每-ref序列。

### 3.3 Graphics-Informed Local Attention

从同一 ref 的 Es 12×12中层特征取得 learned local value，池化至4×4，每 ref 16个位置：

\[
a_{i,p}=E_s^{12}(R_i)_p.
\]

同时从 ref 原图/SDF提取固定局部结构描述：

\[
u_{i,p}=[\text{ink},\text{radius},\cos2\theta,\sin2\theta,
\text{endpoint},\text{junction}].
\]

正式主路线采用严格 Key/Value 分离：

\[
k_{i,p}=P_k(u_{i,p}),
\qquad
v_{i,p}=P_v(a_{i,p}).
\]

目标字符由固定中性锚点 \(B_{0,c}\) 产生局部结构 Query；全文 Content 与 \(B_{0,c}\) 指同一张 native96 neutral render：

\[
q_{c,x}=P_q([E_c(B_{0,c})_x,u_{0,c,x}]).
\]

然后：

\[
r_{c,x}=\operatorname{CrossAttn}
(q_{c,x},\{k_{i,p},v_{i,p}\}_{i,p}).
\]

结构描述作为 Key，learned Es feature 作为 Value：目标字符的竖笔位置可以读取 refs 中相似方向/半径/端点的笔触风格，而不是按中文与拉丁的绝对坐标硬对应。主路线不把 \(x,y\) 放入 Key；含坐标版本只能作为消融，避免位置捷径。

局部分支使用独立的 PrimitiveCrossAttention(Q,K,V,mask)，不把 raw local tokens 与 global tokens 混入同一个既有 context：

- \(A:[B,N,16,C_s]\)，\(U:[B,N,16,6]\)；
- 对每个指定 up block \(l\)，\(Q_l:[B,H_lW_l,d_l]\)，\(K_l,V_l:[B,16N,d_l]\)；
- 输出 reshape 为 \(R_l:[B,C_l,H_l,W_l]\)，经独立 zero-init \(1\times1\) gate 加到该 block；
- global \(n\) tokens 仍走既有 up-path style cross-attention；
- down-path旧 style map保持接口不变，但在 CFG drop 时与所有 Ref 条件同步清零。

首版推理 ref8 共128个 local K/V。这样“graphics key / learned style value”是可编码的机制约束，而不是事后解释普通 context attention。

### 3.4 为什么这一修改仍符合主叙事

- Ref token 全部来自原 episode ref8，没有增加观测预算；
- graphics descriptor 只帮助跨字符匹配局部原语，不提供目标字体未观察到的新设计；
- learned Value 承载目标字体的实际笔触，而不是把 bank donor appearance 当作目标风格；
- Delta 与 Ref 不共享 Value：Delta 是几何可能性，Ref 是目标风格证据。

### 3.5 Ref 训练约束

- Es/Ec 首轮继续冻结、cache-only；新增 Es12 cache 绑定 F0 encoder SHA；
- 一次 CFG Bernoulli drop 必须同步关闭 global/local ref、旧 down-path style map、Alpha bias和整个 Delta branch；所有 bias 经 mask/零门控检查，不允许目标字体条件泄漏。若未来选择保留 Delta，只能称 partial-condition CFG，并另写公式；
- local token 投影采用独立初始化 RNG；
- attention 必须导出按 ref、方向、radius、endpoint类别的使用统计；
- 不启用 own-font support；
- 不在第一版同时增加 Es48纹理 token、VGG、FFT loss或新 ref selector。

### 3.6 Style 监督

主损失保留 diffusion epsilon loss。首版固定 \(\lambda_s=0\)。只有 R3 接口本身阳性但笔触仍弱时，才把 family-balanced local style contrastive loss作为第二阶段独立 treatment：

\[
L=L_\epsilon+\lambda_pL_{perc}+\lambda_oL_{offset}
+\lambda_sL_{local-style}.
\]

- 正样本：生成 x0 与同字体 refs；
- 负样本：同字符或相近墨量的不同字体，避免靠字符/黑度作弊；
- 只在预先固定的低/中噪声区间计算，防止高噪声 x0 估计主导训练；
- \(\lambda_s\) 只在 val 上用短 sweep 冻结，不能看 test16；
- 必须另有等容量无 style loss 对照，避免把 loss 与接口贡献混在一起。

如果时间不足，第一阶段不加该 loss，先测试局部接口本身；接口阳性但笔触仍弱时再加，而不是同一 run 同时改两处。

### 3.7 Ref 必要对照

| ID | Ref 条件 | 回答的问题 |
|---|---|---|
| R0 | global9 mean | 当前旧基线 |
| R1 | per-ref pooled h | ref轴去平均的增量 |
| R2 | R1 + Es12 local tokens，无 primitive keys | 普通 local token 增量 |
| R3 | R1 + graphics-keyed Es12 values | 跨字符原语匹配是否有额外价值 |
| R4 | R3，但在相同 ink/radius bin 内 shuffle key | 是否真正使用结构键 |
| R5 | R3 + local style loss | 监督是否改善笔触读取 |

最小论文采用 R0/R1/R2/R3/R4，并加入等预算的 FSFont-style learned local K/V 强基线；R2/R3需等维、等参数。附加 key 消融为 primitive-only（主路线）、learned-only、primitive+learned、with/without xy。R5只有在 R3 接口阳性但风格仍弱时启动。

### 3.8 Ref 停止条件

- R1 不优于 R0：不继续把 ref-axis averaging 宣称为主因，转查 Es 表示与监督。
- R2 不优于 R1：Es12局部信息没有形成可用增量，停止 local 路线并做层级 probe。
- R3 不优于 R2，或 R4 与 R3相同：primitive key 没有被利用，删去 graphics key，保留简单 local tokens。
- 只改善字重而不改善预注册的端点/连笔/轮廓属性：只能称粗风格增强。
- 身份准确率、伪影或可读性超过预注册非劣界：即使 style preference 上升也不得进入主方法。

## 4. Delta 与 Ref 的联合接口

最终生成器的信息流为：

```text
目标 Content/B0 ───────────────→ 字符身份与空间 Query
       │
       ├─ train bank同目标字符 → Set-Delta几何变化集合 → warp/value geometry
       │                         ↑
目标 ref8 ── Es检索 Alpha ──────┘ 仅排序/缩小候选
       │
       └─ global + graphics-informed local Ref ───────→ style attention
```

统一生成式写为：

\[
\hat y_c=
G_\theta(
x_t,
C_c,
\mathcal D_c(R),
\mathcal R(R)
),
\]

其中 \(\mathcal D_c(R)\) 是候选结构变化集合，\(\mathcal R(R)\) 是观测风格场。

这不是“Delta、Ref、Support 三个模块并列”，而是一个完整推断过程：

1. bank residuals propose；
2. ref-conditioned Alpha 缩小 proposal，独立 Ref path 约束最终视觉实现；
3. diffusion realizes。

## 5. 最小联合实验矩阵

### 5.1 零训练诊断

Phase 1A 使用已入库 F2@80k checkpoint，固定 Content、ref、噪声、sampler、CFG：

- Delta：mean / zero / top1 / oracle legal donor / wrong-char；oracle legal donor 只在冻结 val 题本中，从具备该目标字符的 train-bank donor 里按预注册 GT-SDF 距离选最小者，仅作 upper bound，不参与正式方法选择；
- Ref：true ref / wrong-font ref / single-ref / mean8；
- 记录 source→offset→warp correction→epsilon/x0→最终图的逐段响应。

Phase 1B 在 R1 最小公共底座实现后执行 per-ref mask、条件清零和 CFG drop 干预；不得把旧 F2 不具备的接口写成其 checkpoint 已支持的能力。

该阶段只定位问题，不能替代训练 treatment effect。

### 5.2 20k工程筛查：先 Delta，后 Ref

为避免一次同时改变 Delta 与 Ref 而无法归因，筛查分两轮串行进行。

**Round D：固定 R1 per-ref global，只比较 Delta。**

| Arm | Delta | Ref | 核心比较 |
|---|---|---|---|
| DS-A | D0 no-Delta | R1 | 空分支基线 |
| DS-B | D1 old Ec mean | R1 | 旧 Delta 总效应 |
| DS-C | D2 geometry mean | R1 | DS-C vs DS-B：去 donor appearance |
| DS-D | D3 geometry Set-Delta | R1 | DS-D vs DS-C：保留候选轴 |

**Round R：固定 Round D 的预注册赢家，只比较 Ref。**

| Arm | Delta | Ref | 核心比较 |
|---|---|---|---|
| RS-A | D* | R1 per-ref global | global-only基线 |
| RS-B | D* | R2 learned local Es12 | RS-B vs RS-A：局部信息增量 |
| RS-C | D* | R3 graphics-informed local | RS-C vs RS-B：graphics Key增量 |

Round D 的赢家 D* 按固定顺序选择：先满足 Identity/伪影非劣约束，再比较预注册 style/geometry主指标；若差异不足最小效应阈值，选择更简单的 arm。Round R 同理。

所有 arm：同 F0、同 seed3407、同40k scheduler horizon、同 batch/data order、同 source/CFG draw；新模块初始化使用独立 generator。20k 是同一40k run的中期检查，不在20k看图后重开并挑选新超参。单 seed筛查只决定是否继续，不作为论文稳定性结论。

### 5.3 正式裁决矩阵

正式训练前冻结唯一 parent checkpoint SHA、trainable/frozen module清单、optimizer、scheduler、batch/accumulation、数据 manifest、题本与 primary endpoint。下面的40k机制/消融矩阵先使用筛查 seed3407；只有最终主比较子集运行至少3个预注册 seeds。单 seed 结果只能支撑机制筛查，不能称稳定性结论。

论文证据集合必须覆盖以下项目，但不要求20k已经明确失败的探索臂全部继续到40k：

- Delta：D0/D1/D2/D3/D4/D6；若保留 dual path，再加 D7；D3-D2为 Set 增量，D3-D4为 residual 增量；
- Ref：R0/R1/R2/R3/R4与等预算 FSFont-style local attention；R3-R2为 graphics descriptors 增量；
- 识别性：wrong-donor × wrong-ref 2×2、正确/错误字符 Delta、路径清零与 value/warp交换；
- 公平性：关键对照 matched parameter/FLOPs、相同有效 refs、相同随机 draw和训练 endpoint。

40k集合固定包含 D0，以及每个保留主张的直接 treatment/control 对；若声称 residual 必要性则加入 D4，若声称 Delta利用目标字符则加入 D6，若保留 graphics Key则加入 R4和等预算 FSFont-style baseline，若保留 dual path则加入 D7。未入选路线保留其20k结果并明确标为 screening negative。

推荐确认顺序：

1. 每个拟保留主张的 treatment 与直接 matched control 必须成对从20k继续到40k，不重启：若保留 D3 主张则至少继续 D2+D3；若只保留 D2则继续 D1+D2；若保留 R3主张则继续 R2+R3，若只保留 R2则继续 R1+R2；D0作为完整方法总效应基线保留；
2. 按保留主张补条件性必要对照：residual→D4，Delta→D6，graphics Key→R4与FSFont-style baseline；
3. 若 dual injection 保留，补 D7 warp-only/value-only/dual；
4. 最后只对主方法和最强必要基线补3个预注册 seeds；不为已经明确失败的探索臂补种子。

Go/No-Go 数值阈值、bootstrap CI和 identity/伪影非劣 margin 必须在看正式结果前写入独立 protocol；本设计稿不凭空替代 pilot variance 给出数值。

## 6. 评价指标

### 6.1 主指标

- 盲法 family/style preference，按字体聚类 bootstrap；
- Identity accuracy / OCR；
- E12 family compatibility，仅在验证通过后进入正式主表；
- LPIPS、SSIM、L1只作像素诊断。

### 6.2 几何与笔触分层指标

- boundary Chamfer / SDF distance；
- hole、connected component、endpoint error；
- skeleton stroke-radius distribution；
- slant/orientation distribution；
- 局部属性盲评：端点、连笔、空心轮廓、笔重对比、飞白/干刷；
- 纹理必须与抗锯齿和破碎伪影分开标注。

### 6.3 信息缺失分层

- **ref-observable**：refs 已展示的属性，评价是否迁移；
- **bank-supported but ref-unobserved**：评价兼容性/多候选，不要求唯一复刻 GT；
- **bank-unsupported**：标记不可判，不把失败归咎于模型读取。

### 6.4 角色识别诊断

- 用 Delta 单独预测 donor/font ID、字重、衬线和端点类别，量化轮廓风格泄漏；
- 对 warp/value输出分别清零、置换，按“可连续 warp”与“需创建/消失”题本报告响应；
- 交换字体 A 的 graphics keys 与字体 B 的 learned values，检查输出 appearance是否跟随 Value；
- 在 ref-observable / ref-unobserved 属性上分别报告，避免把缺少信息误判为读取失败。

## 7. 创新性防御

最接近的强先例包括 FSFont/VQ-Font 的 local style cross-attention、CF-Font 的 basis/content fusion、DG-Font 的 deformable skip、NTF 的创建/消失建模和 VecFontSDF 的 SDF 字体表示。因此本文不靠组件名宣称新颖；创新结论只绑定于下列可证伪增量：候选轴保留到空间聚合是否优于等容量 mean/global/Conv，固定中性锚点 residual 是否优于 absolute set，graphics descriptors 是否优于等预算 learned local attention。

### 攻击一：只是 retrieval + cross-attention

答辩：本文保留同一目标字符相对固定中性锚点的 donor candidate axis，直至目标空间位置上的条件聚合。核心对照是 mean-Delta、global/local aggregation、absolute/residual set和等容量 Conv，而不是仅比较有无 retrieval。deterministic 路线不称严格多模态。

### 攻击二：local style attention 已有先例

答辩：不把 local attention 单独列为贡献。Graphics Key 与 Style Value 分离，是为了让中文参考与拉丁目标通过方向、半径、端点等跨字符原语建立对应；它承担目标字体可观察证据的独立注入，而不宣称显式选择 Delta donor。

### 攻击三：SDF/骨架不是新方法

答辩：承认表示原语已有。本文的贡献不是发明 SDF，而是用规范化几何 residual set 隔离 bank 的结构先验与目标 ref 的 appearance evidence，并用 matched controls 检验该分解是否必要。

### 攻击四：仍然无法知道作者真实会怎样设计 i/1

答辩：接受不可辨识性。方法输出与 refs 兼容、受 bank 支持的候选，而不是宣称恢复不可观察的唯一真值；多解由 Set-Delta 的 mode/sample 输出表达。

## 8. 实施顺序与交付物

### Phase 0：同步与冻结合同（预计0.5–1天）

1. 审阅 `1dba686e` 已入库的 P 方案训练/model/mask实现；补齐正式 `sample.py` 的 `style_seq`接线和专用 mask/CFG回归测试，并把执行机当前 run状态、step、checkpoint路径与启动命令写入 `provenance/runs/<RUN_ID>.json`，摘要更新 `PROJECT.md`；没有活跃 run或checkpoint时显式填 `NONE`。P仍只作为接口参考，新方法不继承P权重；
2. 从已入库 F0@100k 建立独立 variant，冻结 parent SHA、数据 manifest、trainable/frozen modules、optimizer、scheduler、batch/accumulation、drop draws和 seed；
3. Support 从论文配置和默认 CLI 中退出，额外观测模式必须显式命名；
4. 落地真实 attention mask、CFG 对 global/local Ref、Alpha和 Delta 的同步 drop，以及新分支独立初始化 RNG；
5. 冻结 val 题本、ref-observable 属性标签、Identity/伪影非劣约束、主指标排序和20k继续/停止规则。

**交付物：** 新 variant 骨架、配置样例、shape/mask/drop/RNG 回归测试、preflight 输出和执行机状态登记。上述项目未通过，不启动训练。

### Phase 1：现有接口的零训练因果诊断（预计1天）

1. 固定 checkpoint、输入、噪声和 sampler，比较 mean/zero/top1/oracle/wrong-char Delta；
2. 对 Es 3×3、pooled h、Es 12×12做 ref-observable 属性 probe；
3. 比较 true/wrong-font ref、single/mean ref，并逐段清零旧 F2 已存在的 source、offset、warp correction和 condition；
4. 把主要问题归入 Delta mixing、Ref representation、读取接口或信息缺失，决定后续优先级。

**交付物：** 干预响应表、固定样例图、probe 结果和一页决策记录。该阶段用于定位，不宣称训练收益。

### Phase 2：R1公共底座、Delta实现与 Round D（预计编码2–3天，随后训练至20k）

1. 将已审 P 实现中的 per-ref pooled h与真实 padding mask移植到新 variant，作为所有 Round D arm 共用的 R1；补齐同步 CFG drop并完成 Phase 1B 无训练干预；
2. 生成带 renderer/font/B0/code hash 的 geometry cache；
3. 实现 D2 geometry mean，验证 cache 和现有 RSI 接口；
4. 再实现 D3 Set-Delta，保持 donor candidate axis 到48/24局部融合；
5. 首轮先接 warp path；value path 独立成 D7 开关，不能与 Set-Delta 同时无对照地引入；
6. 运行 DS-A/DS-B/DS-C/DS-D，同一40k horizon 在20k做预注册中期筛查。

**交付物：** D0–D3配置、单元/梯度/干预测试、20k指标与可视化、D*选择记录。赢家继续训练，不从20k重启。

### Phase 3：Local Ref实现与 Round R（预计编码2–3天，随后训练至20k）

1. 复用 Phase 2 已验证的 R1，不再重新实现或训练同配置基线；
2. 构建 Es12 cache并实现 R2 learned local K/V；
3. R2相对R1出现局部属性信号后，再实现 R3 primitive Key + learned style Value；
4. 固定 Round D 赢家 D*，运行 RS-B/RS-C；RS-A 与 Round D 的 D*+R1 是同一配置，直接复用原 run/checkpoint，不产生重复训练；首轮 local style loss 权重固定为0。

**交付物：** R1–R3配置、mask/置换/梯度测试、20k指标与可视化、R*选择记录。

### Phase 4：40k必要对照与联合确认

1. 将每个保留主张的 treatment/control 对从20k继续到40k：D3+D2或D2+D1，R3+R2或R2+R1，并保留 D0 总效应基线；
2. 按保留主张补 D4 residual-vs-absolute、D6 wrong-character、R4 matched-key-shuffle与FSFont-style baseline；
3. 若保留 dual injection，补 D7 warp-only/value-only/dual；
4. 完成 wrong-donor × wrong-ref、路径清零、key/value交换和 donor assignment 一致性诊断。

**交付物：** 40k冻结表、matched parameter/FLOPs 核对、机制消融和 Go/No-Go 记录。只有单模块增量成立，联合配置才进入正式候选。

### Phase 5：正式结果

1. 只对完整方法与最强必要基线组成的正式主比较子集运行至少3个预注册 seeds；其余40k机制消融保持单 seed3407；
2. 正式测试集只在配置冻结后运行一次，按字体聚类 bootstrap；
3. 同时报告 Identity、几何/笔触分层指标、盲评和 provenance；
4. 按 ref-observable、bank-supported but unobserved、bank-unsupported 分层解释结果。

**交付物：** 主表、消融表、盲评材料、可复现实验清单和最终 claim ledger。

## 9. Go / No-Go 总判决

最终完整方法进入论文主线需要同时满足：

1. Set-Delta 相比 geometry mean 或旧 mean 在预注册风格/几何指标上有稳定增量；
2. anchor-relative residual set 相比 absolute set 有可辨优势，才能保留 residual 必要性声称；
3. graphics-informed Ref 相比普通 learned local token 有增量，或删除 graphics key 后以简单 local Ref进入系统；
4. wrong-char Delta、shuffled primitive key产生符合方向的性能下降；
5. Identity和无伪影达到非劣界；
6. 所有正式方法只读取相同 ref8目标字体图像；
7. 结果能够按 ref-observable、bank-supported、unsupported 三类解释。

若其中2失败，主叙事收缩为“retrieved same-character prior”，不再强调 residual。若3失败，删除 graphics key但不影响 Delta 主叙事。若1失败，保留 Ref路线并停止把 Delta 作为核心贡献。负结果不会迫使整个项目同时崩塌。

## 10. 效果可验证性与预期

该方案不能在训练前保证最终视觉质量一定提升；它能保证的是每个变化都具有可观测的机制响应和独立对照，因此不再依赖“换了模块但不知道输出为何变化”。

必须先通过的工程/机制信号包括：新分支存在梯度和参数更新；true/zero/wrong 条件产生方向一致的响应；mask、CFG drop和候选置换行为正确；warp与value路径的输出范数、空间位置和最终影响可追踪。通过这些检查仍不等于视觉质量提升，视觉收益只由20k/40k matched runs与盲评确认。

当前优先级判断：

| 修改 | 最可能改善 | 预期与不确定性 |
|---|---|---|
| R1 per-ref global | 粗粒度字重、斜势、圆方 | 实现直接，预期较稳，但不能解决细笔触 |
| R2 Es12 local | 端点、连笔、局部轮廓 | 有中等成功可能，取决于 Es 中层是否保留属性 |
| D2 geometry mean | donor纹理/颜色串入 | 对降低 appearance 冲突最直接 |
| D3 Set-Delta | 模式相消、局部候选选择 | 是核心研究变量；是否优于 D2 必须实测 |
| R3 graphics Key | 跨字符局部对应 | 可能有增量，但必须超过等预算 learned-local 才保留 |

主成功标准不是单一 L1 下降，而是 Identity/伪影不劣，同时 ref-observable 风格属性、几何完整性或盲评至少一个预注册主端点改善。

## 11. 不及预期时的优化树

1. **分支几乎无响应：** 先查 gate/output projection、梯度、CFG同步 drop、mask和注入层；不先加损失或扩大模型。
2. **Delta改变了错误风格：** Ec residual切换为TSDF/gradient/mask，降低 Alpha bias，做尺度/半径归一化；若 value path 主要改墨量，退回 warp-only。
3. **结构改善但笔触仍平：** 按 R1 → R2 → R3递进；若 Es12 probe阳性但生成端无收益，再试 Es24/48或局部 style loss，而不是同时更换编码器。
4. **风格改善但Identity或伪影变差：** 降低注入 gate，限制边界带，分离 warp与value职责，并增加几何一致性约束。
5. **Set不如geometry mean：** 先测 global/local soft、whole-glyph top1、hard/Gumbel与 assignment consistency；仍无增益则回退 D2。此时主线仍可保留“检索同字符几何先验”，但删除“候选轴是必要贡献”的声称。
6. **所有local Ref均无增益：** 做 Es层级 probe；若中层确实无相关属性，再训练轻量局部 style encoder。若 refs本身未展示该属性，则转为 bank-supported多候选问题，不把它误判为读取失败。

回退顺序保持主叙事稳定：先简化 Set为 geometry mean，再简化 graphics Key为 learned local，最后才判断 Delta 或 local Ref 整条路线无效。

## 12. PI REVIEW REQUIRED

下列项目会改变计算量、正式口径或实现接口，当前作为明确提案，不隐藏成默认事实：

| ID | 待拍板项 | 推荐默认 | 影响 |
|---|---|---|---|
| PR-1 | 完整方法是否保留 warp + value 双路径 | 实现先 warp-only，value以 D7 独立加入；仅在拓扑题本有增量时进入完整方法 | 决定接口与正文消融数量 |
| PR-2 | graphics Key 的主配置 | primitive-only Key + learned Es12 Value；R2 learned K/V为等预算基线 | 决定创新点是否能独立归因 |
| PR-3 | 筛查顺序 | 先 Round D，后固定 D* 跑 Round R；20k是同一40k run中期点 | 降低联合改动的归因歧义 |
| PR-4 | 正式多种子预算 | 40k机制矩阵用seed3407；仅完整方法和最强必要基线补至少3 seeds | 平衡稳定性与训练成本；旧F1/F2/F3单seed决定不约束新方法 |
| PR-5 | Go/No-Go数值阈值 | 先批准制定流程；读取冻结 val pilot variance 后、40k结果前锁定具体数值 | 目前不能无依据填造 margin |
| PR-6 | P方案与新方法的关系 | 审计已入库P接口，补正式sample、回归测试与run provenance；新方法仍以仓库F0@100k为唯一parent | 复用R1工程资产但不继承P权重或未裁决结果 |
| PR-7 | 正式 seed 编号 | 数量先定为至少3；具体编号由PI预注册 | 避免沿用旧F1/F2/F3单seed决策时产生歧义 |

在 PI 回复前，可执行 Phase 0/1与不依赖上述选择的 cache/interface 工作；不得把推荐默认写成已批准结论或启动正式主表。

## 13. 最终推荐配置

```yaml
method: hrfont_setdelta_graphicsref
target_observation: ref8_only
support: false

delta:
  representation: tsdf_grad_mask
  tsdf_clip_px: 8
  scales: [48, 24]
  candidates_k: 10
  preserve_candidate_axis: true
  alpha_role: attention_logit_bias
  fusion: per_location_donor_attention
  implementation_order: [rsi_warp, residual_value_ablation]
  final_injection: PI_REVIEW_REQUIRED

reference:
  global: per_ref_pooled_h
  local_source: es_12x12
  local_grid_per_ref: [4, 4]
  graphics_key: [ink, radius, cos2theta, sin2theta, endpoint, junction]
  style_value: learned_es_local
  train_n_refs: uniform_1_8
  inference_refs: ordered_ref8
  attention_mask: required

training:
  parent: same_f0
  screening_seed: 3407
  formal_seed_count: 3
  formal_seed_ids: PI_REVIEW_REQUIRED
  endpoint: 40000
  interim_review: 20000
  screening_order: [delta_round, ref_round]
  encoder_runtime: cache_only
  independent_init_rng: true
  matched_data_and_drop_draws: true
  local_style_loss_weight: 0
  cfg_drop_scope: [global_ref, local_ref, old_style_map, alpha_bias, delta_branch]
```

一句话最终版本：

> **HR-Font 保留同一目标字符相对固定中性锚点的 donor residual 轴，直到目标空间位置上的条件聚合；同时以图形描述符引导的局部 reference attention 提供目标字体可观察的 appearance evidence，从而生成受 bank 支持、与 refs 兼容的跨语系设计。**
