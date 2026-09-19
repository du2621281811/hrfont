# K6 假设与实验方案：Donor 解耦、反 shortcut 排序监督与 Bucket 相对约束

日期：2026-09-20  
状态：proposal / next-step review  
前置实验：K1、K5-A、K5-B  
目标：在 K5 已经优于 K1、但仍未达到“复刻特效”级别的前提下，判断当前瓶颈究竟来自 RSI / Bank / Donor 的结构先验污染，还是来自训练目标仍允许模型选择“通俗、安全但不够像 GT”的 shortcut，并设计最小代价验证。

---

## 0. 当前结论

基于 2026-09-19 的 K5 固定协议与逐 step 对比，当前人工判断先冻结为：

1. **K5-A 与 K5-B 相比 K1 有提升。**
2. **K5-A 与 K5-B 差距不大，主观上 K5-B 略好。**
3. **两者都没有达到真正复刻目标字体特效 / 局部设计语言的级别。**

因此下一阶段不应该继续只做同方向的轻量 recipe tuning，而应优先验证两个更根本的问题：

- RSI 注入的 Delta 是否被 Bank / Donor 自身风格污染，导致模型拿到的是“某个 donor 的具体实现”，而不是可迁移的结构变化；
- 当前 reconstruction / change-aware supervision 是否仍然允许模型停在一个离 GT 不够近、但视觉上安全和通俗的局部最优。

另一个重要工程事实：当前 K5 的固定协议记录了 donor pool 规模和 family guard，但**没有逐样本归档真正被选中的 donor 列表 / donor 权重 / router 贡献**。因此在修改模型之前，必须先补齐 donor 可观测性。

---

## 1. Idea A：RSI Delta 是否过度依赖 Bank / Donor

### 1.1 核心假设

当前 Dynamic Delta / RSI 路径的期望角色是：

> 从 bank-supported variation 中提供“目标字符应该如何发生结构变化”的 prior。

但如果 Delta 的形成强依赖具体 donor，而 donor 中又混入明显不同的字体风格，则实际注入可能变成：

> target structure prior + donor-specific appearance contamination

尤其在 complex / decorative / hollow / broken-stroke 等字体上，这种污染可能直接和 TC / target style evidence 冲突。

如果这个假设成立，则 K5 中即使增强了 target-conditioned consumption，模型仍然需要一边使用 target reference，一边抵消 donor 带来的错误方向，最终容易回到保守解。

### 1.2 先做诊断，不先改训练

先增加逐样本 provenance：

- selected donor ids / font names；
- 每个 donor 的候选分数；
- router / attention 权重；
- 每层 Delta norm；
- aggregate Delta norm；
- donor family / script / style bucket；
- donor 与 target font 的 style distance；
- donor 与 GT change direction 的一致性统计。

当前只记录 eligible donor 数量是不够的。

### 1.3 最小验证：Donor Swap Sensitivity Audit

冻结 K5-B@10k，不训练。

对同一个：

- content；
- target font / reference；
- target character；
- noise / seed；

仅替换 donor set，构造四类 donor：

1. **Matched donor**：与 target style / family 距离较近；
2. **Neutral donor**：普通、无明显特效的 donor；
3. **Random donor**：合法候选中随机；
4. **Adversarial-style donor**：结构合法，但视觉风格与 target 明显不同。

观测：

- 输出图像两两距离；
- L1 / SSIM；
- D_change / D_add / D_remove；
- GT feature distance；
- style feature distance；
- 特效区域是否随 donor 改变。

### 1.4 判定

如果在 target refs 完全不变时，仅 donor swap 就能显著改变最终视觉风格，尤其改变空心、断笔、装饰、terminal 等局部特征，则说明 RSI 路径承担了过多 appearance 信息。

理想情况应是：

- donor 改变可以影响“结构变形可行方向”；
- 但不应该显著改变 target-specific appearance。

### 1.5 如果假设成立，优先尝试的修正

不建议第一步就删 Bank，而是按下面顺序：

**A1. Delta normalization / centering**

将 donor delta 做幅值归一或相对 anchor 中心化，削弱 donor 的绝对 style amplitude。

**A2. Multi-donor aggregation**

单 donor / 少数 donor 改为多 donor 聚合，提取共同变化方向，降低单个 donor 的 idiosyncratic style。

**A3. Target-aware donor weighting**

target reference 只用于选择 / reweight donor structural evidence，不直接让 donor appearance 进入最终 representation。

**A4. Donor-invariance consistency**

同一个训练样本使用两组合法 donor：

[
\mathcal{L}_{donor-cons}
=
d(\hat y^{(D_1)}, \hat y^{(D_2)})
]

但该 loss 必须只约束 donor 不应影响的 style-invariant / target-conditioned 部分，不能简单全图强行一致，否则可能把 Delta 本身压没。

更稳妥的实现是对：

- target style embedding；
- unchanged region；
- 或 TC-aligned appearance feature

做一致性，而允许必要的结构 residual 有小幅差异。

---

## 2. Idea B：让输出逼近 GT，同时显式远离“通俗解”

### 2.1 判断

这个方向**值得做，而且比继续单纯增大 GT reconstruction 权重更直接**。

但它本质上不应该直接叫 DPO。当前没有 policy/reference model + preference pair 的概率比优化，更贴切的是：

- margin ranking loss；
- triplet loss；
- contrastive preference-style supervision。

核心不是“GT 更近”本身，而是：

> 对同一个字符，模型输出不仅应该接近 GT，还应当比 generic / neutral / shortcut solution 更接近 GT 一个 margin。

### 2.2 Negative 的定义最关键

建议 negative 不使用任意其他字体，而使用**same-content generic negative**：

- neutral content；
- K0/K1 的 generic-like prediction；
- 普通字体渲染；
- 或当前模型弱化 target condition 后得到的 prediction。

必须保证：

- character identity 一样；
- layout 一样；
- difference 主要来自字体设计。

否则模型会通过改变字形身份或几何位置来“远离 negative”。

### 2.3 推荐 loss

在冻结 feature extractor 的空间中：

[
d_{pos}=d(f(\hat y), f(y_{GT}))
]

[
d_{neg}=d(f(\hat y), f(y_{generic}))
]

[
\mathcal{L}_{rank}
=
\max(0, m + d_{pos} - d_{neg})
]

目标是：

[
d(\hat y, GT) + m < d(\hat y, generic)
]

这比“单纯把输出推远 generic”安全，因为它始终以 GT 为锚点。

### 2.4 不建议直接在 raw pixel 上做 repulsion

pixel-level negative loss 很容易诱发：

- 过度加粗；
- 人为断裂；
- 边缘噪声；
- 夸张装饰；
- 通过无意义像素差异满足 margin。

建议 feature 组合优先使用：

- multi-scale edge / gradient feature；
- frozen perceptual feature；
- 当前已有 target-character appearance / Reader feature；
- change-region masked feature。

### 2.5 必须做 GT-distance gating

如果某个 GT 本身就非常接近 generic content，则不应该强迫它远离 generic。

定义：

[
g = \mathbb{1}[d(f(GT), f(generic)) > \tau]
]

最终：

[
\mathcal{L}=\mathcal{L}_{base}+\lambda_{rank} g \mathcal{L}_{rank}
]

只在 target 确实有明显字体设计差异的样本上启用。

### 2.6 这个方案最可能解决的问题

它直接针对目前的失败模式：

> 输出“没错”，但太普通；  
> GT 明明有明显特效，模型却只做一点粗细 / 宽窄变化就交差。

因此它比继续加普通 L1 / perceptual loss 更符合当前现象。

---

## 3. Idea C：Bucket 内做相对约束，防止不同字体输出趋同

### 3.1 原始 idea 的可行性

“先分 bucket，然后 bucket 内额外监督不同样本的推理结果不应该高度相似”这个方向是成立的。

但不建议直接做：

[
- d(\hat y_i, \hat y_j)
]

即无条件鼓励 pairwise prediction 互相远离。

因为这样会造成“为了不同而不同”，不能保证变化方向和 GT 一致。

### 3.2 更推荐：Relative Style Geometry Matching

Bucket 应围绕**same content / different target font**构造。

例如一个 bucket：

- character 相同；
- script 相同；
- content 完全相同；
- target fonts 不同；
- GT 风格跨度可控。

然后约束：

> prediction 之间的相对距离，应当匹配 GT 之间的相对距离。

例如：

[
d_{pred}^{ij} = d(f(\hat y_i), f(\hat y_j))
]

[
d_{gt}^{ij} = d(f(y_i), f(y_j))
]

[
\mathcal{L}_{rel}
=
|d_{pred}^{ij} - stopgrad(d_{gt}^{ij})|
]

也可以只做 margin 版本：

[
d_{gt}^{ij} > \tau
\Rightarrow
d_{pred}^{ij} > m
]

### 3.3 这比普通 contrastive 更适合 HRFont

因为它不是假设所有不同字体都要“尽可能远”，而是保留真实字体空间的几何关系：

- GT 相近，prediction 也允许相近；
- GT 差异大，prediction 必须拉开；
- 差异大小由真实 GT 决定。

### 3.4 和 K1 已有 same-content pairing 的关系

K1 已经做过 same-content / different-font pairing，但当时明确：

> pairing 只改变 batch 构成，不增加 contrastive loss。

因此 Idea C 是一个自然的下一步：

- 保留 same-content pairing；
- 在此基础上真正增加 relational supervision；
- 用 GT pair distance 作为教师，而不是简单 repulsion。

它可以被视为对现有 sampler 的有效利用，不需要重做整个数据体系。

---

## 4. 三个 idea 的关系

这三个方向并不互斥，它们对应三个不同层次：

### A：输入先验是否被污染

解决：

> RSI / Delta 给模型的信息是否本身就带错方向。

### B：单样本是否存在 generic shortcut

解决：

> 即使条件正确，模型是否仍然选择“普通但安全”的输出。

### C：batch / 数据分布层面是否发生 style collapse

解决：

> 对不同目标字体，模型是否系统性收敛到相似输出。

因此不建议一次把三个东西全部塞进一个训练 run；否则无法判断提升来自哪里。

---

## 5. 推荐实验顺序

### K6-0：Donor Sensitivity Audit，0 training

**目的：先回答 Idea A。**

冻结 K5-B@10k，仅修改推理 / logging。

输出固定 panel：

[
Target refs固定
\times
{Matched, Neutral, Random, Adversarial\ donor}
]

重点看 complex / decorative / hollow / broken / unusual connection。

**Go 条件：**

如果 donor swap 明显改变目标风格，则优先处理 RSI / donor 解耦。

**No-Go 条件：**

如果 donor swap 对 appearance 很稳定，则不要浪费主实验预算重构 Bank；直接进入 B/C。

---

### K6-A：GT-vs-Generic Margin Ranking

以 K5-B 为 parent / recipe baseline，只新增：

- generic negative；
- gated margin ranking loss。

其余：

- sampler；
- TC；
- RSI；
- donor policy；
- base losses

保持不变。

建议先 2k 观察：

- generic distance 是否增大；
- GT distance 是否下降；
- identity 是否稳定；
- 是否出现人为夸张 artifact。

如果趋势正确再到 10k。

---

### K6-B：Same-content Bucket Relative Loss

仍从 K5-B recipe 出发，只新增：

- same-content bucket；
- GT relative geometry loss。

重点验证：

- 不同 target fonts 的 output diversity 是否增加；
- diversity 是否和 GT diversity 同方向；
- 是否减少“不同字体看起来像一个模板”的现象。

---

### K6-C：组合

只有 K6-A / K6-B 至少有一个单独成立后，再组合：

- ranking loss；
- relative bucket loss；
- 若 K6-0 显示 donor contamination，再加入最小 donor correction。

不要直接三项一起上。

---

## 6. 推荐优先级

### P0：K6-0 Donor Sensitivity Audit

原因：成本最低，而且决定是否需要动 RSI / Bank 的方法定义。

### P1：K6-A GT-vs-Generic Ranking

原因：最直接对应目前“比 K1 好但仍不够像、仍然糊弄”的现象。

### P1：K6-B Relative Bucket Supervision

原因：和已有 same-content pairing 高度兼容，工程改动可控，而且可以直接约束 style collapse。

### P2：RSI 架构重构

只有 donor audit 确认污染后再做。

---

## 7. 新增评估项

后续仅看 L1 / SSIM 不够，需要增加下面三类指标。

### 7.1 Generic Shortcut Gap

[
G = d(\hat y, generic)-d(\hat y, GT)
]

希望 G 为正且扩大。

### 7.2 Pairwise Style Recovery

同 character 不同 font：

[
R_{pair}
=
corr(
d(\hat y_i,\hat y_j),
d(GT_i,GT_j)
)
]

用于衡量 prediction 是否恢复真实 GT 的相对 style geometry。

### 7.3 Donor Sensitivity

同 target / ref，仅 donor 不同：

[
S_{donor}
=
E[d(\hat y^{D_i}, \hat y^{D_j})]
]

并分别统计：

- all region；
- change region；
- appearance feature；
- identity feature。

理想状态不是绝对 0，而是 appearance sensitivity 明显低于 structure-relevant sensitivity。

---

## 8. 实现纪律

1. K6-0 必须先补实际 donor provenance，避免继续只有 pool-size logging。
2. K6-A 与 K6-B 都必须单独跑，禁止一开始合并。
3. negative / pairwise loss 只在 GT 差异足够明显的样本启用。
4. 不使用 raw pixel 无锚点 repulsion。
5. identity / readability 必须保留 hard guard。
6. 不因为新 loss 数值下降就判成功，最终仍以 fixed matched visual board 为主。
7. complex / decorative panel 必须单独 review，不能被大量普通字体平均掉。
8. 新增这些 loss 先视为 optimization strategy，不自动升级为新的论文 contribution。

---

## 9. 当前推荐判断

目前最值得验证的不是“再加强一点 K5”，而是：

> **模型究竟是拿到了被 donor 污染的结构条件，还是拿到了正确条件但 objective 仍允许 generic shortcut。**

因此下一步建议严格按：

[
K6	ext{-}0 donor audit
\rightarrow
K6	ext{-}A ranking
\rightarrow
K6	ext{-}B relative bucket
\rightarrow
必要时组合
]

执行。

如果 K6-0 证明 donor contamination 很强，则先修 Delta；如果 donor sensitivity 很低，则直接把主要火力放到 B/C，因为那会更符合当前“模型会做一点，但不愿意真正复刻特效”的失败模式。

---

## 10. 一句话定义

**K6 的核心不是继续增加 reconstruction pressure，而是分别排查“错误先验”与“安全 shortcut”，并用 GT 锚定的相对监督迫使不同目标字体真正拉开。**
