# 故事与 Idea 总结（2026-09-11，Delta / Ref 最终设计口径）

> 本文是当前写作口径的一页式入口。完整方法、实现合同、消融和停止条件见
> [HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md](HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md)。
> F1/F2/F3/F3b 是已完成的旧实现证据；Set-Delta / Graphics-Ref 是下一版设计，尚未实现或训练。

## 1. 问题

给定未见字体的少量中文参考字（主协议 ref8），生成同一字族的拉丁、数字、假名或注音。目标字符身份由中性 Content 给出，但 refs 没有直接展示这些字符的具体设计；问题因此不是普通的 reference imitation，而是**在目标字符设计不可直接观察时，怎样结合 bank-supported variation 与 ref-observable appearance，生成合理且与字族兼容的字形**。

## 2. 方法：先验提出，参考实现

**① Set-Delta Variation Prior（核心）**

- 用目标 ref8 在 train-bank 中检索 Top-K 相似字体，Alpha 只承担候选 proposal 与 attention bias；
- 对每个 donor 保留“同一目标字符相对固定 Noto 中性锚点”的规范化边界 residual，不在网络外提前平均；
- 候选轴保留到目标空间位置后再聚合，并通过 geometry adapter 的 warp/value 路径影响生成；
- 因而 Delta 回答：**bank 支持这个目标字符怎样变化**。

**② Graphics-Informed Local Reference Attention（互补核心）**

- 每张 ref 保留独立 global token，不沿 ref 轴提前平均；
- 从 ref 的 Es 中层特征读取 learned local style values；
- 用 ink、stroke radius、orientation、endpoint、junction 等图形描述符作为跨字符匹配 Key；
- 因而 Ref 回答：**目标字体已经展示了怎样的视觉实现**。

**③ Support 的位置**

Support 不进入论文主方法。own-font support 只作为额外观测工程模式或 oracle upper bound，并披露额外目标字体图像预算。

## 3. 与 FontDiffuser 的关系

FontDiffuser 是 inherited denoising backbone，不是本文要重新发明的部分。关键区别是条件信息与推断对象：

\[
\text{FD}: p(y_c\mid B_{0,c},R),\qquad
\text{HR-Font}:p(y_c\mid B_{0,c},\mathcal D_c(R),\mathcal R(R)).
\]

- FD 用 reference glyph 自身同时提供 style 与结构形变线索；
- HR-Font 显式引入同目标字符的 bank residual candidates，并把 bank-supported variation 与 ref-observable appearance 分路建模；
- 新模块称 Target-Character Variation Adapter，不再写成“Delta-RSI：只替换 FD 的 RSI 输入”；DCN 若保留，只是 geometry adapter 的 inherited operator。

推荐论文句式：

> Unlike reference imitation, cross-script completion must design a glyph that is absent from the references. HR-Font addresses this ambiguity by combining a retrieved target-character variation prior with independently observed reference appearance evidence.

## 4. 评测：合理性优先（独立贡献）

现有评测只重像素级相似度（LPIPS 等），**忽视跨语言风格迁移最重要的「合理性」**。我们构建：

- **E12 独立评测器原型**：与方案编码器隔离的 φ_s2、ID-CLS、membership verifier；当前 T2 gate failed，过门前不进入正式主表；
- **三轴指标**：Identity / Style（SC-R、SC-Gap、Rank@1）/ Quality（coverage、LPIPS）；GT 仅作 positive control；
- **设计师人工评测（E11）**：分层 2AFC + MOS，终审「合理性」。

## 5. 声明阶梯（按证据强度）

| 等级 | 声明 | 状态 |
|---|---|---|
| 机制级 | 候选轴保留到空间聚合；bank variation 与 ref appearance 分工 | 📋 最终设计，待正式实现 |
| 结果级 | 相比 FD/ref-only 与 mean-Delta 改善跨语系合理性和风格 | 📋 待 matched retrain |
| 性质级 | bank 策展形成稳定可控的设计偏好 | 📋 待 K/Alpha/curation 干预 |

## 6. 合作者 Review 重点

1. Set-Delta 是否应以 TSDF+gradient+mask 为正式表示，还是先用 geometry mean 做一天级接口验证；
2. Target-Character Variation Adapter 的 warp/value 双路径是否保持，还是先做单路径快速筛查；
3. Graphics-Ref 的 primitive-only Key 是否足够，R2 learned local K/V 是否作为强基线；
4. 正式实现必须包含 D0/D1/D2/D3/D4/D6 与 R0/R1/R2/R3/R4，并把 FD/ref-only、mean-Delta 和 Set-Delta 放在同一主表；
5. 论文主图应把 FD 画成灰色 inherited backbone，把 Set-Delta 与 Graphics-Ref 画成本文方法，避免“只换 RSI source”的误读。
