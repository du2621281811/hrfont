# Delta 减中性 Content 与 RSI 接入的设计理由

## 结论

应使用

\[
\Delta_c=\sum_s\tilde\alpha_s E_c(A(B_s,c))-E_c(A(B_0,c))
\]

作为 RSI 的结构源。这里必须是**同字符、特征空间内加权后相减**：先分别编码各个 `A(B_s,c)`，再按 alpha 混合多尺度特征；不能把像素先混合再编码。Delta 不是第二份内容输入，而是相对固定中性底 B0 的、与目标字符空间对齐的变化条件。

## 问题一：为什么 Delta 要减去中性 content？

### 1. 移除身份/内容公共分量

绝对特征 `Ec(A(B_s,c))` 的主导因素仍是字符 `c` 的身份和几何结构，而 MCA 已从 `Ec(C)` 获得同一个 `c` 的身份条件。若 RSI 再接收绝对内容特征，就会重复输入“这个字长什么样”，并把身份信息混入“这个字需要怎样改变”。减去同字符的中性 `Ec(A(B0,c))`，会抵消两者共享的大部分内容分量，避免 RSI 与 MCA 争夺身份建模职责。

### 2. 把信号集中到相对外观变化

差分回答的是：在 alpha 选出的风格近邻中，同一个 `c` 相对中性底偏离了什么。它把能量集中到笔画宽度、曲率、端点、字腔和局部比例等外观差异上；当近邻与中性底一致时，Delta 自然趋近 0。固定 B0 也使不同字体、字符和 run 的变化都处在共同坐标系中。

### 3. 与 CFG/unconditional 语义一致

Delta-drop 时将该分支置零，模型应退化到“没有额外形变引导”的路径；zero-init 的新 offset 接入使 step 0 也近似该基线。这一零点语义清晰：`Delta=0` 表示不提供相对中性底的变化，而不是提供某个仍带字符身份的绝对结构。

### 反事实：不做减法会怎样？

若直接把 `sum alpha Ec(A(B_s,c))` 送入 RSI，其数值与空间结构会主要表现为 `c` 的内容特征。RSI 便退化成官方 `Ec(S)` 接线的同类问题：用另一份绝对内容骨架驱动 offset，只是把汉字 `S` 换成库中的 `c`；alpha 的风格近邻信息容易被强内容公共分量淹没。此时 drop 分支也不再对应自然的“无变化”原点，RSI 与 MCA 重复，无法清楚归因收益来自相对风格变化。

## 问题二：这样的 Delta 用 RSI 嵌入合适吗？

### 为什么架构上匹配

RSI 的 `OffsetRefStrucInter + DCN` 不是通用风格编码器；它要求结构源能与内容 skip 的空间位置建立有意义的对应，再据此预测 deformable-convolution offset。官方同语系场景用 `Ec(S)`，成立的隐含前提是 S 与 C 是同一个字符。跨语系时 R 是汉字、C 是拉丁字母，`Ec(R)` 的汉字骨架与拉丁 skip 错位；交叉注意力据此产生的 offset 没有正确的位置语义，这是官方接线跨语失败的直接机制。

Delta 则由**同一个目标字符 c** 的近邻字体特征减去中性特征得到。它的多尺度空间网格与 identity skip 指向相同字符部位，RSI 的交叉注意力可以在正确位置上问“这里相对中性应怎样移动”，再由 DCN 形变 skip。因此 Delta 是 RSI 结构源的合适替换；用户参考 R 的真实风格仍独立走 `Es(R)` 风格交叉注意力，不由 Delta 取代。

### 三处必须诚实保留的张力

1. **Delta 并非纯 where。** 它是同字符特征差，offset 的方向和幅度本身已经包含粗粒度的“怎么变”。论文只能表述为“架构主要将 Delta 用作定位/对齐信号”，不能宣称它完全不含 what；应由 E9 接入与破坏实验验证。
2. **尺度分布改变。** Delta 能量通常比绝对 `Ec(S)` 小，且各层分布不同。offset 接入/末端必须 zero-init，并允许经预注册的 scale adapter 校准；Delta-drop 保证无 Delta 时可退化。任何破坏实验都须逐层匹配正确 Delta 的 RMS，避免能量差成为混杂因素。
3. **Ec 的西文字形质量未知。** 官方 Ec 在拉丁字符上的局部对应是否可靠尚未实证。E0/E9 的 wrong-char Delta、wrong-style Delta、spatial shuffle 与 magnitude-only 消融，以及 offset 定位指标，必须承担这项验证，不能仅凭架构直觉下结论。

### “Delta 适合 RSI”的预注册判定条件

只有 E9 同时出现以下模式，才支持较强结论：

- spatial shuffle 在严格 per-layer RMS 对齐后显著伤害主要指标，说明空间布局确实被使用；
- magnitude-only 保留大部分正确 Delta 的收益，说明 RSI 对变化强度/粗粒度方向有效，而不是仅记忆库字符身份；
- offset 对 `|GT-B0|` 轮廓变化区的定位 AUROC 高，并有一致的 AUPRC/IoU/correlation 证据。

若 spatial shuffle 不伤害，Delta 更可能只是全局调制；若 magnitude-only 丢失大部分收益，则需要承认细粒度空间 what 也是关键；若定位 AUROC 不高，则不能把 offset 解释为变化区域对齐。

## 论文可直接采用的表述

“We construct a character-aligned residual condition by subtracting the neutral-font content features from an alpha-weighted mixture of same-character neighbor features. This removes the dominant identity component already supplied by MCA and gives zero a natural no-deformation semantics.”

“We replace the cross-script-misaligned `Ec(R)` structure source of RSI with this same-character residual, while retaining `Es(R)` for style cross-attention. We describe Delta as primarily an alignment cue—not a pure where-only signal—and test that interpretation with RMS-controlled interventions and offset localization.”
