# RSI 作用、数据流与损失审查（2026-09-05）

## 结论

1. **RSI 是 FontDiffuser (FD) 自己提出的模块**，不是前序论文已有的同名模块。FD 明确对比的是 DG-Font 的 Feature Deformation Skip Connection (FDSC)；两者都使用 deformable convolution (DCN)，但 offset 的条件来源不同。
2. **FD 原本就使用异字参考。** 官方数据集从目标字体中随机选择参考图，并排除当前目标图。因此，不能把“参考字与目标字不同”写成 FD 的错误。
3. **RSI 的准确作用是形变 UNet 的低层 skip feature。** 它不是直接生成字形，也不是额外的 style encoder；它从参考图的多尺度 \(E_c\) 特征与当前 skip feature 的 cross-attention 中预测 DCN offset，再用该 offset 重采样 skip feature。
4. **offset loss 只约束偏移不要过大，不监督正确方向。** 有用的 offset 主要通过最终扩散重建损失和感知损失间接学习。
5. 我们的 Δ-RSI 仍有合理动机，但应改写为一个**待验证假设**：跨语系时参考字结构与目标字的局部结构共享可能更弱；用字体库估计的同目标字符残差，可能比绝对的汉字参考结构更适合作为 RSI 条件。不能预先写成已证实机制。

## 1. 从 DG-Font 到 FD-RSI

### DG-Font：以“同字跨字体对应”为形变依据

[DG-Font (CVPR 2021)](https://arxiv.org/abs/2104.03064) 的 FDSC 对 content encoder 的低层特征做 DCN 形变。其 offset/mask 由两部分拼接后经 CNN 预测：

\[
\Theta=f_\theta(K_s,K_c),\qquad K'_c=f_{\mathrm{DCN}}(K_c,\Theta).
\]

其中 \(K_c\) 是内容图低层特征，\(K_s\) 是 decoder/mixer 中已经通过 AdaIN 注入 style code 后的 guidance map。论文的几何依据是：**同一个字符在不同字体中通常存在逐笔画的近邻对应**。这里的“同一个字符”指 content 输入与最终生成结果的字符身份相同，并不要求 style reference 本身也是这个字符。FDSC 不读取 style reference 的空间 \(E_c\) 特征，而是直接形变目标字符自身的 content skip。

DG-Font 官方代码还有三个具体特征：

- 两个 FDSC 分别处理 content encoder 的 `skip1/skip2`，每个尺度只执行一次；
- 使用 modulated DCNv2，同时预测 18 通道二维 offset 和 9 通道 modulation mask；
- offset/mask prediction convolution 采用 zero initialization，所以初始 offset 为零、mask 经 sigmoid 后为 0.5。

### FontDiffuser：将 offset 条件改为参考字结构

[FontDiffuser (AAAI 2024)](https://arxiv.org/abs/2312.12142) 明确写道 “we propose a Reference-Structure Interaction (RSI) block”。它沿用 DCN 形变 skip feature 的思路，但不再只靠 style guidance map + content feature 经 CNN 预测 offset，而是：

- 参考图 \(x_s\) 经 \(E_c\) 得到多尺度结构图 \(F_s\)；
- UNet 当前 skip feature 记为 \(r_i\)；
- 以参考结构为 Query、skip feature 为 Key/Value 做 cross-attention；
- attention 输出经 FFN/投影得到 \(3\times3\) DCN 的 18 通道二维 offset；
- 用 offset 对 \(r_i\) 做 DCN 重采样，再与 decoder feature 拼接。

FD 论文承认参考结构与 UNet feature 存在 spatial misalignment，并把 cross-attention 作为处理这种错位的机制。因此，RSI 的原始语义不是“同字精确配准”，而是“从参考字结构中寻找与当前内容特征相关的区域，并据此形变 skip feature”。

FD 与 DG-Font 的区别可归纳为：

- **结构条件来源不同**：DG-Font 用目标字符的 content skip 与 style-conditioned decoder map；FD 额外编码参考字，直接用 \(E_c(x_s)\) 的绝对空间结构。
- **offset predictor 不同**：DG-Font 用局部 CNN；FD 用 reference-as-Query 的全局 cross-attention + FFN。
- **被形变对象不同**：DG-Font 形变 content encoder 的低层特征；FD 形变 diffusion UNet down path 的 skip feature。后者已经包含 noisy target 表征并受 MCA 等条件影响，不等同于原始 \(E_c(x_c)\)。
- **DCN 类型不同**：DG-Font 同时预测 offset 与 modulation mask；FD 的 `torchvision.ops.DeformConv2d` 只传 offset，没有 mask。
- **执行次数不同**：DG-Font 是两个尺度、共两次 FDSC；FD 是两个 RSI up block，每个 block 的 3 个 resnet 层都预测一次 offset，共 6 次。
- **初始化不同**：DG-Font 明确 zero-init offset/mask predictor；FD 的 RSI projection 没有对应的显式 zero-init。我们的 E2 又继承 E1@100k 的非零 offset head。
- **训练框架不同**：DG-Font 是 GAN，使用 adversarial/content-consistency/image-reconstruction/offset losses；FD 是 diffusion，使用 noise MSE/perceptual/offset losses，Phase 2 再加 SCR。

因此，二者虽然都把 DCN 放在 skip connection 上，但不能把 RSI 简化成“将 DG-Font 的 FDSC 搬进 diffusion”。FD 的真正新增点是：**用另一个参考字符的显式空间结构，经长距离注意力来产生 offset**。

## 2. 官方代码中的实际数据流

### 采样

官方 [`dataset/font_dataset.py`](../code/official/FontDiffuser/dataset/font_dataset.py)：

- target：目标字体 \(f\) 的目标字符 \(c\)；
- content：中性字体上的同一字符 \(c\)；
- style/reference：从字体 \(f\) 的其他图中随机抽取，并先移除 target path。

所以训练三元组是：

\[
x_c=(B_0,c),\qquad x_s=(f,r),\ r\neq c,\qquad x_{\text{target}}=(f,c).
\]

### 条件编码

官方 [`src/model.py`](../code/official/FontDiffuser/src/model.py) 对同一参考图做两次编码：

\[
x_s \xrightarrow{E_s} e_s \quad\text{(style condition)}
\]

\[
x_s \xrightarrow{E_c} F_s \quad\text{(RSI structure condition)}
\]

内容图另经 \(E_c\) 得到 \(F_c\)，供 MCA 注入内容身份与多尺度细节。

### RSI 内部

官方 [`OffsetRefStrucInter`](../code/official/FontDiffuser/src/modules/attention.py) 的代码方向与论文一致：

\[
Q=\Phi_q(F_s),\quad K=\Phi_k(r_i),\quad V=\Phi_v(r_i),
\]

\[
F_{\mathrm{attn}}=\operatorname{softmax}(QK^\top/\sqrt d)V,
\quad \delta=\operatorname{Proj}(\operatorname{FFN}(F_{\mathrm{attn}})).
\]

随后 [`StyleRSIUpBlock2D`](../code/official/FontDiffuser/src/modules/unet_blocks.py) 执行：

\[
r'_i=\operatorname{DCN}(r_i,\delta),
\]

再将 \(r'_i\) 与 decoder hidden state 拼接。官方 UNet 有两个 RSI up block；按当前实现每个 block 含 3 个 resnet/offset 单元，共产生 6 组 offset。

一个需要谨慎解释的细节：attention 输出序列长度由 Query \(F_s\) 决定，reshape 后仍落在参考结构的空间网格上，再作为 DCN offset 作用于 \(r_i\)。因此，结构条件的空间语义确实重要，但 cross-attention 允许它从 \(r_i\) 全局取值，并非逐像素硬对应。

## 3. Loss 到底约束了什么

### FD Phase 1 / 当前 E1、E2、E2b

\[
\mathcal L=
\mathcal L_{\mathrm{diff}}
+0.01\mathcal L_{\mathrm{perc}}
+0.5\mathcal L_{\mathrm{offset}}.
\]

- **Diffusion MSE**：预测噪声与真实噪声的 MSE；直接训练完整生成网络。
- **Content perceptual loss**：由预测的 \(x_0\) 与 target 经 VGG16 三层特征计算 MSE；同时约束内容与局部视觉重建。名称虽为 content perceptual，它使用的是目标字体真值，不只约束字符身份。
- **Offset loss**：

\[
\mathcal L_{\mathrm{offset}}=\operatorname{mean}|\delta|.
\]

代码先在每个 RSI 单元对 offset 取 mean absolute value，再在 block 内平均、跨两个 block 平均。它是**幅度正则项**，把 offset 拉向零，防止 DCN 因字符图大片同色区域出现任意、过大的采样解。

关键边界：没有 offset ground truth，也没有 loss 指定“哪一点应移动到哪里”。offset 的方向和空间模式只能通过 \(\mathcal L_{\mathrm{diff}}\) 与 \(\mathcal L_{\mathrm{perc}}\) 对最终图像的反向传播间接学得；\(\mathcal L_{\mathrm{offset}}\) 只负责约束幅度。

### FD Phase 2

官方 Phase 2 额外加入 \(0.01\mathcal L_{\mathrm{sc}}\)（SCR style contrastive loss），用于生成图与目标字体的风格一致性。**我们的 E1 与当前 Stage A 均为 SCR off**，所以当前训练没有这项显式风格监督。

Stage A 中 \(E_s/E_c\) 与 α/Δ 都冻结、缓存化；只有 UNet（含 RSI offset head）更新。因此：

- α 是否选对邻居，没有训练 loss 直接修正；
- Δ 是否真的表示目标字形变，没有显式监督；
- 网络只能通过最终生成损失学会使用、弱化或忽略 Δ。

## 4. 对 Δ-RSI 的正确解释

### 合理之处

官方 RSI 的 \(F_s=E_c(x_s)\) 是参考字符的**绝对结构特征**。在中文→西文时，我们改为：

\[
\Delta_c=\sum_{s\in\mathrm{Top10}}\alpha_sE_c(B_s,c)-E_c(B_0,c).
\]

它有两个明确变化：

1. **字符坐标对齐**：两项都是目标字符 \(c\)，不再把汉字参考结构直接作为 RSI Query；
2. **绝对量改为相对量**：减去与 MCA content 相同的中性 \(c\)，旨在削弱重复的字符身份公共分量，突出字体变化。

这与 DG-Font “形变目标字符自身低层特征”的原始几何动机更接近，但 Δ 来自相似字体库的估计，并不是未知目标字体 \(c\) 的真值。

### 尚未证明、不能提前写进结论的部分

- 不能写“FD 要求参考字与目标字相同”；事实相反。
- 不能写“汉字结构必然导致 RSI 失败”；FD 报告过中文→韩文的定性泛化，且 RSI 就是为处理错位设计的。
- 不能写“Δ 是纯 where 信号”；Δ 同时包含笔画强度、局部形状与方向信息。
- 不能写“减法已经去除身份”；只能说旨在抵消公共分量，是否做到要测。
- 不能写“Δ=0 就是零形变”。E1@100k 的 RSI 中 GroupNorm affine、projection 和 output projection 均有非零 bias；即使输入 Δ 全零，offset head 仍可输出非零 offset。Delta-drop 只表示“不给 Δ 条件”，不等于数学上的 identity deformation。

## 5. 当前实验能回答什么

- **E2 vs E2b**：在相同 n-shot style、相同初始化与训练预算下，比较 Δ 与官方 \(E_c(R[0])\) 结构源。这能回答“换结构条件是否改善最终生成”，不能单独证明 RSI 的注意力确实按预期定位。
- **E1c vs E2b**：主要回答 1-shot 与 n-shot style 条件变化的影响。
- 当前 best 使用单步 diffusion validation loss，只适合训练过程选点；不能作为“RSI 对齐正确”或论文主结论的证据。

若要支持机制解释，后续至少需要：

1. zero / wrong-character / wrong-style / spatial-shuffle Δ，并逐层匹配 RMS；
2. 记录各 RSI 层的 offset magnitude，比较 E2 与 E2b；
3. 检查 offset 是否集中在 \(|GT-B_0|\) 的真实变化区域；
4. 用完整采样图的内容、风格与人工评测判断，而非只看 diffusion loss。

## 6. 审查后的最终表述

> FontDiffuser 的 RSI 使用参考字符的 content features 与 UNet skip features 进行 cross-attention，据此预测 deformable-convolution offsets，以建模参考字体相对中性内容的结构变化。官方训练本来就允许参考字与目标字不同。我们进一步假设，跨语系会削弱这种参考结构的可迁移性，因此用相似字体库中同一目标字符相对中性字体的残差替代参考结构。该改动保留 RSI 内部与所有损失不变；其有效性由 E2/E2b 配对实验验证，机制解释则需要受控的 Δ 破坏和 offset 定位实验。

## 来源

- Yang et al., “FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning,” AAAI 2024. [arXiv:2312.12142](https://arxiv.org/abs/2312.12142), [DOI:10.1609/aaai.v38i7.28482](https://doi.org/10.1609/aaai.v38i7.28482).
- Xie et al., “DG-Font: Deformable Generative Networks for Unsupervised Font Generation,” CVPR 2021. [arXiv:2104.03064](https://arxiv.org/abs/2104.03064), [DOI:10.1109/CVPR46437.2021.00509](https://doi.org/10.1109/CVPR46437.2021.00509).
- FD 官方代码镜像：[`code/official/FontDiffuser`](../code/official/FontDiffuser)。
