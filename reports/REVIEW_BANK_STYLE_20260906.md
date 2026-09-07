# HR-Font bank 失配与 Es 风格信息审查（2026-09-07）

> 审查对象：用户指定的 `275fba3`，并核对当前工作树 `5f52c9b`。当前 HEAD 实为 `5f52c9b`；二者在本审查涉及的训练代码、`hrfont_delta_v2.py`、F123 决策记录上无 diff，`5f52c9b` 只新增了被审叙事 `BANK_STYLE_NARRATIVE_20260906.md`。因此代码结论适用于 `275fba3`；对叙事的修订建议适用于当前新增草案。
>
> 证据分级：下文“代码事实”来自当前仓库静态审查；“数值例”是严格按 `softmax(score/0.07)` 算出的假设分数形状，不是实测 bank 统计；仓库未找到正式 V1/V2/V6 输出或 cosine 分布产物，故不把 gate 阈值当成已通过结果。

## 一、Q-A：训练 228 / 推理子集 bank

### 1. 训练期到底依赖 bank 的什么

训练时 library 是 split manifest 的 `train` 列表，排序后构成 `_LibraryEs`（`train.py:480-486`）。它把每个 train 字体、全部 338 个 style 字符的 pooled Es 缓存在 RAM，张量明确为 `[228,338,D]`（`train.py:93-105`）。对一个 episode 的参考字符集合 $R$：

1. 目标字体的参考由 Es cache 读取：空间图逐元素平均产生 style condition；pooled 向量逐字归一化成为检索 query（`train.py:128-136`）。
2. `library.prototypes(rchars)` 从完整表抽出相同 R 字，得到 `[228,n,D]`（`train.py:106-108,180-183`）。
3. `compute_alpha` 再归一化 query/candidate，对每个字符分别做 cosine，最后沿 R 求均值，得到每个候选字体一个分数（`scripts/hrfont_delta_v2.py:88-101`）。所以不是“先把 n 字平均成一个 prototype 再 cosine”。
4. 训练字体作为 query 时先 leave-one-out；val/test 字体不在 `library.font_index`，`exclude=None`（`train.py:179-183`；`hrfont_delta_v2.py:102-108`）。
5. 活跃配置是 `mode=topk, k_top=10, tau=.07`（`train.py:174-176`；parser 默认见 `configs/fontdiffuser.py:33-39`）。实现取 `k=min(10,N_valid)`，按分数降序稳定选择，再**只在选中的 k 个上**做 `softmax(score/.07)`（`hrfont_delta_v2.py:119-130`）。D-P4 的“无绝对阈值”在 topk 分支属实；`eps_alpha` 不参与该分支。
6. 选中索引只用来读取这些字体的 `Ec(target,font,c)`；随后每尺度计算 
   \[
   \Delta_c=\sum_{s\in\mathrm{Top}k}\alpha_s E_c(B_s,c)-E_c(\mathrm{Content},c).
   \]
   见 `train.py:186-194` 与 `scripts/hrfont_feature_cache.py:201-210`。完整 228 字体的 Ec 并不全混入；完整 bank 影响的是候选竞争，直接进入 Δ 的只有 top-k Ec。

### 2. 推理换成子集时逐步发生什么

| 环节 | 训练：完整 train228（训练字体为 query 时有效候选 227） | 推理：100 字体或策展子集 | 是否自动校正 |
|---|---|---|---|
| 候选排名池 | 在 228（或 LOO 后 227）中排 | 只在子集中排 | 必然改变 order statistics；不等同于从原 top-10 简单删项 |
| top-10 成员 | 全库最相似的 10 个 | 子集内部最相似的 10 个；原邻居不在子集时由更低分者补位 | 同一 query 也可得到不同邻居与不同 Ec 方向 |
| softmax 域 | 10 个 | 若子集 $N\ge10$，仍为新 top-10；若 $N<10$，为全部 $N$ 个 | 有。`k=min(k_top,N_valid)` 后重新 softmax，权重严格和为 1（`hrfont_delta_v2.py:120-125`） |
| Δ | 训练邻域 Ec 的凸组合减 neutral | 新邻域 Ec 的凸组合减同一 neutral | 组合仍归一，但均值、RMS、方向、层间统计均可能变；没有 RMS/方向对齐或 OOD 校准 |
| offset head 输入 | 见过完整池诱导的 Δ 分布，以及 25% 的全零 source | 见到子集诱导的非零 Δ 分布 | `source_drop` 只训练“正常 Δ / 零 Δ”，不训练“任意策展 Δ” |

因此草案 §5 所说“剪掉候选后权重和不等于 1”不符合 `compute_alpha` 语义。正确推理实现应对子集**重新排名并在新 top-k 上重新 softmax**；若执行机只是从全库已算权重中删项却不重算，它不是仓库语义。

还要强调实现边界：F123 的 `sample.py` 仅把单张 `style_images` 与 `content_images` 交给旧 pipeline（`sample.py:126-161`），没有 Es cache、α、Δ、support 或预计算 structure 接口。仓库内只有训练/validation path 会调用 `_structure_features`（`train.py:378-405`）；所谓“执行机 Δ 推理”不在本仓库，故无法确认它是否真的执行上述重排与重归一。P1/P2 开跑前必须先做 inference parity：同一 batch 用完整 bank 时，执行机产生的 indices、weights、逐尺度 Δ 与训练函数逐元素一致。

### 3. τ=.07 的浓度：能算什么，不能算什么

softmax 对分数整体平移不敏感；“分数在 0.2–0.6”本身不能决定 top-1 mass，决定量是 top-10 的**间距形状**。以下为精确假设例（effective-N 同时给 Shannon perplexity $e^H$；tail-5 是第 6–10 名总质量）：

| 假设 top-10 分数 | top-1 mass | tail-5 mass | effective-N $e^H$ |
|---|---:|---:|---:|
| 0.60→0.20 等距 | 0.471 | 0.040 | 4.30 |
| 0.60→0.42 等距 | 0.264 | 0.193 | 7.56 |
| 0.40→0.20 等距 | 0.284 | 0.170 | 7.17 |
| 0.60→0.55 等距（很挤） | 0.139 | 0.402 | 9.75 |
| top-1=0.60，其余均 0.50 | 0.317 | 0.380 | 8.38 |

结论是：τ=.07 并不天然令 top-1 垄断。若 top-10 跨 0.4，约 3–4 个有效邻居；若只跨 0.05，则接近 10 个都活跃。因而草案把“错误邻居”简化成“小权重尾部”没有数据支撑：错误者也可能排第一，或 top-10 很挤而承载显著尾部质量。

仓库只有验证定义，没有正式输出：V1 的正式 gate 是 pairwise AUC≥.80，V2 是跨脚本 AUC≥.75，并非“V2 AUC 0.80”（`scripts/hrfont_validate_e1_encoders.py:281-290`）；V6 会记录 LOO top cosine 与 top-1 mass，但明确没有 semantic neighbor labels（`hrfont_validate_e1_encoders.py:212-224`）。未找到这些指标的正式 JSON。因此目前不能给出“通常 top-1 占多少”、真实 effective-N、subset turnover 或 Δ shift 大小。AUC 也不是候选为正确风格的校准概率，不能转成污染质量上界。

### 4. 判定与实验有效性

**(a) 是否等价于“用不同 bank 训练”？** 不等价。换推理 bank 是对已经学好的 $f_\theta(\Delta)$ 做输入干预；换训练 bank 会改变 80k 全程每一步的邻域、Δ、梯度和最终参数。二者共享“bank 改变 Δ”这一机制，但 estimand 不同。

**(b) shift 多大？** 方向上确定存在，数值大小未知。随机 100/228 子集可能保留部分原 top-10，策展 serif-only 则是有意系统性替换，通常风险更大；但不能在没有 Jaccard/turnover、weight divergence 和逐尺度 Δ cosine/RMS 数据时称“大”或“小”。identity-safe RSI 的实际式子是 `skip + zero_conv(warped-skip)`，zero-conv 1×1 权重/偏置零初始化（`unet_blocks.py:647-680,760-764`；`DECISION_F123_20260906.md:16-25`），保证 step 0 等价 F0、且始终保留显式 skip 残差通路。`source_drop=.25` 在训练时把整个 Δ 置零（`train.py:180-193,565-579`），使模型见过无 Δ 情形。这两点提供失效缓冲，但**不构成**对任意非零 OOD Δ 的鲁棒性证明；训练后 zero-conv 已非零，错误方向仍可驱动 offset。

**(c) P2 的因果口径。** 固定 checkpoint、ref、content、noise，只换候选 bank，可以干净识别“该已训练模型对推理期 bank 干预的总响应”，所以称 **inference-time bank steering** 是有效的因果干预描述，不需要为证明“输出会不会被推向某方向”而重训。但它不能证明：（i）训练学到了对策展 bank 稳健且可泛化的控制器；（ii）curated bank 在 matched 使用下提高质量；（iii）bank 越多越接近设计原则。

若要写后两类强主张，需要 matched retrain：从同一 F0 初始化训练 `F2-full/full` 与 `F2-curated/curated`，两者同 seed、batch/order、R、步数、drop、K、τ、cache encoder，只改训练和推理 bank；另加与 curated 同规模的多个随机子集控制，分离“风格构成”与“规模”。若 curated bank 少于 10，须预注册实际 $k=N$。最好再做 2×2 `train bank {full,curated} × infer bank {full,curated}`，直接估计 mismatch 与 steering 的交互。

**(d) P1/P2 在现有 F2/F3 checkpoint 上的有效性。** 

- P1 可作为固定 checkpoint 的 sensitivity/OOD probe：报告 bank size 对 indices Jaccard、turnover、top-1/tail mass、effective-N、逐尺度 Δ RMS/cosine、输出三轴的影响。它**不能**单凭推理子集扫支持“bank 越大，训练出的先验越好”或“228 训练优于 100 训练”。
- P2 可作为固定 checkpoint 的 inference-time steering demo/probe；必须有 full bank、同规模随机子集、serif-only、sans-only 等对照，并同时报告 identity/quality non-inferiority。它可以写“改变 bank 导致可测的方向性输出变化”，不能写“免费且可靠的 controllable generation”或“训练过的可控性”。
- “bank 多样性产生设计师通用原则”“matched curated bank 提高质量/控制稳定性”需要上述新训练臂和独立评测/人评。

### 5. F3 support 是否混入这项问题

不会混入 α library 本身。`_support_tokens` 从 support manifest 按目标字符查最多 8 个 support 字符，然后读取**当前目标字体自身**的 `Ec(style,font,scp)`，多尺度池化后进 adapter（`train.py:219-248`）；D7 也明确运行中 F3 bank 是每个 target 字对应的同字体 8 字，而非 E0 跨字体 bank（`DECISION_POINTS_20260906.md:56-63`）。support token 只拼到 up-path style context（`model.py:49-66`），Δ 则作为 `structure_features` 进入 RSI（`model.py:68-73`）。所以 P1/P2 改 α library 只直接改 Δ 流，support 流不依赖 228 库。

但需披露两个交互边界：F3 是 joint-trained，UNet 可学会组合两路，因此最终输出响应并非统计独立；同一训练循环还分别采样 `source_draw` 与 `support_draw`（`train.py:565-581`）。另，exec-spec 当前 §2 仍写 “Support 图不来自目标字体”，与 D7 和运行代码相反，应以稍后的实际运行决策为准并修文档。

## 二、Q-B：Es-based α 是否泄漏非参考风格

### 1. 完整信息流

```text
目标字体可见参考 R
├─ Es spatial(R1...Rn) ──逐元素均值──> [B,1024,3,3] ──展平──> 9 style tokens
│                                              ├─ MCA/style conditioning
│                                              └─ 与 F3 support tokens 拼接后进 up-path cross-attn
├─ Es pooled(R1...Rn) ──逐字 cosine mean 对 train228 Es pooled prototypes
│                       └─ LOO → top-10 → softmax(/.07) = α（Es 数值到此为止）
│                                      └─ α × Ec(library font, target char c)
│                                           − Ec(Content,c) = Δ multi-scale
│                                                        └─ RSI attention → offset → residual warp
└─ F3 support manifest(q for c) ──> Ec(style, target font, q) ──pool/adapter──> support tokens

Content glyph c ──> Ec(Content,c) ──> identity/content multi-scale；同时作为 Δ 的 neutral
```

逐项代码证据：

- **R 直接进入 style condition：** 每张参考的 Es spatial map 做均值（`train.py:128-136`）；model 将 `[B,C,3,3]` 展成 9 tokens（`model.py:38-50`），送 UNet style context（`model.py:66-73`）。训练/验证均使用 cache，不在线 forward encoder（`train.py:504-508`）。
- **Es pooled：** query 是目标字体自身 R 的 pooled Es；library prototype 是同 R 字的 train228 pooled Es（`train.py:93-108,128-136,180-183`）。它只产 indices/weights；代码随后没有把 library Es 张量传给 model。
- **Ec/Δ：** indices 选择 library 的 `Ec(target,font,c)`，α 混合后减 `Ec(content,"",c)`（`train.py:186-194`；`hrfont_feature_cache.py:201-210`）。Δ 作为 `structure_features` 送 UNet（`model.py:57-73`）；RSI 用 Δ 作 attention query、skip 作 context，并投影 offset（`attention.py:278-330`），最后残差 warp（`unet_blocks.py:760-764`）。
- **F3 support：** 当前目标字体其他 support 字的 Ec(style) 经 pool/adapter 成 tokens（`train.py:219-248`），与 9 style tokens 拼接（`model.py:60-64`）。它不参与 α，也不读取 library font。

### 2. “泄漏”分别能与不能指什么

| 潜在通道 | 审查结论 |
|---|---|
| library 字体的 Es feature 直接进生成器 | **没有。** 它们只参与 cosine 排名与 α 数值；生成器不接收其 Es map/vector。|
| library 字体的 Ec feature 进入生成器 | **有，且是方法明确定义的 bank prior。** 进入的是目标字符 c 的 Ec 特征，字符身份对齐；跨字体变化仍包含字重、衬线、比例、曲线等 style/structure，因此确实是“非参考字体的信息”，不能说纯参考。|
| query 使用隐藏的目标字体信息 | **没有超出 R。** query 只由当前 episode 可见 R 构成；target glyph/GT 不参与 α。|
| test16 进入候选池或 cache 学习 | **没有。** library 只取 manifest 的 train228（`train.py:480-486`）；训练字体 LOO（`train.py:182-183`），val/test 只作 query。exec-spec 也明确 test16 禁止进入候选池（§7，当前文件 `:125`）。|
| support 引入另一字体风格 | **当前 F3 没有。** 它读取目标字体自己的 support glyph；这是额外可见 exemplar 信息，必须在 few-shot protocol 中计入参考预算，但不是外字体污染。|

“错误风格质量”若定义为真实风格标签与 reference 不同的邻居集合 $W$，其污染质量就是 $m_W=\sum_{s\in W}\alpha_s$，天然在 `[0,1]`；这是唯一无标签假设下的数学界，毫无紧致性。τ 表只能展示分数间距如何分配质量，不能判断哪些字体是错的。V2 只比较“同字体汉字/Latin”与“异字体”的成对分数（`hrfont_validate_e1_encoders.py:146-157`）；即使实测过 gate，也不能对 top-10 中 $m_W$ 给概率上界，更何况仓库没有正式结果。V1/V2 的 AUC 是排序概率，不是 calibrated precision；V6 更明确没有语义邻居标签。因此“V2 AUC 0.80 隐含污染受控”不可写。

### 3. 错误成分是 bug 还是 feature

它不是实现 bug，而是 D-P4 选择的建模偏置：无拒识、relative top-K，α 同时 routing+weighting；弱匹配也会被强制采用，必须披露。soft average 本身可以被称为 **bank-conditioned prior**，但“平均即去噪”“尾部就是 prior variance”只是解释性假设：当前只有一个确定性凸组合，没有显式方差变量，尾部也可能造成系统偏差而非有益方差。

诚实口径是：bank 给出一个由 reference-conditioned routing 选择的、目标字符对齐的 population prior；非参考成分被凸权重限制且可逐样本测量，但目前没有语义污染的经验界。建议新增无需训练的 probe：

1. 为 bank 建独立于 Es 的 serif/sans/calligraphic/family 标签或 φ_s2 邻近定义，逐样本报告 wrong-style mass、top-1 是否 wrong、tail-5 wrong mass。
2. 固定 checkpoint/noise，构造 `top1-only`、`top1 removed`、`tail-5 only`、`wrong-only`、`correct-only` Δ；每个变体逐尺度 RMS 对齐到原 Δ，避免幅度混杂。
3. 用独立 φ_s2 测输出 style displacement，回归到 top-1 neighbor 与 tail-5 barycenter 的方向；同时报告 ID/quality。这样才能回答输出更跟随 top-1 还是 soft tail。

### 4. 换成 Ec-based α 会不会消除问题

不会，只会移动 retrieval error profile。若用同字符 Ec 排名，它可能更敏感于字符结构、字重与局部形状，也可能因 Ec 以内容身份为主而弱化跨字符风格一致性；这需要独立 V1/V2/标签检索对比。无论排名用 Es 还是 Ec，混合项仍是 library 的 `Ec(B_s,c)`，所以“非参考 library Ec 进入 Δ”的结构完全相同。Ec-based α 最多改变谁被选、各自权重和污染质量，不能让污染通道消失；且若 query 只能来自汉字 R、candidate 却要 target Latin c，还会重新引入跨字符/跨脚本比较定义问题。结论：切 Ec 是 matched retrieval-source ablation，不是泄漏修复。

### 5. 是否存在 reference 自身的非常规泄漏

没有看到超出声明参考集的 test-side 信息。9-token condition 是 R 的 Es spatial 均值；α query 也是 R 的 pooled Es。Es/Ec 权重由 official P1/F0 训练体系在 train domain 学得并冻结、cache SHA 与 init encoder 强绑定（`train.py:80-90,504-508`）。一个冻结 encoder 从训练字体学到先验，属于标准 learned feature extractor / retrieval prior，不叫 test leakage；应在论文中披露其训练域、冻结点、candidate=train228、test exclusion，并让最终指标使用独立 φ_s2，避免以 Es 自证。

可写：“reference-conditioned Es routing selects and weights target-character-aligned library Ec residuals”；不可写：“Δ 只包含 reference style”“library 不向生成器提供风格信息”“AUC 保证错误风格质量很小”。与 CF-Font 的关系最多写作“沿用 learned retrieval/prior 的一般先例”，若要写具体 precedent 仍需正式文献核验，本次代码审查不替代引用审查。

## 三、必须由 PI 决策的新增点

### D-B1｜P2 的论文强度

- A. 只做现有 checkpoint 的 inference intervention，定位为 sensitivity/steering demo。**推荐，近期最低成本且因果口径成立。**
- B. 写“策展 bank 是可靠控制机制/提升质量”，补 full-vs-curated matched retrain 及随机同规模控制。
- C. 直接把推理 P2 当作训练 bank 结论。不推荐。

### D-B2｜是否为 bank mismatch 补 2×2

- A. 只在 full-trained checkpoint 上扫 infer bank，明确 OOD probe。**推荐作为主线最小集。**
- B. 补 `train {full,curated} × infer {full,curated}`，能分离训练 bank、推理 bank 与 mismatch interaction；成本高，适合强 controllability claim。

### D-B3｜“错误风格”操作定义

- A. 先冻结独立标签/φ_s2 距离定义，再报告 wrong-style mass 与 top/tail 干预。**推荐。**
- B. 仅凭 Es cosine 或 V2 AUC 称邻居正确/错误。不推荐，会循环定义且 AUC 不给污染界。

### D-B4｜推理实现准入

- A. 在 P1/P2 前强制 full-bank parity（indices、weights、每尺度 Δ）并记录 bank manifest/SHA；子集按新池重排、重 softmax。**推荐，属于证据准入。**
- B. 使用仓库旧 `sample.py` 或执行机未审计路径直接出图。不推荐；当前 `sample.py` 没有 Δ/F3 推理语义。

### D-B5｜support 数据契约冲突

- A. 修订 exec-spec，把运行 F3 明确为“目标字体 own-font 8-char support”，并计入可见参考预算。**推荐，符合代码与 D7。**
- B. 坚持“support 不来自目标字体”，则当前 F3 不符合规格，必须另建/重训对应臂。

## 四、对 `BANK_STYLE_NARRATIVE_20260906.md` 的修订建议

1. §2 “227 个库字体被加权平均”改为：“完整 228/LOO 227 是**候选池**，每个样本实际只混合 top-10；完整池通过候选竞争影响这 10 个。”
2. 删除或降格“平均即去噪”“保留下来的恰是设计师共识”“bank 越多样越接近通用原则”。这些尚无定理或实验支持；多样性也可能使近邻更好，但 top-10 与 τ 意味着 218 个未选字体不直接平均，策展还可能带来系统偏置。
3. “免费的 controllable-generation 旋钮”改为“无需更新参数的 inference-time bank intervention；控制方向、强度、质量与 OOD 稳健性待 P2 验证”。
4. §3 P1 不要同时改 bank size 与 `K∈{10,30}`；这是两个因素。主 P1 固定 K=10、只变候选池；K 消融另表。子集需多个固定随机种子/嵌套 manifest，避免单个子集偶然性。
5. §5 删除“剪掉候选后权重和≠1”：仓库语义会对子集新 top-k 重 softmax；改成“执行机必须验证重排/重归一 parity”。
6. §6 “泄漏只可能经排序误差+soft 尾部”不完整。即便排序完全正确，library Ec 仍是有意注入的非参考 population prior；且错误邻居可能是 top-1，不限于尾部。改为“非参考信息通道是 α 加权 library Ec；Es 决定其选择误差与质量分布”。
7. 把“V2 AUC 0.80”改为实际事实边界：代码 gate 是 V1≥.80、V2≥.75；正式结果未在仓库找到。AUC 不可换算污染上界。
8. 增加 F3 注记：own-font support 与 α library 分流，但 joint model 可能产生功能交互；P2 在 F2 与 F3 都应跑，以检验 support 是否缓冲或放大 bank steering。
9. 增加当前 HEAD/推理缺口：F123 `sample.py` 尚无 Δ/support 路径；P1/P2 必须依赖经过 parity 审查的执行机实现，不能把训练函数语义自动外推为已实现推理。

## 五、审查结论（可直接给 PI）

1. **训练 228、推理子集不是“换 bank 重训”。** 它是固定已训练 offset/UNet 后改变 Δ 输入分布；新池会重排邻居并在新 top-k 内归一。shift 的存在确定，大小暂无仓库实测。identity-safe residual 与 25% source-drop 提供旁路/无源鲁棒性，但不保证任意策展 Δ 的正确响应。
2. **现有 P1/P2 仍值得跑。** P1 是 bank-size/OOD sensitivity，P2 是 inference-time steering intervention；两者不能单独证明“更多 bank 学出设计师原则”或“matched curated training 更好”。这些强 claim 要 matched retrain，最好做 2×2。
3. **Es 本身不把 library feature 直接喂给生成器，也没有 test-side 泄漏。** 真正进入模型的非参考信息是 α 加权的 library Ec(target char)；这是 bank prior 的定义，不是代码 bug，但弱/错误邻居被强制采用是 D-P4 必须披露的风险。
4. **τ=.07 未必接近 one-hot。** top-10 分数跨 0.4 时 top-1 约 47%，跨 0.05 时仅约 14%；没有真实 cosine/V6 输出就不能声称尾部污染很小。V2 AUC 更不能给污染质量上界。
5. **改用 Ec 排 α 不会消除非参考信息，只会改变检索误差。** 应当把它当 retrieval-source 消融，而不是“防泄漏修复”。
