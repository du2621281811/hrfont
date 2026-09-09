# E12 v5 与 F3 Support 设计核查（2026-09-07）

> 审查快照：HR-Font `afe839b55980a1144063bce19991a65e2f735f51`。本文把“当前代码事实”与“v5 待实现规格”严格分开；未改任何代码、配置或其他文档。

## 0. 结论先行

1. 当前 `φ_s2` 是单图共享 ResNet18：训练输入是一张中文 glyph 与同 family 的一张 Latin glyph，输出 512-D L2-normalized embedding；对称、单正例 InfoNCE 把 batch 对角线当唯一正例。它不是 ref-set-conditioned verifier，也不读预测图或 GT（`scripts/eval_framework/data.py:166-177`；`scripts/eval_framework/models.py:10-20`；`scripts/eval_framework/train_style_encoder.py:43-55`）。当前另有冻结 `φ_s2` 的 membership MLP，但它只读 query 与固定 ref 字符全集，以 mean+max 聚合 ref，单独用 BCE 训练（`scripts/eval_framework/models.py:29-40`；`scripts/eval_framework/train_membership.py:31-38`）。
2. v5 应改为共享单图主干 `f` + **DeepSets mean** 语境聚合 + 低秩双线性语境条件打分，联合 `L_ctx + L_mem + λ_gt L_gt`。主推理只需 `(R,P)`；`G` 只作训练正锚及校准集可选辅助，绝不能成为主判定必需输入。
3. style 条件从 9 token 增至 `9+64` local token 不属于“很难训”的结构变化，但 cross-attention 的 K/V 长度在相关层上从 9 增至 73（该部分约 8.11×）；应只加 64 token、zero-init 残差门、短 warmup，并先做等预算 matched probe。
4. F2 的 arm 契约明确 `support=False`，所以“support 去平均/per-glyph 化”不要求重训 F2，只要求重训 F3；若把 support 纳入 F2，则已变更方法主线，F2/F3 都必须从同一 F0 重训（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:46-49`；`scripts/launch_cn2west_f123.py:39-42`）。
5. 当前 F3 support 是目标字体自身的固定最多 8 个 support 字，经 `ec|style|font|cp` 取 Ec；不是 Ec 最近邻，也不是跨字体检索（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:219-248`）。D7 已把它记录成优先 ref8 的固定表，但仓库 `scripts/` **没有**可复现的 `support_bank.json` 构建脚本；同时 exec-spec 仍要求 support 不来自目标字体，D-B5 尚未在权威 spec 中落地（`reports/DECISION_POINTS_20260906.md:56-63`；`.cursor/rules/hrfont-execution-spec.mdc:57-62`）。

---

## Q1. E12 v5：模型、输入、输出与 loss

### 1.1 当前实现核对（不是 v5）

#### 输入、配对与输出

- cache glyph 读成灰度 `[0,1]`，再按 `channels` 复制为 3 通道（`scripts/eval_framework/data.py:155-163`）。
- `CrossScriptPairDataset` 的一个样本为 `(a_CN, b_Latin, family)`：索引固定到 `family×中文字符`，Latin 字符由 `index % len(latin)` 固定选择；没有 ref 集、预测图、GT 槽位或随机多正例 episode（`scripts/eval_framework/data.py:166-177`）。
- `PhiS2` 使用无预训练权重的 ResNet18，移除分类头，接可选线性投影并 L2 normalize；当前 `feature_dim=512` 时 ResNet18 输出维度同为 512，投影实际为 Identity（`scripts/eval_framework/models.py:10-20`；`scripts/eval_framework/configs/phi_s2.yaml:9-11`）。

#### loss 与验证

- 训练得到 `za=f(a_CN), zb=f(b_Latin)`，相似度矩阵为 `za @ zb.T / 0.07`；两个方向各做一次 cross-entropy 并平均，target 是 batch 对角线。因此它是**单正例对称 InfoNCE**，不是 family-aware SupCon（`scripts/eval_framework/train_style_encoder.py:52-60`；`scripts/eval_framework/configs/phi_s2.yaml:11-11`）。
- checkpoint 按 held-out anchor 的 pair AUC 选；wrong 字体由异 typeface pool 选取，最多检查 128 个 anchor（`scripts/eval_framework/train_style_encoder.py:13-38`；`scripts/eval_framework/train_style_encoder.py:61-69`）。
- 当前 style eval 对 ref 字符的 embedding 作 mean 后 normalize，再以 cosine 读 `SC-R`、Rank@1/MRR/margin；`sc_gt` 实现却直接复用了同一个 `z@proto[f]`，并没有独立编码一张 GT（`scripts/eval_framework/eval_style.py:13-24`）。

#### 当前 membership verifier

- `MembershipDataset` 返回一张 query 与一组固定 `ref_chars`；偶数 episode 同 family、奇数 episode 异 family，负例保持 query 字符不变（`scripts/eval_framework/data.py:188-200`）。
- verifier 冻结 `φ_s2`，ref embedding 用 `mean+max → MLP → normalize`，再把 `q, r, |q-r|, q*r` 拼接进 MLP 输出一个 logit（`scripts/eval_framework/models.py:29-40`）。它只以 BCE 训练，然后在 val episodes 上拟合 temperature（`scripts/eval_framework/train_membership.py:18-24`；`scripts/eval_framework/train_membership.py:34-38`）。

**诊断结论：** 现状是“先学单图 cosine embedding，再冻结 encoder 训练二分类头”；v5 必须把 ref 语境、多正例与 membership 联合起来，而不是只给现有 verifier 多塞一张 GT。

### 1.2 v5 输入三件套及训练/推理同构

设单图共享主干为 `fθ: R^{3×96×96}→R^d`，所有 R/P/G glyph 共用同一权重与图像预处理；共享权重能避免三路各自形成不可比较的坐标系。

| 槽位 | 训练期 | 主推理期 | 作用 |
|---|---|---|---|
| ref 集 `R={r_j}_{j=1..k}` | 同一 lineage/family 随机抽 `k∈{2,…,8}` 个 glyph；与 candidate/GT 字符去重；允许跨语系，主比例保持 CN ref | F3 口径：目标字体 CN ref，正式固定 ref8；兼容 1..8 张 | 定义“这套参考字的风格语境” |
| candidate `P` | **没有生成预测图**；用同 family 的另一 glyph 作 positive candidate proxy，用异 lineage 的同字符 glyph 作 negative/hard-negative proxy。字符与 R 不同，跨字符、跨语系，包含 digits | 待评价的模型预测图 | 被判断对象 |
| 可选 `G` | 同 family、且不同于 R/P 的另一 glyph，作为额外正锚；若对 P 有同字符渲染，可同时构成配准后的 `(P,G)` 校准 pair | 默认缺省；仅在有 GT 的校准/诊断集提供同字符 GT | 训练增强或可选辅助，不定义主分数 |

训练 candidate 代理与推理预测图有 domain gap，故在不接触 HR-Font/A260/任何方法输出的前提下，对外部字体 glyph 加预注册的轻度仿生成扰动（抗锯齿、轻线宽、±3°、≤5% 平移/尺度、轻局部缺损），并把 clean 与 corrupted view 均列为同族正例。E12 独立性仍要求外部池与 A/260、方法输出、Es/Ec 完全隔离；现执行规范也要求 E12 encoder 与方案 encoder 隔离（`.cursor/rules/hrfont-execution-spec.mdc:93-98`）。

### 1.3 模型形态

#### 共享编码与语境聚合

```text
h_j = fθ(r_j),  h_P = fθ(P),  h_G = fθ(G)
u_j = ψ(h_j)                       # 2-layer projector, d→d→d
z_R = norm( ρ( mean_j u_j ) )      # DeepSets mean
z_P = norm(ψ(h_P)); z_G = norm(ψ(h_G))
```

选择 **DeepSets mean**，不选 attention pooling，原因是：

1. `k=2..8` 时 mean 天然置换不变且对缺张稳定；
2. 参数更少，便于在 ≥60 lineage、单 seed、全池 CV 下控制过拟合；
3. 与现有 SC-R prototype 的 mean 语义连续（当前实现也是 ref embedding mean+normalize，`scripts/eval_framework/eval_style.py:13-20`）；
4. attention pooling 可作为冻结后的 matched ablation，只有 macro-family AUC/校准同时改善才升级。

建议 `d=512`，`ψ=Linear(512,512)→GELU→Linear(512,512)`；backbone feature 与 projector feature同时落盘，但所有主指标预注册只读 projector 输出。

#### 语境条件打分与 membership 概率

采用低秩双线性而不是任意大 MLP：

```text
a_R = norm(A z_R),  a_X = norm(B z_X), A,B∈R^{r×d}, r=128
q(X|R) = a_Xᵀ a_R + b
ℓ_mem(X,R) = γ q(X|R) + b_mem
p_mem(X|R) = sigmoid(ℓ_mem / T)
```

`q` 是未校准 membership logit 的核心；`T>0` 只在 CV 的 validation fold 上拟合。若担心非对称 `A/B` 过拟合，第一版令 `A=B`；与 cosine baseline 作等预算 matched control。

### 1.4 可直接编码的联合 loss

一个 batch 采 `F` 个 lineage，每 lineage 采一个 ref context 与 `M+` 个 positive candidates；负例来自 batch 中所有其他 lineage，sampler 保证 family-balanced、lineage 去重。

#### `L_ctx`：语境条件多正例 NCE / SupCon

对 candidate anchor `i`，其 context 为 `R_i`；令 `Pos(i)` 是 batch/queue 中所有与 `i` 同 lineage 的 glyph view（跨字符、跨语系、digits、可含 G），`Neg(i)` 只含异 lineage。定义：

```text
e_i = norm(B z_i)
c_i = norm(A z_Ri)
score(i,j | R_i) = e_jᵀ c_i + e_jᵀ e_i
L_ctx = -(1/|I|) Σ_i (1/|Pos(i)|) Σ_{p∈Pos(i)}
          log exp(score(i,p|R_i)/τ) /
              Σ_{a∈Pos(i)∪Neg(i)} exp(score(i,a|R_i)/τ)
```

这里 `e_jᵀc_i` 使候选的可接受性由 R 调制，`e_jᵀe_i` 保留同族多 view 聚合；第一版可令两项系数各 1，再只在 inner-val 做冻结小网格 `{0.5,1}`。同 family/lineage 的所有合法 glyph 都必须从 denominator 的负类 mask 中移除。当前计划已经把“同族任何字形为正、异 lineage 为负、digits 入训”冻结为目标重定义（`reports/E12_OPTIMIZATION_PLAN_20260907.md:168-179`）。

更简实现可直接把每个 `(candidate_i, context_i)` 的正 logit `q(P_i|R_i)` 与所有异 lineage candidate-context 交叉组合做 multi-positive InfoNCE；但必须保留同族 mask，不能退回 batch 对角线唯一正例。

#### `L_mem`：membership BCE 联合训练

构造 1:1 正负 episode；负例优先同字符、异 lineage，使 classifier 不能靠字符内容判断：

```text
L_mem = BCEWithLogits(ℓ_mem(P_i,R_i), y_i),  y_i∈{0,1}
```

与当前“冻结 encoder 后另训 BCE”不同，v5 默认联合反传到 `θ,ψ,A/B,head`；同时保留 `stopgrad-encoder membership` 作为消融，以判断收益来自共同表征还是更强 head。

#### `L_gt`：GT 训练增强与可选辅助

训练中有 G 时做两件事：

1. 把 `G` 作为同 lineage positive 加入 `Pos(i)`；
2. 若 `P` 是 G 的扰动 view/同字符 paired proxy，加入一致性：

```text
L_gt = 1[G]·{ 1 - cos(z_P,z_G)
              + β·Huber(ℓ_mem(P,R)-ℓ_mem(G,R)) }
L = L_ctx + λ_mem L_mem + λ_gt L_gt
```

首版冻结 `λ_mem=1, λ_gt=0.1, β=0.5`；仅 inner-val 可在 `{0,0.05,0.1}` 选 `λ_gt`，outer fold 不调参。

**主判定绝不要求 GT。** 否则 evaluator 会把“在 R 语境下是否合理”退化成“是否像唯一 G”，惩罚合理的字腔替代、端点/笔画变化，并与执行规范“GT 仅正控/校准参照，禁止把像素贴近单一 GT 当唯一正确答案”冲突（`.cursor/rules/hrfont-execution-spec.mdc:93-98`）。

主判定：

```text
p_main = sigmoid(ℓ_mem(P,R)/T_main)
```

GT-辅助判定仅用于有 GT 的校准/诊断子集：

```text
d_PG = 1-cos(z_P,z_G)
ℓ_aux = ℓ_mem(P,R)/T_main - η·standardize_val(d_PG)
p_aux = sigmoid(ℓ_aux)
```

`η≥0` 与标准化均值/方差只在 calibration folds 冻结；必须并排报告 `p_main` 与 `p_aux`，不得用 `p_aux` 替换无 GT 方法主表。若 `η=0` 被 CV 选中，就明确报告 GT 未提供额外校准价值。

### 1.5 校准与全池 CV

- 数据：全部外部字型都参与训练组件；目标 ≥60 个经人工 lineage 去重的独立 typeface。单 seed；不再保留固定 5-family test。计划 v2 已冻结“全池训练思想 + LOFO/分层 K-fold 测量 + 单 seed”（`reports/E12_OPTIMIZATION_PLAN_20260907.md:156-166`）。严格实现时，“全部入训”应解释为**每个 outer fold 的模型使用除该 fold 外全部可用 lineage**，否则被测 family 同时入训会使 CV 无意义；跨 folds 合并即覆盖全池。
- 推荐 5-fold stratified group CV（每 fold ≥12 lineage；若池恰好 60），按 lineage 分组并平衡 serif/sans/hand/display 与 script coverage；LOFO 成本可接受时作为敏感性分析。
- 每个 outer fold 内再从 train lineages 划 calibration split；模型/超参只看 inner train/val，`T_main` 用 validation logits 最小化 NLL。当前 temperature scaling 的可用实现是对 `log T` 用 LBFGS 最小化 BCE（`scripts/eval_framework/train_membership.py:18-21`）。
- outer fold 只生成一次 raw logits；跨 fold 报 macro-family AUC、family-cluster bootstrap 95% CI、Brier、ECE，并分 Latin/digit。门仍为主 AUC≥0.90、稳定性下限≥0.85（`reports/E12_OPTIMIZATION_PLAN_20260907.md:164-166`）。

### 1.6 输出协议：同一模型的三种读法

1. **单张合理性**：`p_main(P|R)`；这是温度校准后的 membership probability。输出 raw logit、T、probability、fold/checkpoint SHA。
2. **SC-R / SC-Gap**：
   - `SC-R = cos(z_P,z_R)`；
   - `SC-G = cos(z_P,z_G)`（仅 G 存在，不能再把它复制成 SC-R）；
   - `SC-Gap = SC-R - max_{R^-∈C_neg} cos(z_P,z_R^-)`，负 context 集按预注册 lineage 全候选或 hardest-k 规则；另报 `Gap_GT = SC-R-SC-G` 仅作诊断。当前代码的 margin 已是 true prototype score 减最高错误 prototype score，可复用其语义而不能复用当前错误的 `sc_gt`（`scripts/eval_framework/eval_style.py:18-24`）。
3. **方法 paired 排序**：同一 `(font,char,R,noise)` 下比较方法 A/B 的 `Δp=p_main(P_A|R)-p_main(P_B|R)`；按 font 聚类做 paired bootstrap，并报告 win/tie/loss。不得跨未配对字符或不同 R 比均值。人评仍是终审，E12 是过门后的辅助证据（`reports/E12_OPTIMIZATION_PLAN_20260907.md:181-184`）。

三者共享 `f,ψ,z_R,z_P`：合理性读校准 head，SC 读 embedding，paired 排序读同一个主概率的样本内差；名称和统计量必须分开，不能把 membership AUC 称为 cosine T2。

### 1.7 数据与 F3 评测对齐

- `R`：训练随机 `k∈{2,…,8}`，正式推理固定目标字体 CN ref8；candidate：任意语种，训练含 Latin upper/lower + digits，正式主报 Latin、digits 分报。
- positive：同 lineage 的任意其他字符，跨语系/跨字符；negative：异 lineage，先去除 weight/style/衍生设计假负例，再优先同字符 hard negative。当前 group 正则只能剥常见 weight/style 后缀，不能替代人工 lineage 审计（`scripts/eval_framework/data.py:15-34`）。
- 推理口径与 F3 一致为 `CN ref → Latin candidate`；F3 当前任务与 support 说明也明确是查询字体汉字参考生成/评价西文（`reports/F123_STATUS_20260907.md:5-18`）。

---

## Q2. style 条件加细、support 去平均与 Δ 正交性

### 2.1 style token 从 9 增至 `9+64`：会不会难训？

#### 当前代码事实

- 96×96 StyleEncoder 的通道/分辨率序列为 `64@48²,128@24²,256@12²,512@6²,1024@3²`；最终 `style_emd=h`，所以是 `[B,1024,3,3]`（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/style_encoder.py:293-298`；`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/style_encoder.py:428-442`）。
- n-shot 训练先对多个 spatial map 逐元素 mean，仍为 `[B,1024,3,3]`（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:128-136`）。模型把 H×W 展平为 9 个 1024-D style token（`code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py:49-50`）。
- MCA down/mid 路径也把同一 3×3 map 展成 9 token 做 style cross-attention（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:217-230`）。up-path 读取 `style_hidden_states`（`code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py:60-71`）。
- UNet 配置含 2 个 MCA down block，每 block 2 层；mid 1 层；2 个 RSI up block，每 block `layers_per_block+1=3` 层。因此全局 style map 被 4+1+6=11 个 style-attention 层消费；若 local token 只附加到 `style_hidden_states`，则只增加 6 个 up-path attention 的 context（`code/variants/cn2west_f123_rsi/FontDiffuser/src/build.py:8-35`；`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:126-188`；`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:655-712`）。

#### 推荐实现与参数/计算量

推荐先做 **9 global + 64 local**，不直接上 128：从 `512×6×6` residual map 取 36 原生位置仍偏少，因此更合适从 `256×12×12` map自适应池化到 8×8，再以共享 `Linear(256,1024)` 投影出 64 token。投影层参数为 `256×1024+1024=263,168`；另加一个标量/逐通道残差门最多 1024 参数。若直接投影 `512→1024`，则为 `525,312` 参数。两者都远小于主 UNet，参数量不是主要难点。

cross-attention 的 K/V 长度由 9 变 73，相关 attention matmul 近似按 token 数线性增长：`73/9=8.11×`，即该部分约 `+711%`；若 128 local，则 `(9+128)/9=15.22×`，约 `+1422%`。这不是整个 UNet 8–15×，因为卷积、Q 生成、FFN、down/mid 等不同比例；但 6 个 up attention 的显存/耗时会明显增加。当前 attention 计算显式形成 `QKᵀ` 后 softmax 再乘 V（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/attention.py:250-263`）。

#### 稳定性策略

- 保留旧 9-token 路径不动，新增 local branch：`context=[global9, g·local64]`。
- `g` 或 local 投影末层 **zero-init**，使 step 0 语义严格回到旧 9-token 模型；仓库 SupportAdapter 已用“末层 Linear zero-init”模式（`scripts/hrfont_support_adapter.py:11-24`），identity-safe RSI 也用 zero-conv 保 step0 旁路（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:677-680`）。
- 因 zero-init 会让上游 local projector 第一步梯度为零、门先打开，建议 scalar gate `g=0` + projector 常规 Xavier；或末层 zero-init 后允许其先学习。前 1k–2k step 对新增参数线性 warmup，旧参数 lr 可为主 lr 的 0.25–0.5；监控 local/global attention mass、梯度范数和生成身份。
- 第一阶段只把 local token送入 6 个 up attention，避免同时改变 11 层；若有效，再 matched 比较 down+mid 也使用 local token。

**结论：** 不会因参数量而“难训”，主要风险是 context 变长造成优化竞争与计算上涨。预期收益通道是把原 3×3/空间平均丢失的局部端点、衬线、转角、粗细变化送入生成器；风险是 local token 记字符内容、压过全局风格、以及注意力稀释。必须以同 F0 初始化、同数据/order/RNG/预算的 9-token control 做 matched probe；只在 style 提升且 identity/quality non-inferior 时升级。

### 2.2 support 去平均：F2 是否重训？

- F1/F2/F3 arm 白名单分别是 `(official,off)/(delta,off)/(delta,on)`；不匹配直接 fail（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:46-49`；`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:488-493`）。
- `_support_tokens` 在 adapter 为 `None` 时立即返回 `None`；F2 不创建 adapter，只有 `args.support` 才创建（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:219-226`；`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:495-502`）。
- 训练循环即便 support off 也抽同序 `support_draw`，正是为了保持 F2/F3 RNG 对齐（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:564-576`）。

因此：

- 若“去平均”只指 F3 把每个 glyph 保持为独立 support token（当前 `_support_tokens` 实际已经逐 glyph `_pool_ec` 后 stack，并没有跨 glyph 求平均，`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:214-248`），则 **F2 不重训，F3 必须重训**；旧 F3 checkpoint 不可能无训练地学会新 token 语义。
- F2↔新 F3 的归因仍须：同一 F0 parent、同 batch manifest/order、同 diffusion/source/support/CFG draw 顺序、同 drop、steps、lr/scheduler/warmup；exec-spec 的 matched set 也要求这些量一致（`.cursor/rules/hrfont-execution-spec.mdc:100-104`）。若复用已完成 F2，只能在能证明新 F3 的 batch/order 和每步 RNG 与原 F2 完全复现时称 matched；否则重跑两臂更稳妥。
- 若未来把 per-glyph support 变成主线基础，使 F2 也消费 support，再用另一因素定义 F3，则方法定义已变；**F2 与 F3 都重训**，并重新定义唯一差异。

### 2.3 与 Δ 的关系及唯一交互

Δ 路径没有读取 support：先由目标字体 R 的 normalized Es 与 library per-char Es 做逐字 cosine 后平均，再 top-k/softmax（`scripts/hrfont_delta_v2.py:79-110`）；随后读取 top-k donor 的 `ec|target|font|cp`，减同字符 `ec|content||cp`（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:174-194`；`scripts/hrfont_feature_cache.py:201-210`）。所以 support 的聚合/逐 glyph 改动与 `Δ=Σ_s α_s Ec(B_s,c)-Ec(Content,c)` **构造正交**。

现有数量：Δ 不是 10 个 attention token。top-10 donor 在进入 UNet 前已按 α 混成每尺度**一张** Δ feature map；support 则最多 8 个独立 1024-D token，拼到原 9 个 style token 后形成最多 17-token up-path context（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:180-194`；`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:233-248`；`code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py:60-66`）。RSI 的 Δ 通过两个 up block各 3 层的 offset interpreter消费，而 support/style context 通过相邻的 up-path cross-attention消费（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:750-782`）。

因此唯一实质交互不是“Δ token 与 support token 直接竞争”，而是同一 up block 中先由 Δ 改 skip，再由 style+support attention 改 hidden state；joint training 可使两路功能耦合。若将 support 从 8 增到 64/128，support 在 context 中的名义槽位占比由 `8/17=47.1%` 升至 `64/73=87.7%` 或 `128/137=93.4%`，容易压低 9 个 global style token 的 attention mass，间接改变网络对 Δ 后特征的处理。

缓解优先级：

1. 独立 support cross-attention，再用 zero-init gate 加回，不与 global style 共用 softmax；
2. 若必须拼接，保持 K≤8/16、对 support key/value 乘固定较小 gain（如 0.25，需 val 冻结），并记录每层 global/support attention mass；
3. matched 比较 `8 token`、`64 token` 与等参数 no-support adapter，避免把新增容量误判为 support 信息收益。

---

## Q3. F3 support 字的来源：不是 Ec 最近邻

### 3.1 bank 构建与 provenance

当前训练入口只把 `artifacts/f0/support_bank.json` 当外部必需文件：F3 缺文件就 fail closed，然后将路径透传给 `train.py`（`scripts/launch_cn2west_f123.py:49-62`；`scripts/launch_cn2west_f123.py:91-103`；`scripts/launch_cn2west_f123.py:141-176`）。`_load_support_bank` 只解析 JSON 的 `support` dict，不构建、不检索、不验证 lineage 或 ref8 provenance（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:197-211`）。

对 `scripts/` 全量代码核查的结果是：**没有任何脚本写出当前 `support_bank.json`**。旧 `scripts/hrfont_support_utils.py` 是另一条 E0/Stage-B 检索实现：它依赖 `data/hrfont/e0_bank`，以 gap、cover、MMR 和跨字体 α 选 support（`scripts/hrfont_support_utils.py:11-24`；`scripts/hrfont_support_utils.py:87-108`；`scripts/hrfont_support_utils.py:139-165`），但 F123 launcher/train 没有调用它。

当前 bank 的构建规则只在 D7 落档：每个 target 字映射到 8 个同字体 style-pool 字符，优先 `永和书风骨韵天地`，用于查 `ec|style|{font}|{cp}`；并明确“不是 E0 跨字检索”（`reports/DECISION_POINTS_20260906.md:56-63`）。这是一项**provenance 缺口**：规则有报告，无可复现 builder、输入 manifest SHA、输出 SHA/覆盖统计。不能回答成“由某个 scripts 下脚本构建”；事实答案是“仓库没有该脚本，疑似一次性生成”。

### 3.2 训练消费逻辑

对 batch 中每个 `(font,target_cp)`：

1. 从 `bank[target_cp]` 顺序截前 `support_k`，默认 8；
2. 对每个 support cp 读取 `ec_cache.features("style", font, scp)`，即 key `ec|style|{font}|{scp}`；缺 key 就跳过；
3. 每个 glyph 的 5 尺度 Ec 各做空间均值并 concat，得到 `3+64+128+256+256=707` 维向量；
4. 对 glyph 维 stack/pad，再用 `SupportAdapter(707→1024)` 得到 `[B,K,1024]` tokens（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:48-49`；`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:214-248`；`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:498-502`）。

这些 token 只拼进 up-path 的 style cross-attention context，MCA down-path继续读未修改的 style map（`code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py:60-71`）。

### 3.3 判定与契约冲突

**判定：当前 F3 是固定同字体最多 8 字（约等于 ref8）的 own-font support；不是按 Ec 最近邻检索，不是跨字体检索。** `bank` 甚至只以 target cp 为 key，不接收当前 font embedding 或距离；唯一与 font 相关的动作是取该 font 对应 support cp 的 Ec（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:228-238`）。

冲突仍在：exec-spec 明写“Support 图不来自目标字体，q≠c 且 q 非当前 style ref”（`.cursor/rules/hrfont-execution-spec.mdc:57-62`）；运行代码与 D7 则相反。审查文档 D-B5 推荐修订 exec-spec 为 own-font 8-char 并计入可见参考预算，但当前权威 spec 尚未被修订，故不能称“D-B5 已拍板并落档”；最多说“D7 已形成运行决策，D-B5 给出推荐，规范冲突未闭环”（`reports/REVIEW_BANK_STYLE_20260906.md:157-160`）。

### 3.4 隐含冗余与检索改造边界

- style 条件与 support 都来自目标字体 own-font refs；若 bank 优先 ref8，二者通常是同一 8 字。style 路把各 ref 的 `[1024,3,3]` 先逐元素 mean，形成 9 global token（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:128-136`）；support 路则把同批 ref 的 Ec 各自 pool 成 8 token。信息并非逐元素相同（Es vs Ec、平均 vs per-glyph），但可见图预算完全重叠，F3 的增益应称“同 ref 的额外 Ec/per-glyph 通道”，不能称“新增检索证据”。
- 若改成“Ec 最近邻/跨字体检索”，至少需要：E0 bank（候选字体×字符的冻结 Ec/Es 与 manifest/SHA）、目标/库同空间的 query 定义、目标字体 LOO、lineage 排除、distance/temperature/top-k 的 calibration-only 冻结、coverage/MMR 或同字符约束、cache parity 与错误检索干预。旧工具展示了 gap/MMR/跨字体 α 的一套不同语义（`scripts/hrfont_support_utils.py:32-55`；`scripts/hrfont_support_utils.py:111-136`），不能无验证地接到 F3。
- D7 已明确当前 F3 80k 不能当 E0 检索对照（`reports/DECISION_POINTS_20260906.md:60-63`）。要比较 fixed-own-ref8 与 Ec-kNN/cross-font support，必须从同一 F0、同预算/随机契约分别重训 matched arms。

---

## 4. 决策清单（待 PI 冻结）

### D-SD1｜v5 ref 聚合器

- A. DeepSets mean + projector。**推荐**：置换不变、少参数、与 SC-R prototype 连续。
- B. attention pooling。仅作 matched ablation；小池下更易过拟合。

### D-SD2｜v5 主打分头

- A. rank-128 共享/低秩 bilinear + temperature。**推荐**：可解释且参数受控。
- B. `q,r,|q-r|,q*r` 大 MLP。表达力高，但更易记渲染捷径。

### D-SD3｜GT 的地位

- A. 主判定只读 `(R,P)`；G 只进训练正锚/一致性和校准集可选 `p_aux`。**推荐且不可越线**。
- B. 主推理强制 `(R,P,G)`。不推荐：改变 estimand 并退化为 GT 保真。

### D-SD4｜CV 解释

- A. group 5-fold/LOFO；每 fold 除 held-out lineage 外全量训练，跨 fold 覆盖全部外部池，单 seed。**推荐**。
- B. 同一个全池模型在见过的 family 上“CV 打分”。不推荐：不是泛化测量。

### D-SD5｜v5 loss

- A. `L_ctx + L_mem + 0.1L_gt`，family-balanced、同族全多正例、异 lineage 负例。**推荐**。
- B. 保留对角线 InfoNCE，只换更大 backbone。拒绝：未修同族假负例。

### D-SD6｜style local token 数与注入层

- A. `9 global + 64 local`，local 只进 6 个 up attention，zero-init gate。**推荐首测**。
- B. `9+128` 且同时进 down/mid/up 11 层。不推荐首测：注意力成本与归因同时放大。
- C. 保持 9 token。作为强制 matched control。

### D-SD7｜local 初始化与训练

- A. projector Xavier + residual scalar/channel gate zero-init，新增分支 1k–2k warmup。**推荐**。
- B. 全分支随机初始化直接全 lr。风险较高，step0 不再保持旧语义。

### D-SD8｜support 去平均的重训范围

- A. F2 不变，只从同 F0 重训新 F3，并证明 batch/RNG 可与 F2 复现。**推荐**。
- B. 把 support 也纳入 F2；则 F2/F3 都重训并重写唯一因素。仅在 PI 决定改主线时采用。

### D-SD9｜support softmax 竞争

- A. support 独立 cross-attn + zero-init residual gate。**推荐于 K>8**。
- B. 继续与 9 global token concat。仅适合 K≤8/16，并强制报告 attention mass。

### D-SD10｜当前 F3 support 的论文命名

- A. “own-font ref8 Ec exemplar tokens / 同参考集额外 Ec 通道”。**推荐且符合事实**。
- B. “Ec nearest-neighbor support retrieval”。禁止，当前代码没有检索。

### D-SD11｜D-B5 契约闭环

- A. 修订权威 exec-spec 为 own-font 8-char，并把 8 张计入总 ref budget；补 builder、manifest/SHA/覆盖统计。**推荐，匹配已跑 F3/D7**。
- B. 坚持 support 不来自目标字体；则当前 F3 判为 deviation，另建合规 arm 并重训。

### D-SD12｜support bank builder

- A. 新增独立、确定性的 builder（本任务只设计，不实施）：输入 style charset/ref8/过滤表 SHA；输出 target_cp→ordered support_cp、规则版本、coverage、SHA；缺字 fail closed。**推荐**。
- B. 继续依赖一次性 JSON。拒绝进入正式复现实验。

### D-SD13｜fixed support vs 检索 support

- A. 当前 fixed own-ref8 F3 保持为主线定义；E0 Ec-kNN/cross-font 另做 matched arm。**推荐**。
- B. 把当前 F3 事后重命名为检索对照。禁止；D7 已明确不可如此使用。

## 5. 最小实施顺序

1. 先冻结 D-SD1–D-SD5：外部 ≥60 lineage manifest、fold、episode schema、loss/temperature/output schema；实现 v5 后先过数据 leakage 与同族 mask 单测。
2. 在现有 9-token 生成器不变时验证 E12 v5；主门通过前不以其选择生成方法。
3. 再做 D-SD6 的 `9 vs 9+64` matched probe；不要与 support 检索改造同时发生。
4. 关闭 D-B5：要么修 spec + 补 builder，要么将旧 F3 标 deviation。之后才能设计 fixed ref8 vs Ec-kNN 的正式 support 对照。

