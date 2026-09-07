# E12 独立风格评测器优化计划（2026-09-07）

> 审查对象：HR-Font `84f3eafd`；本计划只设计后续实验，不改动现有代码、配置或门限。
>
> 结论先行：当前 T2 失败不是单纯“26 套仍太少”，而是 **训练目标与 family-style 不一致（同族假负例）+ 正式测试含训练未见的数字 + 只有 5 个固定 held-out 字型且每族只配一个负字型**。第一注应押在“family-aware supervised contrastive / multi-positive batching + 与正式 query 对齐的字符轴 + family-clustered 多 split 诊断”，而不是盲目延长训练或先换大 backbone。0.90 不应事后放宽；应扩大独立字型池，并把统计验收口径现在就冻结。

## 0. 证据边界与当前快照

- v4：26 个 manifest face/group、4316 张 96×96 glyph（104 汉字 + 52 Latin + 10 digits）；三个训练 seed 的 T2 为 `0.791/0.795/0.725`，均值 `0.770`、seed SD `0.039`；T1 为 `0.729/0.773/0.533`，T3 均为 `0.977`（62 类），T4 均通过。[v4 复盘](./E12_SELFTEST_V4_REVIEW_20260907.md)
- v2 的表面 T2 `0.828–0.910` 不可与 v3/v4 横比：它按 face 名切 split，使 Noto 同一 typeface 的不同 weight 横跨 train/test；负例又常是相邻 weight。[v2 复盘](./E12_SELFTEST_V2_REVIEW_20260905.md)
- v3 修复为 typeface-group 隔离与跨 group 负例后，T2 `0.694/0.704/0.798`，且 per-font AUC 呈粗字重近 1、轻/常规近随机的双峰，说明此前主要学到笔画粗细。[v3 复盘](./E12_SELFTEST_V3_REVIEW_20260906.md)
- 本报告中的“预期增益”是用于排序实验的区间，不是承诺；没有新的 ablation 数据时不能把它写进论文结论。

## 1. T2 失败诊断

### 1a. T2 到底测什么、如何 held out、历史泄漏是否已消除

正式 T2 的计算是：对每个测试字型，用 8 个中文参考字 `永和书风骨韵天地` 的 φ_s2 embedding 均值归一化成 prototype；将该字型的 52 个 Latin + 10 个 digit query 与正确 prototype 的 cosine 放入 positive 集合，并与一个确定性选择的“错误 typeface” prototype 的 cosine 放入 negative 集合；最后把所有 positive/negative **跨字型池化**计算二元 ROC-AUC。它不是逐 pair 的正确排序率，也不是 membership verifier AUC。[`self_tests.py:10`](../scripts/eval_framework/self_tests.py#L10)

当前 split 不是 family-name hash，也不是逐样本 hash。实现先用 manifest `group`（否则用正则剥离 weight/style 后缀）形成 typeface units，对排序后的 group 列表用 Python `random.Random(split_seed).shuffle`，再按 0.70/0.15/0.15 切分；v4 的 26 group 因此是 18 train / 3 val / 5 test。三份 v4 配置的 `split_seed` 都固定为 3407，变化的 3407/08/09 只是初始化、batch shuffle 等训练 seed，所以三次都测试同一组五族：GenRyuMin2TC、GenYoGothic2TC、Iansui、LXGWNeoZhiSong、LXGWXiHeiCL。[`data.py:103`](../scripts/eval_framework/data.py#L103)；[`v4 config`](../configs/e12_phi_s2_v4_s3407.yaml)

v2 的直接泄漏渠道已经修掉：`assert_disjoint_splits` 同时检查 face 与 group 交集；T2 的 wrong 也要求不同 group。[`data.py:128`](../scripts/eval_framework/data.py#L128) 但仍有四个效度问题：

1. 每个测试 family 只由 family-name SHA256 在全池选 **一个** wrong prototype；结果对这五个偶然 pair 很敏感，且 wrong 可来自 train、val 或 test。后者不是 checkpoint 泄漏，却使难度取决于“抽到哪个错族”，不能代表对候选池的普遍 family discrimination。[`data.py:139`](../scripts/eval_framework/data.py#L139)
2. T2 prototype 会为 `all_f`（包括测试族）读取 8 个 reference glyph。这符合实际任务“给定目标字体 ref8 再判断 query”，不是标签泄漏；但必须明确 estimand 是 **reference-conditioned membership on unseen typefaces**，不是无参考的字体识别。
3. 训练时 early-stop val 的 anchor 属于 held-out val，但 wrong 来自 train∪val，且最多只评 128 个固定 pair；不触碰 test，所以没有 test-selection leakage，却是一个方差较大、负例分布与正式 T2 不完全一致的选择器。[`train_style_encoder.py:13`](../scripts/eval_framework/train_style_encoder.py#L13)
4. 26 个 manifest 条目虽按文件名成为 26 group，却包含多个 LXGW/ButTaiwan 衍生或近缘设计，以及 variable/single face 混合；“26 个文件”不自动等价于 26 个统计独立的设计谱系。下一版必须人工审计 lineage、script/region、serif/sans/handwriting/display 与 weight，而不能只信 stem/group 正则。[字体来源清单](../artifacts/e12/fonts_manifest.json)

### 1b. self-test JSON 真正包含什么

三份 JSON 只含：模式、是否 group split、负例类型、五个 test family 名；T1 median/p5；**一个 pooled T2 AUC**；T3 总 accuracy；T4 mean drop/t/n；同 typeface 诊断均值/n；逐门布尔值与总 pass。它们不含 raw same/different scores、每字符/每 family AUC、所选 wrong family、置信区间、bootstrap、混淆矩阵或 checkpoint 的 val 曲线。因此仅凭落盘 JSON 无法判断 v4 是否仍为 v3 的“双峰族”，也无法做正确的 family-cluster bootstrap。[v4 JSON 目录](./e12_v4_results/)

可从 JSON 确认的分布摘要只有 T1：三个 seed 的 median `0.729/0.773/0.533`，p5 `0.487/0.499/0.301`；T2 只有 `0.791/0.795/0.725`。每 seed 的 T2 名义上有 5×62=310 positive 和 310 negative，但同一 family 共用一个 prototype 和一个 wrong，字符也共享 encoder，绝不能当作 310 个独立观测。T3 的 0.977 同样只有 aggregate；训练脚本另会产生 per-class accuracy/confusion matrix，但 self-test JSON 没有搬入它们。[`train_id_cls.py:11`](../scripts/eval_framework/train_id_cls.py#L11)

### 1c. 为什么 T3 饱和而 T2 停滞

这是两个难度与不变量完全不同的任务。

- T3 是单图 62-way 字符分类：A 与 B 的拓扑差异跨字体稳定，ResNet18 可用轮廓/部件直接识别。训练集约 18×62=1116 张、3000 step，测试虽换字体，字符标签仍具强共享结构；0.977 因而主要证明 content 可读，不证明跨 script family style 可分。
- φ_s2 从零训练的正对是“同一 family 的一个中文 glyph + 一个 Latin glyph”，必然是不同字符；`keys` 为 family×104 汉字，Latin 由全局 index `%52` 固定映射，每个 family 每个 Latin 恰好重复约两次。代码保存了 `seed` 却未用于配对，所以所谓不同训练 seed 没有改变 pair graph。[`data.py:166`](../scripts/eval_framework/data.py#L166)
- 最关键的是当前对称 InfoNCE 把 batch 对角线视为唯一正例，batch 内所有其他项均为负。18 个 train family、batch 32 时同一 family 通常会出现多个中文 glyph；这些本应共享 family style 的样本被互相推开。粗略地，均匀采样下每个 anchor 期望约 `(32-1)/18≈1.72` 个同族假负例。目标实际鼓励“记住具体 glyph-pair/内容”而不是把一个 family 聚成簇。[`train_style_encoder.py:52`](../scripts/eval_framework/train_style_encoder.py#L52)
- φ_s2 训练 query 只有 52 Latin，正式 T2 却额外评 10 digits；这 10 类未参与 style alignment。T3 训练了 digits，故两门字符覆盖并不对称。应分别报告 Latin-only 与 digit-only，不能让 OOD 字符静默改变 T2。
- 26 group 中训练仅 18 个；v3→v4 扩池却几乎未动 T2，说明“池小”仍影响覆盖和统计功效，但已不是唯一解释。若 loss 持续制造同族假负例，再加字体只会稀释冲突，未必学到 ref-conditioned family invariance。
- ResNet18 的 512-D 表达能力足以把 62 个字符分开，甚至可能太容易保留 content；当前 projection 是 Identity（backbone 本身即 512-D），没有非线性 projector 来让 contrastive loss 的内容捷径留在 projector 而保留可泛化 backbone feature。[`models.py:16`](../scripts/eval_framework/models.py#L16)
- 渲染采用每字体 height-fit、居中、单一 96px、无 augmentation；它减少尺度差但也留下 ascent/bbox/墨量等字体级捷径。正式 A-like 图像若有抗锯齿、缩放、平移或灰度差异，当前训练没有验证鲁棒性。[`data.py:48`](../scripts/eval_framework/data.py#L48)

### 1d. 0.90 门是否适合 N=26：功效与决策

`AUC≥0.90` 作为“可进入主结论的效度下限”并不因当前失败而失当；它回答尺子是否有足够强的 family discrimination，而不是显著高于随机。不能用 v4 的 0.77 倒推放宽门。

问题在验收设计的统计功效，而非 0.90 这个效应量：N=26 经固定切分只有 5 个真正独立的 test family。即使每族 62 字，family/style 是聚类单位；五族下 family bootstrap 极粗、任一难族可移动整体 AUC，且三个模型 seed 复用五族只测模型方差、不测字体抽样方差。当前三 seed 均值 0.770 的常规 t 区间本身也很宽（n=3），更不能声称接近 0.90。理论上 AUC=1 在 N=26 也“可达到”，所以 N 不决定性能上限；N 决定我们能否可信估计总体泛化。

建议 **不设置 pool-size-conditional 的更低效应门**。现在预注册一个“证据充足性条件”而非放宽门：

- 主门仍为 pooled macro/family-balanced AUC ≥0.90；同时要求预先冻结的多个 outer group splits 上三 encoder seed 的中位 AUC ≥0.90，且最差预注册 split 不低于 0.85（后者是稳定性护栏，不替代主门）。
- 以 family 为 resampling unit 报 95% cluster-bootstrap CI；字符级 bootstrap 只作补充。若池仍为 26，结果标记为 pilot，不把 CI 下界硬设为 0.90（五族几乎不可能有有用精度），但也不准进入主结论。
- 正式证据池优先扩大至 **≥60 个经 lineage 去重的独立 typeface，test ≥15（更理想 ≥20）**，分层覆盖设计类别；锁定 pool/分层/splits 后再训练。若资源限制只能用 26，则采用 leave-group-out/repeated 5-fold 估计并明确是有限外部池性能，不能把偶然五族 pass 当泛化证明。

这不是 post-hoc relaxation：0.90 保持不变，新加的是在看下一轮结果前冻结的抽样与不确定性规则。

## 2. 优化选项评估

所有“独立性=是”的方案均只允许外部 OFL/free typeface、公开渲染和人工预先定义的类别；禁止 A/260 glyph、生成结果、方法 Es/Ec 或任何 method encoder feature 参与训练、hard-negative mining、早停和选型。Mac 成本按 96×96、MPS prototype 的相对量级估计；正式三 seed/多 split 应转 GPU。

| 选项 | 机制 | 预期 T2 增益（绝对 AUC） | 主要风险 | 努力 / Mac(MPS) | 独立性 |
|---|---|---:|---|---|---|
| 扩池到 ≥60、lineage 去重 | 增加设计谱系与 held-out family 数，覆盖 serif/sans/kai/song/hand/display、简繁日韩与可靠 Latin companion | +0.03–0.10；更重要是 CI 可解释 | 若只是同源衍生/更多近重复，收益很小；授权/coverage 审计耗时 | 高；cache 可在 Mac，单 run 小时级、全矩阵宜 GPU | 是 |
| **family-aware multi-positive SupCon（首选）** | batch 内同 family 的所有 CN↔Latin 配对均为正；不同 family 才为负；用 family-balanced sampler | **+0.08–0.18** | 太强聚类会只学墨量；需 held-out 验证 | 中；ResNet18 MPS 数小时级可原型 | 是 |
| 难负例课程 | 先随机跨 group，再加入同 script 近似但非同 lineage、serif/sans 内近邻、跨地区 lookalike；保留同字重为鲁棒诊断而非负类 | +0.03–0.10 | 人工“近似”标签带偏；假负（实际同源） | 中高；MPS 可训，mining 只用外部标签/像素 | 是 |
| T2 全候选负例/分层报告 | 每 query 对所有错误 prototype 打分，报 macro per-family AUC、hardest-k 与类别分层，不再每族一个 hash wrong | 不直接提升；可消除约 ±0.05 的配对偶然性 | 计算与相关性上升；必须 cluster CI | 低；Mac 分钟级 | 是 |
| query/ref 字符轴对齐 | 训练加入 digits；formal 分报 Latin/digit；若主任务含 kana，再先冻结 kana endpoint 后扩展 | +0.01–0.05，digit 子集可能更大 | 字符覆盖不全；新增 script 改变 estimand | 低；MPS 可 | 是 |
| 双向/多字符 cross-script episode | 同 family 随机采 CN↔Latin（及预注册 kana/digit），一个 ref-set 对多个 query；不固定 `%52` pair | +0.03–0.08 | 采样复杂；字符频次失衡 | 中；MPS 可 | 是 |
| A-like 轻增强 | ±3°、±5% scale/translation、轻灰度/抗锯齿/线宽扰动；范围由公开渲染协议先验冻结 | +0.01–0.04，主要降域偏移 | 过强会抹去真实笔画风格；不可用 A/260 调范围 | 低中；MPS 约 +20–40% | 是 |
| 延长至 6k–12k step | 在修正目标后扩大预算，并按 family-balanced val 早停 | 现状 0–0.02；修正 loss 后 +0.01–0.04 | 当前目标下加剧假负/记忆化 | 低；MPS 小时级，正式 GPU | 是 |
| SupCon vs triplet/margin ablation | SupCon 利用多正多负；batch-hard triplet 直接优化正负 margin | +0.04–0.12 | triplet 对 margin/miner 敏感；后验挑最好 | 中 | 是 |
| τ 调度（0.12→0.05） | 前期降低过硬负例梯度，后期增强分离；只在 val outer-train 调 | +0.01–0.04 | 小 val 上过拟合 | 低；MPS 可 | 是 |
| 非线性 projector | 512→512/256 MLP+BN/ReLU，测试 backbone、projector 或预注册融合 | +0.02–0.06 | head 吸收任务但下游取错层；BN 小 batch 不稳 | 低中；MPS 可 | 是 |
| LR/warmup/早停重审 | cosine LR；3e-4/1e-4，小网格；warmup 5–10%；early stop 用多负例 macro-family AUC，patience 按完整验证周期 | +0.01–0.04 | 多重比较；现 val 仅3族 | 低 | 是 |
| encoder 三 seed ensemble | 锁定三 checkpoint，先平均 L2 embedding 再建 prototype；同时报告单模型分布 | +0.01–0.04 | 不能把同 test 上挑 seed 当 ensemble；3×推理/存储 | 低中；Mac 推理可 | 是 |
| ResNet34/50 | 增加局部笔画建模容量 | +0.00–0.05 | 18 train family 下更易过拟合；MPS 慢 | 中；R34 可，R50 较慢 | 是 |
| ViT-S/patch 8 | 利用全局布局与笔画关系 | +0.02–0.08（数据足够时） | 96px、小池、从零训练不稳；最好需外部通用预训练，须审计来源 | 高；MPS 可原型但慢 | 条件是（预训练不得含 A/260/方法） |
| 多尺度特征 | 低层笔画纹理 + 高层整体比例，分层池化后归一 | +0.02–0.06 | 更易吃渲染捷径；实现/校准复杂 | 中 | 是 |
| ID+style 多任务 | 同 backbone 加 content adversarial removal，或 style branch + ID branch 正交约束 | -0.02–+0.06 | 普通 ID 辅助反而强化 content，伤 T2；不是首选 | 高 | 是 |
| frozen φ 的 family probe | 在 outer-train family 上学 bilinear/MLP pair scorer，outer-test 完全未见；比较 cosine 基线 | +0.02–0.08 | 若 head 类别式记忆 train family，不泛化；不能在 test 校准 | 中；MPS 易 | 是 |
| learned bilinear vs cosine | 学 `qᵀWr` 或对角 Mahalanobis，正则化、nested CV | +0.01–0.05 | 参数多、校准/过拟合，削弱“简单尺子”叙事 | 中 | 是 |
| φ_s2 + ID-CLS 联合 verifier routing | verifier 同时吃 style embedding、ID logits/embedding，用 ID 判断/置信度作条件或拒答 | family T2 预期 0–+0.03；综合 membership +0.02–0.07 | content shortcut；若用于 style 分数会混轴，且 ID-CLS 是另一独立模型而非方法模型 | 中高；MPS 可 | 是，但不得把联合分数称纯 style |
| DeepSets membership 替代 pairwise cosine | 冻结 φ，对 ref embeddings 做 mean/max 聚合，融合 `q,r,|q-r|,q⊙r`；训练 hard-negative episodes | +0.03–0.10（membership AUC） | **不能拿它的 AUC冒充原 T2**；须新增门、nested held-out 和校准 | 中；MPS 可 | 是 |
| membership 与 pairwise 双路 | cosine T2 保留核心效度门；verifier 作为预注册次指标，要求二者方向一致 | T2 不变；提升实用判别 | 指标增多/选择性报告 | 中 | 是 |

补充判断：**先加数据但不改 loss 不够**。v3 的 8 group 扩到 v4 的 26 group，T2 只从约 0.69–0.80 到 0.73–0.80；这既说明覆盖仍不足，也强烈提示 pairing/negative objective 是当前更近的瓶颈。数据扩池应与 family-aware loss 同一阶段做，但 cheap ablation 先在现池证明方向。

## 3. 分阶段执行计划

### Stage 1（1–3 天）：便宜诊断 + 数据/目标修复

先冻结一份新 protocol registry（在任何新结果出现前）：外部 pool manifest/lineage、字符集合、5 个 outer group splits、每 split 的 train/val/test、所有错误 prototype 或预注册 hardest-k 规则、macro-family AUC、family-cluster bootstrap、三训练 seed 的聚合与 gate 规则。现有 v4 结果不重解释、不改 gate。

然后依次做：

1. 导出 raw score 表：`encoder_seed, split, test_family, query_char, query_script, true_score, wrong_family, wrong_score`；给出 per-family、Latin-only、digit-only、style-category AUC，以及 family bootstrap CI。
2. 运行无需重训的 T2 重算：单 hash wrong、all-wrong、同类别 hard negatives、macro-family vs pooled；检查五族是否双峰、数字是否显著拖累。
3. 在相同 ResNet18/3000-step 下做最小 matched 2×2：原 InfoNCE vs family-aware SupCon；固定 pair vs 随机 multi-positive；所有 run 用相同 outer split/seed。加 digits 作为预注册对齐版本。
4. 审计 26 字型：去除 Han-only/fallback、实际缺字替代、同源衍生与异常 italic/mono；按设计谱系和类别标注。并行准备 ≥60 pool，但不得混入 A/260。

预期：协议修复后的 ResNet18 T2 `0.84–0.90`；如果只重算 all-wrong，数值可能升也可能降，其作用是降偏而非“刷分”。Mac：单个 3k-step run 预计数小时，先一 split×一 seed；正式矩阵用单 GPU 约 1–2 天墙钟。**Go**：至少 2/3 训练 seed 在预注册 pilot splits 上相对原 loss 提升 ≥0.05，且 Latin/digit、per-family 改善不是由一族驱动；进入 Stage 2。**No-go**：增益 <0.03 或只改善 easy families，则停止堆 step，优先扩池/重审字体 correspondence。

### Stage 2（3–7 天）：训练与架构的受控搜索

只在 Stage 1 胜出的 family-aware protocol 上做小型、预登记网格：

- loss：SupCon 与 batch-hard triplet 二选一；τ 固定 0.07 vs `0.12→0.05`；
- projector：Identity vs 2-layer MLP；LR `1e-4/3e-4`，6k cap，10% warmup；
- backbone：ResNet18 vs ResNet34；仅当 ≥60 pool 到位且 R34 仍欠拟合，再试 ViT-S；
- scorer：cosine 为必须保留的主门；bilinear/frozen DeepSets 只作 nested outer-test 的候选次指标；
- 最终可冻结三 seed embedding ensemble，但验收同时必须披露单 seed。

所有选择只看 outer-train/val；outer-test 每候选只开封一次。预期最佳 cosine T2 `0.88–0.93`，DeepSets membership AUC `0.90–0.95`（二者不得混称）。Mac：R18/R34 单 run 数小时至半天；ViT-S 与多 split 正式搜索转 GPU，约 2–4 GPU-days。**Go**：冻结方案在 ≥15 个 lineage-independent held-out families、预注册 outer splits 上达到主 AUC ≥0.90，稳定性下限 ≥0.85，T1/T3/T4 同时过门，且 family-cluster CI/分层结果无灾难子群。**No-go**：最佳诚实 outer-test 仍 <0.88，或 ≥1/3 families 接近随机，则不再通过 scorer 复杂化追门，进入 Stage 3 fallback。

### Stage 3（立即预备，Stage 2 后定案）：验收与论文有效性

路径 A（T2 过门）：锁定 checkpoint/config/code/pool/split SHA；E12 只读所有方法输出；主表可报告 φ_s2 SC-R/SC-Gap，并与盲化人评交叉验证。若 E12 被用于 α，则它不再是对该 α 完全独立的评测证据；只能按设计稿做 matched 消融，并以人评 + 不共享的 ID/quality 为主。[E12-Δ 审查](./DESIGN_E12_DELTA_20260907.md)

路径 B（T2 仍约 0.8，诚实 fallback）：

1. 将设计师 **family-match 2AFC 设为预注册 co-primary perceptual endpoint**：参考区固定显示 ref8；同一目标字符、同一字体、paired noise 下比较预先冻结的方法对；左右随机、方法匿名；按 font×gap strata 分层抽样；每题由 3–5 位合格设计师独立评价，允许“无法判断/二者皆不可用”。题本、排除规则、样本量与分析在揭盲前冻结。
2. 主分析用 mixed-effects logistic model（方法为固定效应，font/character/rater 为随机效应）或按 font 聚类的 paired bootstrap，报告 win rate、95% CI、效应量与多重比较校正；设计师间一致性（Fleiss κ）和身份/可用性题独立报告。GT 只作正控，不作唯一审美答案。
3. φ_s2 明示为 **未过效度门的预注册辅助/探索性诊断**：可在附录报告 raw 分数、与 2AFC 的相关性和失败分层，但不得进入自动主结论、不得用于挑方法/挑 checkpoint/挑题、不得把其显著性当 family-style 成功证据。
4. 主文仍可基于 2AFC 得出“设计师在本预注册样本上更常判断方法 X 与参考家族匹配”，并结合独立 ID（字符正确）与 quality（覆盖/伪影/可用性）形成三轴证据；不可声称“自动 evaluator 验证 X 更有风格一致性”“全 295 字/全字体普遍提升”或“E12 客观证明合理性”。
5. 如果 E12 曾参与 α/routing，则 φ_s2/membership 对该臂只能叫 coupled proxy；方法选择必须由不共享表征的人评和 ID/quality 完成。当前推荐仍保留 Es-α 主线，E12-α 仅在 T1/T2/T4 通过后做 20k matched ablation。[`DESIGN_E12_DELTA`](./DESIGN_E12_DELTA_20260907.md)

Stage 3 的 **Go** 是 2AFC 预注册核心比较达到计划样本量、质量控制/一致性可接受，并给出按 font 聚类的 CI；论文以人感知结论为主。**No-go** 是 κ<0.4、CI 跨越实质等效区间或身份/质量明显恶化；此时只能报告无定论/权衡，不能用未过门 φ_s2“救结论”。人力：题本与 pilot 1–2 天，正式收集依评审可用性约 3–7 天；计算成本低。

## 4. 首选方案（只押一个）

**首选：family-aware multi-positive supervised contrastive learning（配 family-balanced batch），同时把 digits 纳入配对但不先换 backbone。**

理由是它直接修复目标函数的自相矛盾：E12 要同族跨字符聚合，当前 InfoNCE 却平均给每个 anchor 约 1.72 个同族假负例并将其推远；增加训练步数会重复错误监督，换 ResNet50 会更好地拟合错误监督，单纯扩池只会稀释而不会消灭它。这个改动保持数据、参数和方法模型完全隔离，ResNet18/MPS 即可做 matched 原型，若能把 T2 从约 0.77 推到 ≥0.85，再与 ≥60 个 lineage-audited pool 合并最有希望跨过 0.90。

实现语义应先冻结：一个 family 在 batch 中的所有合法 CN↔Latin/digit views 都是 positives；只有不同 lineage/typeface 才是 negatives；同字符不要求出现（跨 script 本来无同字符），但每个 query 应跨 epoch 配多个中文字符/ref subsets，避免固定 `%52` pair 记忆。近似字型 hard negatives 只在 lineage 审计后加入。

## 5. 决策清单与不可越线

- [ ] 不改 `0.90`；先冻结新 estimand、outer splits、macro/CI 和稳定性规则。
- [ ] raw score + per-family/all-negative/Latin-vs-digit 诊断先于任何调参。
- [ ] 修 family false negatives 与固定 pairing；matched 证明 ≥0.05 增益后再扩大搜索。
- [ ] 外部池扩到 ≥60，按设计 lineage 去重，test ≥15；A/260、方法输出、Es/Ec 永不接触训练/挖负例/选模。
- [ ] backbone 先 R18→R34，ViT-S 只在数据够大后；不以复杂 scorer 替换 cosine T2 偷换门。
- [ ] 三 encoder seed × 多 group split；训练 seed 与 split seed 分开报告，不再把相同五族的三 seed 当字体泛化重复。
- [ ] E12 若用于 α，T1/T2/T4 与 SHA lock 必须在生成训练前完成；否则 E12-α 不启动。
- [ ] T2 不过：φ_s2 仅辅助/探索，人评 2AFC co-primary；禁止自动风格主张、全量外推和用 E12 选最好方法。

