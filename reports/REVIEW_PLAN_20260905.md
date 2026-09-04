# HR-Font 2026-09-05 PI 新方案独立审查

审查对象：HEAD `01e6330`；审查日期：2026-09-05。本文只评价实验设计、可归因性与实现可行性，不把尚未接线的 YAML 当作已实现事实；PI 决策文件自己也明确说明当前 Stage-A 仍是 1-token、在线 Ec、soft-ε 的旧代码（`reports/PI_DECISIONS_20260905.md:3-6,68-71`）。

## 1. 总评

**结论：方向合理，但不建议按当前文字直接批准开跑；建议“有条件批准设计、阻断正式 80k”。** 三臂是回答两个局部问题所需的最小骨架：在 n-shot 条件下比较 E2/E2b 可估计 RSI source 的简单效应，在 official-RSI 条件下比较 E1c/E2b 可估计 style protocol 的简单效应；三臂并不是完整 2×2 factorial，因为缺少 “1-shot + Δ”，所以不能估计 style×RSI 交互，也不能把 n-shot 下的 Δ 效果自动外推到 1-shot（`reports/PI_DECISIONS_20260905.md:10-23`）。

当前最大的三类风险如下。

1. **E1c 身份不诚实。** E1 的 100k 阶段是 UNet+Ec+Es 全参数训练，而 E1c 从 100k 权重重新开一个 warmup=2k、只训 UNet+offset、冻结 Es/Ec 的新阶段；它是 “E1 初始化的 frozen-encoder official-RSI control”，不是优化轨迹意义上的 “true E1 continue”（`.cursor/rules/hrfont-execution-spec.mdc:17,68-70`；`configs/e1c_ft_continue_s3407.yaml:9-11,25-38,39-57`；`code/variants/cn2west_ft_v2/FontDiffuser/train.py:149-165,207-225`）。
2. **P2 的方法叙事自相矛盾。** 新决定称候选无需“足够相似”、α 只是把 227 路缩到 K 路，但现有 rationale 又把 α 定义为风格相似度、风格近邻和经过 V1/V2/V6 验证的检索机制；而且 α 的 softmax 权重显然不仅做筛选，还直接决定 Ec mixture（`reports/PI_DECISIONS_20260905.md:25-36`；`reports/DELTA_RSI_DESIGN_RATIONALE.md:19-21,65-75`；`scripts/hrfont_delta_v2.py:99-137,157-189`）。
3. **P3 不是只换 DataLoader。** Stage-A 虽已有 `style_features` 和 `structure_features` 入口，但 MCA 的 Content 仍在 model forward 内在线跑 Ec；DPM forward 也一样。因此 cache-only 还必须新增独立的 cached content-pyramid/MCA 路径，并同步改训练、CFG、DPM wrapper、sample/val，而不只是把 RSI 的 `structure_features` 换成缓存（`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:26-68,87-124`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/unet_blocks.py:194-217,307-339`）。

此外有两个立即阻断归因的实现问题：E2 的 Δ 分支按 25% mask 置零 structure，E2b official 分支完全不应用该 mask，因此当前 E2/E2b 同时改变 “source 类型” 和 “source dropout”；YAML 的 `k_top: 10` 也没有进入 parser/launcher/`DeltaConfig`，运行时仍会使用默认 K=3（`code/variants/cn2west_stage_a/FontDiffuser/train.py:108-167,279-297`；`scripts/hrfont_delta_v2.py:30-43,117-126`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:43-67`）。

## 2. 逐项审查

### 2.1 P1 — 三臂归因

#### P1(a) “E1c vs E2b isolates style protocol” 是否成立

**结论：按“整个 style protocol package”可以成立；按“只隔离 n-shot 空间均值”目前不成立。** E1c 从 338 池用 `random.choice` 抽一张，单图同时进入 Es 与 Ec；E2b 先抽 `n~U{1..8}`、再无放回抽有序 R，style 是 R 的 n 张 Es map 均值，而 RSI 取 `R[0]`（`code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py:83-108`；`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:114-149`；`reports/PI_DECISIONS_20260905.md:12-18`）。因此该 contrast 一次改变了：参考数、聚合方式，以及 “Es style 与 Ec structure 是同一单图” 到 “Ec 只取均值集合的第一图” 的联合关系；这些可统称 style protocol，但不能把结果写成纯粹的 “多图平均增益”。

更严重的是两条现代码的 CFG 语义不同：E1 variant 在 joint drop 时把 raw content/style image 置为 1 后再跑编码器；Stage-A 则把 content pixels 和预计算 style feature 置为 0，并保留 official structure（`code/variants/cn2west_ft_v2/FontDiffuser/train.py:279-292`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:279-297`）。所以若 E1c 继续走 `cn2west_ft_v2`、E2b 走 `cn2west_stage_a`，即使都写 `cfg_joint_drop=.10`，E1c/E2b 也不只差 style protocol（`configs/e1c_ft_continue_s3407.yaml:39-57`；`configs/e2b_ft_continue_s3407.yaml:45-63`）。

**建议：**若 PI 优先追求三臂最小因果设计，应让三臂共享同一个 cache-level conditional/CFG/source-drop 实现；E1c 仅把 `n=1`，且使用同一 episode manifest 的 `R[0]`。此时它应改称 “1-shot official-RSI matched arm”，不再称原 E1 forward。若必须保留原 E1 forward/CFG，则 E1c vs E2b 只能解释为 style+conditioning package，需要第四臂才能拆开。

#### P1(b) E1c 是否是 “true continue-80k baseline”

**结论：不是；冻结编码器确实破坏 “E1 优化继续” 的身份，但不破坏 “conditional forward 拓扑沿用 E1” 这一较窄说法。** E1 的优化器接收 `model.parameters()`，Es/Ec 与 UNet 都被训练并被逐 checkpoint 保存；项目计划也明确记录 E1 “UNet+Ec+Es 全训”（`code/official/FontDiffuser/train.py:72-85,123-141,249-256`；`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:102-110`）。E1c 则明确冻结 encoder、只训 UNet+offset；同时它重启 AdamW 与 linear schedule、warmup=2k，而不是恢复 E1 的 optimizer/scheduler state（`configs/e1c_ft_continue_s3407.yaml:25-38,39-57`）。

从方法约束看，E1c 若解冻 Es/Ec 就与 cache-only 直接冲突：缓存被绑定到 arm 的 encoder SHA，而训练中 encoder 每步变化会让 cache 从第一步起陈旧（`reports/PI_DECISIONS_20260905.md:38-52`）。它还会让 E1c/E2b 同时改变 style protocol 和可训练参数集合，彻底失去第二个局部 contrast（`.cursor/rules/hrfont-execution-spec.mdc:62,68-71`）。

**建议：**保留冻结版本以服务最小三臂归因，但删除 “真基线/true continue/原 E1 再训” 字样，统一改为 “E1@100k 初始化、冻结编码器的 1-shot official-RSI continuation control”。若 PI 必须回答 “把 E1 原训练配方再跑 80k 会怎样”，应另加 E1d：解冻 Es/Ec、允许在线编码器、重置优化阶段并单独披露；它不能替代 frozen E1c，也不能进入 E1c/E2b 单变量 contrast。

#### P1(c) E2 vs E2b 是否只隔离 RSI source

**结论：设计意图接近正确，当前 runtime 尚不满足。** 两臂从同一 dataset 取得同一有序 R；E2b official 路径明确使用 `refs[0]`，E2 的 Δ 路径使用同一 R 做逐字 query、top-K 邻居与 Content subtraction；额外的邻居 feature fetch 和减法属于 “Δ source 的构造”，不是独立 treatment（`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:122-149`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:108-167`）。UNet 的改动也只是把原 `encoder_hidden_states[3]` 改成显式 `structure_features`，RSI 仍在相同两个 up blocks 通过同一 offset/DCN 接口消费它（`code/official/FontDiffuser/src/modules/unet.py:264-285`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/unet.py:265-298`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/unet_blocks.py:534-587`）。

但当前 `_structure_features` 在 Δ 分支会按 `delta_draw` 把整个多尺度 structure 置零，official 分支在返回前没有相同处理；这使 E2 相对 E2b 还多了 25% structure-source dropout（`code/variants/cn2west_stage_a/FontDiffuser/train.py:118-167,279-285`）。要声称 “ONLY RSI source”，必须把相同 source-drop mask 应用于 E2 的 Δ 和 E2b 的 `Ec(R[0])`，或者两臂都设为 0；仅“消费同一 RNG draw”不够。

还有一个配置歧义：E2 写 `init.new_layer=zero`，E2b 写 `null`，但现代码声明 direct Ec features 没有 adapter，并保存空 `delta_layers.pth`（`configs/e2_stage_a_s3407.yaml:7-9`；`configs/e2b_ft_continue_s3407.yaml:7-9`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:170-177`）。若坚持“无新增参数、复用 E1 offset”，应把两者都写 `null`；若真实现 zero-init 新层，该层本身就是 source 之外的混杂。

#### P1(d) 新设计相对旧 E2b 身份是否丢失信息

**结论：没有丢掉有效的 Δ 主对照，反而纠正了旧 E2b 一臂承担两个身份的问题。** 2026-09-04 审查已经指出 E2b 虽复用 E1 offset head，却因 n-shot pooled/1-token style 而不等价 E1 forward，并建议只把它称为 E2 的 matched official-RSI control（`reports/REVIEW_ALIGNMENT_20260904.md:49-61`）。新计划保留 E2b 作为 E2 matched control，又增加 E1c 和 E1@100k anchor 进入主表，因此能分别观察额外训练、style protocol 和 Δ package（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:9-24,127-130`）。

真正仍缺的是两点：没有 “1-shot+Δ” 就不能估计交互；若 E1c 被错误宣传为 literal E1 continuation，仍无法回答全参数 E1 再训。建议在论文中把三个 estimand 限定为 “official 下的 style simple effect”“n-shot 下的 source simple effect”“相对 frozen 1-shot control 的 package effect”。

#### P1 的额外统计风险

E1 已在 train228×295 上训练 100k，且 val loss 的最好点在 98k、100k 仅非常接近；再训 80k 有继续平台或过拟合的现实风险（`provenance/runs/E1-FTV2-A-S3407.json:7-18,24-44,57-64`）。该风险对 E2/E2b 的 matched source contrast是共同的，但可能让 E1c 退化，从而放大 “E2 package 相对 E1c” 的表面收益。建议 E1@100k 始终保留为 anchor，并把固定 80k endpoint 作为主 estimand、各臂从 10k 起独立选出的 best 作为次要 estimand；当前三份 YAML 都声明 best eligibility=10k，但训练器尚未实现 val/best（`configs/e1c_ft_continue_s3407.yaml:62-71`；`configs/e2_stage_a_s3407.yaml:68-78`；`configs/e2b_ft_continue_s3407.yaml:68-78`；`reports/REVIEW_ALIGNMENT_20260904.md:63-66`）。

相同 seed 也不会自动让 E1c 与 E2/E2b 抽到可配对的参考：前者每样本调用一次 `random.choice`，后者先 `random.randint` 再 `random.sample`，Python RNG 调用图不同（`code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py:83-92`；`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:114-131`）。E2/E2b 在不间断且相同 batch 顺序时可共享调用图，但现 checkpoint 不保存 RNG/sampler cursor，resume 后仍会失配（`code/variants/cn2west_stage_a/FontDiffuser/train.py:170-192,245-259`）。建议不用“消耗相同数量随机数”的脆弱办法，而用 `(seed,font,target_cp,global_sample_index)` 的 stateless ordered-R manifest；E1c 只读取同一 R 的第一项。

### 2.2 P2 — top-K α

#### P2(a) 内部一致性与相对旧 soft-ε 的优劣

**结论：top-10→softmax 是更清楚、不会因平坦分布被 ε 全删空的主方法，但当前字段语义和接线仍不一致。** 旧 `soft` 先对全部候选 softmax，再只保留严格大于 ε 的权重；227 个近均匀候选时每个约 0.0044，`ε=.01` 可把全部删除（`scripts/hrfont_delta_v2.py:64-76,107-129`）。新 top-K 分支先选固定 K，再只在 K 内 softmax，库不足就全取，因而只要 leave-one-out 后有候选就不会空（`scripts/hrfont_delta_v2.py:117-137`）。

不过 `eps_alpha` 在现有 topk 分支完全未使用，所以 “≤1e-6 numerical floor” 不是当前代码语义；数值稳定的 softmax 本身也不需要用 ε 截权重（`scripts/hrfont_delta_v2.py:109-126`）。建议 topk 模式直接冻结 `eps_alpha=0`，或明确规定它只用于 `clamp_min` 后再归一化，并补 finite/sum=1 检查，不能保留一个看似生效、实际 dead 的字段。

更直接的 bug 是 YAML 同时写 `k_max=10,k_top=10`，但 launcher 只传 `--delta_k_max`，parser 没有 `delta_k_top`，训练器构造 `DeltaConfig` 时也不传 `k_top`；因此当前 `mode=topk` 实际会回落到 dataclass 默认 K=3（`configs/e2_stage_a_s3407.yaml:30-39`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:50-54`；`code/variants/cn2west_stage_a/FontDiffuser/configs/fontdiffuser.py:33-39`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:122-132`；`scripts/hrfont_delta_v2.py:30-37`）。正式代码应只保留一个语义明确的 `k` 字段，避免 `k_max/k_top` 双源漂移。

#### P2(b) 第 10/11 名边界不稳定

**结论：有真实统计问题；固定 cache 下的 deterministic tie-break 只能解决精确并列，不能解决近并列。** 当前排序按 `(-score,index)`，而 library 先排序，因此精确同分时稳定；但 R 子集变化或 fp16 量化让 10/11 名交换时，整个 Ec path 会离散更换一套字体（`code/variants/cn2west_stage_a/FontDiffuser/train.py:108-132`；`scripts/hrfont_delta_v2.py:117-126`）。

建议预注册但不事后改方法：按 n 报告 `s10-s11`、R 子集 bootstrap 的 top-10 Jaccard/neighbor turnover、top-1 mass、entropy 和 effective-N；K={3,10} 消融照常报告。若 near-boundary 很多，应把结论写成 “relative top-K routing is noisy but aggregate generation is stable/unstable”，而不是在看完结果后换回 full-pool。

#### P2(c) n~U{1..8} 与 n=1 噪声

**结论：top-K 会放大 n=1 的 winner-selection noise，但不建议未经诊断就删除 n=1。** 相似度是 n 个逐字 cosine 的均值；在近似独立条件下其均值方差按 `1/n` 缩放，所以 n=1 是方差最大的 regime，而 hard ranking 会把估计噪声转成离散邻居变化（`scripts/hrfont_delta_v2.py:88-101,117-126`）。现规范均匀抽 n，意味着只有 1/8 episode 正好是 eval 的 n=8，训练与固定 n=8 评测也存在分布差异（`configs/e2_stage_a_s3407.yaml:40-44`；`.cursor/rules/hrfont-execution-spec.mdc:70-71`）。

建议先保留 n=1 以保留真正 few-shot 能力，但必须分 n 报告边界稳定性和结果；若 n=1 的 top-10 turnover 或性能显著异常，再把 `n_min=2` 作为预注册敏感性实验，而不是主结果跑后删除困难样本。

#### P2(d) τ=.07 是否合理

**结论：先验上可用，但只看 cosine 的绝对范围 0.2–0.6 无法判断浓度；softmax 只取决于分数差。** 当前实现计算 `softmax(score/.07)`（`scripts/hrfont_delta_v2.py:99-123`）。由该公式可得：分差 0.01/0.05/0.10/0.20 对应 odds ratio 约 1.15/2.04/4.17/17.4；若十个分数线性覆盖 0.10，top weight 约 18%、effective-N≈8.34；覆盖 0.20 时约 28%、effective-N≈5.84；覆盖 0.40 时约 47%、effective-N≈3.24。仓库现文档只有 “V6 应报告 nearest cosine/top-1 mass” 的验证计划，没有提交实际 score-gap 分布，因此不能从仓库证实 0.2–0.6 或 τ=.07 的实际浓度（`reports/DELTA_RSI_DESIGN_RATIONALE.md:73-75`）。

建议在任何训练前用正式 E1@100k Es cache 做只读 calibration，分 n 报 `s1,s10,s11,span,entropy,effective-N`；τ=.07 可保持预注册值，但必须用这些统计说明它实际是近均匀、适度混合还是近 one-hot。

#### P2(e) “邻居不必相似” 的叙事张力

**结论：当前表述冲突，必须改。** PI 决策称 α 只缩小 mixture search space、不要求邻居足够相似；rationale 却称其为 “按风格相似度找邻居”“风格相近的库字体”，并要求 Es 作为经验证可用的风格相似度量（`reports/PI_DECISIONS_20260905.md:25-28`；`reports/DELTA_RSI_DESIGN_RATIONALE.md:65-75`）。同时 Δ 的机制解释本身也说 “在 α 选出的风格近邻中” 学相对中性变化（`reports/DELTA_RSI_DESIGN_RATIONALE.md:19-21`）。

推荐统一成：**“α 用冻结 Es 对 train228 做相对排序，取最匹配的 top-K 并加权；不设绝对相似度阈值，因此 top-K 可能仍是弱匹配。α 同时承担 routing 与 mixture weighting，弱匹配程度由 score/gap/稳定性统计披露。”** 这既不夸大 absolute similarity，也不否认 α 的 style-ranking 作用。

### 2.3 P3 — cache-only + 9-token

#### P3(a) 9-token 是否与 E1 一致

**结论：是；新 9-token 修正了 Stage-A 的真实偏离。** 96×96 StyleEncoder 的五级输出通道为 64/128/256/512/1024、空间为 48/24/12/6/3，最终 `style_emd` 在 adaptive pooling 前保留为 `[B,1024,3,3]`（`code/official/FontDiffuser/src/modules/style_encoder.py:293-307,428-442`）。official/E1 model 把该 map permute+reshape 成 `[B,9,1024]` 进入 attention（`code/official/FontDiffuser/src/model.py:26-47`）。

旧 Stage-A `_style_conditions` 取第二返回值 pooled，先逐图 L2 normalize、对 R 求均值，得到 `[B,1024]`；model 再扩成 `[B,1024,1,1]`，即一个 token（`code/variants/cn2west_stage_a/FontDiffuser/train.py:54-61`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:36-48`）。因此 P3 的 map 逐元素均值后展平为 9 token，既保持 attention length，也只让 n 改变 map 估计，科学上更干净（`reports/PI_DECISIONS_20260905.md:54-58`）。

仍需记录每个 n 的 style-map channel norm/RMS：raw spatial maps 未做 L2 normalize，跨图平均会因字符相关成分和抵消改变条件幅度；虽然 token 数恒为 9，不能未经测量就把 n 的影响只描述成抽象的“方差下降”（`code/official/FontDiffuser/src/modules/style_encoder.py:428-442`；`reports/PI_DECISIONS_20260905.md:56-58`）。建议保持 raw element-wise mean 以最接近 E1，并把 per-n norm 与 attention 输入统计作为 smoke/诊断，而不在看结果后追加 normalization。

#### P3(b) cache-only 的模型改造范围

**结论：`structure_features` 只够 RSI，不够 MCA。** Stage-A model 已允许外部传 `style_features` 和 `structure_features`，但无论训练还是 DPM forward，都会对 `content_images` 在线调用 content encoder，再把 residual pyramid 交给 MCA（`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:26-68,87-124`）。MCA down/mid blocks按 index 从 `encoder_hidden_states[1]` 取不同尺度，故必须新增 `content_features`/named pyramid 参数，而不是只给一个 final tensor（`code/official/FontDiffuser/src/modules/unet.py:76-122,240-262`；`code/official/FontDiffuser/src/modules/unet_blocks.py:194-217,307-339`）。

E1c 的 `cn2west_ft_v2` 更没有预计算入口：其 forward 内同时运行 Es(style)、Ec(content)、Ec(style)（`code/variants/cn2west_ft_v2/FontDiffuser/src/model.py:26-58,77-110`）。因此 P3 需要对两个 variant 的 train model 和 DPM model 都做接口重构，或先由 PI 改成三臂共用一个统一 conditional core；后者更利于 matched 审计。

#### P3(c) 存储量与覆盖

**Es：**原始 payload 是 `260×338×1024×3×3×2 bytes = 1,619,804,160 bytes ≈1.51 GiB`；考虑索引、manifest、分片和文件格式开销，文档写 “~2 GiB” 是合理预算（`reports/PI_DECISIONS_20260905.md:42-45,52`；`code/official/FontDiffuser/src/modules/style_encoder.py:293-307`）。现有 builder 只缓存 train split 的 normalized pooled fp16 `[1024]`，不是 spatial、也不覆盖 val/test，必须重写（`scripts/hrfont_es_cache.py:69-89,90-118`）。

**Ec 完整 contract：**96×96 ContentEncoder 的 residual+final 实际为 `[3,96,96]`、`[64,48,48]`、`[128,24,24]`、`[256,12,12]`，再追加一次 final `[256,12,12]`；最后一项与最后一个 residual 是同一 h 的重复 contract（`code/official/FontDiffuser/src/modules/content_encoder.py:333-350,374-405,426-435`；`code/official/FontDiffuser/src/model.py:39-47`）。按 fp16 每 glyph 是 322,560 values=645,120 bytes。于是 target `228×295` 约 40.41 GiB，style `260×338` 约 52.80 GiB，Content×295 约 0.18 GiB，合计约 **93.39 GiB**，尚未含容器/索引开销；“40–90 GiB” 的下界相当于只算 target，上界略低估完整 target+style（`reports/PI_DECISIONS_20260905.md:42-52`）。

可以无损降到约 **64.06 GiB payload**：RSI 的两个 up block 实际只消费 64×48×48 与 128×24×24 两层，MCA 额外需要 256×12×12；缓存按 named scale 保存这三层并在接口重建 contract，避免保存 raw input 和重复 final（`code/official/FontDiffuser/src/build.py:15-35`；`code/official/FontDiffuser/src/modules/unet.py:127-161,255-285`；`code/official/FontDiffuser/src/modules/unet_blocks.py:534-561`）。该优化必须以 online-vs-cache 逐层及最终 noise/offset parity test 为出口条件。

覆盖表本身是对的：Delta bank 只需 train228 target；val/test target 是 GT，不应进 Δ 库；但 E1c/E2b 在 val/test 仍需 official `Ec(style)`，所以 Ec style 必须覆盖全部 260×338，当前 P3 表已这样写（`reports/PI_DECISIONS_20260905.md:42-49`；`.cursor/rules/hrfont-execution-spec.mdc:43-46`）。

**I/O 建议：**不要用数十 GiB 的 Python dict+`torch.load`；当前 Stage-A 会把整个 Es dict 读入 CPU，现 launcher也只检查这一份 `.pt`（`code/variants/cn2west_stage_a/FontDiffuser/train.py:245-254`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:31-37`）。应使用连续 indexed arrays、按 role/scale 分片、mmap/read-only batched fetch，manifest 冻结 key order、offset、shape、dtype、file SHA、encoder SHA、dataset/split SHA；GPU 只接收当前 batch 的 content/style/source tensors。

#### P3(d) fp16 cache 的正确性

**结论：可行但不是 bit-exact replay，尤其会影响 hard top-K 边界。** Es/Ec 主干使用 DBlock；spectral-normalization power iteration 只有 `training=True` 才更新 u/sv，eval 模式不会产生运行状态漂移，Style final 还有 stateless InstanceNorm（`code/official/FontDiffuser/src/modules/style_encoder.py:110-150,382-408`；`code/official/FontDiffuser/src/modules/content_encoder.py:111-159,394-406`）。这支持 “固定权重+eval mode 可离线复算”。

但把未量化的 reference feature 序列化成 fp16 后再 fetch 不会与 reference pass bit-exact；对生成分支通常可用 tolerance 验证，对 α 的第 10/11 名 hard boundary 则可能改变成员。建议 spatial/Ec cache 用 fp16，所有混合/减法/normalize/cosine 在 fp32；另存 Es pooled fp32（约 0.34 GiB）专供 α，避免从 fp16 spatial 再池化放大 ranking 抖动。必须预注册逐层 max/mean error、cosine drift、top-K agreement、noise/offset parity 阈值。

#### P3(e) inference 复杂度

**结论：运行时计算更简单，接口改造更复杂。** 当前 sample 只输入 raw content/style，DPM pipeline 的 cond/uncond 各只有两项；DPM wrapper 也只拼接这两项，虽然 model 支持 `cond[2]/cond[3]`，它们实际上从未送到 solver（`code/variants/cn2west_stage_a/FontDiffuser/sample.py:126-161`；`code/variants/cn2west_stage_a/FontDiffuser/src/dpm_solver/pipeline_dpm_solver.py:42-83`；`code/variants/cn2west_stage_a/FontDiffuser/src/dpm_solver/dpm_solver_pytorch.py:330-340`）。必须在采样前按 cache key 一次性构造 style map、MCA content pyramid 和 official/Δ structure，并让 CFG wrapper 对每一层 condition 正确拼 batch；否则 DPM 每一步仍会在线跑 encoder或 structure 默认为零（`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:95-120`）。

完成后，20-step solver 可重复使用同一批 cached conditions，不再每次模型评估重复跑 Es/Ec；但 cache-only 对未预注册的新字体不再是即时 few-shot：需要先完成一次 encoder/cache preprocessing。论文和 demo 必须明确 cache-only 是 benchmark protocol，还是对任意用户字体也强制先建 cache。

#### P3(f) Δ-drop、R draw 与 9-token CFG

cache-only 不应改变语义：R 仍按 episode 抽字符/keys，E1c 取一 key，E2/E2b 取 n keys；Δ-drop 仍以样本为单位把整个多尺度 source 置零，而不是改邻居 K 或重抽 R（`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:122-149`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:157-167,279-297`）。9-token style-drop 应一次把完整 `[1024,3,3]` map 置零，不能随机丢 9 个 token 中的一部分，否则改变了 CFG treatment。

应再冻结 “先构造 α/记录 top-K 诊断，再应用 source-drop” 的顺序：当前代码先为所有样本构造 plans 和邻居，再在逐样本 feature 阶段检查 dropped mask（`code/variants/cn2west_stage_a/FontDiffuser/train.py:116-167`）。cache-only 可以对 dropped 样本跳过昂贵的 Ec fetch/mix，但不应因此跳过 R/K 统计、改变 RNG 或让稳定性报告只覆盖未 drop 的 75%。

但 content uncond 仍未冻结：E1 原代码对 raw image 置 1 后编码，当前 Stage-A 对 normalized content pixels 置 0 后编码，而 cache-only 又可选择 “整层 feature 置零” 或 “读取 Ec(blank) cache”；三者不等价（`code/variants/cn2west_ft_v2/FontDiffuser/train.py:279-292`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:279-297`）。这必须在 PI 批准前与 source-drop 一并定义，且三臂共享同一实现才能支持 P1 的单变量措辞。

### 2.4 commit 一致性审计

#### E2/E2b YAML whitelist

结构化比较当前两份 YAML，差异**恰好四项**：`experiment.id`、`init.new_layer`、`model.rsi_source`、`model.delta.enabled`；这与 exec-spec 白名单文字一致（`configs/e2_stage_a_s3407.yaml:3-9,23-39`；`configs/e2b_ft_continue_s3407.yaml:3-9,23-39`；`.cursor/rules/hrfont-execution-spec.mdc:57-63`）。新增的 `encoder_runtime/style_condition/style_tokens` 在两臂相同，Ec/Es cache path、n-shot、训练和 eval 字段也文本相同（`configs/e2_stage_a_s3407.yaml:10-78`；`configs/e2b_ft_continue_s3407.yaml:10-78`）。

但这只是静态 YAML 成立；launcher 没有做 structural diff，也忽略 `encoder_runtime/style_condition/style_tokens/ec_cache_path/k_top/eval.best_eligible_from_step`，因此 matched whitelist 还不是可执行防线（`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:27-72`）。此外，若 “无新层” 是最终设计，四项白名单中的 `init.new_layer` 应删除而不是被永久合理化。

#### E1c YAML 可运行性

`variant=cn2west_ft_v2` 当前**不支持**该 YAML 的 cache-only/freeze/eval 语义。其 parser 没有 config/cache/freeze/nshot/eval-ref 参数；train 总步数参数本身支持 80k，且能从指定 checkpoint 加载权重，但优化器仍接收全部 model parameters（`configs/e1c_ft_continue_s3407.yaml:4-11,23-38,39-71`；`code/variants/cn2west_ft_v2/FontDiffuser/configs/fontdiffuser.py:4-36,45-85`；`code/variants/cn2west_ft_v2/FontDiffuser/train.py:149-165,207-225`）。现有 E1 launcher 还硬编码 E1 run 身份/100k 默认值，并不读取 E1c YAML（`scripts/launch_cn2west_ft_v2_e1.py:31-47,77-127`）。

最小 deltas 是：新增 config-aware E1c launcher/schema；dataset 返回 cache keys；train/model/DPM 接受 cached style/content/official structure；冻结/审计 encoder 且 optimizer 只收 UNet；接 Es/Ec SHA/coverage validator；实现固定 eval manifest、5k val、best≥10k；补 exact resume/STOP/heartbeat。`--max_train_steps=80000` 本身无需改，只需正确传入（`code/variants/cn2west_ft_v2/FontDiffuser/configs/fontdiffuser.py:52-58`；`code/variants/cn2west_ft_v2/FontDiffuser/train.py:248-266,340-378`）。

#### 其他直接接线缺口

- Stage-A 只验证 Es checkpoint SHA，不验证 Ec SHA；P3 已把 D-A1 扩展到每份 Ec cache（`code/variants/cn2west_stage_a/FontDiffuser/train.py:77-99,245-254`；`reports/PI_DECISIONS_20260905.md:42-51`）。
- Stage-A 没有 Ec cache loader/builder；现有 `_structure_features` 仍把图片 batch 在线送进 Ec，MCA 也在线 Ec(content)（`code/variants/cn2west_stage_a/FontDiffuser/train.py:108-167`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:50-54`）。
- checkpoint 只保存 step/optimizer/scheduler，没有 Python/NumPy/Torch/DataLoader/AMP RNG state；E2/E2b 中断恢复后不能继续保证 R、batch、drop、noise、timestep matched（`code/variants/cn2west_stage_a/FontDiffuser/train.py:170-192,245-259`；`.cursor/rules/hrfont-execution-spec.mdc:84-87`）。
- E1c 和 E2/E2b 的 eval/best 配置均未接入有效 sample/val；当前 DPM cond 不含 cache feature/structure（`configs/e1c_ft_continue_s3407.yaml:62-71`；`configs/e2_stage_a_s3407.yaml:68-78`；`code/variants/cn2west_stage_a/FontDiffuser/src/dpm_solver/pipeline_dpm_solver.py:42-83`）。

#### REGISTRY.md

E1c 行的 experiment ID 与 YAML 一致、状态 `planned` 合理、指针也正确（`provenance/REGISTRY.md:22-34`；`configs/e1c_ft_continue_s3407.yaml:4-8`）。但 registry 的 code variants 表没有登记 `cn2west_ft_v2`，尽管 E1/E1c 都引用它；E2b 的 ID 仍叫 `E2B-FT-CONTINUE-S3407`，与其已冻结的 “n-shot matched official-RSI、不是 E1 continuation” 身份冲突，建议在开跑前改名，因为当前尚无 provenance/run 可迁移（`provenance/REGISTRY.md:14-20,30-34`；`.cursor/rules/hrfont-execution-spec.mdc:68-71,93-100`）。

## 3. 必须 PI 决策的新增点

### D-P1 — E1c 优先服务哪种身份

- A. **三臂可归因优先（推荐）：**三臂共用 cache/CFG/source-drop core；E1c=n=1+official RSI，冻结 Es/Ec；改名为 1-shot official-RSI matched arm，不称 true continue。
- B. E1 fidelity 优先：E1c 保留 raw-image CFG、解冻 Es/Ec、允许在线 encoder；它只作 literal-ish 续训参考，另增一个 frozen 1-shot matched arm。
- C. 保持当前文字：冻结 encoder 但继续称 “原 E1 真续训”。**不推荐。** E1 全参数训练和 E1c freeze 的差异已有明确记录（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:102-120`）。

### D-P2 — structure-drop 与 CFG null 的统一语义

- A. **推荐：**定义通用 `source_drop=.25`，E2 的 Δ 与 E2b/E1c 的 official source 都用同一 mask；joint CFG 把整个 9-token style map 与全部 MCA content scales 置零，同时保留 structure；三臂同实现、独立可恢复 RNG。
- B. 两臂都不做 source-drop；只保留 joint CFG。
- C. 只给 Δ 做 drop。**不推荐：**这会让 E2/E2b 的 source contrast混入 regularization 差异；当前代码正是此状态（`code/variants/cn2west_stage_a/FontDiffuser/train.py:151-167,279-297`）。

### D-P3 — E1c 的固定评测参考

- A. **推荐：**冻结 shared ordered-R eval manifest；E1c 用 `R[0]=永`，E2/E2b 用完整 ref8，使 1-shot 集合严格嵌套于 8-shot。
- B. 冻结并提交 E1 旧 val 的 seeded-random 1-shot manifest；可复现但不与 E2b ref8 嵌套。
- C. `eval_refs=null`、每次重抽。**拒绝：**当前 YAML 正是 null，无法定义可复现 estimand（`configs/e1c_ft_continue_s3407.yaml:34-38`）；E1 历史 val 至少每次固定重置 seed 后按固定顺序抽样（`scripts/build_e1_train_dashboard.py:568-597`）。

### D-P4 — α 的论文定位

- A. **推荐：**“relative Es-ranked top-K；无 absolute threshold；同时做 routing+weighting；弱匹配需披露”。
- B. “邻居无需相似，α 只是降维”。**不推荐：**与现有 rationale 和 α-weighted mixture 不一致（`reports/PI_DECISIONS_20260905.md:25-28`；`reports/DELTA_RSI_DESIGN_RATIONALE.md:65-75`）。
- C. 恢复 absolute similarity threshold。会重新引入空邻域语义，不符合 P2。

### D-P5 — n=1 与 top-K 边界稳定性

- A. **推荐：**保留 U{1..8}；预注册按 n 的 `s10-s11/Jaccard/turnover/effective-N`，K={3,10} 附录，不设事后 fallback。
- B. 主训练改 U{2..8}，n=1 只做 robustness eval。
- C. gap 小时动态扩大 K/切 full-pool。**不推荐：**这新增未冻结的样本依赖方法分支。

### D-P6 — cache layout 与精度

- A. 存完整 residual+final fp16，约 93.39 GiB payload；α pooled 单存 fp32。
- B. **推荐：**只存 named consumed scales，约 64.06 GiB payload；α pooled fp32；用逐层与 end-to-end parity gate 证明语义等价。
- C. 全 fp32，约翻倍；最稳但 I/O/磁盘成本最高。当前 content/RSI 的实际索引只需要三个唯一尺度（`code/official/FontDiffuser/src/modules/unet.py:240-285`；`code/official/FontDiffuser/src/modules/unet_blocks.py:534-561`）。

### D-P7 — cache-only 的产品/论文适用范围

- A. **推荐：**正式 benchmark 绝对 cache-only；新字体先走单独的 preprocessing/cache-build stage，生成时不在线编码。
- B. benchmark cache-only，但 demo/外部用户允许在线 encoder，并明确它不属于正式计时/评测协议。
- C. 宣称任意 unseen font 即时 few-shot，同时禁止任何预处理。两者不可兼得，因为当前所需 Es/Ec key 必须先存在（`reports/PI_DECISIONS_20260905.md:38-49`）。

### D-P8 — checkpoint estimand 与 seeds

- A. **推荐：**固定 80k 为主比较，best≥10k 为次要；3407 只作先行，最终 matched 结论补 3408/3409，并按 font cluster 做推断。
- B. best≥10k 为主、80k 次要；仍补三 seeds。
- C. 只报 seed3407 的各自 best。**不推荐。** 全局规范仍写 seeds=3407/3408/3409、3407 先跑，主分析已有 font-cluster bootstrap/mixed effects（`.cursor/rules/hrfont-execution-spec.mdc:16-19,70-74`；`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:127-130`）。

### D-P9 — E2 的 `new_layer`

- A. **推荐：**E2/E2b 都设 `new_layer=null`；RSI 只换 source，复用同一 E1 offset head。
- B. 两臂都加同构 zero-init adapter。
- C. 只给 E2 加 adapter。**拒绝：**不能再声称只隔离 RSI source；现代码也明确没有 adapter（`configs/e2_stage_a_s3407.yaml:7-9`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:170-177`）。

## 4. 文档残留清单

下面是指定词及常见空格/中文别名的完整结果；排除本报告自身，避免结果递归包含本节：

```text
$ rg -n -i --no-heading -g '!REVIEW_PLAN_20260905.md' '1[ -]?token|one[ -]?token|mean[- ]pool\s*\(?Es|pooled[^\n]{0,50}1[ -]?token|mode\s*=\s*soft|eps_alpha\s*[=:]\s*0\.01|full[- ]pool|soft 全池|全池加权|D-A4|D-A6' reports/ .cursor/rules/
.cursor/rules/hrfont-execution-spec.mdc:26:- D-A4：**已作废**，由 §1.6 的 9-token 空间均值覆盖。
.cursor/rules/hrfont-execution-spec.mdc:28:- D-A6：**已作废** 空邻域 fail-fast/重定 ε；由 §1.6 必取 top-K 覆盖。
.cursor/rules/hrfont-execution-spec.mdc:39:- **style token：** n 张 `style_emd` `[1024,3,3]` 逐元素平均 → 9 token。禁止 pooled 1-token。
reports/ICLR2027_HRFONT.md:132:| 邻域 | **soft 全池** | softmax(cos/τ) 加权 + ε/K_max 截断；top-3 为消融开关 |
reports/ICLR2027_HRFONT.md:133:| α 空邻域 | **fail-fast** | 空率必须为 0，否则训练中止（D-A6） |
reports/REVIEW_ALIGNMENT_20260904.md:56:### F05 — D6 的 “mean-pool Es(R)” 还隐含了未明说的归一化/空间降维决定
reports/REVIEW_ALIGNMENT_20260904.md:59:- **问题：** 代码语义不是简单的 “mean-pool Es(R)”：它取 StyleEncoder 第二返回值（spatial average 后的 pooled vector），先逐张 L2-normalize，再跨 R 求均值，且不对最终均值再 normalize；随后只给 cross-attention 一个 token。exec-spec 没冻结这四个细节。它们对 E2/E2b 相同，所以不破坏二者 pair，但会形成相对 E1 的额外 architecture/scale transition。
reports/REVIEW_ALIGNMENT_20260904.md:154:### D-A4 — D6 style condition 的精确定义
reports/REVIEW_ALIGNMENT_20260904.md:156:**选项：** A. 当前“逐张 normalized pooled vector→mean→1 token”；B. mean raw pooled；C. mean spatial style maps。**推荐 A（尊重已实现 D6），但必须将该额外 E1→E2 transition 写入方法与控制解释。**
reports/REVIEW_ALIGNMENT_20260904.md:162:### D-A6 — soft-α 空邻域
reports/REVIEW_ALIGNMENT_20260904.md:307:  > style 条件=mean-pool(Es(R))、α query=同 R 的 Es mean-pool、gap/support 同 R。
reports/REVIEW_ALIGNMENT_20260904.md:406:3. **E2b 同头但不等价 E1：**它确实复用 E1 offset head；但 style condition 已由单张 spatial map 改为 normalized pooled n-shot 的 1-token 均值。E2/E2b 仍是 structure-source matched pair，E2b 应改称 matched control。
reports/REVIEW_ALIGNMENT_20260904.md:410:**待 PI 决策：**D-A1 cache/init SHA 强绑定；D-A2 文档/patch mass-update；D-A3 E2b 定义为 matched control；D-A4 冻结 n-shot style 精确表示；D-A5 best 从 10k 起；D-A6 空 α 处理；D-A7 V1/test-null 口径；D-A8 E2c retrieval space；D-A9 CFG inference structure 语义；D-A10 proto manifest 兼容策略。推荐项均见第 3 节。
reports/DELTA_RSI_DESIGN_RATIONALE.md:75:没有额外手工风格特征或另训检索器：α 只用冻结 $E_s$ 的 ref8 编码；具体实现把目标字体八个已归一化向量 `[8,D]` 与各库字体的同字 per-char 向量逐字做 cosine、再对字平均（per-char cosine 平均，合作者对齐），随后 softmax/τ 全池加权 + ε/K_max 截断（[`scripts/hrfont_delta_v2.py:80`](../scripts/hrfont_delta_v2.py#L80)–[`95`](../scripts/hrfont_delta_v2.py#L95)、[`225`](../scripts/hrfont_delta_v2.py#L225)–[`239`](../scripts/hrfont_delta_v2.py#L239)）。这里的 Es 是 official P1 初始化后、经 E1 在 A/train228 上跨语系联合适配并最终冻结的 style encoder；E1 训练 Ec/Es/UNet，E2 再冻结 Ec/Es（[`.cursor/rules/hrfont-execution-spec.mdc:47`](../.cursor/rules/hrfont-execution-spec.mdc#L47)–[`49`](../.cursor/rules/hrfont-execution-spec.mdc#L49)），所以 α 是方法内部的、与生成模型共享表征的检索机制。用 Es 选 α 不构成评测上的循环论证，因为最终风格结论由隔离于方案编码器的 $φ_{s2}$ 与 T1–T4 门控承担（[`.cursor/rules/hrfont-execution-spec.mdc:56`](../.cursor/rules/hrfont-execution-spec.mdc#L56)–[`60`](../.cursor/rules/hrfont-execution-spec.mdc#L60)）；只有拿 Es 自己给最终方法打分才会循环。Es 是否“够准”由验证电池实证回答：V1 报检索 rank/Recall 与 pairwise AUC，V2 报跨语系同字体对异字体 AUC，V6 报 leave-one-out 最近邻 cosine 与 top-1 α mass（[`scripts/hrfont_validate_e1_encoders.py:109`](../scripts/hrfont_validate_e1_encoders.py#L109)–[`147`](../scripts/hrfont_validate_e1_encoders.py#L147)、[`202`](../scripts/hrfont_validate_e1_encoders.py#L202)–[`214`](../scripts/hrfont_validate_e1_encoders.py#L214)）。因此论文只需主张 Es 提供了**经验证可用的风格相似度量**；向量不具有人可解释轴并不削弱该机制，但 V1/V2/V6 不过门时不能把 α 当作已验证可靠。
reports/DELTA_RSI_DESIGN_RATIONALE.md:107:PI 2026-09-05 已冻结：α **必取 top-10** 再 `softmax(s/0.07)`（`mode=topk, k_top=10`）；`eps_alpha` 只作 `≤1e-6` 数值地板，**不得**把邻域截空。K=3 仅附录消融。相似度聚合为同字 cosine 再平均（[`scripts/hrfont_delta_v2.py:88`](../scripts/hrfont_delta_v2.py#L88)–[`101`](../scripts/hrfont_delta_v2.py#L101)）。当前 Stage-A 代码仍可能是 soft-ε / 1-token / 在线 Ec，以 YAML 与 [`PI_DECISIONS_20260905.md`](./PI_DECISIONS_20260905.md) 为准，代码待审核后修改。
reports/PI_DECISIONS_20260905.md:4:本提交 **只改文档与 YAML 口径**，Stage-A 训练代码仍是旧实现（1-token / 在线 Ec / soft-ε），审核通过后再改代码。
reports/PI_DECISIONS_20260905.md:54:**style 条件（覆盖 D-A4）：**
reports/PI_DECISIONS_20260905.md:58:- **不用** pooled 压成 1 token。n 只影响这张 3×3 的估计方差，不改变 token 数。
reports/PI_DECISIONS_20260905.md:62:- D-A4 原文「pooled → 1 token」。
reports/PI_DECISIONS_20260905.md:63:- D-A6 把空邻域当合法 fail-fast/重定 ε（改为 top-K 保证非空；cache/数据坏才中止）。
reports/PI_DECISIONS_20260905.md:64:- 主方法 `mode=soft` + `eps_alpha=0.01`。
reports/PI_DECISIONS_20260905.md:70:当前 `cn2west_stage_a` 仍是：live Es pooled 1-token、live Ec、soft-ε、无 E1c 入口、sample 不送 Δ。
```

分类：exec-spec 与 PI_DECISIONS 中的命中是明确的 supersession/禁用说明，不应删除；`REVIEW_ALIGNMENT_20260904.md` 是历史审查记录，可保留但应在文件顶部加 “superseded by 2026-09-05”；真正会误导当前方法叙事的活跃残留是 `reports/ICLR2027_HRFONT.md:132-133,137,176` 的 soft 全池/fail-fast/soft 加权，以及 `reports/DELTA_RSI_DESIGN_RATIONALE.md:75` 仍写 “全池加权+ε/K_max 截断”（`reports/DELTA_RSI_DESIGN_RATIONALE.md:73-75,105-107`）。

## 5. 如果批准，代码改造清单

工作量量级：S=半天内，M=约 1–2 工程日，L=约 3–5 工程日，XL=跨多个模块且需完整回归；仅表示相对量级。

| 顺序 | 改造 | 依赖 | 量级 | 验收出口 |
|---|---|---|---|---|
| 0 | 把 D-P1…D-P9 写入唯一规范，清理 active stale docs；决定三臂共享 core 还是保留两个 variant | PI | S | 无身份/CFG/drop/new-layer 歧义；`reports/PI_DECISIONS_20260905.md:10-66` 与 exec-spec 一致 |
| 1 | 统一 schema/resolved config/structural diff；接通 `k/style/cache/eval` 字段，未知/dead 字段 fail；E2/E2b whitelist 可执行 | 0 | M | launcher 不再忽略 `k_top/ec_cache/style_tokens/best`；当前忽略范围见 `code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:27-72` |
| 2 | 建 Es spatial+pooled 与 Ec named-scale cache builder；覆盖 260/228/Content；分片 mmap、manifest、原子发布 | 0–1 | L | key/shape/dtype/finite/count/file SHA/encoder SHA/dataset SHA 全过；现 Es builder 仅 pooled train228（`scripts/hrfont_es_cache.py:80-118`） |
| 3 | cache validator 与 parity suite：online fp32 vs cache fp16、pooled fp32、top-K agreement、最终 noise/offset | 2 | M | 预注册误差阈值全绿；第10/11边界统计产出 |
| 4 | 生成 stateless/shared episode manifest：batch order、n、ordered R；独立 CFG/source/noise/timestep RNG streams；exact resume 保存 cursor/scaler/RNG | 1 | L | E2/E2b 前 N batch/key/mask/noise checksum 相同，中断/恢复与不中断一致；规范要求见 `.cursor/rules/hrfont-execution-spec.mdc:84-90` |
| 5 | 重构 model conditional API：显式 `style_map/content_pyramid/structure_pyramid`；训练与 DPM 均禁止调用 Es/Ec；用 named scales 替代 magic list index | 2–3 | XL | hook 证明 MCA/RSI 收到正确层；模型对象不在线调用 encoder；现在线调用见 `code/variants/cn2west_stage_a/FontDiffuser/src/model.py:36-54,99-111` |
| 6 | α 实现：唯一 K=10、top-K 内 fp32 softmax、finite/sum检查、stable tie、LOO、K不足全取、零候选 fail；记录 s10/s11/entropy/effective-N | 1–4 | M | synthetic+正式 cache 单测；现 K=3 fallback 缺口见 `scripts/hrfont_delta_v2.py:30-43,117-137` |
| 7 | 统一 source-drop 与 CFG cache semantics；9-token map 整体 drop；E2b official source 使用同一 source mask | 0,4–6 | M | 三臂 mask checksum；E2/E2b 只差 source tensor构造 |
| 8 | E1c 入口：config-aware launcher、frozen optimizer、shared R[0]、official cached structure；若仍独立 variant则同步 conditional core tests | 1–7 | L | 80k dry-run、encoder SHA 不变、与 E2b 除 style factor 外逐项 diff |
| 9 | cache-only sample/DPM/CFG wrapper；一次构造 cond 后供全部 solver steps；paired-noise manifest | 5–8 | L | E2/E2b/E1c inference hook、cond/uncond/source-drop 语义测试；当前 cond 只有两项（`code/variants/cn2west_stage_a/FontDiffuser/src/dpm_solver/pipeline_dpm_solver.py:59-83`） |
| 10 | val/best/ops：固定 ref manifest、每5k val、80k 主 endpoint、best≥10k 次要、1k/5k checkpoint、STOP/heartbeat | 4,8–9 | L | val16×295 可复现、best.json、DONE/provenance、STOP/resume tests；当前 trainer 只保存 ckpt（`code/variants/cn2west_stage_a/FontDiffuser/train.py:312-330`） |
| 11 | 端到端 matched smoke 后才启动 seed3407；通过后补 3408/3409 | 1–10 | M | cache-only 违规扫描=0、YAML diff=allowlist、draw checksum、online/cache parity、20-step sample、5k val 全绿 |

**批准门槛建议：**PI 可先批准 P3 9-token 与离线 cache 的方向，但必须先拍板 D-P1/D-P2/D-P3/D-P4/D-P8；正式 80k 的工程门是表中 1–11 全部通过。P2 的 K=10 与 τ=.07 可预注册不变，但必须先交付正式 cache 上的非调参诊断，证明实际浓度和边界稳定性可被诚实报告。
