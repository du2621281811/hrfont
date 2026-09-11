# 风格偏弱：最小改动裁决与执行门槛

评审日期：**2026-09-10**。文件名按任务约定保留 20260909。状态：**设计交付，未实现、未生成探针、未训练，不等于 PI 批准开跑**。

## 0. 快照、证据与裁决

- 本机实际 HEAD=`54a693521d39e5b6b98e724e0b3eaa4232feb043`，不是任务给出的 `81103a58`。已尝试 `git pull --ff-only`，因 `.git/FETCH_HEAD: Operation not permitted` 失败；不能声称已同步远端。本文仅依据本地快照，执行机须先同步并重核引用及 resolved config。本次只写本文。
- 唯一推荐 **R-L128：保留 global9，从每张 ref 的冻结 F0 Es 第三个 down block 取局部特征，独立池化为 4×4，经一个共享 Linear(256,1024)，8 refs 合计 128 token，拼入已有 up-path style cross-attention context**。不新增 attention、SupportAdapter、几何分支或损失；不改 down-path、F1/F2、Δ/RSI。
- 这是 A2-local 的“逐 ref 保留、有限预算”修订，不是把原 local64 原样搬进来。A1 只作零训练诊断，不升格另一条正式方法。**A1 有信号才许可 R-L128 20k；20k 面板过 PI 才继续累计 80k。**无信号不硬开训练。
- 原诊断认为 ref 均值与低空间分辨率是风格变平的候选原因，画布不是主因；此为待干预验证的机制解释，不能把眼检直接当因果定论。依据：`reports/STYLE_REGULARITY_AND_CANVAS_20260909.md:9`、`:23`。

### 0.1 哪份 F3b 定义有效

本任务锁定的 **F3b-S = own-font Ec + topology bank（SHA 前缀 6cdefe70）+ 现有 SupportAdapter → up-path style attention + global9，不带 local**。依据最新诊断 `reports/STYLE_REGULARITY_AND_CANVAS_20260909.md:16`、`:29`，及执行文档追加裁决 `reports/DESIGN_F3B_20260908.md:115`。这里只确认文档定义，**未核实执行机当前是否仍排队或已完成**。

必须显式排除的旧条款：

| 本地冲突 | 本次裁决 |
|---|---|
| `reports/DESIGN_F3B_20260908.md:9`、`:44` 仍写 cross-font / 新 mid attention | 不继承；采用上述 own-font / 既有 up-path 的任务锁定版 |
| 同文 `:56` 写逐 ref 投影后平均为 local64 | 废弃本方案中的 ref 平均；该条也不得混入已排队 F3b-S |
| `.cursor/rules/hrfont-execution-spec.mdc:61` 的 support 外部来源旧口径 | 不能据此把执行版 own-font 改成跨库来源；差异须记入执行批准记录 |
| 设计文档 `:65` warmup5k，与速查 `:17` 其他臂 warmup2k | 不在风格臂自行选值；取经过核实的 F3b-S resolved config，并核对与 F2 的 matched 契约 |

**数据来源和论文身份必须分开说。**执行版 own-font support 确实读取该字体额外 CN 字（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:276`、`:308`）；不能写成实际张量来自跨字体库。论文仍只赋予它“结构条件化支持/库先验”的功能，不拿它的笔触承载能力冒充本方案的 ref 个体证据。必须披露 own-font 扩展访问预算，不能把整套系统说成只访问八张图；R-L128 的新增输入严格限于原 episode refs，没有新增目标字体观测。跨库版本的“同 J 同 support”论证也不能直接套到 own-font 实现。故事边界依据 `reports/IDEA_ICLR_SCORE_20260908.md:90`、`:210`、`:232`。

## 1. 为什么这一刀最小且对题

当前编码器在 96 输入、ch=64 时各 down 输出分辨率为 48/24/12/6/3、通道为 64/128/256/512/1024（`code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/style_encoder.py:293`）。forward 返回 `style_emd`、pooled vector、残差特征；最终 map 在 pooling 到向量前取出（同文件 `:428`）。训练确实对每个 ref 的最终 spatial map 执行 `spatial.mean(0)`（`code/variants/cn2west_f123_rsi/FontDiffuser/train.py:133`）。这会丢失 ref 间差异；但“丢了哪种笔触、是否主因”尚需实验。

现有 down-path 在 `code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py:227`、`:341` **解包四维 map 后才展平**，不能直接塞 `[B,72,1024]`。真正的最小接线点是 `code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py:49`、`:60`：已经有独立的三维 `style_hidden_states`，support 只拼进这里，down-path 仍拿旧 map。因此 R-L128 只扩展这一条已有 context，不改 down-path 消费合同。

中层 12×12 保留的空间采样较细，但 stride≈8 **不是感受野只有 8px**，更不保证顿笔仍可辨。4×4 池化又会损失信息；选择它是成本优先的可证伪折中，不宣称充分表达所有笔触。保留每张 ref 独立 token，至少不再强制把不同汉字相同网格位置的斜笔/竖笔平均成“中庸笔触”。

## 2. A1 / A2 逐项裁决

依据候选原文 `reports/GENERATOR_STYLE_PLAN_20260909.md:19`、`:35`、`:45`。

| 候选 | 保留 | 砍掉/修改及原因 |
|---|---|---|
| A1 少平均 | 固定 checkpoint/noise 的 k=2、k=3 子集探针 | “不平均”须纠正为**子集内仍 mean → 9 token**；否则同时改 k 与 token 数。禁止按生成结果挑最好 refs；禁止用 test16 决策 |
| A1 8×9 concat | 原 8 张逐 ref 最终 map 展平，72 token，零新增参数 | 仅替换 up-path 的 global9 为72，support 不变，down-path/global pooled 检索不变；不能向旧四维接口硬塞 token。加重复均值 token 的等长度控制，不能把长度效应当信息恢复 |
| A2-geo | PNG 几何量保留为离线诊断指标 | **砍生成分支及4 token MLP**。绕过 Es 不满足本任务限定；CN 字 ink/孔洞/斜势易与字形结构混杂，也不能证明端点收笔。不得以拉丁 I/l/1 当训练中文 ref 输入 |
| A2-local | 冻结 Es 第三 down block、离线缓存、逐 ref 保留、复用 style attention | 砍 ref-axis attention、砍任何 ref 等权平均；不做 geo+local。修成每 ref 16 token，共128，而不是“每 ref64却最后只有64”。只进 up-path，避免同时改两个注入位置 |
| A3 独立 attention / 拉丁 refs | 不纳入本时间窗 | 新模块或新数据协议均超出最小改动；没有证据前不扩大方案 |

**为什么不唯一推荐 72-token 最终层 concat？**它是零参数、很有价值的诊断，但不能补回最终层已经压掉的局部事件；将它再开成正式训练臂会增加筛选路线。本文只授权一个训练候选 R-L128，A1 成功也不等于已经证明 R-L128 会成功。

## 3. 零训练 A1：题本、操作与写死的 gate

### 3.1 题本与固定项

使用 **val16，不用 test16**。在看任何候选输出前锁定16字体×12目标字×2噪声种子（3407/3408）=384个配对格；目标字固定 `I/l/o/1/A/a/あ/の/れ/し/は/ぬ`，先做缺字审计，不得看到结果后替换题目。显示相同 ordered ref8、Content；GT 仅在第二轮揭示作 positive control，不以像素贴近 GT 定义个性。测试隔离依据 `.cursor/rules/hrfont-execution-spec.mdc:125`。

第一轮直接用已有 F2 固定 checkpoint（若实际为75k，明确标75k，只作探针）；F3b-S 可用后在其固定 checkpoint 复核相同题本，不冒用旧失效 adapter 的 F3。F0/F2/F3b-S 面板可帮助定位从何处变平，**不同步数面板不能作为训练效果归因**。所有分支固定 sampler、20 steps、CFG7.5、初始噪声、Content、Δ/J/α、support 选字与张量；仅改 Es→up-path 的 style context。原 global mean 及 pooled query 单独保留，不让 k 干预改变检索。

预注册五格（输出盲化、左右顺序固定随机化）：

1. **P8**：ref8 mean，9 token，基线。
2. **P2**：`永、风` 的 map 均值，9 token。
3. **P3**：`永、风、骨` 的 map 均值，9 token。
4. **PC72**：ref8 各9 token concat，72 token，既有 K/V 投影原样复用。
5. **PR72**：把 P8 的9 token 重复8次，72 token，控制长度及 support 的 attention 竞争。F2 无 support 时，在无 attention dropout、eval 模式下其输出应与 P8 数值近似相等；F3b-S 有 support 时重复 style 会改变相对 softmax 质量，**不要求其与 P8 相等**。

P2/P3 用固定中文字符，不按目标 GT 或候选结果挑字；其阴性只能排除这两个子集，不能穷尽所有子集。每个探针从离线 Es cache 读数；关闭 online encoder fallback，推理 `eval()`。

### 3.2 通过判据（提出的预注册值，不是已测结果）

用户/PI 为主要评审，另两位评审独立盲判。每格回答：**是否更像 refs 那种“有脾气”，而非只是更斜、更黑或更破碎？**按倾斜、笔触粗细对比、连断/收笔、字重四项标记“匹配更好/平局/更差”，另判字符身份与伪影。先只看 refs 和成对输出，后看 GT 作解释，不按 GT 唯一答案打分。

在揭示输出前，评审仅据 refs 给每字体标注可辨的属性；不可辨属性记 NA，不把它算失败或成功。以下条件须同时满足：

- 主比较 PC72 对 PR72：用户/PI 的总体 ref-style 偏好率（胜=1、平=.5）**≥60%**；另两人合并也≥60%；按字体聚类 bootstrap 的总体偏好差相对50%的95% CI下界>0。若只有P2/P3改善而PC72失败，记“少ref诊断阳性”，**不自动通过 R-L128 训练 gate**，须固定子集复核而非选最好结果。
- 至少 **10/16字体** 净偏好为正；拉丁和假名各自偏好率>50%；至少两个有足够样本的属性类别净改善，不能全靠加粗。每个属性至少覆盖4字体，否则该项不可判，整组不满足两属性要求则暂停。
- 身份正确率与无明显伪影率相对基线的差值，字体聚类95% CI下界均 **≥−2个百分点**；同时用户/PI未发现可复现的系统性错字/断裂。不能以风格胜率冲抵错字。
- 辅助指标：同字符配对报告 ink率、骨架宽度CV、ink二阶矩斜势、连通域/端点数的变化，并与盲判属性方向对照；这些量只支持描述，不是普适跨脚本 style 距离。若独立 φ_s2/E12 已过 T1–T4和同输入不变量检验，补报 SC-Gap、family-match、ID-CLS及字体聚类CI；否则标“探索性”，**不以E12高分替代眼检**。依据 `reports/GENERATOR_STYLE_PLAN_20260909.md:74`、`.cursor/rules/hrfont-execution-spec.mdc:95`、`reports/IDEA_ICLR_SCORE_20260908.md:157`。

阈值不在 test16 上调整；CI 太宽也算未过 gate，而不是自行放宽。P2/P3为机制辅助，不和PC72做“挑一个显著即成功”的多重择优。A1 在 F2 阳性即可准备20k；若已有 F3b-S 探针明确显示增益消失，则暂停，优先检查 support 竞争。

### 3.3 阴性怎么处理

若上述 A1 无信号：**工程决策按“3×3 已无可用笔触信息，必须中层图”处理，停止对最终层聚合的继续调参；但这不是信息论证明。**训练时只见均值的 attention 可能不会消费 concat，refs 也可能根本没展示该属性，故阴性不能严格推出3×3不含任何信息。原候选的强结论见 `reports/GENERATOR_STYLE_PLAN_20260909.md:29`，本文限定其证据边界。

此时不绕过“有信号才训练”：只允许下一轮零训练的中层缓存可视化/同字跨字体近邻诊断，确认 ref 中确有笔触、12×12相对3×3更能保留盲标属性，再由PI另立 gate。不能把随机投影塞入 checkpoint 后的噪声输出称为中层无效，也不能凭缓存图好看直接开80k。**本轮保持 F3b-S，R-L128不启动；9/25窗口内不保证救回。**

## 4. R-L128：可直接转为实现规格

### 4.1 特征、token、接线

设 episode refs 数为 n，`G_r∈R^(1024×3×3)`、`H_r∈R^(256×12×12)`：

```
G = flatten(mean_ref(G_r))                         # [B,9,1024]，旧路径不变
L_r = flatten_spatial(adaptive_avg_pool2d(H_r,4))  # [16,256]，每ref独立
L = concat_ref(Linear_256_to_1024(L_r))            # [B,16*n,1024]
C_up = concat(G, S_existing, L)                   # 推理 9+8+128=145
C_down = old_global_map                          # [B,1024,3,3]
```

`H_r` 精确取 `StyleEncoder.blocks[2]` 内第三个 DBlock **完整输出（含shortcut及downsample之后）**，不是 pooled vector，也不依赖 `residual_features` 列表位置碰巧一致。离线 hook 后 assert `[B,256,12,12]`；最终map assert `[B,1024,3,3]`。层结构依据 `code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/style_encoder.py:382`、`:428`。

新增可训练参数仅一个共享 `Linear(256,1024,bias=True)`，共 **263,168** 参数；所有 refs 共用，不加位置/ref-ID embedding、LayerNorm、gate、ref-attention或第二 adapter。新投影使用框架标准随机初始化，**不 zero-init、不压缩输出幅度、不冻结预热、不加防御性 schedule**。训练现有 UNet/RSI/SupportAdapter 的可训练集合与 F3b-S 相同，只多该投影；Es/Ec继续冻结。

`model.forward` / 采样wrapper新增独立 `local_style_tokens` 与有效mask参数；保持旧 `style_features` 四维语义。在 `src/model.py:64` 的 support concat 后附加L，再沿既有 `input_hidden_states[2]` 传递；down-path `[0]` 不动。训练和DPM推理两条wrapper必须同改，不能只改训练。局部投影在 `no_grad` 外执行；`train.py:657` 已体现 support 对这条纪律的要求。

### 4.2 cache-only

离线缓存角色暂命名 `es_local_f0_block2_pool4`，每个 `(split,font,ref_cp)` 保存 `[16,256]` fp16；FP32池化后转换存储。需要更细分辨率的后续诊断可重新离线建独立版本，禁止训练时临时跑Es。完整260×338字池约 **686.6 MiB payload**（不含索引/校验元数据），可按字体分片；训练只读train角色，val/test仅在对应评测角色读，test不用于筛选。

manifest必须含：完整F0 Es权重SHA256、encoder代码SHA、hook路径、原始及保存shape、池化参数、dtype、RGB归一化、A协议dataset/split/render SHA、字池/索引SHA、payload SHA。**F0 Es权重SHA要与global cache一致**；不能用当前训练checkpoint的Es重新混建。预处理始终96×96 RGB PNG、无resize。逐条随机抽查离线hook→池化与cache读取误差及shape，校验缺字为失败而非零填补。正式训推若试图调用Es/Ec立即报错。依据 `.cursor/rules/hrfont-execution-spec.mdc:38`、`:52`、`:59`，及 `reports/DESIGN_F3B_20260908.md:53`。

### 4.3 n、k_s、padding、drop 与 RNG

- **n是style refs数，k_s是support字数，不能混用。**style训练沿用主线 `n~Uniform{1..8}`，推理固定同一ordered ref8；L训练为16n、推理128。所有有效ref都保留，不挑“最有风格”的单字。训推聚合算子相同，n=8属于训练支持域；不声称训练分布等于推理固定值。依据 `.cursor/rules/hrfont-execution-spec.mdc:50`。
- support的训练k_s/选字分布与F3b-S逐样本完全相同，推理同用拓扑前8。设计候选写U{4..16}（`reports/DESIGN_F3B_20260908.md:25`），实际代码受bank meta控制（`train.py:287`）；必须读取执行机meta和resolved config确认，不按文档猜。如果F3b-S实际固定8，新增臂也固定8。style改动不能借机“修正”support采样。
- 变长L pad到128必须**真正屏蔽padding logits**，不许仅补零。当前 `src/modules/attention.py:209`、`:223` 显示mask参数被忽略，`train.py:317`也仅补零support；因此不能宣称现有padding安全。最小实现是把context有效mask传过现有wrapper，在既有attention logits上mask；无新增模块类型。新增臂只屏蔽L的pad，旧G/S语义保持不变；F3b-S同样使用该实现但mask全有效，必须做all-valid parity。若要修旧support padding，须两臂统一另核实，不得单边夹带。
- 新L仅受既有 **CFG joint drop=.10** 控制：conditional有效；unconditional全禁用L（屏蔽而非让Linear bias泄漏）。旧global9仍按旧方式置零，保证至少有有效context、不产生全masked softmax。**source_drop=.25只管原结构源，不额外drop L；support_drop=.20也不管L。**旧S究竟随source/CFG关闭必须复用配对控制的真实行为。
- 不能照抄设计文档 `:46` 就声称当前S满足该契约：`train.py:649`–`:661`分别抽source/support/CFG，调用support时只传support_draw。执行前须明确“保留历史行为做matched”还是“双方统一修复新control”；本文不授权仅在R-L128修S。无法提供有效配对control则停训。
- 新投影初始化在隔离且可恢复的RNG域；不消耗共享noise/timestep/source/CFG/SCR或DataLoader流。L无随机抽样/drop新draw。support现有 `train.py:297`、`:302` 用共享CPU随机抽样，与文档独立RNG建议不符；本方案不新增消耗，逐步比较两臂noise、timestep、refs、J/α、source/support/CFG mask及support选字摘要。若另修RNG必须控制臂同修，不能据同seed就宣称matched。
- resume保存原Python/NumPy/Torch CPU/CUDA、loader generator/worker/sampler cursor、AMP/optimizer/scheduler，加初始化/新流状态；有效batch不变。依据 `.cursor/rules/hrfont-execution-spec.mdc:102`。

以上是开训前合约检查，不是本次修改代码。mask/all-valid parity、关L的旧路径parity、cache-only、非零投影梯度/实际参数更新、同batch共享RNG摘要、DPM训推接口一致性全部通过，才算实现验收。

## 5. 对照链与20k/80k归因

```
F1 ↔ F2                         保留既有Δ主线，不重跑/改定义
F2 control → F3b-S             support组合效应，global9共同不变
           → F3b-S + R-L128    固定support时，新增ref局部style条件的总效应
           → 同模型L错ref干预  条件敏感性诊断，不另称训练臂
```

**“接在F3b-S之后”指比较链与批准顺序，不是从F3b-S@80k再续训20k。**R-L128从与F3b-S同一F0初始化，seed3407、80k总scheduler horizon；比较双方20k与双方80k。F3b-S支持部分采用相同标准随机初始化参数（隔离新Linear的初始化RNG）；既有分支不得因构造顺序漂移。共同80k优化参数以获批resolved config为准。

F2 checkpoint只有init/步数/有效batch/lr/scheduler/warmup/data order/drop/预算全部可核对才叫matched control；75k不能代替80k。若现成F2/F3b-S没有合格20k checkpoint，先查归档；不足则只报告描述性面板，并请求PI另批补control，**不违反“不动F1/F2”而私自重训，也不伪称三臂matched**。

为什么不并入已排队F3b-S：F2→full会同时增加support与style表示，失去support-only锚点，且影响已冻结日程。保持F3b-S原样，才能把新增条件效应放在唯一改变的style分支上。即使L改变S的attention权重竞争，也属于这个增量的作用机制，不是“support输入也换了”的实验混杂；但**不能因此宣称两者独立或协同**。

R-L128对F3b-S的胜出仍包含容量、token数和信息的总效应。20k面板再加同checkpoint“错字体L、global/Δ/S固定”的等token诊断（同中文ref字符，匹配投影后RMS，eps=1e-8，scale clip [.25,4]，越界排除并报告），检查个体ref条件是否真被使用；纯token顺序shuffle在无位置编码时基本不改变attention，**不是破坏笔触信息的有效控制**。若要宣称信息优于容量，应另批同参数/长度的无信息训练控制；若要宣称support×local协同，应另批2×2、local-only格。9/25前默认都不扩张，只称条件增量。依据 `reports/IDEA_ICLR_SCORE_20260908.md:125`。

20k使用同一锁定题本，对R-L128 vs F3b-S沿用§3的≥60%、≥10字体、两属性、两语系及身份/伪影非劣门槛；并要求用户/PI明确签字。没有gate通过就停止，不以loss下降或挑单张best放行。错ref干预无效时撤回“已利用个体笔触”的强解释，PI不得只凭E12升分放行该解释。通过后从20k完整恢复到**累计80k**，不改scheduler；若误用了20k horizon，正式80k必须重新从F0开并披露100k成本（`reports/DESIGN_F3B_20260908.md:74`）。主终点固定80k，best≥10k只作次要；seed3407单seed不外推稳定性。

## 6. 成本、风险与9/25窗口

### 6.1 可计算的增量与不能伪报的吞吐

只计算新增up-path style cross-attention，令query长度Q、head数h，scores与AV成本约正比 **Q×context长度**，不是context长度平方。支持字数推理k_s=8时：

| context | token数 | 相对F3b-S的该attention成本 |
|---|---:|---:|
| F3b-S | 9+8=17 | 1× |
| A1 PC72 / PR72 | 72+8=80 | 4.71× |
| R-L128 | 9+8+128=145 | 8.53×，即增加753% |
| 原每ref64全部concat | 9+8+512=529 | 31.12×，本轮拒绝 |

训练n=1..8时新增16..128 token；若均匀n，未pad有效新增均值72。但dense padded attention仍按128成本，不能拿平均72冒充节省。若k_s=16，最坏context153。projection新增263,168参数；以参数fp32、梯度fp32、Adam两动量fp32估算约4 MiB，不含AMP额外副本。token本体batch8×128×1024×2字节≈2 MiB。

新增单层attention score内存约 `B*h*Q*128*bytes`：举例B=8、h=8、Q=24²、fp16约9 MiB；Q=96²约144 MiB。**这是假设示例，不是实测层shape或总显存**；反传、softmax fp32、多个层/CFG倍batch会再放大。现实现显式K/V及attention的依据为 `code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/attention.py:209`。不改down-path可限制影响，但不能推出整机只慢某个固定百分比。

正式预算必须在执行机独占时做配对smoke：双方同batch/96协议/precision运行50步预热+200步计时，记录p50/p95 step、峰值allocated/reserved、cache吞吐、真实Q/h和验证耗时。不得为挤显存单边改effective batch、gradient checkpointing或精度；需要调整则两臂同条件重核预算。OOM先停，不悄悄减refs或改pool尺寸。

### 6.2 截止日倒排（计划，不是排程已实现）

按评审日2026-09-10到2026-09-25约15天安排；**9/23前完成主训练，9/24–25留评测/审图/归档**。旧F1独占约1.0–1.4s/it仅是文档观察，不是R-L128速度（`reports/STYLE_REGULARITY_AND_CANVAS_20260909.md:31`）。

| 实测step time假设 | 20k | 续60k | 累计80k |
|---|---:|---:|---:|
| 2 s | 11.1 h | 33.3 h | 44.4 h |
| 4 s | 22.2 h | 66.7 h | 88.9 h |
| 6 s | 33.3 h | 100.0 h | 133.3 h |

以上是steps×秒数算术，不含排队/cache/验证/停机。计划9/10–12完成A1、缓存及实现验收，9/13前拿到F3b-S同端点控制与20k窗口，9/15–16完成20k PI审图；能否如期取决于实际F1队列及单卡资源，本文未访问远端核实。**预计剩余用时=队列等待+缓存/验收+剩余steps×实测p95×1.3+至少48h评测余量**，超过9/25则不启动80k。

若实测≤4s、9/16前20k过门、控制臂可用且无需重跑，余60k约66.7h、乘1.3约86.7h，时间窗有条件可行；6s仍须逐项核算队列与验证，不能保证。若字面执行“20k筛查+从F0另跑80k”，总100k在4s时111.1h，必须额外披露，不作为默认。F3b-S未完成时将其剩余时间串行加上，禁止为了赶时间抢F1 I/O或隐去control成本。

### 6.3 三条主要风险与处置

1. **笔触信息仍不足/内容泄漏。**12×12经4×4池化可能仍丢顿笔，也可能把中文部件带进目标；用属性盲判与身份非劣挡住。失败保留F3b-S；更细每ref64只作另批后续，不在同一run调大token继续挑结果。
2. **context竞争与分布变化。**L多于global/S，标准随机投影和变长mask可能扰乱原条件；用重复均值控制、错L诊断、真实mask和20k gate检查，不用零初始化/渐进开门掩盖失败。
3. **吞吐及归因债务。**attention增量显著，旧support RNG/drop和文档冲突可能使历史control不合格；先测成本、冻结真实合同。无法在9/25前取得matched端点就交support-only与阴性报告，不包装成完整方法胜出。

## 7. 三句叙事接线

目标字符对齐的检索残差提供相对中性字形的变化方向，结构条件化support提供补全偏好而非目标个体设计的证明。我们仅补充同一组中文refs经冻结Es保留的逐ref局部style token，使其自身的笔触约束更有机会进入现有生成通道，不把库先验偷换成个体证据。以家族兼容性代理和独立盲评检验这一组合增强，维持“检索残差先验＋合理性评测”两核心贡献，不新增第三项架构创新声称。

## 8. 决策清单（待PI确认，推荐已唯一化）

| ID | 选项 | 推荐与批准条件 |
|---|---|---|
| D-SW1 | 72-token正式训练 / geo4 / R-L128 / 不改 | **唯一训练候选R-L128**；A1仅探针；无gate则不改 |
| D-SW2 | 改down+up / 只改up / 新attention | **只改既有up context**，一共享Linear，不加模块类型 |
| D-SW3 | 每ref64共512 / 每ref16共128 / ref平均local64 | **每ref16共128**；不沿ref平均，不默默换容量 |
| D-SW4 | test16挑图 / val16固定题本 | **val16预注册**；PC72对PR72过§3全部门槛才20k |
| D-SW5 | A1阴性照训 / 继续最终层调参 / 中层诊断后再批 | **停本轮训练，转中层零训练诊断后再批**；阴性不作信息论证明 |
| D-SW6 | 并入排队F3b-S / 独立从F0配对 / 从S@80k续训 | **独立从同F0配对**，保持F3b-S不含local |
| D-SW7 | style固定8训练 / 沿用n=1..8；support随意改k_s / 原样 | **沿用style分布、推理8；support原合同**，mask必须真实有效 |
| D-SW8 | 单边修support drop/RNG / 原样配对或双方统一核实 | **不夹带单边修复**；无法证明matched就停并请求控制臂批准 |
| D-SW9 | zero-init/gate / 标准随机 | **标准随机**，隔离初始化RNG，检查真实参数更新 |
| D-SW10 | 20k后重开80k / 80k horizon续到80k | **20k审图通过后续至累计80k**，固定端点、不隐去筛查成本 |
| D-SW11 | 声称协同 / 条件增量 | **只称support存在时ref-style条件总增量**；协同须另批四格 |
| D-SW12 | 无论进度赶9/25 / 按实测倒排停止 | **按p95×1.3＋48h余量倒排**；不满足则交F3b-S与诊断，不强行开长训 |

交付验收：本文仅为方案评审；代码、配置、cache、运行状态均未由本次修改或验证。执行前优先解决HEAD同步和F3b-S真实配置/对照证据核验，再由PI批准D-SW1..12。
