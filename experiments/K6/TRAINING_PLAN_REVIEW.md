# K6 训练执行计划 — 待用户 Review

日期：2026-09-20。状态：仅方案；未授权启动、未启动训练。
依据：远端 main 5a5b01b8636cf7d5aba3da90c06493efec78c518，reports/K6_HYPOTHESIS_AND_EXPERIMENT_PLAN_20260920.md。原文副本见同目录 UPSTREAM_PROPOSAL_20260920.md。

## 1. 本次最重要的修订

K6 所有训练组均独立加载 K0/G0b step10000，与 K5 相同。K5-B 只作为结构与配方基线，绝不是训练初始化权重。新 adapter/router/gate 按 K5-B 原始初始化规则和相同 seed 初始化；optimizer、EMA、scaler、RNG 与步数从零开始。A、B、C 不互相续训。只有同一 run 的故障恢复才加载该 run 的完整断点。

K6-0 是冻结 K5-B@10k 的推理诊断，不是训练组，因此它不适用“从 K0 开始训练”。不能为了统一名称把它改为 K0 donor 诊断。

原文“K5 优于 K1、B 略好”作为合作者的人工判断记录，不作为本计划已经证明的 matched 结论。新增 loss 作为优化策略，不自动列为新贡献点。

## 2. 实验矩阵和顺序

| 组别 | 网络/初始化 | 唯一主要干预 | 预算 | 执行条件 |
|---|---|---|---|---|
| K6-0 | 冻结 K5-B 10k | donor swap 和路由审计 | 不训练 | 批准后先完成 |
| K6-A | K5-B 双分支结构，独立 K0 | GT-vs-generic gated ranking | 10000 成功更新 | 工程预检通过 |
| K6-B | 同上，独立 K0 | 同字符、异字体 GT 相对距离监督 | 10000 成功更新 | A 之后，独立实验 |
| K6-C | 同上，独立 K0 | A+B 两项组合 | 条件性 10000 成功更新 | 单项结果 review 后另行批准 |

推荐顺序：K6-0 → K6-A → K6-B → 单项报告/review → 决定 K6-C。
2k 是中期视觉检查点，不是因视觉不佳自动终止点；A/B 默认完整 10k。若 K6-0 提示 donor 污染，先交证据与单独修复计划，不把 donor 修复偷偷混入 A/B/C。C 首轮仅组合 A+B，donor correction 必须另命名隔离。

## 3. 所有训练组固定项

- V2 的原冻结 train/val/test；0913 membership 不变，0917 合法 GT 规则不变。不得用 val/test 校准训练门限、负例、loss 系数或 donor。
- 同一 V2 train bank、编码器/预处理 hash、self 与 weight-only 排除；原 Top10 Chinese-reference alpha。推理严格匹配训练 bank 和过滤规则。
- K5-B 的 Ec/Es 独立 router、offset 和层级门控；冻结 Es48/24、Ec；保留 FP32 attention 和 Es/gate 数值修复；不新增 decoder 外观支路或 SCR。
- Seed3407；global batch64，8 rank × microbatch8。原 KSampler 保留：western32/kana24/bopomofo8；detail配额19/14/5；同字符配对4/3/1。
- AdamW：backbone2e-5，新模块1e-4，betas(.9,.999)，eps1e-8，多维参数 decay.01，其余0。
- warmup500、flat至5000、cosine至10000的0.1；TC ramp1000；联合 CFG drop.02、source drop.05。
- base loss：epsilon1、VGG.01、offset.25、completion.01、detail.05；原时间权重、EMA和成功更新计数不变。
- EMA留存2k/4k/5k/6k/8k/10k；滚动完整断点含optimizer/scaler/RNG/sampler/所有rank状态。先测磁盘占用，不删旧证据腾空间。

## 4. A/B 共用的新增监督接口（本次待 review 的具体建议）

首版使用现有冻结 VGG 的空间特征，复用原 perceptual 分支；不把全局 pooled Es 当作唯一风格评判器，也不训练新的特征网络。每层保留空间位置，计算逐元素平均 L1，再按层平均。各层尺度用固定 V2 train 校准清单的 GT-neutral 距离中位数归一化，设置数值下限；清单、数值与特征权重 hash 在训练前冻结。无单图独立标准化。

定义 d 为上述非负对称距离，delta_i=d(GT_i,neutral_i)。按 script 在 train 校准清单上取 delta 的中位数为 tau；delta<=tau 不启用新损失。门限与尺度在 A/B 共用，正式训练不按结果追调。校准采用固定最多4096合法 train 样本，记录覆盖和缺项，不用测试集。

新增 loss 施加到现有 raw_x0 产生的去噪估计 clean（特征输入沿用原VGG的[0,1]截断）。它不是完整20步采样图，因此新loss下降不能代表最终生成改善。记录截断比例和有效梯度；门限、距离、hinge全部FP32。

为降低高噪声 clean 的不可靠性：只对 alpha_bar>=0.5 且条件未联合drop的样本计算新增监督；附乘 sqrt(alpha_bar)，不改变原 t/noise 抽样。新项统一前1000更新线性ramp，建议初始系数各0.05；这是待批准的试验超参，不宣称已优化。固定用global batch64归一，pair项按原8对归一，无有效项返回可微0。预检只检查finite/梯度量级；若量级显著压过base，报数值与建议再review，不自动改系数。

限制：原独立 t/noise 仍会给 pair 距离带来噪声。首版保留它以避免改变K5基线；使用两端都满足时间门控与较小的sqrt(alpha)权重。记录每步有效pair数，不能用随机噪声制造出的差异宣称风格恢复。若几乎无有效对，预检必须报错，不偷偷改成新sampler或短步rollout。

## 5. K6-A：GT 锚定排序

首轮只用当前 neutral content 对应同一字符的渲染作为 generic negative；不混入 K1 预测或在线弱条件预测，避免多种负例与额外网络引入混杂。

m_i = 0.2 * stopgrad(delta_i)
L_rank_i = max(0, m_i + d(pred_i,GT_i) - d(pred_i,neutral_i))
L_A = L_base + 0.05 * ramp * gated_time_weighted_mean(L_rank)

所有GT/negative特征detach，梯度只回生成端。m随真实差异缩放，避免固定margin大于可达GT-negative差距。计算正负距离时使用同一特征/尺度/空间范围。首轮不加不对称 change-mask。

须同时看 d(pred,GT) 是否下降、identity是否稳定；只把 generic 推远不算成功。普通字体delta小则新项关闭，保留base监督。不把字母原有孔洞当作“空心字体风格”标签。

## 6. K6-B：GT 相对距离监督

复用 KSampler 已抽到的8组同字符不同字体配对，不新建会改变采样分布的大bucket。源码目前会 shuffle 全局64样本，且不返回pair成员映射；实施时额外返回基于出现位置的稳定pair_id，不额外消耗随机数，不按dataset index错误合并重复样本。

L_rel_ij = |d(pred_i,pred_j) - stopgrad(d(GT_i,GT_j))|
L_B = L_base + 0.05 * ramp * gated_time_weighted_mean_over_8_pairs(L_rel)

同script的GT pair距离超过校准tau才启用，两端均需有效条件与alpha_bar>=.5。该损失并不识别风格分配：交换两字体输出仍可能保持距离，因此必须保留原逐样本GT监督，并检验风格对应性。

跨rank实施：仅对原pair成员做支持autograd的特征gather，按全局pair_id一次配对；GT路径无梯度。严格核对DDP平均与pair loss缩放，8卡梯度和单进程global64参考一致。禁止普通all_gather后detach预测，禁止只在单卡随机找同字符，禁止重复计入跨rank pair。可等价减少通信，但须验证梯度一致。

## 7. K6-0：补全 donor 诊断

现有K5六组诊断576图已经生成，原始selected donor/alpha已记录；不是从零再做Top10/Top50/all。尚欠路由、表示与严格donor swap证据。复用仅限权重、precision、样本、noise、refs和协议完全一致部分。

冻结K5-B10k，原24cases×2seed×4组donor=192输出，每组10个合法donor（不足明确报告，不复制/越界填满）。Matched、neutral、random、adversarial四组都先执行self/weight-only合法过滤。

- Matched只用目标可见中文refs的相似度；不得用目标西文GT选候选。它只代表“中文参考近邻”，不宣称西文也相似。
- neutral/效果bucket来自train donor可见图像的预先审定标签；adversarial按参考可见风格与train标签定义，仅操作性假设。
- 四组统一均匀alpha隔离donor set因素；原Top10-alpha保留为另一个背景基线。随机组固定seed。
- 记录每层/采样时间的实际routing与alpha差异、偏移与Delta norm、候选覆盖及重复。GT方向一致性和距离只用于事后分析，绝不能反馈进检索。
- 对最终图的风格、结构、GT误差、普通字体退化分别比较；低敏感度也可能代表路径没被使用，不能单独证明bank正确或无污染。

## 8. 预检与失败恢复

启动前验收：K0权重来源hash/参数加载覆盖、独立新模块初始化一致性、split和bank一致性、train-only校准、0差异/GT近neutral的关闭测试；新项反向确实到UNet/router且冻结特征无梯度。

B额外验收：真实pair跨rank、重复样本pair_id、全无有效pair、单卡/8卡梯度缩放一致。A验收margin可达性、negative同字同预处理。A/B共享确认新增loss禁用时与K5-B输出/base梯度一致。

各组真实8rank两步→保存完整状态→恢复两步，验证finite、门控启动后的Es梯度与rank同步；零gate初步零梯度不误判，但后续必须有效。正式运行异常保留输入、日志、代码hash，最小工程修复、复验、完整状态恢复；科学配方改变重新review，禁止跳坏数据、伪造完成或确定性故障无限重试。

## 9. 评估与成功判据

2k/5k/10k固定视觉panel（空心/装饰/断笔/端点及普通字体，固定字/ref/noise，train与val分开）；2k/5k不能据此称收敛。10k按K5原train固定查询 + full frozen V2 val/test，1/2/4/8shot、DPM++20/order2/CFG1。测试集不用于决定C或调参，C决策基于train诊断及val；test仅最终报告。

对照：K0仅1shot；现有K5-B10k为直接结构/配方基线，复用须核实代码/权重/bank/样本/precision；K5-A作为次要结构背景。必要时仅重推对照，不默认重训K5。若实施改变base噪声配对或采样等行为，则原K5-B不再严格matched，必须另提从K0的K6-R基线。

指标：原L1/SSIM/D_change/add/remove/high/out；按shot/script/difficulty/font宏平均分层；新增generic gap、pred-GT距离、pair距离误差及Pearson/Spearman。字体为单位bootstrap区间，避免把重复case/pair当独立样本；pair相关性在GT距离方差不足时标NA。

训练用VGG指标不是独立质量证据。最终以盲化模型标签的同字GT/ref对照图，检查空心/局部细节恢复及字符身份；冻结Ec身份距离只作辅助，不能代替可读性人评。原文identity hard guard具体落实为配对样本身份一致预检与阶段图板人工错误清单，未验证OCR不能伪装hard guarantee。

成立条件：目标特效在固定val panel更接近GT，且不是加粗/噪点/断裂伪装、普通字体和身份无系统退化；gap或diversity增加本身不算成立。结论不明确则报告不明确，不自动组合C。验收表同时列有效监督比例、梯度占比、成本和失败案例。

## 10. 时间、资源与本次 review 决策

已完成K5-B实测13567秒≈3小时46分/10k（8V100）。A规划4–5小时，B4.5–6小时，A+B纯训练约8.5–11小时；可选C约4.5–6小时。此为工程规划值，尚未实测新增特征gather与监督；前200成功更新重估。实现/预检/校准/诊断约半日至一日的工作量，最终全量推理/指标另计，不能把训练ETA当交付ETA。

最近data1约9GiB、data2约11GiB，未证明足够容纳多组完整断点和推理。批准后先算单组完整状态与EMA/输出峰值，选有足够空间的隔离存储；不得删除K5/K4历史来满足计划，不足时如实报所需额外容量。

请review的建议：批准K6-0→A→B；A/B各完整10k、全部K0独立初始化；首版neutral单负例、VGG空间距离、上述门限/系数/时间门控；C保留为单项结果后的另次决定。当前不启动、不更新现有自动化为K6训练队列。
