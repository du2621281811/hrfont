# G 系列：独立目标字外观补全与效果优先排程

**后续状态入口（2026-09-14 18:11）：** TC-G2/TC-G2RL 已真实完成+5k；本文件下述“尚未真实训练”是当日早期实现快照。当前8卡队列、Ref8批次区别和PI待review事项统一见 [G_NEXT_8V100_PLAN_20260914.md](G_NEXT_8V100_PLAN_20260914.md)。TC-v2模块规格保持不变，本文件早期单卡/阶段排程由新计划覆盖。

日期：2026-09-14。状态：**Luna 已加入本地实现并进行独立 review/修复；尚未真实训练**。本文件不表示执行机队列已变更。CPU 检查范围见[实现报告](TC_V2_LUNA_IMPLEMENTATION_REVIEW_20260914.md)，完整命令与待执行 GPU smoke 见[执行交接](TC_V2_EXECUTION_HANDOFF_20260914.md)。不把代码提交等同于生成效果验证。

2026-09-14 本轮首次 fetch 的执行信息基点为 `baa9b2fc`；随后本轮方案与实现提交不代表执行机进度更新。该基点仍记录 G0 starting、G2/G1/G2-PRL queued，没有新的完成证据；未登录执行机核实实时状态。旧实验目录和 provenance 保留。

PI 输入：Set-Delta 已弃用；G 系效果优先，无需复刻 F 系训练预算；PI 视觉判断 F2-RL 优于 F2-PRL。该判断用于方向选择，不改写为已完成的盲评统计。

## 1. 设计收敛：TC-v2

TC = target-character appearance completion，目标字外观/风格补全。此版本替代 2026-09-13 的 Es-teacher + Delta-input 候选；不再并行验证这两个旧选择。

| 项 | 旧候选 | 本轮实施规格 |
|---|---|---|
| H 输入 | Ref Es + Content + Mean-Delta | Ref VGG 外观统计 + Content Ec |
| H 监督 | 目标字 pooled Es | 目标字 VGG 多尺度外观统计 |
| Delta/alpha/donor | H 的输入之一 | 不进入 H；原结构路径保留 |
| 接入基线 | 未锁定 | 优先 RL（global9 + local），由 clean val 确认 |
| 注入 | 泛指 style residual | 只给现有 global9 tokens 加残差，local 与 down-path 不改 |

复用 `src/criterion.py::VGG16` 的三个输出，通道 64/128/256。固定同一权重、96px 图像和现有 VGG 输入归一化，不改变渲染协议：

`psi(x) = concat_l(mean_HW(phi_l(x)), std_HW(phi_l(x)))`，共 `2*(64+128+256)=896` 维。std 使用 correction=0，在 FP32 中计算。每维按 clean train 目标统计做标准化并给 std 下限；统计只由 train 拟合，记录于 cache manifest。

每个中文 Ref 保留一个 896 维描述；共享投影到 256 维。Ec(Content) 五尺度池化拼接为 707 维，投影成目标 query。一次 masked cross-attention + 小 MLP 输出 896 维预测 `a_hat`。不使用 font-ID embedding，不输入目标 GT、Delta、alpha、donor 或扩散 noisy state。

教师 `a_star = standardize(psi(GT))`。描述预测为 `a_hat = H(psi(Refs), Ec(Content))`，`L_comp = SmoothL1(a_hat, stopgrad(a_star))`。VGG 统计是外观代理，不是纯风格真值；它可能遗漏空间细节，因此特征误差下降不是生成效果验收。

接入：`G'_j = G_j + W_out(a_hat)`，j=1..9，W_out 的权重与偏置均初始化为零。已有 local tokens L 不改；down-path Ref mean map 不改；Mean-Delta RSI 不改。生成器用 `[G', L]`，若 RL 未胜出则使用 `[G']`。不追加零 token 冒充恒等初始化，不把预测替换成训练 GT 条件。

Es 仍被原模型与 alpha 使用；只有新增 H 分支摆脱 Es 输入/teacher 的直接依赖。此设计不承诺整个生成器与 Es 无关。

## 2. 训练数据与实现要求

- clean train 有效 56,429 pairs；既有 train/val/test 字体划分不变。50/38/12 目标语种 sampler 保持。1–8 个中文 Ref 随机取样，验证固定 true-1 / true-8 两种输入。
- 教师只缓存 train 与供评估使用的 val 描述；test 描述不得参与 H 拟合、归一化、选参或模型选择。真值不得进入采样条件。
- train Ref 统计与 target 统计可离线算；train target FP16 原始896维约0.10GB，Ref/val/索引另计。Ec 使用与 G0 权重匹配的缓存，不重建大 Ec 库。
- Ec/Es/VGG 冻结并 eval；H 和 W_out 在 no_grad 外。VGG 教师与 cache 提取才放 no_grad。H 的 optimizer 参数组与保存/恢复必须完整。
- 训练用原始未丢弃输入计算 H 的监督；注入生成器之前按现有 CFG mask 关闭补全残差，避免偏置泄漏。现有 source_drop=0.25 继续作用于 Delta；H 不受 donor 丢弃影响。
- 补全描述与扩散 t 无关，推理每个样本算一次；训练、validation 与实际 sampler 必须走相同注入接口。
- 统计初始化不使用测试字体；cache manifest 绑定数据 mapping、VGG 权重、图像归一化和统计量指纹。

实现检查：小 batch overfit；H 梯度非零；W_out 首步有梯度；冻结编码器无梯度；W_out=0 时与对应 RL/base 数值一致；CFG 关闭时无条件泄漏；checkpoint 重载后采样一致。不能沿用只为 G0→RSI 新建分支设计的 parity gate 来要求已训练 RSI 为零。

## 3. 先选能画好的底座，再验证 TC

下面是新的计划 ID/别名，**不是已经注册或运行的实验**。落地前分配独立 run_id，不覆盖既有 G2/G2-PRL。

| 顺序 | 计划别名 | 从哪里开始 | 预算/动作 | 下一步条件 |
|---|---|---|---|---|
| 0 | G0/G2 existing | 现有记录 | 保留已有成果；未完成则先拿到可用 clean G2 与对应 G0 cache | 执行机核验 checkpoint/READY/val 图 |
| 1a | G-RL-pilot | 选定 G2 权重 | 加载现有 UNet/RSI；新增 local projection；先 +2k，值得则到 +5k | clean val 笔触/整体风格改善 |
| 1b | G-base-continue | 同一 G2 权重 | 同样 +2k / +5k | 避免把正常续训收益全算给 RL |
| 并行准备 | TC-pretrain | H 随机初始化 | 冻结全部大网络，bs256、lr3e-4、最多2k步 | val 优于同字符均值/简单 Ref 预测基线 |
| 2a | G-TC-pilot | 阶段1胜出底座 | 加载预训练 H，W_out=0，联合 +2k，值得则到 +5k | 图像胜过同起点续训，不只 L_comp 降 |
| 2b | G-best-continue | 同一胜出底座 | 同样续训，作为效果锚点 | 与2a同验证面板比较 |
| 3 | G-TC-main | 胜出 TC checkpoint | 以新增5k为一段继续；先预留额外10k，不要求40k | 连续验证仍改善才继续 |
| 后置 | TC evidence | 冻结主方案后 | 同容量H无补全监督；Delta×TC必要消融 | 只给确有效的方案补论文证据 |

阶段1的 RL 使用现成 F2RL arm 的 G9+L 设计，不因旧 F2-RL 胜出就直接拿 dirty F checkpoint 当 clean parent。PRL/G1 若已完成则评估和保留；若未启动，后续计划中降低优先级，不再作为 TC 的前置条件。是否暂停已运行作业必须先看执行机状态，不能凭旧 Git 状态杀进程。

后置 Delta×TC 消融须有对应训练设置；在已经带 Delta 训练的模型上推理时关闭 Delta 只是敏感性诊断，不能冒充 no-Delta 训练基线。

两个短训练可以在确认资源允许时并行；VGG 缓存/H 预训练不需要占满8卡。不要为用满GPU并发打开多个94GB Ec读取进程。若 I/O 出现竞争，顺序跑短分支。墙钟需实际吞吐测量，不按旧 F 日志承诺小时数。

## 4. 第一轮超参（起始值，不是调参结论）

- 网络阶段：单卡 bs8、fp16+GradScaler、原 UNet lr1e-5；新 local projection / H / W_out lr1e-4；AdamW 沿用既有其余参数；梯度裁剪1.0。
- H 预训练：bs256、lr3e-4、warmup100、最多2k；FP32计算统计与loss。
- 网络 pilot：新增步数最多5k，warmup200，之后恒定lr；2k做readout，每1k保存并验证，`best_min_step=1000`。延长阶段重新登记学习率日程，不偷偷恢复已衰减到零的 scheduler。
- TC：`L = L_diff + 0.01*L_percep + 0.5*L_offset + 0.01*L_comp`；先不同时加入边缘loss、更换alpha、重训Ec/Es或改渲染。
- 单个既定 seed 沿用3407。数值超参只有验证给出问题时再调整，不开大网格。
- warm-start = 只加载模型权重，保留训练过的 RSI/local projection，新 optimizer/scheduler/阶段step；不能把不同参数组的新模块实验作为旧 trainer_state 的原样 resume。记录 parent SHA、parent步数、新增步数。已加入 weight-only warm-start 接口和 H checkpoint 读写；可执行参数模板见[执行交接](TC_V2_EXECUTION_HANDOFF_20260914.md)。

## 5. 视觉优先的选择标准

固定 clean val 字体与合法目标字符，1-shot/8-shot各生成一套。以字体为单位，遮住方法名，比较整体字体一致性、笔触/端点、断笔/粘连、可读性。所有方法用同一 Ref、采样配置和噪声；不挑各自最好看的样本。

全体 clean val 的 L1/LPIPS/SSIM及分语种统计是辅助，不由单个 diffusion val_loss 决定选择。既有 test Demo 已被看过，保留展示但不继续拿它作新调参面板；最终使用 clean test704口径评估，输出所有语种的有效样本数。

2k小面板8个val字体、每字体最多12个合法目标字，只用于快速方向判断。进入5k/主训练前扩展到全部有效val字体与更广字符，不能用一张小图板宣布泛化。连续两个间隔无视觉改善则回到已保留的最好checkpoint，不覆盖旧文件。

若 H 缓存学习失败：先查统计尺度、同字符GT间是否有外观差异，再考虑增加局部描述；不硬开长训练。若 H 特征预测好但图像无收益：查注入幅度、梯度、训推一致性；不直接加大L_comp。若 TC 始终不如底座：保留RL/基础胜出方案，TC不进入主方法。

## 6. RL/PRL判断与实际队列

代码 `_pack_up_style`：RL = mean global9 + local；PRL = per-ref pooled h + local，无up-path global9。两者 down-path 都保留均值3×3。因此 PRL 不是“RL加更多信息”。全局空间信息的丢失、pooled归一化差异与token竞争都是候选原因，尚无单因素证据。

本轮执行顺序建议取代“必须先 G2→G1+PRL 再考虑其它”的研发依赖，但**`scripts/watchdog_g.py` 仍是原队列，尚未修改/重启**。执行者必须先核实真实进度，再显式更新队列；仅阅读/同步本报告不会改变正在运行的进程。

该计划采用 ml-training-recipes 的短pilot、分组学习率、冻结大编码器、先看生成再延长原则。与F系不同的训练预算可用于效果探索；最终方法增量仍保留同起点续训与必要消融，不能将所有训练配方变化都归因于TC。
