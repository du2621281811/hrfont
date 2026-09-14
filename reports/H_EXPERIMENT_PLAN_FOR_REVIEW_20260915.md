# H 系列：效果主线与消融执行计划（待 PI review）

> 已被2026-09-15后续用户决策部分替代：每个正式模型新增10k，核心70k，效果优先启动H3。实际预算、10k学习率和执行顺序以 `H_EXECUTION_20260915.md` 为准；下文180k/20k/40k保留为历史设计，禁止据此启动额外预算。

状态：设计稿，不是执行指令。2026-09-15 北京时间核对；未启动 H、未终止 G、未提交本稿到 Git。H 的预算独立于原 G-REF 25000-update 上限，必须由 PI 批准后才生效。01:15 后补充：用户授权的旧 G 存储清理已完成，见第10节；“训完要推”按训完立即推理出图纳入排期，H 自动执行钩子尚未实现。

## 0. 决策摘要

- 主线目标：完整 H3 训练 **新增40k真实optimizer updates**；20k为第一完整评审节点，不再用2.5k短训或10k终点认定方法已经收敛。
- 效果与归因同时做：H3与去补全监督H4都到40k；其余核心消融先到20k，与H3@20k比较。未稳定的关键对照按成对方式延长，不能把较短消融直接对比长训主线后归因于模块。
- 默认共同父模型 P：`G-CONT-G2-8gpu-V0914-A-S3407/global_step_5000`，其谱系为G0b基础模型→G2新增10k→CONT新增5k。选终点而不是凭旧val loss挑best，避免旧验证RNG和新协议混用。H0–H5均从P的共同UNet/Es/Ec出发，新增模块重新初始化；不串接不同实验结果来假装独立消融。
- Delta严格对照用另一共同父模型 P0：`G0b-F0-V0913-BS256-A-S3407/global_step_10000`，两臂分别从H阶段起启用/禁用Delta。不要把已经训练过Delta的G2权重关掉Delta，写成完整“从未使用Delta”训练消融。
- 主架构：保留Es block2的12×12局部证据，目标Ec查询→每ref对齐→等权融合→固定144-token目标外观memory。旧R与广播TC不叠加进来；目标补全是这个memory的训练任务。
- 先复用冻结Es/Ec与已有全局/结构缓存，不贸然重训编码器；CFG joint-drop一致化与EMA在所有H臂统一应用，不作为新模块收益。

## 1. G 资产实机盘点与复用方式

以下路径均以 `/root/projects/hrfont/runs/` 为根。DONE优先于遗留heartbeat；已完成目录中的heartbeat仍写running不意味着作业未结束。

| G资产 | 实际完成情况 | H中的使用 |
|---|---|---|
| `G0b-F0-V0913-BS256-A-S3407/global_step_10000` | DONE10k，global batch256 | 现有Es/Ec/cache基座；H-D+/H-D−共同初始化 |
| `G0c-F0-V0913-BS256-A-S3407/global_step_20000` | 从G0b续到累计20k，DONE | 更长encoder训练候选；不是默认父模型，须新建匹配缓存后才能比较 |
| `G1-F1-V0913-A-S3407/global_step_10000` | DONE10k，official RSI | 可续训作旧结构方案参照；不是no-Delta纯移除对照，因为它有official RSI |
| `G2-F2-V0913-A-S3407/global_step_10000` | DONE10k | 可直接续训；H默认使用它的CONT终点以复用已完成5k |
| `G2-RL-V0913-A-S3407/global_step_10000` | DONE10k | 可续训旧RL效果基线；不是H核心消融的异源初始化 |
| `G-CONT-G2-8gpu-V0914-A-S3407/global_step_5000` | 从G2@10k新增5k，DONE；另有2500和best | **默认P，H0–H5共同UNet/encoder初态** |
| `G-CONT-G2RL-8gpu-V0914-A-S3407/global_step_5000` | 从G2-RL@10k新增5k，DONE | 可直接续训的成熟RL候选；用于复用已有结果/效果备份，若作为H1正式父模型则所有可比臂应重新对齐，不能仅给H1换父模型 |
| `G-TC-G2-8gpu-V0913-A-S3407` / `G-TC-G2RL-8gpu-V0913-A-S3407` | 各新增5k，DONE；2500/5000/best存在 | 旧TC参照。896维全局H权重不能直接当成新空间reader权重；其VGG teacher规范可复用 |
| `G-REF-A0-V0914-S3407/global_step_2500` | DONE2500，面板完成 | 当前新recipe的短训控制；可续训，但不因新近就优先于P |
| `G-REF-A1-V0914-S3407` | 00:52读取约800步，未DONE | 待完成后作为R@短训对照；不能列为已完成父权重 |
| `G-REF-B0/B1` | 读取时尚未开始 | 不纳入可用资产；未来如完成再登记 |

严格区分：**warm-start**复用模型权重并重建H optimizer/scheduler；**resume**仅用于同一个H run中断恢复，必须恢复optimizer、scaler、EMA、schedule与rank RNG/采样状态。H不是把旧trainer_state换个目录继续计数。

Es SHA：G0b `2b1f65fe018fb4b869c0c12260f71cb0accc58c0398920671b5b989cb9ed2174`，G0c `fe4195c74a6a5e21a10867808e717d63cf20cae0b783a6b9af13b2c6375b67b4`。
Ec SHA：G0b `a780e70b88ea52751dbbc454b731706c2d94ebf1c0971b57f049212f6984e88c`，G0c `3c58096a0d2825713e041829160bcfe16fa9a7885a14a04b579351f30bd2e0f3`。
当前 `artifacts/g0/{es_spatial,ec_multiscale,es_local}` 都绑定G0b。禁止G0c权重配G0b缓存。更长G0c不是免费替换。

## 2. 为什么不把旧10k当收敛结论

- G2旧val：7500约0.00209577、10000约0.00220756，但best记录在10000才开放；best@10000不是整个轨迹最优。
- G-CONT两臂均完成新增5k；best在2500，最后5000的val又升高。这支持“需要更可靠的评测和学习率后段”，不支持无条件认为训练越长必然越好。
- G-REF修正了固定val RNG和实际update调度，因此其val loss绝对数不能直接与旧日志当作完全相同测量比较。
- H记录实际样本呈现数而不只记录step：batch64时20k=1.28M episodes，40k=2.56M；当前clean train有56429对，分别约22.7/45.4次全表大小的样本呈现量。由于分组加权和随机ref，不称精确epoch。

## 3. H矩阵：每行是独立实验，不串联初始化

| ID | ref信息/融合/补全 | Delta | 共同父权重 | 新增updates | 优先级/问题 |
|---|---|---|---|---:|---|
| **H0** | 旧G2 mean global9；无local、无新TC | on | P | 20k | 判断原方法长训后的效果，给新方案一个充分训练的基线 |
| **H1** | 旧RL pool4×4 + N；无新TC | on | P | 20k | 低分辨局部证据基线；local projection从共同初始化种子新建 |
| **H2** | raw12×12 local + N，仍直接拼接 | on | P | 20k | H2 vs H1检查额外空间证据；相同projection参数初态，记录额外计算 |
| **H3** | raw12×12→目标对齐→144-token融合，**有L_comp** | on | P | **40k，20k中评** | 完整方法，效果主线 |
| **H4** | 与H3完全同一网络/初始化，**L_comp=0** | on | P | **40k，20k中评** | 最重要的TC监督消融；辅助readout可以保留但不对主表示反传补全loss |
| H5 | 与H3相同目标144 queries，但ref先pool4×4，L_comp同H3 | on | P | 20k，可选 | 在同一reader参数量下验证ref输入分辨率，与H3@20k配对 |
| **H-D+** | 与H3同配方 | on | **P0** | 20k | Delta严格对照的full侧，不能拿P谱系H3代替 |
| **H-D−** | 与H-D+同配方，结构残差置零 | off | **P0** | 20k | 只移除Delta，不偷偷换成official RSI |

H0/H1/H2/H3的差别不都是单一参数量控制，H2→H3同时改变融合/监督；因此融合收益应主要看H2→H4，监督收益看H4→H3，输入分辨率严格控制看H5→H3。表中明确这一点，不把逐行递增当作完美单因素消融。

可选H-FuseMean：与H3保持Q、投影、输出长度、L_comp一致，但先平均ref feature maps再做一次目标attention；与每ref先对齐后均值比较。仅在需要更强融合机理证据时加20k，不默认启动。

**预算**：核心H0/H1/H2各20k + H3/H4各40k + H-D+/D−各20k = **180k八卡updates**。H5加20k→200k；H-FuseMean再加20k→220k。启动预检/LR pilot上限额外8k，单列，不计入正式模型训练史。相比G计划是显著扩展，等待用户批准。

## 4. H3/H4具体模块与迁移

- Ref输入：冻结Es第三DBlock输出 `[B,K,256,12,12]`，不作4×4池化，不经过末层InstanceNorm。最多8个ref，padding mask严格生效。
- Query：冻结Ec的目标12×12特征，经Linear投影到256维，加二维目标位置编码；144 queries，4 heads，head_dim64。
- K/V：ref局部向量共享投影到256维；Q/K匹配可LayerNorm，V不做单位范数归一化，保留幅度。每ref内部softmax读取，再对有效ref响应均值融合，1层轻量residual MLP。
- Memory：`[B,144,256]`，生成消费投影256→1024；不使用旧896维全局TC广播，不叠加旧R模块。目标Ec不依赖扩散噪声，推理前计算一次memory。
- Teacher：复用现有冻结VGG的enc_2，GT输入96px时输出128×24×24，平均池化到128×12×12。训练期readout `Linear(256,128)`逐token预测；teacher不更新，只读GT；train通道均值/尺度固定后标准化。不能宣称12×12监督能恢复GT里所有单像素纹理。
- H3从开始就对memory给L_comp梯度；H4同初始化但lambda=0。冻结encoder不冻结reader。readout只看memory，不读GT/noisy target/第二份ref。
- 迁移初期所有新接口臂使用相同2000-update过渡：up-attention输出 `Y=(1-g)A(Q,G_old)+g A(Q,M_new)`，g从0线性到1；共用原attention权重，g是固定日程而非新学习模块。g=0复现旧父模型，g=1删去旧up fallback计算。H1/H2的M_new为各自global+local上下文；H3/H4/H5为目标memory。记录g，验证step0一致、2000后新通路有梯度且确实被用到。
- **down-path旧global9暂保留**，所有正式臂一致，不在本轮同时改MCA/down结构。H主线不是“全网络都只剩一个144-token输入”，而是“单一的可训练目标外观补全/ref up-path”；没有第二个全局/空间补全器。移除down全局是可选后续，不伪装已实现。

H2的N必须泛化：k_valid=有效local token数/T_per_ref，T_per_ref=144；H1为16。不得沿用当前硬编码除16逻辑。H3固定目标memory、不拼raw local，不使用这项global/local计数校正。

## 5. 统一训练配方（提议默认）

### 数据与batch

- 冻结现有v0913_clean及donor mask：train56429对、val4123、test3880；保留现有Latin/kana/bopomofo的50/38/12采样权重。
- 96px原生输入；首轮不做翻转、旋转、独立ref/GT形变、膨胀腐蚀等改变字体语义/风格的增强。不混入新增脏font或另一个清洗版本。
- 每episode随机k∈{1,...,8}均匀采样，同字体ref无放回，参考顺序不参与语义；同一microbatch可按最大k padding。随机参考组合本身就是支持集增强。
- 单个8-rank DDP作业，global batch64。默认每卡4、累积2；若所有臂实测稳定且余量足，统一改每卡8、累积1。H2若OOM优先activation checkpoint/attention slicing，必要时统一改每卡2、累积4，不偷偷改变global batch。

### 优化器和LR

AdamW beta=(0.9,0.999)、eps=1e-8、weight_decay=0.01；bias和norm不衰减。单一seed3407作为执行配置，不在论文突出。

| 参数组 | 0–1000更新 | 1000–20000 | 20000–40000 |
|---|---|---|---|
| 继承UNet/RSI参数 | 从0线性warmup至 **2e-5** | 保持2e-5 | cosine到2e-6 |
| 新reader/local projection/消费投影/readout | 从0线性warmup至 **1e-4** | 保持1e-4 | cosine到1e-5 |
| Es/Ec/VGG | 冻结 | 冻结 | 冻结 |

所有臂从一开始使用同一40k日程；20k停止的臂是共同轨迹的截面，不给其单独提前decay后再与H3@20k比较。需继续时可按原日程resume到40k，不能重置warmup。40k仍进步时，先review再做成对新增10k的低LR延长，不预先承诺无限训练。

正式训练前做两组2000-update LR pilot：UNet1e-5/新模块5e-5，以及UNet2e-5/新模块1e-4；其他因素固定，在H3样式上比较数值、可读性、细节趋势与耗时。其余最多4k诊断预算用于单batch、断点恢复、H4小规模补全off。按val确定正式LR后全核心臂共用，不给单个消融单独调参。默认选2e-5并非保证更好。

### Loss

`L = L_eps + 0.01 L_VGG + 0.5 (L_offset/2) + lambda_comp L_comp`。

- 保留epsilon预测、原noise schedule，不同时换FM/v-prediction。
- H3/H5/H-D+/H-D−：lambda_comp在前2000次update从0升到0.01；teacher标准化后的SmoothL1按全部有效维度平均。H4=0；H0/H1/H2无补全项。
- 监测各loss及其reader梯度，不只打印总loss。若pilot发现补全梯度持续压过生成梯度两倍以上且生成退化，正式启动前统一将lambda固定为0.003；不在正式臂里自动反复调权。
- 先不额外加D/对抗loss/新style contrastive loss，避免补全贡献和新loss收益混在一起。

### CFG一致化、EMA和AMP

- joint CFG dropout=0.1，同时关闭ref、content、Delta以及外观memory；独立source_drop=0.25仍保留，对Delta的最终关闭mask用joint_mask OR source_mask。无Delta臂仍消耗同样随机draw以对齐数据噪声轨迹。不要把dropout修正只给H3。
- 增加EMA覆盖UNet和所有参与推理的可训练ref/reader参数；不更新冻结teacher/encoder。更新系数 `min(0.999, 1-1/(u+1))`，u为成功optimizer更新数。EMA仅在未AMP跳步时更新，与checkpoint一起保存。所有H臂统一使用EMA做主评测，raw权重在固定诊断集另存对照。
- V100使用fp16+GradScaler、FP32主参数/optimizer states；attention logits/softmax和loss reductions保持FP32，clip global grad norm=1.0。不要用bf16或假设V100能用FlashAttention2。

## 6. 正确性预检与训练健康标准

正式启动前必须通过：

1. 1-ref与重复8-ref、参考置换、padding不变性；测试reader输出和最终生成（固定Delta）；N的16/144-token两版本测试。
2. 8-rank各有不同训练样本；只在accumulation完成且未AMP skip时增加global_step/LR/EMA；跨rank新模块参数更新后hash/数值一致；不能仅以某rank有梯度证明DDP同步正确。
3. 相同父模型和g=0的输出一致；新模块RNG隔离，不改episode/noise流；g=1后旧up fallback确实退出。
4. 单batch拟合、小数据过拟合；正确ref vs错字体ref，排查只凭Ec生成平均字体；目标GT只进入teacher/loss，不进入推理。
5. 断点恢复短测：完整optimizer/scaler/EMA/随机状态/采样位置恢复；原子写checkpoint，无半写文件被当作best。
6. Es/Ec/cache SHA和native96px预处理一致；新raw12缓存在线随机抽查；不得修Es尾层重复调用后配旧cache。

训练监控：每100成功updates记录分项loss、每参数组LR、unclipped/clipped grad norm、reader/消费投影更新范数、EMA差异、AMP skips、loss scale、显存峰值、data time、update time、rank间差异。

- NaN/Inf、OOM、rank不同步、越界/错配cache：立即保存安全状态/停止该臂，先修复再新run，不带病长训。
- 过渡结束后新路径梯度持续为零，或连续1000更新AMP skip>1%，或95%以上更新都触发clip：标记需要诊断，不把阈值当万能发散判据。
- 数值正常但20k时仍较5k/10k有明显图像收益：继续主线；数值正常但训练小集都没有目标外观学习迹象：检查条件消费/teacher，而非仅加步数。

## 7. 评测、保存与“训够”判断

- 每2k：固定val子集64对 × k1/k4/k8的原图小面板（一个参考组）及分项指标，固定噪声；validation可一次缓存条件。避开当前每500更新重复全val带来的高开销。
- 20k/40k及每臂结束：固定分层val面板，16字体各选16个有效目标字符（Latin/digits/kana/bopomofo按预先冻结规则覆盖），共256对 × 4组ref × k1/2/4/8，4096张。10k保留192张快照，不默认追加4096张。每字体存在性须先按clean mask校验；不足时按固定规则补选，不能看结果选字。各组预先从ref charset确定，组内k嵌套。主表优先Latin及digits，同时分列kana/bopomofo。此项替代此前每个里程碑全量val的高成本安排。
- 全量val仅用于进入最终比较的候选；每个checkpoint为4123对 × 16条件 = 65968张。最终test每个checkpoint为3880对 × 16条件 = 62080张；先按val锁定checkpoint再测试。默认工期包含主模型和一个基线各一次完整val/test；若论文要求更多方法全量test，每加一个checkpoint追加推理预算，不隐含在训练时间里。
- 主比较按同一update切片、EMA、采样器20step/CFG7.5。必要的CFG微调在val上统一扫描；test不选checkpoint/CFG/配方。
- 图像review看空心/描边、尖收笔、飞白、装饰回卷、字重和字形，不用“变粗”代替细节恢复；报告前景/边缘误差、L1、SSIM、字符可读性，以及固定k跨ref组合波动。E12新版可用并完成校准审查后补充，当前不把旧评测器分数作为唯一选模依据。
- 每2k原子保存last_state（含EMA/optimizer/scaler）；5k/10k/20k/30k/40k保存推理权重与固定面板，保留按既定val选择的best。原始loss最低与视觉best分开记录，不再只允许末步成为best。
- 30k–40k连续几个面板均无可见改善、定量指标改善不足约1%且跨ref稳定性无进展，视为暂时平台，避免无限延长；1%是review提示不是自动停训/显著性阈值。若仅主线延长，报告对应新预算，关键消融需同预算延长再比较。

## 8. 存储与工期

新ref raw12 fp16 cache：87880×256×12×12×2 bytes，约6.0GiB；从已有Es重算，不可由pool4缓存插值伪造。新teacher可直接复用生成loss中GT的VGG特征，避免再保存全量teacher cache；若需预计算约2GiB另列。原Ec约大规模缓存只读复用，不复制。

初始只有约19GiB空闲。用户于本轮授权清理旧G后，已释放27.09GiB，实机空闲45.91GiB（df约46G）。没有删除日志/推理图/指标/cache，也没有改动活动G-REF正式臂。满足至少40GiB的初始门槛，但训练中仍须监控：raw12缓存、活动run的EMA/optimizer恢复、两个原子保存版本和多臂里程碑都占空间；不能调用自动删除旧cache的历史rebuild脚本。

G-REF A0的2500步训练含验证约43分钟，折合约1.0秒/update，只能作为旧架构量级参考。H2更长上下文、H3/H4新增reader和teacher消费、accumulation和评测改变都会影响吞吐。180k核心updates在假设1–2秒/update时约50–100小时（8卡同时工作），不含缓存/预检/完整图像评测；这不是实测工期承诺。先测H1/H2/H3连续100–300稳态updates，再给正式预算。

### 8.1 交付时间与训后推理（新增）

以下从H切换获批、八卡可用开始计时，不是从文档创建时自动倒计时；不包含当前G队列继续运行造成的等待。开发/调试时间是预算估算。旧CFG扫描256张约80秒（含初始化），只提供推理量级参考，不能视作H实测吞吐。

| 阶段 | 暂估八卡墙钟时间 | 交付 |
|---|---:|---|
| 实现、单测、raw12缓存、DDP预检、LR pilots | 6–12小时，出现实现问题则重估 | 可运行H、梯度/恢复/吞吐报告 |
| 单臂20k训练 | 5.6–11.1小时 | EMA/raw里程碑，随后立即推理 |
| 单臂40k训练 | 11.1–22.2小时 | EMA/raw终点，随后立即推理 |
| 首轮H0@20k vs H3@20k，含准备与面板 | 累计约18–36小时 | 第一组充分训练的基线/新方法对比 |
| 核心180k全部训练 | 净训练50–100小时 | H0–H4及Delta双臂 |
| 核心训练+准备+中评+主模型/基线完整val/test | **约4–8天** | 可review模型、shot对比、主要消融和完整结果 |

推理不等所有实验训完才启动：每2k的192张快照预算2–5分钟；每个20k/40k里程碑的4096张面板预算0.5–1.5小时；全量val或test每个checkpoint各暂估6–12小时。所有数字都需用H实际吞吐重估。核心矩阵共9次4096张面板，约4.5–13.5小时；最多90次192张快照约3–7.5小时，去重后总中评按8–21小时预留。若新增其他里程碑全量推理，另计预算。完整总时间约88–181小时，按4–8天排期。

H队列应实现状态机：`TRAIN_DONE → QUICK_INFER → REVIEW_PANEL+METRICS → NEXT_TRAIN`。每次推理保存checkpoint/EMA/代码/data SHA、ref列表、seed、采样设置、preds、指标JSON和可编辑/可浏览面板。以文件数和成功状态校验推理完成，不能仅凭训练DONE跳过。推理失败保留checkpoint并标记失败任务，可让下一训练臂继续，稍后补推；不得把缺图的模型记为已交付。最终大规模推理在选模后排队执行。八卡训练时单个8-rank DDP作业；推理时8个独立worker分摊episode，不与训练抢同八卡。

此状态机是H实现验收要求，当前尚未编写或启动H执行器。现有G-REF执行器已有A0_panel等训后推理任务，继续按原计划运行。

### 8.2 基于G2的论文叙事（新增，待review）

一句话：**HR-Font以目标字的结构变化先验和目标对齐的外观补全，将少量中文参考扩展为跨文字系统、风格协调的字体家族。**

三项贡献保持：任务与多维评测；Mean-Delta结构变化先验；目标字外观补全TC。G2/G/H是实验代号，G2-CONT是初始化资产，不是第四项贡献。Mean-Delta回答“这个目标字有哪些可行的形态变化”，TC回答“参考字体的局部外观如何体现在这个目标字上”；alpha仍用于限定候选先验邻域，不承担跨script风格真值判断。

当前paper/iclr2027_hrfont/main.tex的标题、abstract方向、任务定义与Mean-Delta公式可保留。需实质更新TC小节、overview/denoiser接口、图1、训练目标与实验矩阵：将896维VGG均值/标准差预测及9-token共享广播改为144个目标对齐局部tokens和空间teacher监督。旧TC作为历史对照，不在主文沿用已被替换的公式。down-global9仍存在，图文如实呈现。不是叠加新R再叠加新TC，而是同一目标外观表示同时承担ref融合与补全学习。

G2当前证据支持将其用作稳定的结构/生成基座，尚不足以声称笔触、描边已恢复；H3/H4检验补全监督，H2/H4检验融合，H-D双臂检验Delta。文章把最终经验证的机制作为方法主体，不把调试过程写成模块清单。G2 warm-start的Delta历史不能用于严格no-Delta归因，因此保留G0b共同父权重的双臂对照。

## 9. 建议执行顺序与用户需 review 的点

1. H-PREFLIGHT/LR pilots（上限8k），确定共同配方、冻结数据/缓存/初始化指纹。
2. **H0@20k + H3前20k**：先拿到长训基线与完整方法，效果优先。
3. **H4前20k**：检验补全监督。H3/H4均数值正常、有可用趋势时，两者继续至40k；不要因H4更好就丢弃该结果。
4. H1/H2@20k补齐信息量/局部路径参照；H-D+/H-D−@20k补齐Delta贡献。若前期H3明显失败，则暂停大批消融，先定位，不浪费完整180k预算。
5. H5和H-FuseMean按论文证据需要追加。最终最佳不预设一定是H3，用固定val标准选择；三项贡献靠对照成立，不靠实验命名保证。

请PI review：默认核心180k + 预检8k预算；主线40k/消融20k的分层；P选择G-CONT-G2终点、P0选择G0b；冻结Es/Ec；所有H统一joint CFG+EMA；先保持down-global9，只替换up-ref/TC路径；至少40GiB空间安排。

H正式获批时，应明确G-REF剩余队列是否在当前安全checkpoint边界停止；不能让旧自动巡检继续追加N/旧TC而同时启动H抢同八卡。当前仍按原批准计划运行G，本设计稿不改变该状态。

本计划使用 ml-training-recipes 及其 diffusion/DDP/checkpointing 指南，结合当前V100、冻结cache和已有训练循环调整；没有照搬通用LLM LR、bf16或图像翻转增强。

## 10. 旧G资产清理执行记录（2026-09-15）

用户明确要求只保留仍需测试的旧G模型。脚本 `scripts/prune_g_storage_20260915.py` 先执行dry-run，再验证DONE、文件size/mtime、符号链接和活动进程依赖，按完全一致清单删除129个文件；未删除任何run目录。

- 已释放逻辑大小29,084,163,919 bytes = 27.0867GiB；磁盘空闲20,205,965,312→49,290,375,168 bytes。活动A1和dispatcher在清理后仍RUNNING。
- 保留：G0b@10k、G0c@20k、G1/G2/G2-RL@10k；G-CONT-G2/G2RL@2500和5000；旧TC-G2/TC-G2RL@2500和5000及其best路径；pilot@1000、pilot8@2500及best，因这些仍被既有review评测协议引用。保留best路径避免破坏旧评测入口。
- 完整恢复状态额外保留三份：G0b@10k、G0c@20k、默认H父模型G-CONT-G2@5000。其余旧G保留推理和weight-only warm-start所需权重，不再保留历史optimizer。
- 移除：上述完成run的冗余last_state权重和不需要的trainer_state；两个RL pilot未选中的5000终点；两个已完成100-step smoke的实际模型/optimizer文件。smoke编码器符号链接、配置、DONE、日志等小文件保留，不能再将smoke目录当作完整checkpoint。
- 未触碰：G-REF-A0/A1正式臂、未来B臂、F系列、缓存、数据、日志、指标、所有推理结果。best与milestone存在同模型的兼容路径，此次不为节省少量空间冒险重写这些路径。
- 删除的optimizer和未保留的smoke/pilot终点没有另行归档，不可恢复；保留的终点仍支持推理和warm-start。完整审计见 `reports/g_storage_prune_20260915.json`。

本轮仅补充叙事和排期供review，尚未改论文正文或启动H。论文叙事梳理采用ml-paper-writing的统一贡献主线原则，不增加未经验证的结果陈述。
