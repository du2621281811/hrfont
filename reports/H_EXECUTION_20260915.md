# H 系列执行规格：每臂新增10k，效果优先

2026-09-15，用户已明确批准开始训练，并将原180k方案缩减为每个正式模型新增10000个真实optimizer updates。本文件替代 `H_EXPERIMENT_PLAN_FOR_REVIEW_20260915.md` 中所有20k/40k预算、学习率时间表和执行顺序；结构设计与共同父权重原则继续有效。

## 批准的队列

| 顺序 | 模型 | 改动 | 新增成功updates |
|---|---|---|---:|
| 1 | H3 | 完整目标对齐外观memory + 空间补全监督 | 10000 |
| 2 | H0 | G2原global9接口，统一新训练配方 | 10000 |
| 3 | H4 | 与H3同网络、同初始化，补全监督权重0 | 10000 |
| 4 | H2 | raw12局部token直接拼接 + N计数归一化 | 10000 |
| 5 | H1 | pool4局部token直接拼接 + N | 10000 |
| 6 | H-D+ | 从G0b共同起点训练完整方法 | 10000 |
| 7 | H-D− | 与H-D+相同，只将结构Delta置零 | 10000 |

核心上限 **70000八卡updates**。H5/早期均值融合等可选实验不启动，也不自动续到20k/40k。10k是本轮预算和评审终点，不是预先认定收敛。若仍有改善空间，先报告曲线/图像证据，再请用户批准单独延长。

H0–H4共同父权重 `/root/projects/hrfont/runs/G-CONT-G2-8gpu-V0914-A-S3407/global_step_5000`；H-D双臂共同父权重 `/root/projects/hrfont/runs/G0b-F0-V0913-BS256-A-S3407/global_step_10000`。每臂独立初始化，不从前一个H模型串联；10k计数不包含父模型训练史。Es/Ec/VGG均冻结，Mean-Delta与清洗数据保持不变。

## 实现与10k配方

- `scripts/hrfont_h.py`：每ref保留12×12×256特征；目标Ec12×12加二维位置编码；4头、256维空间attention，先逐ref对齐再等权融合，输出144个目标token。一个residual MLP，256→1024消费投影；256→128训练readout预测标准化VGG enc2空间teacher。不叠加旧R或896维广播TC。down-global9保留。
- `scripts/h_runtime.py`：复用现有clean数据、Es/Ec缓存、排除本字体的donor检索和Mean-Delta算法。新raw12缓存与现有Es绑定。训练数据仍56429对，val4123对。
- `scripts/prepare_h_cache.py`：8卡分片缓存87880个ref，fp16共6.03GiB；每rank在线复验8个样本。teacher通道均值/标准差由固定2048个有效训练目标标定，记录逐一文件路径，不读取val/test目标；不是全训练集统计，作为明确的工程近似。
- `scripts/train_h.py`：独立DDP训练入口，不修改旧G trainer。每卡4、累积2，global64；weighted采样遵循clean sample_weights。episode由attempt编号确定，8个rank分摊同一global64列表；ref在1–8之间无放回采样。记录rank样本与参数指纹。
- AdamW：继承参数峰值2e-5，新reader/projection峰值1e-4；betas=(.9,.999)，eps=1e-8，矩阵参数wd=.01，bias/norm不衰减。
- **500步warmup → 保持至5000 → cosine至10000时峰值的10%**。按成功更新数调度，不按8rank或microbatch重复推进。
- 前1000更新线性迁移up-attention：`(1-g)A(Q,G_old)+gA(Q,M_new)`，所有新接口臂一致；之后停止旧up fallback。H3/H-D双臂的补全系数同期0→.01；H4为0。
- loss保持eps + .01 VGG + .5(offset/2) + lambda_comp SmoothL1。teacher复用GT生成loss所需VGG结果，GT不输入reader/生成条件。
- joint CFG .1同步关闭content、ref/up memory、Delta，额外Delta dropout .25；H-D−仍执行相同条件计算和随机draw后置零，避免改变其他因素。
- V100 fp16+GradScaler，初始scale1024；attention logits/softmax、loss及optimizer为FP32；全局clip1。非有限梯度所有rank一致跳过更新并降低scale，成功更新才增加step/LR/EMA。超过10次skip且比例>1%停止诊断。
- EMA覆盖所有推理可训练参数，decay=min(.999,1−1/(u+1))。所有rank同步维护，用同一EMA协议推理。
- 不再跑原上限8k的LR搜索。先做100步固定batch八卡smoke，再从完整断点接续20步，正式H3从共同父权重重新开始；预检预算单列，不混入正式10k。

## 保存、训后推理与验收

- 每500步安全保存model、EMA、optimizer、scaler、各rank RNG、attempt/采样键和step。先写新目录，再原子切换last_state符号链接，只保留本H run最近两个恢复状态；已标记的推理里程碑权重通过硬链接保留。
- 每2000步：固定64对 × k1/4/8 =192张val快照；每臂10000步：固定256对 × 4组ref × k1/2/4/8 =4096张。每字体目标只从clean有效集合选定，参考组内嵌套；种子由font/char固定，不随shot/模型改变。
- 标准DPM++20、CFG7.5、EMA，保存pred、GT、episode元数据、L1/SSIM/边缘误差和HTML。完整test不用于选择checkpoint，本轮先交付固定val对比；最终全量val/test按候选与需求单独安排，不把它的多小时开销隐含在10k训练中。
- 训后推理包含在每臂执行器内；只有10000步且4096张完整生成后才写正式DONE。推理失败则队列停止并报告，不能把只有权重的模型当成已交付。
- 单元测试覆盖reader重复ref、置换、NaN padding屏蔽、梯度与LR端点；实机接口测试覆盖G2父模型g=0复现、固定Delta下重复8ref生成一致性、N的16/144版本和DPM20采样。
- 八卡smoke检查非零reader梯度、DDP权重一致、有限loss、固定batch学习；恢复测试检查optimizer/scaler/step正确接续。仅“可以恢复”不等于已验证与不中断轨迹逐位一致；若该精度测试尚未完成则不宣称bitwise replay。
- 数值故障自动停止；视觉效果由固定面板及巡检检查，不以diffusion loss替代风格效果。发现明显退化先暂停后续队列诊断，不自动增加步数或切换研究方向。

## 执行机交接

- SSH `sitonholy`，权重/报告主目录 `/root/projects/hrfont`。
- H代码隔离快照 `/root/projects/hrfont_h_20260915.AkAWHq`，基础代码为本地 `d0fd6be1b`，新增H文件按run config记录SHA。正式运行中的快照不得热改。
- 新缓存 `/root/projects/hrfont/artifacts/h_20260915`；构建完成，64个在线复验通过。
- 队列入口 `scripts/queue_h_20260915.py`，状态 `/root/projects/hrfont/reports/h_20260915/status.json`；初始queue PID3219789。缓存PID3218045已完成。
- G A1及其面板已完成。后续G B0经STOP机制保存到 `runs/G-REF-B0-V0914-S3407/stopped_step` 后退出；G dispatcher因提前终止B0没有DONE而记录NEEDS_ATTENTION，这是本次切换引起的预期状态，不是H故障。旧G STOP文件保留，不重启其剩余B/N/TC队列。
- 旧巡检已更新为只监控H的10k队列并恢复ACTIVE，每10分钟检查。不得终止不属于本项目的后台GPU上下文。

## 时间预算

旧G约1秒/update不是H实测。10k每臂先按约3–6小时净训练估算；H3/H2的实际吞吐、缓存读取和2k快照须实测更新。7臂净训练约20–40小时，额外7次4096图面板及中途快照通常数小时到十余小时，首轮按约1.5–3天预留，较慢实现需相应延长。首个H3完成时间以其前100–300个成功更新的实测速率为准，不承诺未经测量的时刻。

使用ml-training-recipes的混合精度、DDP、EMA与checkpoint指南；具体采样、teacher、学习率和增强策略按当前字体任务调整，不采用通用分类翻转或LLM配方。

## 正式启动核验（北京时间2026-09-15 01:41）

- H3正式run：`H3-V0915-S3407`，8-rank训练子进程的launcher PID3221526；队列PID3219789。
- 最新已读取成功更新step10/attempt10，8个rank均到step10，reader梯度1.3325e-4，跨rank参数指纹差0，AMP skips0，峰值分配显存每卡约4.8GiB。LR继承参数4e-7、新参数2e-6，符合第10次更新的500-step warmup；gate=.01符合1000-step迁移。
- 近期单次更新约1.11秒，首步约3.89秒。还不能将十步吞吐视为全程稳态承诺；暂估首个H3含推理约3–5小时，首个2k快照约40–60分钟，核心整轮约1.5–3天，依据后续100–300步和首次推理重估。
- 八卡固定batch smoke100→完整恢复→120均完成；100步loss0.027425→0.018438、reader梯度非零、DDP差0、skip0。恢复接续20步完成，用时25.84秒。该测试证明短程训练与恢复可用，不替代正式风格效果评测。
- 真实父模型接口测试：g=0输出最大绝对差0；固定Delta、1ref与重复8ref输出差0；N的16和144 token版本、DPM20调用均PASS。
- 独立推理重试入口为 `scripts/eval_h_checkpoint.py`，使用torchrun八卡，`--checkpoint <run/global_step_N> --out <new review dir>`，默认4096张标准val，`--quick`为192张。不需要重训。
- 多臂存储维护：`scripts/prune_h_rolling_states.py` 默认只预览，`--apply`只清理已完成10k且推理DONE的正式H run里过期rolling state，保留最后完整恢复状态及所有推理milestone。既有G或其他数据不在其白名单。
# 已停止的历史排程

2026-09-15：用户决定H→I。H0/H3各完成10k，H4安全停止9522，其他H臂取消。下文保留为原始执行记录，**不得继续派发**；当前安排见 [I执行规格](I_EXECUTION_20260915.md) 和 [跨系列追踪](EXPERIMENT_TRACKER.md)。
