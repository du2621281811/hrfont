# HRFont 跨系列实验追踪（持续更新入口）

更新：2026-09-15 18:04 +08。当前优先级：**磁盘低于10GiB，I2已在5146步安全暂停，待存储处理与恢复review；I1已完成，I0匹配推理仍待执行**。H 已由用户决定终止，不继续旧七臂队列。

本次审阅入口：[I2重建批准证据](review_20260915/I2_RENDER_APPROVAL_20260915.json)、[I2构建规则](review_20260915/I2_RENDER_REVIEW.md)、[E12-c实验计划（待review、未派发）](E12C_PLAN_20260915.md)、[最新E12代码审核](review_20260915/E12_REVIEW.md)。本次用户明确批准后，已归档解除渲染专用STOP；其他失败/磁盘条件仍需停队列。

| 系列/模型 | 动机与方法 | 进展 | 效果与验证结论 | 详情 |
|---|---|---|---|---|
| F / F2-VEC | 初始跨语种、RSI/Delta及栅格-矢量分支 | 历史实验保留；本轮不续训 | 不把栅格指标等同于矢量质量；旧结果保持原协议 | [历史报告目录](REPORT_HUB.md)、[逐run记录](experiments/RUNS.json) |
| E12 / E12-b | 字体关系/风格评测模型 | 独立评测模型系列，不等同于生成器训练 | 使用各版本自己的验证状态；未核验新指标不混填生成表 | [E12-b](e12_b/STATUS.md) |
| G0/G0b/G0c、G1/G2/G2-RL | V0913基线、Mean-Delta及ref接口 | 历史检查点/推理保留；**仅G0b指定10k别名为I0** | G/TC/CONT是不同lineage；不能因同名G0混用权重 | [G shot板](../G_V0913_SHOT.html)、[历史目录](REPORT_HUB.md) |
| G-TC / G-CONT / G-REF | TC联合监督、匹配续训、ref聚合 | 不自动重启；实际各run记录保存在机器快照 | 先前笔触/特效不突出，推动H/I改造；不写成所有G方法均无效 | [逐run状态](experiments/RUNS.json) |
| H0 | G2-CONT底座延长10k，全局ref控制 | 10k完成，4096最终推理完成 | 最终L1/SSIM整体优于H3；H0不是I0 | [H0/H3匹配](H0_H3_FINAL_MATCHED_REVIEW_20260915.md) |
| H3 | 缓存局部特征、目标对齐读出+补全监督 | 10k完成，4096最终推理完成 | L1四个shot均未超过H0；部分edge改进未形成明显笔触收益 | [最终review](H3_FINAL_REVIEW_20260915.md) |
| H4 | H3结构、去补全监督 | **9522安全停止**；2/4/6/8k面板完成 | 8k对齐比较H3略优，但整体效果仍有限；无H4@10k结果 | [8k匹配](H4_8K_MATCHED_REVIEW_20260915.md) |
| H2/H1/H-D+/H-D− | ref数量归一化与旧结构消融计划 | **未启动、取消** | 没有训练结果，不填写负结果 | [H收尾](H_CLOSEOUT_20260915.md) |
| I0 | 统一预训练底座 | G0b@10000别名，不新增训练 | 需要按I协议比较，不沿用H0作为I控制 | [I规格](I_EXECUTION_20260915.md) |
| I1 | 在线多尺度TC + 动态set-delta | **10k训练及4096最终推理完成，全部4864阶段预测图归档；零跳步、DDP差0、恢复状态完整** | 全量平均L1/SSIM随shot增加改善；与8k匹配192子集：1shot改善，4/8shot小幅波动。尚无I0最终匹配结果，不宣称风格问题解决 | [最终报告](I1_FINAL_I2_START_20260915.md)、[最终原图](experiments/I/I1/step00010000/review.html) |
| I2 | I1架构 + 中文留一目标辅助任务 | **低磁盘STOP：5146/5146安全暂停，完整恢复状态已保存且trainer加载通过；队列及训练进程已退出**；最后周期日志八rank5100、零跳步、DDP差0 | 触发磁盘5.70GiB，停止后5.43GiB；未手工删除数据。无6k新结果，4k效果结论不变；存储处理与明确恢复review后才能接续，剩余4854更新 | [暂停与恢复交接](I2_DISK_STOP_20260915.md)、[4k匹配报告](I2_4K_MATCHED_REVIEW_20260915.md)、[4k原图](experiments/I/I2/step00004000/review.html) |
| E12-c | 轻量跨文字字体相容性评测＋独立人评 | **轻量设计待PI review，未实现/未训练/未占卡** | 单模型：真实字体多正例表征→冻结编码器训练集合头；人评仅验证，不执行旧C0/C1/C2，与I独立 | [交接入口](E12C_HANDOFF_20260915.md) · [完整计划](E12C_PLAN_20260915.md) · [人评协议](E12C_HUMAN_PROTOCOL_20260915.md) |

## 不混淆的规则

1. 唯一身份是 `series / run_id / parent / checkpoint_step / protocol`，短名不能替代 lineage。I0=G0b，不等于H0。
2. `experiments/RUNS.json` 从执行机读取每个run的DONE/STOPPED/config/heartbeat；旧heartbeat不证明仍在运行。正式I活动以队列状态及现场进程核验为准。
3. 推理归档同时保留PNG、GT、protocol、逐样本metrics、可离线HTML。H稳定路径继续保留，I新路径为 `experiments/I/I1/step00002000/` 等；不复制改名旧图冒充新推理。
4. 历史F/G/E评测通过原报告目录检索；未迁移的历史资产标明原入口，不宣称已经重新评测。新旧ref组、clean/dirty、val/test不能混算。
5. 每次里程碑：导出状态 → 同步原图 → 更新本表结论 → 非force Git提交与推送。10分钟巡检负责发现进度/故障，阶段结果到齐后推送。权重留执行机，Git保存配置、哈希及位置，不往Git塞多GB训练状态。

## 更新记录

- 2026-09-15 18:04 +08：18:01磁盘实测5.70GiB，按批准阈值创建队列/运行STOP；I2在5146步安全保存退出，停止后5.43GiB。model/EMA/trainer及complete清单完整，trainer实际CPU加载验证8 RNG与910 optimizer状态条目；无重启。队列缺DONE报NEEDS_ATTENTION是主动暂停的预期后果，不是训练数值故障；保留新STOP，不因旧渲染批准而解除。恢复需先处理存储并review，详见暂停交接。

- 2026-09-15 17:49 +08：八rank4700/4700，skips0、DDP差0；reader/encoder/router梯度.005349/.001258/.00005085，scale4096，gate1、TC系数.01，LR新模块1e-4/继承2e-5。4500恢复目录的model/EMA/trainer及complete清单完整，未做中断恢复测试。磁盘精确余量11,419,795,456字节（10.64GiB），接近但尚未低于10GiB暂停线；无6k推理结果，不热改、不重启、不删除数据。队列的磁盘检查在实验臂启动前，不是训练中连续保护，后续巡检若低于阈值须通过STOP安全暂停并报告。

- 2026-09-15 17:34 +08：I2的4k/192图59.75秒完成，268个归档文件哈希全部通过；与I1@4k、I2@2k各192键严格匹配。八rank4300/4300，skips0、DDP差0，reader/encoder/router梯度均非零、scale4096；4000步model/EMA/trainer及complete清单完整，未做中断恢复测试。磁盘13GiB，继续原队列。Overleaf 1.0已在独立论文更新中发布，本次巡检不修改论文。

- 2026-09-15 16:54 +08：八rank全部3300/3300，AMP跳步0、DDP差0；reader/encoder/router梯度.00655/.00197/.000113，scale2048，gate1、TC系数.01、LR新模块1e-4/继承2e-5正常。3000步model/EMA/trainer及complete清单完整，未进行中断恢复测试。磁盘13GiB；无新推理结果，未热改、重启或删除数据，Overleaf候选稿未触碰。

- 2026-09-15 16:21 +08：I2首2k/192图推理56.22秒完成，268登记文件哈希全部通过；与I1同2k的192键严格匹配，结果略逊但有局部改善。训练已恢复，8rank到2400，skips0、DDP差0、模块梯度非零，gate1/TC系数.01符合计划。磁盘14GiB，未重启、热改或删除数据。导出快照16:20:30的heartbeat为2300，后续现场检查为2400。

- 2026-09-15 15:16 +08：I2 step700日志及全部rank核验通过。reader/encoder/router梯度.00334/.00123/5.86e-5，GradScaler1024、所有已导出记录skips0且DDP差0。`last_state -> state_step_500`包含model/EMA/trainer及complete=true清单；只检查文件完整性，未中断训练做恢复测试。磁盘15GiB，没有新推理里程碑；继续原队列，不删除数据或重启实验。

- 2026-09-15 15:04 +08：I2首100步100/100、skips0、DDP差0，reader/encoder/router梯度分别8.82e-5/1.36e-5/1.19e-4；step300全部8 rank同步、各rank中文辅助loss有限且为正，gate=.3、TC系数=.003，新模块LR6e-5、继承LR1.2e-5。100→300用时442.12秒，约2.21秒/成功更新；10k净训练约6.1h，额外推理/保存另计。磁盘17GiB，高于10GiB停派发线，未删除任何数据；第一份正式恢复状态按计划在500步保存。安全fast-forward纳入协作者5次E12-b说明文档提交至b8dea70fe，未改训练代码或启动E12-c。

- 2026-09-15 14:27 +08：I1完成10k训练，累计训练循环及中间评测9494.97秒（约2.64h，不含最终4096推理）。末步新模块LR1e-5、继承模块2e-6，reader/encoder/router梯度均非零，AMP scale32768；所有rank step10000、skips0、DDP差0。`last_state -> state_step_10000`含model、EMA、trainer和complete=true清单，本轮未做恢复测试。磁盘约18GiB；I2尚未预检或正式启动，保持单队列衔接。最终推理未完成，不把训练完成等同于整臂完成。

- 2026-09-15：用户批准H→I；H4@9522安全保存，H队列STOP生效。H队列因缺DONE显示NEEDS_ATTENTION是主动停止的后果，禁止按训练故障自动重启。
- 2026-09-15：I架构完成梯度/无条件分支/重复ref/候选置换/DPM预检；8卡100次更新无AMP跳步，DDP参数一致。正式状态随后写入队列JSON。
- 2026-09-15：I1正式run=`I1-V0915-S3407`，队列PID3360111，训练协调PID3360112。预检另完成恢复100→120，未用smoke权重启动正式训练。I1正式step100约0.935s/update，峰值8686MiB；初始化后前100步97.3s。I2和I0最终相同协议推理已列入队列。
- 2026-09-15：I1已保存step500完整状态。I2中文输入338张生成完成，CPU真实模型检查target不在refs、Delta offset严格0、在线encoder梯度非零；尚未运行I2正式训练或8卡预检。旧西文重绘审计不完全一致，按新中文锚点版本明确记录，主数据未改。
- 2026-09-15 12:04 +08现场核验：I1日志已到1000/1000，TC ramp完全开放；reader梯度.00184、encoder梯度.000889、router梯度5.78e-5，AMP跳步0、DDP参数差0，step1000约.907秒。首批已push 7eff8eecf，中文扩展及后续记录已push 2d688c88c；当前同步提交以Git历史为准。
