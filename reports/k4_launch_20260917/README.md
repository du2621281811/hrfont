# K4 已启动：C → B → A

用户已批准完整实验并要求 K4-C 排在 K4-B 前。服务器持久队列已启动，三组各训练 10,000 个成功更新，再执行全部模型的 v2 val/test 推理和指标统计。

**2026-09-17 23:18（北京时间）核验：K4-C 已完成 100/10,000 更新，8 张 V100、global batch 64、AMP 跳步 0、八卡参数同步检查差值 0。** 前 100 步用时 107.13 秒，平均 1.07 秒/更新；这是训练循环速度，未计初始化、保存与评估。当前只确认运行健康，尚无 K4 生成质量结论。

| 顺序 | 实验 | 初始化与数据 | 状态 |
|---|---|---|---|
| 1 | K4-C | 同 K1 的 K0 起点；K1 配方；v2 train | 正式训练运行中 |
| 2 | K4-B | 同 K4-C 起点、数据、主采样；增加原 K3 风格差异监督 | 已排队，10k |
| 3 | K4-A | K1 最终完整 EMA；仅 0917 train 增训；新 optimizer/LR 周期 | 已排队，10k |
| 4 | 全量推理与报告 | K0、K1、K3、K4-C/B/A；v2 val/test | 已排队 |

K4-B/C 是相同主任务预算的风格监督对照；B 有额外辅助计算。K4-A 相对共同 K0 已包含先前 K1 的 10k，因此 A 与 C 不是等历史训练预算比较。三组仍使用中文参考到西文、假名、注音，不新增任务方向或 SCR。

## 数据与监督已落实

- B/C：423 个有效训练字体、95,976 个合法 GT 对；A：200 个 0917 训练字体、39,547 对。0913 划分保持冻结，0917 holdout GT 不进入训练或 donor 库。
- donor 逐目标字符和实际参考集合筛选：必须拥有合法训练 GT、全部参考字，且不能是当前目标字体。缺字不替换为其他 GT。
- 已复核 200 个新训练字体的代表字面板，对正例补看中文参考；新增 51 个字体、110 个 font-script 确认标签（西文 51、假名 37、注音 22），旧 22 个组标签保持不变。其余为“未确认”，不是“简单”。这是代表字筛查，不是每个字符的视觉验收。
- 10k 主任务计划共 640,000 episode。B/C 覆盖全部 423 字体、295 个目标字符，实际访问 90,403 个不同 GT 对；0917 占采样曝光 63.85%。A 覆盖全部 200 字体、295 字，访问 39,526 个不同 GT 对。固定更新预算不承诺遍历每个 pair。
- 旧 Es/Ec 只读复用；新增 78,070 条 Es 参考特征、39,547 条训练 Ec 目标特征。Ec 只存实际消费的两个 donor 尺度，保留完整中性内容特征。缓存文件 SHA 已验证。

完整视觉面板保存在本地交付目录 `outputs/K4_LAUNCH_20260917/detail_review/index.html`。Git 中的[标签记录](../../experiments/K4/new_detail_review.json)与[最终标签清单](../../experiments/K4/detail_manifest.json)同步保存。

## 预检与启动证据

| 检查 | 实测结果 |
|---|---|
| 数据清单、GT/参考存在性、donor 范围、10k 采样、缓存哈希 | PASS |
| K0/K1 旧数据新旧适配器 GPU 对齐 | 抽查四组预测像素 MAE 均为 0；K1 覆盖 1/2/4/8-shot |
| 缺参考字体 | FZXLB、ZKTBanQTFU、ZKTMingXTFU、FZZJ-MJZCFU 均实际推理通过 |
| K4-A 初始化 | 所有可训练状态张量与正式 K1 最终 EMA 完全一致 |
| 八卡主训练与续训 | C 连续 120 步；从 100 步完整状态恢复到 120；无 AMP 跳步，同步检查为 0 |
| 首两个恢复更新 | loss 绝对差分别 0、6.56e-7；episode 身份一致 |
| 恢复后最终状态 | 相对 L2 差 8.26e-5；不宣称长期 FP16 逐位一致 |
| B/C 受控采样 | 8 卡前 17 个主 episode/noise/drop 身份逐步一致 |
| K3 完整辅助梯度 | 首个辅助更新的 8 个 rollout 步均有梯度；Reader 有梯度；路由器通过后续检查 |
| A 增训预检 | 两个成功更新，TC beta 保持 0.8 |
| v2 Val192 | 192 条实际 GPU 推理完整完成 |
| 指标依赖 | LPIPS/E12-c 冻结模型 CPU 加载与输入烟测通过；全量指标尚未执行 |
| 正式 C 首批更新 | 8 卡均达到 100 步；首两步与预检 episode 相同、loss 误差 <1e-5 |

预检曾因错误断言暂停：验收要求第一次辅助更新的路由器梯度非零，但 F0 新结构残差层的 zero-conv 初始为零，第一步的生成梯度无法传到上游路由器。已改为检查这一初始零梯度，并要求第 17 步八卡路由梯度能量非零；逐卡梯度必须有限。额外对 120 步状态做八组配对反传，八组路由梯度均非零。早期部分卡的 FP16 小梯度仍可能下溢，这不等于后续链路断开，也不证明风格改善。

此次只改了验收脚本，训练配方和其余 309 个冻结文件未变。旧预检身份和失败记录保留，通过逐文件哈希证明训练源码一致后复用已完成的实际预检；未修改旧 checkpoint/config 来伪装重跑。

证据：[正式运行快照](evidence/LIVE_LAUNCH.json)、[训练与续训](evidence/TRAINING_CHECKS_PASSED.json)、[GPU 对齐](evidence/GPU_PARITY_PASSED.json)、[采样统计](evidence/CONTRACTS_PASSED.json)、[路由补查](evidence/ROUTER_PROBE.json)、[源码连续性](evidence/CONTINUITY.json)、[指标预检](evidence/METRICS_PREFLIGHT.json)。

## 后续自动执行范围

v2 val 7,331 对、test 7,088 对，合计 14,419 对。K0 仅 1-shot；K1、K3、K4-A/B/C 均为 1/2/4/8-shot，合计 302,799 张结果。

已批准复用 2,336 张严格匹配的 K0 图像，逐图及输入 SHA 已验证，预计新生成 300,463 张。所有使用 Delta 的模型统一 v2 train donor 库，因此旧 K1/K3 输出不混入新主评测。统一最终 10k EMA、嵌套参考字、每查询噪声、CFG=1、DPM++ 20 步/order 2、96×96。

每 2k 跑固定 Val192，只作观察，不依质量提前停训或挑选 checkpoint。每 1k 保存完整状态并保留两份滚动恢复点；保留 2k/4k/5k/6k/8k/10k EMA。最终指标输出 L1、SSIM、LPIPS、D_region/D_change/D_high、冻结 E12-c Family，分来源、shot、script、字符子类、冻结几何难度统计，给出 font-macro、pair-micro、来源/脚本平衡分数及配对 lineage bootstrap 区间。Family 是辅助相似度，不是正确率或人评替代；单训练 seed 的区间不代表跨 seed 稳定性。

## 运行交接

- 冻结执行目录：`/root/projects/hrfont_k4_20260917`
- 执行源码 commit：`6ddd2171f50d3ed7f5678a270c5f34f9120cc538`；[完整文件哈希](evidence/K4_CODE_IDENTITY.json)
- E12-c 后处理模型源码由独立 [依赖清单](evidence/K4_POSTPROCESS_DEPENDENCIES.json) 记录；它来自已跟踪 Git 文件，不影响训练源码身份。
- 持久队列 PID：`3848077`；正式 C torchrun PID：`3848098`。队列脱离 SSH 会话运行。
- 队列状态：`/root/data1/hrfont_k4_20260917/control/status.json`
- 队列日志：`/root/data1/hrfont_k4_20260917/control/queue.log`
- C 日志：`/root/data1/hrfont_k4_20260917/control/K4-C_TRAIN_10000.log`
- C 运行目录：`/root/projects/hrfont/runs/K4-C-K1RECIPE-V2-S3407`
- checkpoint、推理、指标：`/root/data1/hrfont_k4_20260917/`
- 全量完成后的远端页面：`/root/data1/hrfont_k4_20260917/review/index.html`，目前尚未生成。

快照时 data1 余量 47.46 GiB，根盘 8.24 GiB。低于 data1 15 GiB/根盘 2 GiB 或发生训练异常时会停止并保留记录；不删除旧实验。队列 `control/STOP` 在阶段边界停止，当前 run 的 `STOP` 在成功更新边界保存状态后停止。不要重复启动队列；锁会阻止并发执行。

预计三组主训练约 10–12 小时，随后全量推理、指标和 I/O 另计；总运行窗口暂按 20–30 小时估算，以实测进度更新。此估计不是完成承诺，也不表示所有后续阶段已验收完成。
