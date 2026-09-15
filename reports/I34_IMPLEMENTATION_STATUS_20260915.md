# I3/I4/E12-c 实施状态

更新：2026-09-16。本页替代上一版“组件已测、入口未实现”的状态。全部预检及队列dry-run已通过，正式后台队列已派发I3；队列PID3457564，torchrun PID3457570。实时进展以 [派发回执](I34_DISPATCH_20260916.json) 和执行机小摘要为准。

## 已完成

- 独立 I3/I4 trainer、实际144-token墨迹监督、输出细节loss、冻结校准难例采样，以及两协议推理/归档入口均已实现并部署至 `/root/projects/hrfont_i34_20260915.bJikwQ`；旧I快照未修改。
- 223个有效训练字体、131,803张native96 PNG全量解码/SHA检查；541个字体—语种组分桶与36组面板实际复核完成，清洗表不变。
- I3真实8卡80步固定batch loss约0.165→0.108；80→84恢复成功，无重放，全部rank AMP skips=0、DDP spread=0，reader/encoder/router/token projection/ink readout梯度非零。
- I4真实8卡10步中文辅助路径通过，主/辅loss有限，全部rank无跳步或参数分歧。
- 7项I34组件测试；E12-c集合/掩码/多正例/BN及AUC测试已落地。预检发现执行环境无sklearn，已改为等价average-rank AUC并补并列分数测试，不修改评测定义。
- E12-c按178/22/23字体实例划分，主生成val/test排除；已知家族接口保留，本次未知家族明确记录，不阻塞已批准实验。
- I2 matched4888与I0 formal4096完成并取回本地；I2 6395登记文件SHA通过，I0 4352预测/GT PNG解码、4372文件登记。
- PI授权新checkpoint存储于 `/root/data1/hrfont_i34_20260915`，仅链接新产物，不移动/删除旧模型；根盘与checkpoint盘均保留10GiB安全线。

## 已通过的最终准入

E12-c A20+B20短流程已完成，全部12个内部验证单元可计算，编码器在B阶段保持不变。真实模型DPM推理及重复ref检查通过，1张与8张重复ref的TC最大误差0.000488，符合FP16数值范围。绑定代码/数据SHA的 [PREFLIGHT_PASSED.json](experiments/I/preflight_i34/PREFLIGHT_PASSED.json) 已归档。I3固定batch前后10步中位loss为0.16437→0.10880；这只证明可训练，不是正式效果结果。

队列默认dry-run已实机通过；正式运行不使用预检权重。发现8个容器不可见的现存GPU上下文（各约1.8GiB、利用率0%），准入保留它们，拒绝新PID并要求8卡各至少12GiB可用、利用率不高于5%；未终止任何现存进程。代码及旧结果已push `c218b7484`，最终准入已push `1edf2f2be`。每小时巡检已更新至新队列，不重复启动任务。

详细实现、路径、命令和恢复约束见 [执行手册](I34_EXECUTION_RUNBOOK_20260916.md)。正式执行顺序仍是 I3 → E12-c → I4；两生成器各10k，不继承预检或I1/I2权重。论文及Overleaf本次未修改。
