# 实验与推理归档

入口：[总追踪表](../EXPERIMENT_TRACKER.md) · [逐run机器记录](RUNS.json) · [I执行规格](../I_EXECUTION_20260915.md) · [历史完整导航](../REPORT_HUB.md)

## I 系列

I0 = `G0b-F0-V0913-BS256-A-S3407/global_step_10000`，仅别名。I1/I2均从I0独立启动。

实时快照：[队列](I/QUEUE.json)。I1/I2结果在完成后归档为 `I/<model>/step########/review.html`，附`ARCHIVE.json`逐文件校验。目录不存在表示尚未完成/导出，不是空白结果。

## H 系列（原始PNG、GT、指标和离线HTML）

| 模型 | 2k | 4k | 6k | 8k | 10k |
|---|---|---|---|---|---|
| H0 | [192图](../h_20260915/H0_step2000/review.html) | [192图](../h_20260915/H0_step4000/review.html) | [192图](../h_20260915/H0_step6000/review.html) | [192图](../h_20260915/H0_step8000/review.html) | [4096图](../h_20260915/H0_step10000/review.html) |
| H3 | [192图](../h_20260915/H3_step2000/review.html) | [192图](../h_20260915/H3_step4000/review.html) | [192图](../h_20260915/H3_step6000/review.html) | [192图](../h_20260915/H3_step8000/review.html) | [4096图](../h_20260915/H3_step10000/review.html) |
| H4 | [192图](../h_20260915/H4_step2000/review.html) | [192图](../h_20260915/H4_step4000/review.html) | [192图](../h_20260915/H4_step6000/review.html) | [192图](../h_20260915/H4_step8000/review.html) | 未到10k，9522停止 |

H0/H3各5个推理阶段，H4四个已完成阶段；合计10496张预测（不含GT与辅助contact sheets）。[收尾结论](../H_CLOSEOUT_20260915.md)。

## G / F / E12 历史入口

- G：[shot面板](../../G_V0913_SHOT.html)，G各变体/TC/CONT/REF的身份和服务器检查点记录见RUNS.json内series=G。
- F：[正式字符板](../f03_test16_strat/index.html)、[向量分支板](../f03_test16_strat/f2vec_eye_board.html)。
- E12-b：[评测模型状态](../e12_b/STATUS.md)，版本定义与生成器系列分开。
- 更多历史协议：[REPORT_HUB](../REPORT_HUB.md)。这些是既有结果的索引，不是本轮重新生成的I基线。

## 同步约定

执行机运行`python scripts/archive_experiments.py`；将`reports/experiments/`及新H面板同步到Git工作树，再提交推送。每次记录时间、parent/config、状态、逐样本协议与文件哈希。生成器权重/优化器状态不进Git，原服务器路径保留。离线看图需要完整git拉取对应PNG；GitHub原生页面不直接执行HTML。
