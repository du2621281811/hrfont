# K4-A 训练状态归档

更新时间：2026-09-18（+08:00）

## 结论

K4-A 当前尚未完成 20k 训练。主 run
`K4-A-K1FT-0917-WEIGHT-S3407` 的首次尝试在 step 6400 失败，最后可恢复
checkpoint 为 step 6000；失败原因是 rank 6 检测到 `Nonfinite loss`，随后 NCCL
watchdog 因分布式进程未推进而终止任务。

09:56 已启动数值故障恢复流程，从 step 6000 续训。当前 8 卡已恢复到
step 14100 / 20000，状态为 **RUNNING / INCOMPLETE**；不能把已有结果当作
K4-A 的最终模型或论文指标。

## 当前进度

| 项目 | 状态 |
|---|---|
| K4-A weight-only 首次尝试 | step 6400 / 10000，FAILED |
| 最后可恢复 checkpoint | step 6000 |
| K4-A 数值恢复 | 从 step 6000 续训，当前 step 14100 / 20000，RUNNING |
| K4-A FAMILY 相关 run | step 714，safely stopped |
| K4-A 固定协议推理 | 2k / 4k / 6k / 8k / 10k / 12k / 14k 已完成 |
| K4-C | 等待 K4-A 恢复完成 |
| K4-B | 等待 K4-A 恢复完成 |
| GPU | 8 卡已占用，恢复进程运行中 |

恢复后的最新 telemetry：`loss=0.0255622`、`detail_loss=0.0416674`、
`D_change=0.0635021`、`update_seconds=0.9482`、峰值显存约 `7848 MiB`，
AMP skip 为 `0`。这些数值只描述训练状态，不构成效果结论。

## VAL 推理归档

当前已有同一 K4-A run 的 7 组中间 VAL 推理，均已完整同步到 Git：

- `step00002000`：192 样本，4 shots，192 张图
- `step00004000`：192 样本，4 shots，192 张图
- `step00006000`：192 样本，4 shots，192 张图
- `step00008000`：192 样本，4 shots，192 张图
- `step00010000`：192 样本，4 shots，192 张图
- `step00012000`：192 样本，4 shots，192 张图
- `step00014000`：192 样本，4 shots，192 张图

每组包含预测图、`metrics.json`、`DONE.json`、逐 rank JSON/JSONL 和进度文件。
这些是中间 checkpoint 的 VAL 结果，不是 20k 最终评测；20k 推理仍待训练完成。

## 可复核位置

- 主 run：`/root/projects/hrfont/runs/K4-A-K1FT-0917-WEIGHT-S3407`
- 主 checkpoint：`/root/data1/hrfont_k4_weight_20260918/checkpoints/K4-A-K1FT-0917-WEIGHT-S3407/state_step_6000`
- 训练日志：`/root/data1/hrfont_k4_weight_20260918/control/K4-A_TRAIN_10000.log`
- Git 元数据：`reports/experiments/K/K4-A-K1FT-0917-WEIGHT-S3407/status.json`

Git 只登记配置、代码/数据身份、日志哈希、进度和 checkpoint 位置；训练权重、
optimizer state 和大体积图片继续保留在执行机，不复制进仓库。

## 复现身份

- identity commit：`779de670cf460f44e6b3e938dd8af22b27178755`
- donor：`v0917`
- seed：`3407`
- global batch：`64`（8 GPU）
- parent：`K1-ORIGINAL-V0917-S3407/global_step_10000/ema.pth`
- donor family guard：`K4-weight-only-v1`，violations `0`

恢复预检已通过：8 ranks、同 RNG、finite backward、故障同步和 healthy path
均通过；恢复策略不改变 recipe、数据和 parent。详细哈希、恢复身份和相关 FAMILY
run 记录见同目录的 `status.json`。
