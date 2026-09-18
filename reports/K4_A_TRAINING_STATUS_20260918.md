# K4-A 训练状态归档

更新时间：2026-09-18（+08:00）

## 结论

K4-A 当前没有完成 10k 训练，也没有进入推理。当前主 run
`K4-A-K1FT-0917-WEIGHT-S3407` 在 step 6400 失败，最后可恢复 checkpoint
为 step 6000。失败原因是 rank 6 检测到 `Nonfinite loss`；随后 NCCL watchdog
因分布式进程未推进而终止任务。该结果应标记为 **FAILED / INCOMPLETE**，不能当作
K4-A 的最终模型或论文指标。

## 当前进度

| 项目 | 状态 |
|---|---|
| K4-A weight-only 主 run | step 6400 / 10000，FAILED |
| 最后可恢复 checkpoint | step 6000 |
| K4-A FAMILY 相关 run | step 714，safely stopped |
| K4-A 固定协议推理 | 未开始 |
| K4-C | 未开始 |
| K4-B | 未开始 |
| GPU | 当前空闲 |

主 run 的最后训练 telemetry：`loss=0.0368855`、`detail_loss=0.0486300`、
`D_change=0.0902226`、`update_seconds=0.7921`、峰值显存约 `8395 MiB`，
AMP skip 为 `0`。这些数值只描述失败前的训练状态，不构成效果结论。

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

详细哈希和相关 FAMILY run 记录见同目录的 `status.json`。
