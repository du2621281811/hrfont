# G 系列进度快照 — 2026-09-14
- 生成时间：2026-09-14T14:44:31.493267+08:00
- 机器：V100 Docker / hrfont

## 8 卡队列 `run_g_tc_8gpu_queue.sh`
| 阶段 | 状态 | 备注 |
|------|------|------|
| `build_tc_v2_cache` | DONE | artifacts/tc_v2_cache · 148432 entries |
| `train_tc_v2_head` | DONE | artifacts/tc_v2_head/tc_head.pth |
| `G-RL-pilot-8gpu-V0913-A-S3407` | DONE | 5k · best@2500 · val_loss@2500=0.002071 |
| `G-TC-G2-8gpu-V0913-A-S3407` | RUNNING | step 3400/5000 · loss=0.0194123312830925 · ts=2026-09-14T14:44:22 |
| `G-TC-G2RL-8gpu-V0913-A-S3407` | QUEUED | 接 TC-G2 后自动启动 |

**ETA（按 ~1.55 it/s）：** TC-G2 约再 17 min → TC-G2RL 约 54 min → **队列收完约 71 min（~1.2h）。**

## 已完成主臂（权重在 `runs/`，不进 git）
- G0b@10k / G0c@20k · G1/G2/G2-RL@10k · 1GPU pilot@5k
- 缓存：`artifacts/g0/*`（不进 git）

## 测试集推理看板（本仓可分析）
- 板：`reports/g_v0913_shot/index.html`（枢纽 `/g_shot/`）
- 协议：`reports/g_v0913_shot/PROTOCOL.json`
- 指标：`reports/g_v0913_shot/metrics.json`（L1/SSIM；LPIPS 空）
- 预测：`reports/g_v0913_shot/preds/` + `gt/` + `refs/`
- 决策备忘：`reports/G_QUEUE_DECISIONS_20260914.md` · `reports/G_STYLE_COMPLETION_PLAN_20260914.md`

## 说明
- 8 卡 TC/G2RL **尚未**写入本看板；队列结束后需再跑评测扩展。
- 磁盘根分区约 96% 满；权重仍仅本地 `runs/`。
