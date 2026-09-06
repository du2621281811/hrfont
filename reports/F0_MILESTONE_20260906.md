# F0 milestone 选择（2026-09-06）

规则预注册于 [`DECISION_F123_20260906.md`](./DECISION_F123_20260906.md) §2，**先写规则后看数**。

## 选择

| | step | val16 loss (n=4720, seed=3407, offset=0) |
|---|---:|---:|
| **选中** | **100000** | **0.031089** |
| 100k 端点 | 100000 | 0.031089（与选中相同） |
| E1 对照 best | 98000 | 0.029764 |
| E1 对照 100k | 100000 | 0.029787 |

曲线：`reports/training_logs/F0-RSIFREE-FT-A-S3407/val_loss.png`。全程近似单调下降（55k、75k 有 3e-5 量级回升），最小点就是终点，没有早期过拟合可挑。

相对 E1@100k，F0 高 **0.00130**。这是去掉 RSI 后在同一协议 A / train228 / 100k / seed 3407 上的代价，后面 F1 要把官方 RSI 加回来、F2 把 Δ 加回来，都从这个 100k 出发。

`runs/F0-RSIFREE-FT-A-S3407/best` → `global_step_100000`。

## 扫描脚本

`python scripts/scan_f0_val_loss.py --device cuda:2`

与 E1 dashboard 的 `eval_val_loss_one` 同损失、同 val 集、同 seed；唯一差别是 `offset_coefficient=0`（F0 训练时就是 0）和 `StyleUpBlockNoRSI`。
