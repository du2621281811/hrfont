# v0913_clean 同步清单

Git 只传映射和脚本。PNG / ckpt / Es / Ec **不进仓**。

## 他机重建（只读映射）

本机已有 dirty 协议 A 盘即可，不必再从 3090 拷 PNG。

```bash
python3 scripts/verify_v0913_clean_map.py --png-root data/fontdiffuser-p253-t295-s338-cn2west-v2
```

| 要有 | 路径 | 进 Git？ |
|------|------|----------|
| pair 表 + 权重 | `manifests/v0913_clean/` | 是 |
| 合同 | `manifests/v0913_clean.json` | 是 |
| 审查漏斗 | `data/p649_v2a_layers/training_map/` | 是（已在仓） |
| dirty PNG | `data/fontdiffuser-p253-t295-s338-cn2west-v2/` | 否，本机已有 ~0.7G |
| 官方 P1 | `code/official/FontDiffuser/ckpt/` | 否，本机已有 |

## F0-clean 还要不要 3090？

**不要。** F0 在线编 PNG，不读 94G Ec，也不要 `es_local`。

不要再拷：协议 A PNG、dirty F0 的 Es/Ec、TTF。

## F0-clean 训完之后（本机现编，仍不用 3090）

| 产物 | 做什么 | 不要用 |
|------|--------|--------|
| 新 Es | 从 **F0-clean** encoder 编 | dirty `artifacts/f0/es_spatial_f0/` |
| 新 Ec | 从 **F0-clean** encoder 编（~94G） | dirty `artifacts/f0/ec_multiscale_f0/` |
| 新 es_local | F2-RL 用，从 F0-clean Es hook 编 | 3090 上的 dirty local cache |

## 只有要比旧臂数字时才向 3090 要

| id | 要什么 | 何时 |
|----|--------|------|
| `05_eval` | 旧 F1/F2/F2-P/F2-RL 的 40k ckpt | 和 dirty 臂做同协议重算 |
| `es_local` dirty | `artifacts/f0/es_local_f0_block2_pool4/` | **仅**在旧 dirty F0 上补跑 F2-RL |

本 V100 没有旧 F1/F2/F2-P/F2-RL 权重。F2-RL 缺的 local Es 也可以用本机 dirty F0 `best` + PNG 现编，不必等 3090。
