# R0 Ink Gate 过滤阈值（已 PI 冻结）

> 状态：**已冻结（2026-09-04）**。  
> 依据：协议 A 全量 ink 扫描；审查产物见 `data/cn2west_v2_abc_review/proto_A_ink/`。

## 1. 主指标

- **字体门**：`mean_ink_ratio = mean(ink_bbox_h × ink_bbox_w / 96²)`
- **次级**：`mean_ink_pixel_ratio`（不对主门）
- **T**：250

## 2. 冻结规则

| 规则 | 结论 |
|------|------|
| `mean_ink_ratio < 0.20` | 字体级 drop 候选 |
| `FZXianZTJW` | **drop**（mean≈7.6%，原 train） |
| 其余仅 B-screen 对照候选 | **pass**（框面积均 ≥24%） |
| 训练样本级 `glyph_bbox < 0.20` | **本轮不上** |

校准文件：`data/cn2west_v2_abc_review/proto_A_ink/ink_gate_calibration.json`  
决策日志：`…/ink_review_decisions.jsonl`

## 3. Split（同期冻结）

PI 覆盖原合作者 237/16/8：

| 项 | 值 |
|----|----|
| 活跃池 | **260**（261 − FZXianZTJW） |
| train / val / test | **228 / 16 / 16** |
| 真源 | `manifests/split_v3_228_16_16.json` |
| 策略 | 保留原 val16 + 原 test8；seed=3407 自 train 再抽 8 字进 test |
| 盘上 | A/B/C/D/F/H 的 Target/Style 已搬家；drop 字体在各盘 `excluded/` |

目标正式 ID：`fontdiffuser-p260-t295-s338-cn2west-v2a-r1-<manifest8>`（目录名可暂仍为 p253）。

## 4. 已勾选（PI）

- [x] 字体门：mean_bbox &lt; 0.20
- [x] `FZXianZTJW` → drop
- [x] 训练样本级面积硬阈值：本轮不上
- [x] 重划为 228/16/16（val=test=16）
