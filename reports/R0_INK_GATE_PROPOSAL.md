# R0 Ink Gate 过滤阈值提案（待 PI 确认）

> 状态：**草案，未冻结**。确认前不得写入训练 config / 正式 p261 manifest。  
> 依据数据：协议 A 全量 261 字体 × 633 字符（165,213 PNG），`T=250`。  
> 产物：`data/cn2west_v2_abc_review/proto_A_ink/ink_ratio_rank.{json,csv}` · 审查页 `review.html`

## 1. 主指标

按合作者 `RENDER_PROTOCOL_ANALYSIS.md`：

- **主排序 / 字体门**：`mean_ink_ratio = mean(ink_bbox_h × ink_bbox_w / 96²)`（框面积占比）
- **次级**：`mean_ink_pixel_ratio = mean(|S|/96²)`（像素填充率；不对主门）
- **探针**：全量 633；HTML 快速首屏另用代表字

## 2. 建议冻结的字体级审查阈值（本轮用户指定）

| 规则 | 作用 | 当前命中 |
|------|------|----------|
| **`mean_ink_ratio < 0.20`** | 自动进入审查候选；默认建议 `drop` | **1 / 261**：`FZXianZTJW`（train，mean≈7.6%，median≈5.0%） |
| `touch_edge` | 审查候选；建议 `rerender` | 见 rank JSON summary |
| `n_empty > 0` | 审查候选；建议 `rerender`/`drop` | 见 rank JSON summary |
| 旧 B-screen `drop`/`review` | **仅对照**，不单独定生死 | 2 drop + 12 review |

说明：

- 绝对阈值 `0.20` 与合作者「初始人工候选：calibration P5 **或** 绝对值 &lt;0.20」对齐；本轮以**绝对值**为主提案。
- train237 上字体 `mean_ink_ratio` 的 **calibration P5 ≈ 0.31**（会扩到约 13 个字体）。若 PI 希望更保守召回，可改为「&lt;P5 **或** &lt;0.20」并重跑候选列表。
- **直接自动剔除**起点仍建议 `<0.12` 且校准零误剔才启用；当前唯一 &lt;0.20 的字体 mean≈0.076，落在可讨论自动 drop 区间，但仍须人工确认。

## 3. 训练样本级过滤（勿与字体门混淆）

执行规格提醒：纯面积阈值会误伤合法细字（`i` / `l` / `1` / `一` 等）。

**本提案不把 `glyph_bbox < 0.20` 直接冻成训练滤除。** 训练滤除待定选项（需另批）：

1. `ink_bbox_height/96` 下限 + 窄字豁免；或  
2. 仅对「已 pass 字体」做空图/触边剔除，不做面积硬切。

全局字形框面积分布参考（A 盘）：median≈48%，**p5≈19.9%** —— 若对**每个字形**砍 &lt;20%，会切掉约 5% 合法样本，不可取。

## 4. 审查工作流（R0）

1. 打开 `http://127.0.0.1:8777/cn2west_v2_abc_review/proto_A_ink/review.html`
2. 默认筛「审查候选」；对 `FZXianZTJW` 等做 `pass` / `drop` / `rerender`
3. 导出 `ink_review_decisions.jsonl` → 追加到仓库同名文件（append-only）
4. 字体/视觉负责人复核全部 drop/rerender + 随机 10% pass
5. 冻结后写 `ink_gate_calibration.json`（阈值、T、charset hash、calibration stems、审查 SHA）并登记 provenance

## 5. 请 PI 勾选

- [ ] 同意字体门：**mean_bbox &lt; 0.20 → 审查候选（建议 drop）**
- [ ] 改用：mean_bbox &lt; train P5（≈0.31）或「P5 ∪ 0.20」
- [ ] 同意 `FZXianZTJW` 预审建议：`drop`（或改为 `rerender` / `pass`）
- [ ] 训练样本级面积硬阈值：本轮 **不上** / 另案

确认后执行：回写 decisions → 更新 p261 manifest `excluded_fonts[]` → 新 dataset ID（rerender 不得覆写）。
