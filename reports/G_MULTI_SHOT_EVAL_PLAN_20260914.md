# G 多 split / 多 shot 评测计划 — 2026-09-14

更新：2026-09-14T15:49:51.944320+08:00

## 已确认（用户）

- **Ref8：** `山水云月金石心文`（弃用旧组 `永和书风骨韵天地`）
- **Shot 子集（均取自同一 Ref8，便于对比）：**
  - 1-shot → `山`
  - 2-shot → `山水`
  - 4-shot → `山水云月`
  - 8-shot → `山水云月金石心文`
- Es 与 Δ 同一 episode / 同一组 Ref。
- **Content：** 全量 **stratified47**（不做只跑 CURATED 裁剪）。
- **加速：** 单卡 **batch 推理**（目标 bs 8，显存允许再试 16）；协议仍 DPM++20 / CFG7.5 / seed3407。

## 字体范围

- test：split v3 **全部 16**
- train5：`FZPTYJW`, `FZDuHJW_Cu`, `FZShuLTJW-H`, `FZYiMSJW-T`, `FZFeiSJW-EL`
- val5：`FZYouHK_511M`, `FZLTHProGBK_H`, `FZJunYTJW-H`, `FZBangSKKXJW`, `FZKANGJW`
- 详情：`reports/g_v0913_shot/ref8_proposal/PROPOSAL.json`

## 模型臂（推理）

- G0b / G0c：仅 1-shot
- G1 / G2 / G2-RL / pilot(1GPU) / pilot-8gpu / TC-G2 / TC-G2RL：1/2/4/8-shot
- 缺臂：pilot-8gpu、TC-G2、TC-G2RL（训练收尾后补齐）

## 工作量与 ETA（batch 后）

- 约 **36,660** 张（26 字体 × 47 字 × 30 臂）
- bs=1 基线 ~2.3 h → **bs≈8 后估 ~45–50 min 纯推理**（乐观 ~25–30 min）
- 另加：等 TC-G2RL 收尾 + 实现 batch + 组板 → **整轮约 1–1.5 h**（相对确认时刻；训练进行中时以 heartbeat 为准）

## 设计备注（解读用）

- TC-v2 注入：**只给 up-path global9 加残差，不动 local**。在 G2-RL 上可能与 L 信号不完全对齐；属计划内保守接法，视觉对比时计入条件。
- 8 卡训练配方相对纸面 pilot：global batch 64、双 TC（G2+G2RL）、缺 G-base-continue 锚点、ckpt 间隔 2.5k。

## 预览

- Ref8 对照页：`reports/g_v0913_shot/ref8_proposal/index.html`（枢纽 `/g_shot/ref8_proposal/`）
