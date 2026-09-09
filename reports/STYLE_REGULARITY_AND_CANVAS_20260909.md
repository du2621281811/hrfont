# 画布、偏方与 F3「太规整」——结论（2026-09-09）

同步给协作者。训练计划不变：**先跑完 F1，再从 F0 开拓扑版 F3b**（own-font Ec，不重跑 F1/F2）。

## 1. 两件事不要混

| 观感 | 主因 | 改画布有用吗 |
|---|---|---|
| 偏方 / 像盖在方章上 | A 协议 96×96 居中 + inner≤84 + Founder 拉丁本身比 Content 脸更宽 | 最多减轻「方印」构图，**救不了风格**；全库重渲 + F0 起重训，现在不要做 |
| 基础风格对、但太平、没脾气 | **8 张汉字 ref 的 Es 3×3 逐像素均值 → 9 个全局 token**；F3 再用同一批 ref8 的 Ec 灌进 up-path，Δ/RSI 继续往中文邻域结构拽 | 几乎无用 |

test16 上 F2@75k / F3@80k 相对 GT 的 ink 宽高比 Δ 均值约 0 / −0.015，不是生成器额外把 GT 压成方块。假名 Target 均值已经 ≈1.0；`I/l/i` 跨字体 CV 最大（0.40–0.53），且目标均值明显宽于 Content（`I`: 0.13→0.32）。表：`reports/samecontent_spatial_variation/`。

## 2. 不改的

- **不重跑 F1/F2**：共用 style 通路仍是 9 个全局 token；这次只换 F3b 的 support 选字表。
- **排队的 F3b 不加 local64**：否则 F2 vs F3b 同时变 support 和 style 容量。
- **不改画布比例、不重渲 A 协议。**
- **不加 cross-font Ec 再均值**：结构会更规整，不是更有风格。

## 3. 建议的下一刀（F1 结束之后）

1. 推理期少平均：固定 checkpoint，style 只用 2–3 张最有特征的 ref，对比 ref8 均值。立刻「有脾气」就坐实是平均在杀风格。
2. 眼检同一格 GT / F0 / F2 / F3：GT 已经很正 → 数据；F0 有纹理、F2/F3 变平 → Δ/support 规整化。
3. 若要补风格分辨率：F2（现 9 token）↔ F3b-support-only ↔ F3b-full（+Es 中层 local64），不要把 local token 接到 F1/F2 共用通路。

## 4. 执行状态（本机）

- 旧 F3b `f3b_joint_crossbank_s3407`（stroke-bank）已 STOP @ **10883**，不续训。
- 新 F3b `f3b_topology_ownfont_s3407`：own-font Ec + topology top-16；注入仍是 SupportAdapter → up-path style attention（不是 RSI）；等 F1 `DONE.json` 后从 F0 开 80k。
- F1 单独跑后步速见 `reports/TRAINING_STATUS_F1_F3B_E12.md`（与 F3b 并发时 ~3 s/it，停 F3b 后 ~1.0–1.4 s/it）。
