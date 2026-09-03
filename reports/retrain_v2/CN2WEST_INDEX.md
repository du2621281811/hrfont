# 中→西实验索引（CN→Latin/Kana）

**任务：** 用目标字体的**中文参考**学风格 → 生成**拉丁/假名** → 与同字体 GT 比 **L1↓ / SSIM↑**  
**测集：** Demo-8（8 字体 holdout）· **P1** 拉丁数字 · **P2** 假名  
**网页入口：** [`CN2WEST_INDEX.html`](CN2WEST_INDEX.html)

> 08-14 后主线已切 **中→中字库扩写**；本页只收录跨语阶段实验。中→中见 [`WIKI_EXPERIMENT_SUMMARY.md`](WIKI_EXPERIMENT_SUMMARY.md) / [`index.html`](index.html)。

---

## 0. 文字结论（先读）

| 文档 | 内容 |
|------|------|
| [`RETRAIN_V2_SUMMARY.md`](RETRAIN_V2_SUMMARY.md) | **主表 + 八方法数字** |
| [`PROTOCOL.md`](PROTOCOL.md) | 协议 / 输入差异 / 指标口径 |
| [`STROKE_EXPERIMENTS_ANALYSIS.md`](STROKE_EXPERIMENTS_ANALYSIS.md) | Stroke 两线终裁 NO-GO |

**一句话：** 横比 **GAR ft_p253 最优**；FD cnstyle 可用；LF 崩；stroke 两刀均负向。

---

## 1. 主实验 · 八方法横比（✅ 已结）

| 方法 | 类 | P1 L1 | P1 SSIM | 裁决 |
|------|----|------:|--------:|------|
| **GAR ft_p253** | 官方+ft | **0.070** | **0.670** | **主推** |
| MX ft | 官方+ft | 0.087 | 0.633 | 次强 |
| FCA/GAS/MF | 自训 GAN | ~0.078 | ~0.62 | 被 GAR 超 |
| **FD cnstyle@25k** | 官方+ft | 0.083 | 0.607 | 可用 |
| LF | 自训 | 0.138 | 0.298 | 失败 |

**看图：**

- [`compare_report/`](compare_report/) — 指标表 + 每字 8 方法并排
- [`gallery.html`](gallery.html) — 全量交互图库
- [`wiki_pack/explorer.html`](wiki_pack/explorer.html) — IO 故事板（需数据路径可访问）

---

## 2. GAR 训练阶梯（✅）

| 阶段 | P1 L1/SSIM | 索引 |
|------|------------|------|
| 官方零样本 | 0.127 / 0.405 | 见 `gar_p253` 对照表 |
| P0 · 42 字 LoRA | 0.088 / 0.569 | [`gar_p0_train/`](gar_p0_train/) |
| **ft_p253 · 253 字** | **0.070 / 0.670** | [`gar_p253/`](gar_p253/) · [`gar_p253_train/`](gar_p253_train/) |

---

## 3. FontDiffuser CN-style（✅）

| 跑 | P1 | 索引 |
|----|-----|------|
| cnstyle@42 **25k**（主表） | 0.083 / 0.607 | [`fd_epoch_gallery/`](fd_epoch_gallery/) |
| p253 cnstyle@12k（放大） | 0.081 / 0.616 | 同上 · `FD_CNSTYLE_RESULTS.json` |

---

## 4. Stroke 补风格（✅ NO-GO）

| 实验 | Δ（lite P1 SSIM） | 索引 |
|------|------------------:|------|
| GAR stroke aux λ=0.05 | **−0.030** | [`gar_stroke_preview/gallery/`](gar_stroke_preview/gallery/) · [`gar_stroke_live.html`](gar_stroke_live.html) |
| FD stroke-SCR | **−0.020** | [`stroke_verdict/`](stroke_verdict/) · [`fd_stroke_scr_preview/gallery/`](fd_stroke_scr_preview/gallery/) |

协议：[`STROKE_FAST_PILOT.md`](STROKE_FAST_PILOT.md) · [`FD_STROKE_SCR_PLAN.md`](FD_STROKE_SCR_PLAN.md)

---

## 5. 渲染 / 协议探针（GAR 跨语侧）

| 探针 | 问什么 | 索引 |
|------|--------|------|
| 官方 style 复现 | 官方 PNG vs 自渲 | [`gar_official_style_reproduce/`](gar_official_style_reproduce/) |
| 固定48 vs 搜参64 | 产品渲法伤不伤 | [`gar_toy_a_fixed48/`](gar_toy_a_fixed48/) |
| Demo-8 fixed48 vs CS | A=ft0 渲法错配 | [`designer_cn2cn_demo8_a_cs_render/`](designer_cn2cn_demo8_a_cs_render/) |
| Content 字体 sweep | 哪套 content 更稳 | [`gar_content_font_sweep/`](gar_content_font_sweep/) · [`fd_content_font_sweep/`](fd_content_font_sweep/) |
| 难度 × content | 难在风格还是 content | [`designer_probe_easy/`](designer_probe_easy/) |

**结论摘要：** 训推渲**不一致**伤很大（fixed48 L1 ~0.09 vs CS ~0.13）；干净「一致后再比 R1 vs F48」见中→中 [`render_protocol_ablation/`](render_protocol_ablation/)（非本阶段）。

---

## 6. 未做 / 排除

- MX/LF「CN style + 中性 content」协议重训 — 未完成  
- FCA-Regression、主表 CN→CN — 协议排除  

---

## 7. 与后阶段关系

| 后阶段 | 关系 |
|--------|------|
| E9/E10/fix12/P1b | **中→中**，不复用本表绝对值 |
| Native256 | FD 分辨率迁移，**不是**跨语主实验 |
| `official_paper_metrics/` | 混有中→中 ckpt，**非**跨语主索引 |
