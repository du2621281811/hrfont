# 方法对比测试网页 — 设计与评测范围（2026-09-09）

面向 PI：一眼对比 **P1 / F0 / F1 / F2 / F3(hist) / F3b / E12** 的效果，而不是分散在多个目录里翻图。

---

## 1. 现在已经能看的中间结果（可立刻上网页）

| 资产 | 位置 | 覆盖方法 | 粒度 | 说明 |
|---|---|---|---|---|
| **主生成协议评测** | `reports/f03_test16_strat/`（HTTP `:8767`） | P1, F0@100k, F3@80k, F2@75k；另有 F2/F3 时间线子集 | test16×47=752（全量）或 256（mid） | L1/SSIM/LPIPS **相对 GT 的诊断指标**；`index.html` 并排浏览；`timeline_f2.html` 看 F2 训练曲线式抽检 |
| **像素对比报告** | `RESULTS_COMPARE_20260907.html` | P1/F0/F3（及当时未完的 F2） | 汇总表 + 分组 | 论文前诊断，非最终 claim |
| **F3 历史 ckpt 看板** | `reports/f3_ckpt_dashboard/` | F3 5k–80k | 逐步抽检 | 看 Support 臂中途形态 |
| **E12 眼测探针** | `reports/e12_eye_probe_v4_s3407/`（HTTP `:8766`） | E12 v4 | 错例三联图 | membership/φ 的失败 case；**不是**生成器对比 |
| **E12 v5.1 数值** | `runs/e12_phi_*` / `e12_membership_*` | φ AUC、membership ROC/PR | 标量 | φ best_auc≈0.90；membership test ROC≈0.937 |
| **训练心跳** | `runs/*/heartbeat.json` + watchdog | F1/F3b 进行中 | step/loss | 不进效果对比主页，进「进度条」侧栏即可 |

**缺口（尚不能公平对比）：**

- **F1**：训练中（~24k/80k），尚无 test16×47 全量 pred
- **F3b**：训练中（~9k/80k），尚无正式 pred；历史 **F3≠F3b**（adapter/`no_grad` 与 bank 协议不同）
- **F2@80k**：DONE 有 ckpt，但 stratified 全量目前记到 **75k**
- **E12 不作生成器主指标**：GT 像素分不进主排行；E12 只进「风格一致性 / 眼测」页

---

## 2. 统一评测范围（所有生成方法必须对齐）

### 2.1 主协议 A（唯一默认对比口径）

与 `f03_test16_strat/PROTOCOL.json` 锁定一致：

| 项 | 值 |
|---|---|
| Split | `test` |
| Fonts | 16（test16） |
| Chars | 47（digit / Lu / Ll+扩展 / 假名 / 注音） |
| n | **752** font×char |
| Seed | **3407**，per-item `set_seed`（方法间配对） |
| Sampler | DPM-Solver++ **20**，CFG **7.5**，order-2 multistep |
| Style | ref8 首个可用，优先「永」 |
| 指标 | L1↓ / SSIM↑ / LPIPS↓ vs GT（**诊断**）；另预留 E12 cos / membership logit（**风格**，不进主榜加权） |

### 2.2 中间抽检协议 B（训练未完时用）

| 项 | 值 |
|---|---|
| 子集 | 固定 **4 fonts × 16 chars = 64** 或沿用现有 mid **256**（须在页面标注） |
| 触发 ckpt | 每 **5k / 10k**（与 `ckpt_interval` 对齐） |
| 用途 | 时间线对比、早停直觉；**不得**与 752 全量数字混排在同一表不写 n |

### 2.3 方法槽位（网页上的「列」）

| 列 ID | 含义 | 当前 |
|---|---|---|
| `P1` | 官方基线 | 有 752 |
| `F0_100k` | 无 RSI 父 | 有 752 |
| `F1_*` | Official RSI | 待 eval（建议先 20k/40k mid，完训 80k 全量） |
| `F2_75000` / `F2_80k` | Delta RSI | 75k 全量有；80k 待补 |
| `F3_80k` | 历史 Joint Support | 有 752（标注 **legacy**） |
| `F3b_*` | own-font + stroke bank | 待 20k mid + 80k 全量 |
| `GT` | 真值缩略图 | refs 已有 |

---

## 3. 网页信息架构（一个门户，三个工作区）

入口建议：`reports/compare_portal/index.html`，根目录静态服务一个端口（如 **8768**），内链现有 `:8766/:8767`。

```
Compare Portal
├── A. 总览 Overview          ← 方法状态条 + 主指标表 + 配对胜负
├── B. 字形对比 Glyph Lab     ← 选 font/char，多方法并排 + 差分
├── C. 训练时间线 Timeline    ← 同方法不同 step；F2/F3 已有，扩 F1/F3b
├── D. 风格 / E12             ← AUC、错例墙、与生成器解耦
└── E. 协议与 caveat          ← PROTOCOL 原文 + 「GT 像素≠论文风格 claim」
```

### A. 总览 Overview（默认首页）

**呈现：**

1. **方法卡片行**：状态（DONE / running@step / missing）、可点进 Glyph Lab。  
2. **主表（仅 n=752 或明确标注的子集）**：列=方法，行=L1/SSIM/LPIPS overall + 按 script（Lu / Ll / Other）。  
3. **配对面板**：固定对照  
   - F1 vs F0（Official RSI）  
   - F2 vs F1（Delta）  
   - F3b vs F2（Support，主科学对比）  
   - F3_legacy vs F3b（仅历史参考，灰显）  
   显示：均值差、胜/负格子数、字体级 9/7 类条形（沿用现有分析脚本口径）。  
4. **缺失格**：无 pred 的方法显示「排队评测」而不是 0。

### B. 字形对比 Glyph Lab（核心体验）

**控件：** font 下拉、char 下拉、方法多选（默认 `GT | F0 | F2_75k | F3_80k`，F1/F3b 有了再勾）。

**呈现：**

| 模式 | 布局 | 用途 |
|---|---|---|
| **Strip** | 一行：GT + 各方法 96px 图 + 每格脚注 L1/SSIM | 快速扫 |
| **Diff** | 选「基准方法」A，对其余做 `|pred−A|` 热力小图 | 看谁更像谁 |
| **Hard cases** | 按「方法X−方法Y」有利差排序的 top-K 差例 | 找失败字体/字类 |
| **Script filter** | Lu / Ll / digit / kana / bopomofo | 对齐已知「小写涨、大写跌」叙事 |

数据：`browse_index.json` + `preds/<method>/...` + `metrics_items.json`（已有）。

### C. 训练时间线 Timeline

**范围：** 同方法 `5k…80k` 抽检（现成 F2/F3 mid 256；F1/F3b 按协议 B 补）。

**呈现：** 横轴 step，纵轴可选 L1 或 SSIM；下方固定一个 font×char 的「电影条」多帧。

已有：`timeline_f2.html` → 门户内 iframe/链入，并复制模板给 F1/F3b。

### D. 风格 / E12

**与生成器分栏，避免和 L1 表混读。**

| 块 | 内容 |
|---|---|
| φ | val AUC 曲线摘要、best checkpoint |
| Membership | test ROC/PR/Brier/ECE + temperature |
| Eye probe | 链到 `:8766` standalone；错例 collage |
| （可选）生成器侧 E12 score | 仅当对 pred 跑完同一 probe 后上线 |

### E. 协议页

只读展示 `PROTOCOL.json`、caveat、F3 vs F3b 不可混比声明、seed 语义（同噪声配对 ≠ 多样本方差）。

---

## 4. 评测生产流水线（网页背后要跑的活）

优先级（不打断正在训的 F1/F3b 全卡时，用空闲 GPU 或训完后）：

| 优先级 | 任务 | 产出落到 |
|---|---|---|
| P0 | 门户静态页 + 聚合 `manifest.json`（方法→pred 路径→n→step） | `compare_portal/` |
| P1 | F2@80k 全量 752（补齐与 F0/F3 对齐） | `preds/F2_80k` |
| P2 | F1@20k 或 @25k 协议 B mid；有 `global_step_*` 即可 | `preds/F1_20000` |
| P3 | F3b@20k mid（PI：先看 20k） | `preds/F3b_20000` |
| P4 | F1@80k、F3b@80k 全量 752 | 主表换列 |
| P5 | 可选：对全量 pred 跑 E12 membership score 分布图 | D 区 |

脚本复用：`scripts/eval_f03_test16_strat.py`（已支持多方法 pred 目录）。

---

## 5. UX 细则（方便「对比不同方法」）

1. **同一字符永远同一随机种子图**：URL 带 `?font=&char=` 可分享。  
2. **默认对照预设**按钮：`RSI`（F0|F1）、`Delta`（F1|F2）、`Support`（F2|F3b）、`Legacy`（F3|F3b）。  
3. **数字与图同屏**：点表中单元格 → Glyph Lab 跳到该 script 的最差分例。  
4. **色盲安全**：胜负用位置+符号，不只靠红绿。  
5. **标注 n 与 step**：凡 mid 子集标题强制 `n=256 @step=`。  
6. **不把 E12 分数和 L1 加权成「总分」**（除非 PI 另批）。

---

## 6. 与现有服务的关系

| 端口 | 内容 | 门户角色 |
|---|---|---|
| **8768（新建）** | Compare Portal 首页 | 唯一入口 |
| 8767 | f03_test16_strat | Glyph Lab / 旧浏览的后端静态根（可同机 reverse 或相对链） |
| 8766 | E12 eye probe | D 区深链 |

---

## 7. 验收标准（网页「好用」）

- [ ] 30 秒内能回答：F0 vs F3@80k 在 overall LPIPS 上谁好、差多少  
- [ ] 1 分钟内能打开同一 `font×char` 的 GT|F0|F2|F3 四图  
- [ ] 能区分「全量 752」与「mid 256」且不会误读  
- [ ] F1/F3b 无数据时显示排队，不出现假 0  
- [ ] F3_legacy 与 F3b 有视觉警告条  

---

## 8. 建议实施顺序（短）

1. 上线门户壳 + 链现有 8767/8766 + 嵌入当前 `metrics_summary` 表（**今天可做**）  
2. 加强 Glyph Lab：多方法多选 + Diff（基于现有 preds）  
3. 空闲 GPU 补 F2@80k、F1 mid、F3b@20k  
4. 完训后刷主表为 F1/F2/F3b 终局对比  

本设计不改变任何训练超参；只约束**评测与呈现口径**。
