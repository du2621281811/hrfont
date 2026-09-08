# E12 v5.1 实验计划与执行规格

日期：2026-09-08 ｜ 状态：PI 已 review v5.1 计划，本文为执行规格（Cursor/合作者可照此实现）
前置文档：`reports/E12_OPTIMIZATION_PLAN_20260907.md`（v1+v2 修订）、`reports/DESIGN_E12_DELTA_20260907.md`（D-E1..6）
代码位置：**唯一** `scripts/eval_framework/`。指标不得散落到 variant 或旧脚本。

## 0. 冻结项（不可改）

- **单 seed**（PI 2026-09-07）：一个 encoder + 全池 CV 测 T2。审稿需要时再补第二 seed 报方差。
- 门限：**T2 主 AUC ≥0.90、稳定性下限 ≥0.85**；T3 ≥0.90；T1/T4 同前（T1=0.644 口径的跨脚本一致性、T4=wrong-ref 显著下降）。**门限不事后放宽**。
- 目标定义：E12 = 「生成文字在参考图语境下是否**合理**」的独立评测组件。合理性操作化 = membership 概率（语境条件判别），不是像素接近。
- 独立性：E12 训练只接触外部字型池，**永不接触 A/260 字体库、方法输出、Es/Ec**。
- 主判定输入只有 (ref 集, 候选图)，**GT 不进主判定**（防退化为像素保真）。GT 仅训练增强 + 校准集可选融合。

## 1. 数据扩展（D1）

- 外部字型池：26 → **≥60 个 lineage 去重字型**（复用 `scripts/e12_fetch_external_fonts.py` 的拉取+渲染管线；Google Fonts + OFL/免费商用 CJK 并继续扩展）。
- **渲染字符扩到 target295 全集**（`manifests/charset_cn2west_v2_planned.json`）：10 digits + 52 letters + **27 latin_ext（àáèéêíòóùúüāēěīńňōūǎǐǒǔǖǘǚǜ）** + 83 hiragana + 86 katakana + 37 bopomofo。渲染协议不变（96×96、L 模式、per_font_height_fit、margin 6、无 resize）。
- **字形覆盖矩阵**：每字型 × target295 逐字查字形存在性；覆盖不足的字符子集按池降级报告（Latin/digits/latin_ext 全池判；kana/bopomofo 用有覆盖的子池），**不静默丢字**。
- ref 语境字符：CN 字（沿用现有 8 字 + 可扩展），与主实验 ref 来源对齐。
- cache：`artifacts/e12/external_font_cache/` + SHA 锁定 + `fonts_manifest.json` 升级（覆盖矩阵入 manifest）。

## 2. 训练对协议（D2）

- 样本 = (ref 子集 R, 候选字形 P, 可选 GT G)。R 从族内随机抽 **k∈{2..全池} 变基数**（组件全量用数据，PI 拍板）。
- 正例 = 同族任意字形（跨字符/跨语系/**含 digits 与 latin_ext**）；负例 = 异 lineage（lineage 去重）。
- GT = 同族另一字形：训练时作额外正锚 + 一致性正则；不进主判定。
- 评测/自测：**leave-one-family-out CV**（全池轮转），macro-family AUC + cluster bootstrap CI。

## 3. 主干筛查（D3）

四路 probe，同一 T2 CV 口径判赢家（每路 ~15min GPU/MPS）：

| 路 | 配置 | 来源 |
|---|---|---|
| A | ResNet18 从零训（现主线） | scripts/eval_framework/models.py |
| B | frozen CLIP-ViT-B/16 + linear probe | — |
| C | frozen DINOv2 + linear probe | — |
| D | frozen CoAtNet-font（5330/14603 字体类预训练）+ probe | external_eval/similar_recommend 权重（GitHub Release asset） |

**约束**：换骨干必须重过 T1-T4 门；D 路引入前验证其训练字库与 A/260 零重叠 + 权重许可声明（公司内部代码）。

## 4. 模型架构（D4）

- 共享主干 f(·)：96×96 → z∈R^d。
- ref 聚合：DeepSets（mean / attention pool，以 CV 为准）。
- 打分头：s(P|R) = z_Pᵀ W z_R / τ → BCE membership 头 → temperature scaling（val CV 校准）。
- 输出三件套同源：s(P|R) 合理性分（主判定）、SC-R/SC-Gap 嵌入距离（方法对比）、校准集 GT-辅助分 s_final = w·s(P|R)+(1−w)·sim(z_P,z_G)，w 校准集冻结。

## 5. 训练配置（D5）

- Loss：L = L_nce（语境条件多正例 SupCon，bilinear score，τ=0.07）+ λ·L_mem（membership BCE）+ GT 一致性正则（可选）。
- 超参沿用：ResNet18/AdamW lr3e-4/wd.01/bs64/50ep/cosine+5ep warmup；membership lr1e-4/bs32/30ep+温度校准；device:auto（Mac MPS 可训）。
- 时长预算：1-3 h/模型；总预算（数据+筛查+训练+自测）≤1 GPU·天 或 1 Mac 夜。

## 6. 自测门（D6，全过才准进主结论）

- **T1 跨脚本一致性**：CN-ref → Latin 候选 vs CN-ref → CN 候选的判定一致性（口径沿用 v3/v4）。
- **T2 = 全池 leave-one-family-out CV**：macro-family AUC ≥0.90，cluster bootstrap CI 下界 ≥0.85。**这是 E12 过门的主判决**。
- **T3 membership 校准**：同族/异族二分类 AUC ≥0.90 + temperature scaling 校准曲线。
- **T4 wrong-ref**：错误 ref 语境下合理性分显著下降。
- v4 基线参考：T2 0.791/0.795/0.725（gate_failed）、T3 0.977。修复方向 = family-aware multi-positive SupCon + digits/latin_ext 入训 + 全池 CV（v2 已批准）。

## 7. 集成与口径（D7）

- 主口径：**生成器实际收到的 8-ref 条件**（`永和书风骨韵天地`）；可选全池语境作为上参照探针（单独报告）。
- 锚定趋势：异族负对照（机会线）、GT 正对照（上参照）同框报告；方法间配对 + bootstrap CI。
- **无判定阈值**；绝对判定权归人评（用户 review + 设计师 2AFC 终审）。
- 风格结论准入条件不变：E12 过门前，风格结论不得进主结论。

## 8. 复用件（D8）

- `external_eval/get_stroke_embedding`（VGG19 笔触特征）：**F3/F3b 笔触保真诊断器**——量化「丢连笔/笔触」的用户观察，输出 per-method stroke-style 保持度。仅作诊断。
- skeleton 相似度：拓扑证据，与 F3b support 选字的 primitive 计算**共用实现**。
- CoAtNet：见 D3 路 D。

## 9. 里程碑与验收

| 里程碑 | 验收 |
|---|---|
| M1 数据 | 池 ≥60 lineage 去重；295 渲染 + 覆盖矩阵 + SHA；零重叠校验通过 |
| M2 筛查 | 四路 T2 CV 分数表 + 赢家 |
| M3 训练 | 单 seed 训练完成；三件套输出就绪 |
| M4 自测 | T1-T4 全过（T2≥0.90、CI≥0.85、T3≥0.90） |
| M5 集成 | 8-ref 主口径 + 锚定趋势报告 + 人评协议 |

## 10. 待拍板

- D3 路 D（CoAtNet）权重许可与零重叠验证的责任人。
- 笔触诊断器（D8）是否纳入正式报告指标（建议：先诊断后决定）。
