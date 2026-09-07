# 故事与 Idea 总结（2026-09-07，PI 口径整合版）

> 本文是当前项目叙事的一页式整合（含 PI 最新补充的 IDEA 表述）。证据等级以
> `BANK_STYLE_NARRATIVE_20260906.md`、`ARMS_NARRATIVE_QKV_20260905.md`、
> `REVIEW_BANK_STYLE_20260906.md` 为准。ICLR 视角评估见 `IDEA_ICLR_EVAL_20260907.md`。

## 1. 问题

给定一个未见字体的少量中文参考字（few-shot，主协议 8 张），生成同一字族的拉丁/假名字符。现有方法（FontDiffuser/DG-Font/CF-Font 系）**不能很好地捕捉跨语言的风格表示**：它们依赖同语系的结构对应（参考字结构与目标字结构的空间对齐），跨语系时参考字（汉字）无法有效预测目标字（拉丁）的形变，导致生成的跨语言文本在参考风格内**不合理**。

## 2. 方法：两个互补的机制

**① Δ-RSI（核心）**：FD 的 RSI 以参考字自身结构 `Ec(ref)` 为形变源——跨语系错位。我们把它换成**同字残差结构先验 Δ**：

- 用冻结 Es 对目标字体的 ref 做 per-char cosine 排序，从 228 字体库取 **top-10** 邻居，softmax(cos/τ, τ=.07) 加权；
- Δ = Σ ᾶ·Ec(库字体的目标字) − Ec(Noto Content)——**目标字身份对齐**，风格信息来自库先验；减去中性底得到「相对中性底应如何变」的 change-only 信号；
- 注入方式：RSI 的注意力/DCN **原封不动**，只换结构源；新 RSI 头用 identity-safe zero-conv residual（step0 严格等价无 RSI 的 F0），单通路归因干净。

**② 多 support 支持字机制（F3）**：在 Δ 之外，把支持字样例（当前实现：目标字体 own-font 8 字）经 SupportAdapter 注入 RSI context 流，为模型提供跨语言风格语义补充。F2 vs F3 隔离 support 的贡献。

**③ 附带性质：变化空间可定制**。α 无绝对阈值、输出完全由 bank 邻域构成——**策展 bank 即定制生成偏好**（推理期、零训练）。这说明 Δ 表征不仅指明变化方向，还把「跨语言风格变化空间」显式化了（P1/P2 探针验证中）。

## 3. 训练与因果拓扑

`official P1 → E1(FT 锚) → F0(RSI-free FT) → F1(official RSI)/F2(Δ)/F3(Δ+support)`：
- E1 证明「增益来自机制而非更多数据」；F0 提供无 RSI 的干净底座（无震荡引入 RSI）；
- F1↔F2 隔离 Δ；F2↔F3 隔离 support；全部 matched（同 F0、同预算、同 RNG/drop、identity-safe init）。

## 4. 评测：合理性优先（独立贡献）

现有评测只重像素级相似度（LPIPS 等），**忽视跨语言风格迁移最重要的「合理性」**。我们构建：

- **E12 独立评测器**：与方案编码器隔离的 φ_s2（外部 26 字型训练，T1–T4 自测门）、ID-CLS、DeepSets membership verifier；
- **三轴指标**：Identity / Style（SC-R、SC-Gap、Rank@1）/ Quality（coverage、LPIPS）；GT 仅作 positive control；
- **设计师人工评测（E11）**：分层 2AFC + MOS，终审「合理性」。

## 5. 声明阶梯（按证据强度）

| 等级 | 声明 | 状态 |
|---|---|---|
| 机制级 | 跨语系错位 → 换同字残差源；α 加权库先验；bank 策展改变先验 | ✅ 设计与 matched 消融支持 |
| 结果级 | 合理性/风格指标优于 baseline；few-shot 跨语言 SOTA | 🔄 等 F1/F2/F3 80k + E12 过门 + E3 主评测 |
| 性质级 | 「更符合设计师通用原则」；定制=可靠控制机制 | 📋 P1/P2 + 人评；强 claim 需 matched retrain |

## 6. 关键风险与依赖（诚实清单）

- SOTA 主张需主表落地（当前 F2/F3 ~13.5k+/80k，E12 卡 T2 门待 S3 重训）；
- F3 support 语义=own-font 8 字（与最初「跨字检索」设计不同，D-B5 待拍板）；
- 推理 Δ 路径在执行机实现，仓库 sample.py 未接线（D-B4 parity 准入）；
- 可控性叙事当前是 inference-time intervention，非「训练过的可控性」。
