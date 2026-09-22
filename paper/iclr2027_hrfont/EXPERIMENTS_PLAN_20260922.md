# Experiments 写作方案 v2（按 PI 决策修订）

日期：2026-09-22（v2，取代同日 v1）
依据：PI 5 项决策 + 执行机权威配置（`runs/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2/config.json`、`runs/K7-B-V3-K0-S3407-20K-R2/config.json`、`data/v3_v2_plus_v0921_20260922/v0921_spec.json`）；评测器归属与防御性写作判据按 `references/paper-method-audit.md` §7b/§7c。
目标读者：PI。**本文只给方案与结构；main.tex 未改。**

## 0. 本轮决策与影响

| PI 决策 | 落地 |
|---|---|
| ① 训练参数以 K6-B 为主 | §3 按 K6-B run 的 config 权威值写（旧的「10,000 updates」说法作废） |
| ② 数据写 V3，只讲不泄露 | §3：V3 = v2 + v0921（315 字体、41,663 pairs，train-only、**不进 bank**）；泄露只写一句设定，不写辩护段 |
| ③ 外部复现结果已有、效果低于我们 | §4 主表外部行按已有复现结果填（数字来源待指路） |
| ④ 人评结果后到 | 结构先写；风格列留占位，人评只写协议不写结果 |
| ⑤ 砍掉过细实验 | §5 清单：主文消融压到 3 行；去掉逐层/逐 token/初始化/精度类消融、donor 审计、第二坐标消融、逐字体分解表与逐基线细节枚举 |

---

## 1. 节结构（5 小节，主文约 2.5 页）

| 小节 | 功能 | 篇幅 |
|---|---|---|
| 5.1 Setup and protocol | 数据 V3、协议 A、参考预算、训练配置、评测轴与统计口径、GT 定位 | ~0.8 页 |
| 5.2 Cross-script consistency evaluator | 构造一句 + 验证门一句 + 用途边界一句（结果占位） | ~0.3 页 |
| 5.3 Main comparison | 主表（外部方法 + 我们的方法），每行四项指标 | ~0.6 页 |
| 5.4 Contribution studies | 3 行消融，单变量 | ~0.4 页 |
| 5.5 Reference budgets and qualitative analysis | 1/2/4/8 参考预算曲线 + 三档定性 | ~0.4 页 |

## 2. 论证主线（主张 → 证据 → 落点）

| 主张 | 证据 | 落点 |
|---|---|---|
| 结构变化与外观表达可分开处理；identity 与 cross-script consistency 是两个可分开测的轴 | 双轴主表（identity 自动指标 + 风格相容性）；像素指标只作诊断 | 主表 |
| CRP：库实例化形变空间，参考判定实现，路由在生成中确定结构偏移 | ① 结构条件对照 ② 组合方式对照 | 消融表 |
| TC：从目标内容与源参考预测 target-aligned appearance representation | ③ 外观条件对照 | 消融表 |
| 跨书写系统场景下优于既有方法 | 主表外部行（已有复现结果）+ 参考预算曲线 | 主表、5.5 |

互补性不再单独设行：①与③各自去掉后与完整模型的对比即给出互补证据。

## 3. 数据与训练配置（权威值）

**数据（V3）**
- v2 主体：260 字体 = 228 train / 16 val / 16 test；295 目标字符、338 中文参考；协议 A（96×96 原生、逐字体统一字号、无 resize）。
- 补充集 v0921：315 字体、41,663 训练 pairs，`role = train-only`；bank 冻结在 v2 的 donor 列表（`bank_font_source = frozen V2 donor_train_by_cp only`）。
- 泄露只写一句设定：「补充集只进训练；检索 bank 保持冻结在 v2 的 donor 列表，补充字体不会成为推理期的候选。」

**训练（K6-B run 权威配置）**

| 项 | 值 |
|---|---|
| 结构 | 双支路 Ec/Es + RSI；目标 = endpoint-generation only，无 offset 正则、无 pure-noise 辅助 |
| 初始化 | F0 checkpoint @10k（K6-B：`G0b-F0-V0913-BS256-A-S3407/global_step_10000`；K7-B：`F0-CLEAN-V0913-A-S3407/global_step_10000`） |
| 并行 | 8 GPU × micro-batch 8 × accum 1 = global batch 64 |
| 步数 | 20,000 updates |
| 日程 | warmup 500 / 平台至 5,000 / cosine 衰减至 10% |
| lr | 峰值 2e-5（denoiser 与结构模块）、1e-4（reader 与 Es 支路读出头）；AdamW betas (.9,.999)、wd .01、eps 1e-8 |
| 精度 / EMA | fp16（logits 与 loss 走 FP32）；EMA `min(.999, 1-1/(u+1))` |
| seed / 采样 | 3407；attempt-keyed 全局采样 |
| 损失 | perceptual .01；外观项 .01（ramp `β=.8·min(u/1000,1)`）；细节项 .05；offset 0 |
| 条件 dropout | joint CFG .02；source drop .05 |

## 4. 表格

**主表**：`Method | Identity ↑ | Style pref. ↑ | Family ↑ | LPIPS ↓`
行：外部方法（已有复现结果，效果低于我们）／CRP（fixed weights）／ours。
外部方法一行一句交代参考预算与适配方式；训练数据等细节进附录，不在主文逐方法枚举。

**消融表（3 行）**

| Question | Matched comparison |
|---|---|
| 结构条件 | 完整模型 vs 去掉结构偏移 |
| 组合方式 | routed vs 固定先验权重 |
| 外观条件 | 完整模型 vs 去掉外观条件 |

行名即基线定义；正文不为对照逐行声明"保留了哪些条件"。

**参考预算**：1/2/4/8 曲线（图），可复用 `reports/g_v0913_shot_k1248` 的协议产物。

## 5. 不做 / 下沉附录（避免过细的防御性实验）

**不进主文**
- 逐 token / 逐层 / 初始化方式 / 精度细节类消融。
- donor swap 敏感性审计（机制探针，非论文必需）。
- 第二坐标（Es 中间层读取）独立消融。
- 「同库同字绝对聚合」对照（bank 守卫；放附录，不进主文）。
- 逐字体 / 逐字符分解表（主文一句说明按 script 分层趋势一致）。
- 每个基线的训练数据、预训练、跨语系适配细节枚举。

**下沉附录**：完整训练超参表、评测器细节与门结果、参考预算逐档数字、定性面板字样清单、F0 初始化臂的说明。

## 6. 评测器与人评（结果留空）

- 评测器：一句构造（共享跨书写系统表征 + 语境条件监督 + 温度校准的家族相容性分数）+ 一句验证门 + 一句用途边界（GT 作正对照；不作"越像 GT 越好"的质量尺）。
- 人评：只写协议要点（5 点量表、主维度风格相容性、次维度补全合理性、不展示 GT/方法/分数、每图 3 份有效评分）。**结果占位，不写推断。**
- 统计：逐图配对（同参考集、同采样器、同噪声、同 seed）；sign / Wilcoxon + Holm；报 CI。

## 7. LaTeX 骨架（可直接 splice，数字待填）

```latex
\section{Experiments}
\label{sec:experiments}

\subsection{Setup and protocol}
The corpus combines 260 typefaces, split into 228 training, 16 validation, and 16 test fonts, with a supplement of 315 typefaces used for training only.
The supplement never enters the retrieval bank, which stays frozen on the donor list of the main corpus, so no supplement typeface can appear as a candidate at inference.
Target characters and Chinese reference characters follow the partition of the main corpus.
Protocol A renders RGB glyphs at $96\times96$ with a shared font size per font and neutral Noto Sans CJK Regular content.
Evaluation uses nested reference sets of 1, 2, 4, and 8 characters drawn from the same ordered manifest for every method.

\paragraph{Training.}
% K6-B 配方：F0@10k 初始化；8 GPU $\times$ micro-batch 8 = global 64；20,000 updates；
% warmup 500 / flat to 5,000 / cosine to 10\%；lr 2e-5（denoiser 与结构模块）、1e-4（reader）；
% perceptual .01、appearance .01（ramped）、detail .05、offset 0；fp16；seed 3407；EMA。
% 逐项数值与实现细节见附录 \ref{app:repro}。

\subsection{Cross-script consistency evaluator}
% 一句构造 + 一句验证门 + 一句用途边界；结果占位。

\subsection{Main comparison}
\begin{table}[t]
\centering\small
\caption{Main comparison. Identity is character recognition accuracy, style preference follows the human protocol, and family compatibility is the evaluator score.}
\label{tab:main}
\begin{tabular}{lrrrr}
\toprule
Method & Identity $\uparrow$ & Style pref.\ $\uparrow$ & Family $\uparrow$ & LPIPS $\downarrow$\\
\midrule
% 外部方法（已有复现结果）
% CRP (fixed weights)
% ours
\bottomrule
\end{tabular}
\end{table}

\subsection{Contribution studies}
\begin{table}[t]
\centering\small
\caption{Contribution studies. Each row changes one variable.}
\label{tab:ablations}
\begin{tabular}{llr}
\toprule
Question & Matched comparison & Result\\
\midrule
Structural condition & full model vs no structural offset & \\
Combination & routed vs fixed prior weights & \\
Appearance condition & full model vs no appearance condition & \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Reference budgets and qualitative analysis}
% 1/2/4/8 曲线 + 三档定性面板
```

## 8. 待确认（需你拍板）

| # | 事项 | 推荐 |
|---|---|---|
| D-E1 | 主表用哪条 run 的权重：K6-B（v2 数据、20k 已完成）还是 K7-B（V3 数据、同配方、仍在跑） | 数据段写 V3 → 用 K7-B 权重力求一致；若必须用 K6-B，则数据段需写 v2 |
| D-E2 | 外部复现结果的数字在哪里（本地仓与执行机 `reports/` 均未找到外部基线复现产物） | 你指路后接入主表 |
| D-E3 | F0 初始化臂差异（K6-B 用 `G0b-F0-V0913`、K7-B 用 `F0-CLEAN-V0913`）写正文还是附录 | 附录一句 |
| D-E4 | 是否同意消融压到 3 行、其余下沉附录 | 按 §5 执行 |
| D-E5 | 「同库同字绝对聚合」对照是否放附录 | 放附录，不进主文 |

## 9. 未核实项
主表各行数字（K6-B `DONE.json` 的 `inference_complete=false`，需另跑固定协议推理）、人评采集状态、评测器门结果、K7-B 训练完成度。
