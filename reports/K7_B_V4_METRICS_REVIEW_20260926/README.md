# K7-B / V4 指标审阅包

本目录是独立的指标归档，仅包含表格、字体级差值和评测样本清单；不包含训练源码、模型权重或推理图片。它与代码/图片归档分开存放，避免合作者误读旧版本。

## 评测口径

- V4 共同评测集合：47 个有效字体、8,638 个字体—字符样本；完整清单见 `v4_common_sample_manifest.csv`。清单不含本机文件路径。
- 主表采用已确认的 checkpoint 选择：A1 使用 20K，A2/A3 使用 4K。表中的模型显示名不带步数。
- 主表与按文字体系汇总均先在字体内对字符求均值，再对有效字体等权平均。
- L1、LPIPS、DINO distance 越低越好；SSIM、RSC-8 越高越好。字体配对效果表统一为正值代表 HRFont 更好：误差指标为对照减 HRFont，相似度指标为 HRFont 减对照。
- 字体配对区间以字体为重采样单位，10,000 次 bootstrap，固定种子 3407。按文字体系表分别使用各自有效字体集合，并记录字体数、字符数及字体—字符对数。
- 问卷文件只保留题内成对顺序一致率；30 道正式题等权汇总，按题 bootstrap 10,000 次、种子 3407。字体聚类敏感性区间另列，不含 Kendall 相关系数。

## FontDiffuser RSC-8 标记

FontDiffuser 主表 RSC-8 使用用户确认值 0.613962。依赖该值的字体配对及分层 RSC-8 记录均标记为 `estimated`；其余未标记为估计的记录按表中样本直接计算。

## 文件索引

- `v4_main_table.csv`：主表，含 L1、SSIM、LPIPS、DINO distance、RSC-8。
- `paired_font_effects.csv`：主要模型对的字体级配对效应、区间和胜/平/负比例。
- `paired_font_differences_by_font.csv`：逐字体效应明细。
- `v4_metrics_by_script.csv`：Latin、kana、bopomofo 的字体宏平均和样本计数。
- `v4_paired_effects_by_script.csv`：各文字体系内的字体配对效应及区间。
- `questionnaire_order_agreement.csv`：问卷题内顺序一致率及区间。
- `questionnaire_fontdiffuser_7_supplement.csv`：问卷补充的 7 个 FontDiffuser 样本指标。
- `v4_common_sample_manifest.csv`：V4 共同样本清单。
- `SHA256SUMS`：本目录文件校验值。

此包是审阅用指标快照，不替代训练/推理代码归档，也不含逐样本预测图像。
