# E12 v4 自检复盘（2026-09-07）

上游：[`E12_SELFTEST_V3_REVIEW_20260906.md`](./E12_SELFTEST_V3_REVIEW_20260906.md)。本轮 = **S3 外部字型池**（26 套 OFL/free CJK）+ v3 字符轴 → `artifacts/e12/cache_v4`（SHA 见 manifest）。

## 1. 结果（正式门，不放宽）

| 种子 | T1 中位 ≥0.60 | T2 AUC ≥0.90 | T3 准确率 ≥0.90 | T4 drop≥0.02 | 过门 |
|---|---|---|---|---|---|
| 3407 | 0.729 过 | **0.791 不过** | 0.977 过 | 0.161 过 | 否 |
| 3408 | 0.773 过 | **0.795 不过** | 0.977 过 | 0.194 过 | 否 |
| 3409 | **0.533 不过** | **0.725 不过** | 0.977 过 | 0.126 过 | 否 |

落盘：`runs/e12_self_tests_v4_s{3407,3408,3409}/self_tests.json`（本目录副本见 `reports/e12_v4_results/`）。

相对 v3：T2 略升（约 0.73→0.79），仍远低于 0.90。T3 很高（62 类）。**主结论：仍 gate_failed，不进生成器打分。**

## 2. 数据与配置

- cache：`artifacts/e12/cache_v4` · 26 typefaces · 104 汉字 + 52 Latin + 10 数字 · 4316 PNG · canvas 96
- 字库源：`artifacts/e12/fonts_s3`（gitignore；合作者可用 cache 直接训，不必同步字体）
- 训练：`phi_s2` max_steps=3000, batch=32, lr=3e-4, early_stop on val AUC；再 `id_cls`；再 `self_tests`
- 切分：`split_by_group=true`，跨字型负样本
- 门限：T1≥0.60 / T2≥0.90 / T3≥0.90 / T4≥0.02 且 t≥1.96

## 3. 含义

编码器「认得字体身份」不难（T3），「跨未见字型做风格可分」仍不够（T2）。在过门前，**不要用 E12 分数比较 F0/F1/F2/F3**。

Mac 复训说明：[`E12_MAC_COLLAB.md`](./E12_MAC_COLLAB.md)。
