# E12 cache_v3 自检复盘（2026-09-06）

上游：[`E12_SELFTEST_V2_REVIEW_20260905.md`](./E12_SELFTEST_V2_REVIEW_20260905.md)。本轮实施 PI 批准的 S1/S2/S4/S5，**S3（扩真实字型池）暂缓**。

## 1. 结果

三 seed（3407/3408/3409），cache_v3，正式门。

| 指标 | 门 | v2 | v3 | 判定 |
|---|---|---|---|---|
| T1 风格相似中位数 | ≥0.60 | 0.589 / 0.597 / 0.569 | **0.676 / 0.615 / 0.640** | v2 全败 → **v3 全过** |
| T2 AUC | ≥0.90 | 0.885 / 0.910 / 0.828 | **0.694 / 0.704 / 0.798** | 均未过；**v3 更低是对的**，见 §2.1 |
| T3 身份准确率 | ≥0.90 | 1.000 / 1.000 / 1.000 | **0.907 / 0.921 / 0.895** | v2 无意义 → v3 真实，2/3 过 |
| T4 错配掉分 | ≥0.02 且 t≥1.96 | 0.152 / 0.143 / 0.112 | **0.075 / 0.154 / 0.201**（t = 17.4 / 18.3 / 18.8） | 全过 |

新增诊断 `D_within_typeface`（同字型换字重的掉分）：0.037 / 0.116 / 0.122，稳定小于跨字型掉分——编码器确实把"换字重"看作比"换字型"更小的变化，符合预期。

φ_s2 val AUC（跨字型负样本、留出锚点）：0.956 / 0.979 / 0.968。ID-CLS macro F1：0.866 / 0.873 / 0.827。

**总判定：仍未过门，不进主结论。** 但失败点从"四项里两项废、一项无意义"收敛成"只剩 T2，且原因可定位"。

## 2. 归因

### 2.1 T2 从 0.87 掉到 0.73 是修复的结果，不是退步

v2 的 T2 建立在两个虚高之上：`split_families` 按 family 名切分，`NotoSansCJK-Bold` 在 train、`NotoSansCJK-Regular` 在 test，编码器实际见过测试字型；负样本 `all_f[(i+1)%n]` 又几乎总是同一字型的另一个字重。v3 按字型分组切分 + 跨字型负样本之后，测试集里的字型编码器**一个都没见过**。0.73 是这个更难任务上的真实数字。

### 2.2 T2 不过是判别力问题，不是跨字体标定问题

先排除标定：pooled AUC（全局阈值）与 per-font AUC（每个锚点各自算）几乎相同——

| seed | pooled T2 | per-font AUC 均值 |
|---|---|---|
| 3407 | 0.694 | 0.722 |
| 3408 | 0.704 | 0.720 |
| 3409 | 0.798 | 0.729 |

如果是标定问题，per-font 会明显高于 pooled。它没有。

真正的结构是 **per-font AUC 呈双峰**（seed 3409）：

| 测试字体 | AUC | same | diff |
|---|---:|---:|---:|
| NotoSansCJK-Black | 1.000 | 0.877 | 0.324 |
| NotoSansCJK-Bold | 1.000 | 0.816 | 0.343 |
| NotoSansCJK-Medium | 0.999 | 0.720 | 0.284 |
| gbsn00lp | 0.707 | 0.530 | 0.443 |
| NotoSansCJK-Thin | 0.588 | 0.608 | 0.560 |
| NotoSansCJK-Light | 0.554 | 0.427 | 0.406 |
| NotoSansCJK-DemiLight | 0.496 | 0.531 | 0.533 |
| NotoSansCJK-Regular | 0.486 | 0.599 | **0.605** |

三个重字重接近满分，四个轻/常规字重在随机线附近，Regular 甚至反了（错字型原型比自己的原型还像）。三个 seed 同样的分裂（完整数据 `runs/e12_t2_diagnostic_v3.json`）。

编码器学到的主要是**笔画粗细**。粗细一旦不显著，它就没有别的依据可用——因为训练集只有 5 个字型（TSTanHJW、gkai00mp、ukai、uming、wqy-zenhei），全是中文向的楷/宋/黑，而测试字型 NotoSansCJK 是它从没见过的现代无衬线族。

**这正是 S3。** T2 不是靠调训练预算或负采样能补的；它缺的是训练期见过的真实字型数量。

### 2.3 还没解决的次生问题

按字型分组切分后，8 个字型 → train 5 / val 1 / test 2。两个后果：

1. **InfoNCE 假负样本**：`CrossScriptPairDataset` 把 batch 内其它样本当负例，但 batch=32 而训练只有 5 个字型，平均每个字型 6 个样本，其中同字型不同字的对被当成了负例。
2. **val 只有一个字型组**，组内不存在跨字型负样本。现在从 train∪val 取负样本（**不含 test**，否则选 `best.pt` 时就泄漏了测试字体），但这意味着负样本是编码器见过的字体，锚点才是留出的。

两条都是字型池太小的直接后果，都随 S3 消失。

## 3. 本轮改了什么

| ID | 改动 | 位置 |
|---|---|---|
| S1 | 字符轴 8 → 104 汉字；cache_v3 = 104 汉字 + 52 Latin + 10 数字 × 20 face | `artifacts/e12/cache_v3` |
| S2 | `split_families` 按字型分组切分；`pick_negative` 跨字型；`assert_disjoint_splits` 增加字型组泄漏检查 | `scripts/eval_framework/data.py` |
| S2 | val 负样本池扩到 train∪val（不含 test）；无负样本时直接报错而非产生 NaN | `train_style_encoder.py` |
| S4 | ID-CLS 8 类（A–H）→ 62 类（52 Latin + 10 数字） | `configs/e12_id_cls_v3_s*.yaml` |
| S5 | 按 val AUC early stop（patience=4 次评估）；不再按 train loss 选 | `train_style_encoder.py` |
| — | YAML 裸数字被解析成 int 导致字形查找崩溃；chars 统一 `as_chars()` 强转并在 config 里加引号 | `data.py` 等 |
| — | `/dev/shm` 只有 64 MB，DataLoader worker Bus error；v3 配置 `workers: 0` | `configs/e12_*_v3_*.yaml` |
| **S3** | **未做**：字型池仍是 8 个真实字型（20 个 face 里 13 个是 Noto 的字重） | — |

## 4. 建议

1. **T2 的门先不要动。** 0.90 是合理的；现在过不了是因为池子小，不是门定错了。
2. **S3 是唯一有效路径**：需要 PI 给外部字型来源。目标是训练期 ≥25 个**互不相同的字型**（不是字重），且覆盖衬线/无衬线/手写/装饰。按当前 5 个字型的规模，任何调参都只是在噪声上挑数字。
3. 在 S3 落地前，**E12 的数字不进主结论**，只作为框架自检记录。这一点与 v2 的判定一致。

## 5. 复现

```bash
bash scripts/launch_e12_formal_v3.sh                      # phi_s2 -> id_cls -> T1-T4，三 seed
python scripts/eval_framework/self_tests.py --config configs/e12_self_tests_v3_s3407.yaml
```

产物：`reports/training_logs/e12_*_v3_*`、`runs/e12_t2_diagnostic_v3.json`。
