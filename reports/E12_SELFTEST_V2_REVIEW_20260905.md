# E12 cache_v2 T1–T4 结果与归因（2026-09-05）

配置：`configs/e12_self_tests_v2_s*.yaml` · cache `artifacts/e12/cache_v2` · 产物 `runs/e12_self_tests_v2_s*/self_tests.json`

## 1. 结果

| seed | T1 median (门 .60) | T2 AUC (门 .90) | T3 acc (门 .90) | T4 mean_drop / t (门 .02 / 1.96) | passed |
|---|---|---|---|---|---|
| 3407 | 0.589 ✗ | 0.885 ✗ | 1.000 ✓ | 0.152 / 6.94 ✓ | ✗ |
| 3408 | 0.597 ✗ | 0.910 ✓ | 1.000 ✓ | 0.143 / 9.26 ✓ | ✗ |
| 3409 | 0.569 ✗ | 0.828 ✗ | 1.000 ✓ | 0.112 / 8.07 ✓ | ✗ |

训练侧 φ_s2 best val AUC：0.736 / 0.675 / 0.797。T4 的 `n=24`（3 测试族 × 8 query 字符）。

**结论：正式门未过；当前 φ_s2 指标不得进入主结论。** T3=1.000 不是好消息，是任务过易的信号（见 §2.4）。

## 2. 归因

### 2.1 字族数被字重伪装放大（主因）

`fonts_pool_v2` 有 20 个 face，但按字型（typeface）归并只有 **8 个真实字族**：

```
NotoSansCJK × 7 weights, NotoSerifCJK × 7 weights,
TSTanHJW-R, gbsn00lp, gkai00mp, ukai, uming, wqy-zenhei  (各 1)
```

即 70% 的"族"只是同一字型的字重变体。φ_s2 学到的主要可分轴是**笔画粗细**，不是跨语系的字型身份。

### 2.2 split 存在字型泄漏

`split_families` 按 face 名切分，未按字型分组：

| split | faces | 字型组 |
|---|---|---|
| train | 14 | NotoSansCJK, NotoSerifCJK, TSTanHJW-R, gkai00mp, ukai, uming, wqy-zenhei |
| val | 3 | gbsn00lp, NotoSansCJK, NotoSerifCJK |
| test | 3 | NotoSansCJK, NotoSerifCJK |

**test 的 3 个族全部来自 train 已见过的两个字型组**（`NotoSansCJK-Medium`、`NotoSansCJK-Bold`、`NotoSerifCJK-SemiBold`）。这既高估了泛化，又把测试集限制在同型近亲上。

### 2.3 负样本是同字型相邻字重（T2/T4 的硬上限）

`self_tests.py` / `train_style_encoder.py` 取 `wrong = all_f[(index+1) % n]`（字母序相邻），在当前池上落成：

| test family | wrong family | 性质 |
|---|---|---|
| NotoSansCJK-Medium | NotoSansCJK-Regular | 同字型，相邻字重 |
| NotoSansCJK-Bold | NotoSansCJK-DemiLight | 同字型，跨两级字重 |
| NotoSerifCJK-SemiBold | TSTanHJW-R | 真·异字型 |

3 个负例里 2 个在 96px、8 字符下几乎不可分。T2 AUC ≈0.83–0.91 基本是这个配对决定的，**不是编码器能力的无偏估计**。

### 2.4 T3 过易，不构成有效门

ID-CLS 只有 8 类（A–H），字形差异远大于风格差异，1.000 是天花板效应。它现在无法证伪任何东西。

### 2.5 训练样本量 → 过拟合

`CrossScriptPairDataset` 的 key 是 `family × 汉字`，cache 只有 8 个汉字：

```
train pairs = 14 families × 8 汉字 = 112
5000 steps × bs32 ≈ 1400 epochs
```

训练 loss 收到 `5.5e-5`，而 val AUC 停在 0.68–0.80 —— 典型记忆化。T1 median 0.57–0.60 卡在门线附近也与此一致。

## 3. 解决方案（按性价比排序）

### S1. 扩字符轴（最便宜，无需新字体）
cache_v3 把汉字从 8 → 100（ref 保留 8 字用于 prototype，训练用全量）。train pairs 直接 112 → 1400（×12.5）。现有字体全是 CJK 字型，字符可用性没有障碍。

### S2. 按字型分组切分 + 分组感知负采样
- `split_families` 增加 `group_key`（剥离字重后缀），**按组切分**，禁止同字型跨 split。
- 负样本从**其他字型组**里采，并把 "within-typeface 字重对" 作为单独一档报告（诊断项，不进正式门）。

### S3. 扩真实字型池
目标数十个**独立字型**（不是字重）。当前机器上外网多次 404/TLS 失败，需要 PI 给可用来源或离线包；这是唯一无法靠改代码绕过的项。

### S4. 重设 T3
ID-CLS 类别扩到 52 Latin（+10 digit），门维持 0.90。当前 8 类版本降级为 smoke。

### S5. 训练预算与早停
按新 pair 数重算 step budget，按 val AUC 早停，禁止用训练 loss 判收敛。

**依赖关系：** S1、S2、S4、S5 是代码/配置改动，可立即做；S3 需要 PI 提供字体来源。S1+S2 落地前，T2/T3 的数值都不具备解释力。

## 4. 现状标记

- `E12-*-V2-S3407/08/09`：**gate_failed**，仅作通路与诊断，不进主表。
- cache_v1 / cache_v2 均保留，不覆盖；后续发 cache_v3。
