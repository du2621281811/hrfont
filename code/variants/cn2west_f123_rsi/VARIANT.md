# Variant `cn2west_f123_rsi`

F1 / F2 / F3 的**唯一**代码实现。三条臂共用一条 code path，`--arm` 只切换一个因子。

| arm | `--rsi_source` | `--support` | 相对上一条臂隔离的变量 |
|---|---|---|---|
| F1 | `official`（`Ec(R[0])`） | off | 新增 RSI 容量本身 |
| F2 | `delta` | off | 只换 structure source |
| F3 | `delta` | on，drop=.20 | 只加 Support |

派生自 `cn2west_stage_a`（Δ / α / cache 管线沿用）。逐行差异见 [`docs/patches/cn2west_f123_rsi.diff`](../../../docs/patches/cn2west_f123_rsi.diff)。

## 相对 stage_a 的改动

### 1. `StyleRSIUpBlockIdentitySafe`（`src/modules/unet_blocks.py`）

```
warped   = DeformConv(skip, offset)
skip_out = skip + zero_conv(warped - skip)      # zero_conv: 1×1，weight/bias 全零初始化
```

F0 没有任何 RSI 模块，所以 F1/F2/F3 是在 F0 上**新挂**模块。zero_conv 让 step 0 与 F0 的 raw-skip 前向逐元素相等——这是代数恒等式，与算子实现和数值精度无关。被否掉的备选（零 offset + identity DCN kernel）实测偏差 5.58，见 `scripts/test_identity_safe_rsi.py`。

`rsi_gain() = mean|zero_conv.weight|` 每 `log_interval` 写进 `train_log.jsonl`，直接读出这条臂到底用了多少 RSI。

### 2. `--source_drop` 取代 `--delta_drop`（`train.py`）

旧 `_structure_features` 在 official 分支提前 return，drop 只作用于 delta——这是旧 E2 vs E2b 不 matched 的根因。现在两个分支一视同仁。

`support_draw` 即使在 support=off 的臂上也照抽，否则 F2 与 F3 从第 1 步起 RNG 分叉。

### 3. `_load_parent`（`train.py`）

F0 checkpoint 没有 RSI key，用 `strict=False` 加载，新建的 key 写进 `parent_load_manifest.json`。任何**非** RSI 的 missing/unexpected key 直接抛错。

### 4. Support（`src/model.py`、`train.py`）

`SupportAdapter` 的 token 只拼进 up-path cross-attention 的 context；MCA down-path 看到的 style map 不变，所以 F2↔F3 只差一处。support glyph 取目标字体自己的字（Ec cache `style` role）。

### 5. Parity gate（`train.py`）

非 resume 启动时强制跑：同一 batch 前向两次（`rsi_enabled` True/False），断言 `max|Δ| == 0`，结果写 `parity_gate.json`。

## 前置条件（全部 fail closed）

1. F0 跑满并按 `reports/DECISION_F123_20260906.md` §2 选出 milestone
2. 从该 milestone 重建 Es/Ec cache —— **E1 的 cache 绑在 E1 encoder 上，不可复用**
3. F3 另需 support bank（`data/hrfont/e0_bank` 已不存在，需按新协议重建）

## 用法

```bash
python scripts/test_identity_safe_rsi.py
python scripts/launch_cn2west_f123.py --arm F2 --parent runs/F0-RSIFREE-FT-A-S3407/best --smoke
python scripts/launch_cn2west_f123.py --arm F2 --parent runs/F0-RSIFREE-FT-A-S3407/best --yes
```

配置：`configs/f{1,2,3}_*.yaml`。单 seed 3407（PI 2026-09-05）。
