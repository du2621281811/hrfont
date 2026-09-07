# Variant `cn2west_f0_rsifree`

F0：**完全去掉 RSI / DCN 模块**，不是只把 `offset_coefficient` 设成 0。

| 项 | 官方 / E1 | F0 |
|---|---|---|
| Up block | 带 RSI（offset 解释器 + DeformConv） | `StyleUpBlockNoRSI`：skip 直通，**无** offset / DCN / zero_conv 参数 |
| 从 P1 加载 | 全量 | 丢弃 RSI/DCN 相关 key，只保留可匹配权重 |
| `offset_coefficient` | 0.5 | **0.0**（模块已不存在，offset 项恒为 0；系数再设也无结构可训） |
| 风格条件 | Content + Style 图 + RSI 结构 | 仍用 Content + Style 注意力；**没有** RSI 结构通道 |

入口：`scripts/launch_cn2west_f0_rsifree.py` · run `F0-RSIFREE-FT-A-S3407`。

派生自官方 FontDiffuser / E1 微调设定（协议 A、seed 3407、100k、lr 1e-5、warmup 5000、bs 8）。F0 训完后按 val16 loss 选 milestone（现为 100k），再作为 F1/F2/F3 的**唯一父权重**。

F1/F2/F3 在 F0 上**新挂** `StyleRSIUpBlockIdentitySafe`（见 `cn2west_f123_rsi`），step 0 前向与 F0 raw-skip 逐元素相等。
