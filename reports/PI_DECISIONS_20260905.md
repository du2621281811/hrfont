# PI 决策冻结（2026-09-05）— 请合作者审核

范围：覆盖 2026-09-04 对齐审查中与 **对照臂、α、style token、cache** 冲突的条目。  
本提交 **只改文档与 YAML 口径**，Stage-A 训练代码仍是旧实现（1-token / 在线 Ec / soft-ε），审核通过后再改代码。

冲突时以本文件 + `.cursor/rules/hrfont-execution-spec.mdc` §1.6 为准。

## 已拍板（三条）

### P1 — 三臂归因，E2 vs E2b 仍是 Δ 主对照

| 臂 | ID | style | RSI 结构源 | 作用 |
|---|---|---|---|---|
| **E1c** | `E1C-FT-CONTINUE-S3407` | **E1 原协议**：1-shot，单张 `style_emd` 3×3 → **9 token** | 官方 `Ec(同一张 style 图)` | 「同样再训 80k、不改 RSI」的真基线 |
| **E2b** | `E2B-FT-CONTINUE-S3407` | n-shot（训 n~U{1..8}，评 n=8），n 张 3×3 **空间均值** → **9 token** | 官方 `Ec(R[0])` | 在已换 n-shot style 后，官方 RSI 对照 |
| **E2** | `E2-STAGE-A-S3407` | 与 E2b 完全相同 | **Δ**（top-K 同字 Ec − Content） | 方法臂 |

三臂共用：E1@100k 初始化、冻 Es/Ec、只训 UNet+offset、80k、8×1、seed 3407 先跑、best 从 10k 起。

- E2 vs E2b：只改 RSI 源（Δ 是否有用）。  
- E1c vs E2b：只改 style 协议（n-shot 9-token 均值 vs 原 1-shot）。  
- E1c vs E2：整包（n-shot + Δ），不能单变量归因。  
- **E2c** 仍是附录冷系统（P1 init + P1 cache），不是 matched 对照。

### P2 — α = 必取 top-K，再 softmax；禁止空邻域

Δ 学的是 **同字内容的潜在变化方向**，不要求邻居风格「足够像」。α 只把 227 路收成 K 路，缩小混合搜索空间。

冻结：

- 相似度：同一组 R 上 **逐字 cosine 再对字平均**（已实现）。  
- 选邻：`mode=topk`，**K=10**；在这 K 个分数上 `softmax(s/τ)`，τ=0.07。  
- **禁止** 用 `ε=0.01` 把 softmax 质量截空。`eps_alpha` 仅允许 `≤1e-6` 的数值地板。  
- 空邻域不是方法语义；实现上 top-K 必须永远返回 K 个（库不足 K 时取全部，且 leave-one-out 后至少 1 个，否则 fail-closed 视为数据/cache 坏）。  
- 消融（附录）：K∈{3,10}；不把 soft-ε 截断当主方法。  
- n-shot：**上限已定为 8**（训 `n~Uniform{1..8}`，评测固定 8）。α 与 style 消费同一组 R。

### P3 — 全部离线 cache；style 用 9 token 空间均值

训练和推理 **禁止在线跑 Es/Ec**。渲染已落盘，编码器只在建 cache 时跑一次。

| cache | 键 | 存什么 | 谁用 |
|---|---|---|---|
| Es spatial | `(split, font, style_cp)` 覆盖 **260×338** | `style_emd` `[1024,3,3]` fp16 | E1c/E2/E2b 的 style 条件 |
| Es pooled | 可从 spatial 平均得到，或旁路存 `[1024]` | 归一化 pooled | α query / 库 prototype |
| Ec target | train228×295（val/test 只读、不进 Δ 库） | 多尺度 residuals+final | Δ 邻居 |
| Ec content | ContentImage×295 | 同构多尺度 | MCA Identity 与 Δ 减数 |
| Ec style | 260×338 StyleImage | 同构多尺度 | E1c / E2b 官方 RSI（`Ec(style)` / `Ec(R[0])`） |

- SHA：每份 cache 的 encoder SHA 必须等于该臂 init 的 Es 或 Ec SHA（D-A1 扩到 Ec）。  
- E2c 若做：另建 **P1** cache，禁止混用 E1@100k 表。  
- 体积量级：Es spatial ~2 GiB；Ec target+style fp16 合计约 40–90 GiB，盘上 mmap/按 batch 取，禁止整表进 GPU。

**style 条件（覆盖 D-A4）：**

- 对 R 中每张图取 Es 的 **空间图** `style_emd ∈ ℝ^{1024×3×3}`。  
- 对 n 张做逐元素平均，展平为 **9 个 token**（与 E1 注意力长度相同）。  
- **不用** pooled 压成 1 token。n 只影响这张 3×3 的估计方差，不改变 token 数。

## 明确作废

- D-A4 原文「pooled → 1 token」。  
- D-A6 把空邻域当合法 fail-fast/重定 ε（改为 top-K 保证非空；cache/数据坏才中止）。  
- 主方法 `mode=soft` + `eps_alpha=0.01`。  
- 「E2b = E1 续训 / step0 等价 E1」。E1 续训是 **E1c**。  
- 训练期在线 Es/Ec。

## 实现缺口（审核通过后再改代码）

当前 `cn2west_stage_a` 仍是：live Es pooled 1-token、live Ec、soft-ε、无 E1c 入口、sample 不送 Δ。  
YAML 已改口径；launcher/train 尚未接线。
