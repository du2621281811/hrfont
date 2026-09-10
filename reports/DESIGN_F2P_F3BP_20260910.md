# F2-P / F3b-P 设计规格（P-attn 臂，2026-09-10）

目标：把 style 条件从「多 ref 3×3 图等权平均 → 9 token」改为「逐 ref pooled h → n 个独立 token」，消除平均对细粒度风格（斜势/笔触对比/连笔）的抹除。F2/F3 已证明 9-token 均值是「风格平」的主因（F3 val 35k-80k 平台 + 用户人工判读 + STYLE_REGULARITY 诊断）。

## 1. 臂定义与对比链

| 臂 | 定义 | 状态 |
|---|---|---|
| F2 | Δ-only、global9 均值（带病对照） | ✅ 已 80k（合作者未同步日志；40k 里程碑在盘上） |
| **F2-P** | Δ-only、per-ref h tokens（修复聚合器） | 新训 40k |
| **F3b-P** | Δ + own-font 拓扑 support + per-ref h tokens | 新训 40k |

- 聚合器消融：F2@40k ↔ F2-P@40k（唯一变量 = style 聚合方式）
- support 消融：F2-P@40k ↔ F3b-P@40k（唯一变量 = support 存在性）
- F3b-S（带病 support 版）**取消**——F2/F3 已证明 9-token 问题，不再训带病臂。

## 2. 代码事实与修改点

**现状**（`src/model.py:49-62`）：
```python
style_hidden_states = style_img_feature.permute(0,2,3,1).reshape(B, H*W, C)  # 9 token（均值后 map）
if support_tokens is not None:  # F3b-P 才有
    style_hidden_states = torch.cat([style_hidden_states, support_tokens], dim=1)
input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states]
# [0] 4D map → down-path（不动）；[2] 3D token → up-path style cross-attn
```

**修改清单（全在 train.py 与 model.py 两个文件 + 推理 wrapper）**：

1. `train.py:_style_conditions`（:130-142）：`spatial.mean(0)` 的 9-token 均值**删除**，改为逐 ref 取 pooled h（`es_cache.pooled_tensor`，与 α 检索同一 cache、同一归一化）→ 堆叠 [n, 1024] → **不聚合、不平均**。
2. `src/model.py`：style_hidden_states 的第一段改为接收 `[B, n, D]` 的 h token 序列（D=1024 经共享投影到 cross_attention_dim，若有维度差）；support concat 逻辑不变（F3b-P）；down-path 4D map 原样（保留旧 map 契约）。
3. 新增唯一可训练参数：**一个共享 Linear**（1024 → cross_attention_dim，若维度一致则无新增参数），标准随机初始化、隔离 RNG，不 zero-init。
4. 推理 wrapper（执行机 sample.py）：与训练同步改——同一 h token 输入、同一投影、同一 n=8 固定 ref8。
5. **零新 cache**：pooled h 已在 Es cache（α 检索天天在算），无 687MB 中层缓存开销。

## 3. 关键设计点：为什么「不聚合」是正解

- **平均/加权平均/拓扑加权平均都会合并 ref 轴**——加权平均只是「选谁进平均」更聪明，合并后仍是单一向量，笔触级差异照样丢。用户的判读（「选 ref 再准也避免不了」）在此成立。
- **正解 = 保留 ref 轴，把按需参考交给 UNet 自己的 attention**：h token 序列直接作为 up-path style context，UNet 每个位置的 Q 各取所需（生成 o 的位置自然多看带圈 ref 的 h）——「按需参考」由模型原生实现，不需要额外聚合器。
- **拓扑表的角色**：作为**可选注意力偏置**（每个 ref 加一个拓扑相关度标量权重，弱偏置不硬选），列入消融候选而非主方案——主方案不加，保持最小。

## 4. 技术决策清单（PI 2026-09-10 已拍板）

| ID | 决策项 | 裁决 |
|---|---|---|
| D-P1 | 聚合方式 | **A**：逐 ref h token 直进 UNet（无聚合、无平均）；B/C 仅作消融候选 |
| D-P2 | h 维度处理 | 投影（若 1024≠CA dim）；共享一个 Linear |
| D-P3 | down-path 4D map | **保留**（[B,1024,3,3] 照旧喂 MCA；只改 up-path 3D token 段） |
| D-P4 | α 检索 | **照旧**（8-ref pooled query；检索与生成条件解耦） |
| D-P5 | n 处理 | 训练变长 n~U{1..8} + **真实 attention mask**；推理主协议固定 ref8（n 输入个数由 PI 定）；eval 加 n 扫描面板 n∈{1,2,4,8}（全在训练支撑域，零训练） |
| D-P6 | 拓扑偏置 | **不加**（保持 F2↔F2-P 单变量；留作 F3b-P 后推理期 logit-bias 消融，固定 ckpt 零训练） |
| D-P7 | 预算 | **40k**（PI 拍板：后续全部臂 40k；80k horizon 开跑、40k 评估可续；F1/F2/F3 主线维持 80k 不动） |
| D-P8 | run id | `f2_pattn_s3407` / `f3b_pattn_s3407`；**种子全链一致 3407**（含新投影初始化隔离 RNG 但同 seed 域） |
| D-P9 | matched 契约 | 同 F0 init、seed 3407、source_drop .25 共享、CFG .10、support_drop .20（仅 F3b-P）、40k 端点 |
| D-P10 | 训推一致 | 同 h 形态、同投影、同 mask 规则；20k 面板 → 40k 终评 |

## 5. 验收清单（开训前）

- [ ] F2@40k 里程碑确认存在（合作者同步后核对）
- [ ] h token 形状断言 + 投影梯度非零（真实更新，不以 loss 反传代替）
- [ ] 训练/推理 wrapper 同改（执行机 sample.py）
- [ ] attention mask 真实屏蔽 padding（不复用被忽略的旧 mask 路径）
- [ ] 20k 面板（F2@40k vs F2-P@20k 同题本）→ 用户 review → 40k 终评

## 6. 时间

F2-P 40k ≈ 16h → F3b-P 40k ≈ 17h（串行防 I/O 争抢），+ 面板/评测 2 天 → 全部 ~9/15 前完成。
