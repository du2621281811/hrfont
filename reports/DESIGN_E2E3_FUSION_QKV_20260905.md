# HR-Font：E2/E3 融合、RSI-free FT 与 Δ-RSI QKV 设计建议

**日期：** 2026-09-05  
**代码基线：** `827046d`  
**结论级别：** 方法/实验设计建议；未改代码、配置或既有文档。

## 0. 给 PI 的执行摘要

1. **可以把 Stage A（Δ-RSI）与 Stage B（SupportAdapter，现编号 E5）合成一次端到端训练，但不应把它当成无代价的省时。** 它省一次串行训练和一次 checkpoint 交接，却丢掉现有 Stage B 最有价值的归因：在冻结 Stage A 后，SupportAdapter 是唯一可学习增量。推荐主线采用“一次联合训练 + 两个 matched controls”，而不是只跑一条联合臂。
2. **“RSI-free FT 后加 zero-init RSI，step 0 精确等价 FT”按当前结构不成立。** RSI 不只是 offset head：它在 skip 上插入 `DeformConv2d`。零 offset 只令 DCN 在规则网格采样，仍会经过 DCN 的卷积权重，并不等于普通 `UpBlock2D` 的原样 skip。若 PI 要精确等价，必须采用显式 identity bypass/残差式 RSI，或同时把 DCN 初始化为严格 identity 并通过逐层 parity test。
3. **实际 QKV 是 Q=结构条件（官方为参考字 Ec；E2 为 Δ），K=V=UNet down-path skip。** 现有 RSI 审计与周报在这一点上是对的，不需要纠正方向。代码先把第一个参数投影为 Q，把 `context` 投影为 K/V。
4. **最优先的 QKV 改进不是简单加 LayerNorm。** 当前头已经对 Δ 和 skip 各做 GroupNorm、1×1 Conv、LayerNorm；再加同类归一化大概率重复。优先做：显式 identity-safe offset 残差参数化、Q/K/V 角色交换的 20k 单 seed 筛选、以及 Δ→K/V 的轻量 additive modulation。坐标编码与更多尺度放在附录筛选。
5. **时间风险高。** 旧三臂计划约 **20.19 GPU-days**（若 E1c 也做三 seed，则 23.33）；3 GPU 理想墙钟 6.73 天（或 7.78 天）。新的、因果上仍站得住的最小计划约 **34.47–42.59 GPU-days**，3 GPU 理想墙钟 **11.49–14.20 天**，尚未计工程、cache、验证、评测、失败重跑。距 9/25 仅 20 天，不能同时承诺 100k 联合训练、三臂三 seed、完整消融和从容写作。

---

## 1. Q2：E2/E3 到底指什么

### 1.1 本报告采用的映射

现行执行规格中：

- **E2** 是 Stage A / Δ-RSI 训练；
- **E3** 是 test16 主对比评测，本身无训练；
- **E5** 才是 Stage B / SupportAdapter 训练，且冻结 Stage A、只训练 adapter。

证据是现行拓扑写成 `E2 → E3/E4 → E5 → E6`（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:26-39`）；实验卡也把 E3 定义为 official P1、E1、E1c、E2b、E2 在 test16×295 上评测（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:127-130`），而 E5 明确是从 E2 best 初始化、冻结 A、只训练 SupportAdapter 25k（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:132-138`；`.cursor/rules/hrfont-execution-spec.mdc:87-89,115-116`）。

因此，结合 PI 所说“二者没有架构创新、合并可省训练时间”，本报告把口语中的“E2+E3”解释为 **Stage A（Δ-RSI）+ Stage B（Support）**，即现行编号 **E2+E5**。

**另一种字面解释：** 若 PI 真指现编号 E2+E3，则只能把训练和其后 test16 评测合并为一个流水线/里程碑；E3 无训练可省，方法结构与 GPU 预算几乎不变。

### 1.2 融合得到什么、失去什么

**得到：**

- 从“80k Stage A + 25k frozen-adapter Stage B”变为一次 80–100k 联合训练，减少一次 checkpoint 选择、冻结/解冻切换和独立 launcher；更早得到端到端 Δ+Support 证据。
- Support 可以与 Δ-RSI/UNet 共同适配，理论上避免 adapter 在一个已经冻结的表示上被动补救；`support_drop=.20` 可让无 support 模式仍被训练。
- 若固定总 optimizer steps，墙钟可能低于两段串行 105k；但每步加入 support retrieval/adapter 后并不一定与当前 E2 同速，必须用 1k 实测速率重算。

**失去：**

- 现 E5 的核心不是架构新颖，而是**归因干净**：Stage A 全冻，只有 `Linear→GELU→Linear→LayerNorm` adapter 可更新；故 E2→E5 的差可解释为 support 增量（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:132-138`）。联合训练后，UNet、RSI、SupportAdapter 都可共同变化，不能再说“相对 Stage A 唯一 learned delta 是 adapter”。
- 论文原来的分工“Δ 回答哪里/如何变，Support 回答缺什么局部证据”会被联合优化混合；即使最终增益更大，也无法区分 support 本身、额外训练预算、或 support 改变了 RSI/UNet 表示。
- E5b/E6 的 paired comparison 与 E7 support ablations 都假定存在冻结的 Stage-A anchor；规格明确 E6 比 E2 vs E5、E2d vs E5b，E7 使用同一 E5 checkpoint/noise/retrieval（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:136-142`）。融合后必须重新定义这些 estimands。
- 如果“联合臂”比“不带 support 臂”多了 support dropout 的 RNG、不同有效 batch 或不同无梯度样本处理，也会产生新的非方法混杂。

### 1.3 中间路径及其因果洁净度

可做 **joint E2&3**：SupportAdapter 末层 zero-init，`support_drop=.20`，UNet/RSI/adapter 联合训练；同时保留同初始化、同 batch manifest、同 steps、同 RNG draw 的 no-support matched arm。这个方案在 step 0 可让 **adapter 输出**为零（现有设计本就要求末层 zero-init，`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:136`），但它只能保证“有/无 support package”的随机化对比，不能证明最终差异只来自 adapter 参数，因为联合臂中 UNet/RSI 也会沿不同梯度轨迹更新。

因果洁净度从高到低：

1. **原两阶段：** Stage A 固定后，只训 adapter；最干净，但多 25k 阶段。
2. **联合 matched pair：** 同起点、同预算，support on/off；可估计“允许 support 参与联合优化”的整体处理效应，不能估计“adapter 的纯增量”。
3. **只跑联合臂：** 只能展示最终系统，不能排除多训练/更多条件/更多参数的解释；不适合承担主机制结论。

推荐 2 作为赶截止日的折中，并把原冻结 adapter 小规模复现保留为 appendix sanity check（可仅 seed3407），而不是完全删除冻结设计。

## 2. Q2：RSI-free FT 的结构事实与推荐规格

### 2.1 “FD weights minus RSI”不是简单删一组独立 head

官方 `FontDiffuserModel` 由 UNet、style encoder、content encoder 三个大部件组成；参考图既供 Es，也经 Ec 产生 `style_content_res_features` 并作为 UNet 第四路输入（`code/official/FontDiffuser/src/model.py:14-24,34-54`）。官方 build 在四个 up blocks 中间两层选择 `StyleRSIUpBlock2D`（`code/official/FontDiffuser/src/build.py:8-22`）。

RSI 的可学习参数确实集中在这些 block 内的两组模块：每个 resnet 层各有一个 `OffsetRefStrucInter` 和一个 `DeformConv2d`（`code/official/FontDiffuser/src/modules/unet_blocks.py:423-477,503-506`）。但它们**嵌在 skip pathway 中**，不是旁挂而不改变主干：

```text
RSI block: raw skip -> offset head -> DeformConv -> concat decoder
plain block: raw skip ---------------------------> concat decoder
```

RSI 路径的精确调用是先 `offset=sc_inter_offset(skip, structure)`，再 `dcn_deform(skip, offset)`，最后 concat（`code/official/FontDiffuser/src/modules/unet_blocks.py:545-563`）；普通 `UpBlock2D` 则直接 concat 原始 skip（`code/official/FontDiffuser/src/modules/unet_blocks.py:638-655`）。还要注意 `StyleRSIUpBlock2D` 在 DCN 之后继续执行独立的 style `SpatialTransformer`（同文件 `:493-505,573-579`），而普通 block 没有这层（同文件 `:590-661`）。**推荐的 RSI-free 不是直接换成现成 plain block**，而是新建 shape-compatible `StyleUpBlockNoRSI`：保留原 resnet、style attention 与 upsample，只把 `offset head → DCN(skip)` 换成 raw skip。否则会同时删除 up-path style attention，把“去 RSI”混成更大的架构变化。

### 2.2 无 RSI 时前向能否运行

**可以，但必须构建另一种 up-block 拓扑。** `get_up_block` 已同时支持 `UpBlock2D` 与 `StyleRSIUpBlock2D`（`code/official/FontDiffuser/src/modules/unet_blocks.py:64-108`）；普通 block 直接取 down-path residual 并与 decoder hidden state 拼接（同文件 `:638-655`）。这证明 skip 在没有 offset head 时能运行，但现成 `UpBlock2D` 还会删除 style attention，不能作为严格的“RSI-only removal”。应实现上述 `StyleUpBlockNoRSI`，复用能匹配的 resnet/style-attention 权重；它会改变 state_dict 中 RSI/DCN 专属键和前向签名，需显式 variant/config，而不能在现有 RSI block 中传 `None` 就自然旁路。

Stage-A 当前 `structure_features=None` 只会构造与 content pyramid 同形的零张量（`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:51-67,110-124`），之后仍把零张量交给 RSI；这不是移除 RSI。

### 2.3 step-0 等价：当前提案不成立

需要区分三句话：

1. `structure_features=None -> zeros`：代码成立。
2. “Δ=0 -> offset=0”：对现有 FD head **不成立**。结构分支有 affine GroupNorm、带 bias 的 1×1 Conv、LayerNorm、attention/FFN，以及带 bias 的 `proj_out`（`code/official/FontDiffuser/src/modules/attention.py:278-299,301-330`）。既有审计也正确指出 E1@100k 的非零 head 在零输入时仍可能输出非零 offset（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:156-162`）。
3. “zero-init offset head -> FT 原前向”：仍然**不自动成立**。即使 `offset=0`，代码仍执行 `DeformConv2d(skip, 0)`（`code/official/FontDiffuser/src/modules/unet_blocks.py:553-563`）；零 offset 是规则采样，不是绕过 DCN，DCN 的 3×3 learned kernel 未必是 identity。而 RSI-free FT 的普通 block 是直接 concat raw skip。

要声称精确等价，推荐二选一：

- **首选：identity-safe residual RSI**：`skip_out = skip + g * (DCN(skip, offset)-skip)`，新增支路/最后投影零初始化，使 `g=0` 或 residual=0 时逐元素等于 raw skip；主文把它描述为稳定参数化，而非额外能力。PI 若反对“zero-init gate”，可固定 gate=1、但把整个 residual 输出投影 zero-init；归因上仍是新参数化，必须对 official/Δ controls 同样使用。
- **备选：严格 identity DCN init**：每个 input-output 同通道的 3×3 kernel 仅中心对角为 1、其余为 0、bias=0，同时 offset 输出严格为 0。然后用固定 batch 在 fp32 下检查每个 RSI block 的 raw skip、warped skip 和最终 noise prediction，规定 `max_abs<=1e-6`。这比显式 bypass 更脆弱。

在 parity gate 通过前，报告只能写“从 RSI-free FT 初始化并新加零 offset 头”，不能写“step0 exactly equals FT”。

### 2.4 新 FT 的预算与超参

推荐命名 **F0-RSIFREE-FT-A-S3407**（论文名 `FT-noRSI`），初始化 official P1 的可匹配权重；丢弃官方 RSI/DCN 专属键并保存 missing/unexpected-key allowlist。数据仍为 A/train228，effective batch=8×1，AdamW、fp16、clip=1、CFG=.10、`lr=1e-5`、linear、warmup=5k，与已完成 E1 的冻结 FT 配方对齐（`.cursor/rules/hrfont-execution-spec.mdc:12-18`；E1 实际 provenance 为 bs8、lr 1e-5、warmup5k，`provenance/runs/E1-FTV2-A-S3407.json:23-38`）。

**预算推荐：主终点 100k，不建议未经证据直接缩短。** E1 在 98k 才取得最佳 val loss，100k 非明显恶化（`provenance/runs/E1-FTV2-A-S3407.json:40-44,57-61`），而去掉 RSI 改变了优化问题，不能假定 25–50k 已收敛。为赶期可预注册 60k early-stop：若 50k、60k 连续两个 5k milestone 的主 val composite 改善小于预定阈值且无质量轴退化，则停在 60k；否则到 100k。主表须报告实际 endpoint，不能事后挑步数。

### 2.5 E1 如何继续承担“不是更多数据”的证据

E1 与新 FT、方法臂都使用同一 A/train228；因此“不是更多数据”应由**同数据、同训练样本协议**而非仅 E1 的存在来支撑。最小主表：

| 行 | 初始化/RSI | 作用 |
|---|---|---|
| official P1 | 原官方权重、官方 RSI | 未适配目标训练域的外部 anchor |
| E1@100k | P1→A/train228、官方 RSI | 证明仅域内 FT 能带来多少；已完成，保留不重跑 |
| FT-noRSI | P1 minus RSI→A/train228 | 新方法的直接 baseline；隔离“去 RSI/重新 FT”本身 |
| FT-noRSI+official-RSI | 新增 identity-safe official source | 控制“只加回 RSI 容量/训练”的收益 |
| FT-noRSI+Δ-RSI | 同上，仅 source=Δ | Δ 的 matched effect |
| FT-noRSI+Δ-RSI+Support | 联合最终系统 | 最终增益；需与上一行同预算 matched |

E1 vs official P1 表明同一数据域 FT 的增益；FT-noRSI vs E1 表明删除 RSI 的影响；最终系统 vs FT-noRSI 表明机制 package 在**相同数据**上增加的收益；Δ vs official source 才是机制的单变量证据。不能用“E1 数据相同”替代这些训练预算/架构 controls。

### 2.6 正在运行 E2/E2b 怎么办

两臂均从带官方 RSI 的 E1@100k 初始化，且规范要求复用同一 E1 offset head（`.cursor/rules/hrfont-execution-spec.mdc:51-53,111-112`）；新拓扑改为 RSI-free FT + 新增 identity-safe head 后，它们不再是新主线的 matched arms。

截至 13:55，E2b=27.8k、E2=9.2k，同日启动（`reports/WEEKLY_20260905.md:58-62`）。按观测 1.7/5.0 s/step，已投入约 **0.55+0.53=1.08 GPU-days**；跑完当前 seed 还需约 **1.03+4.10=5.13 GPU-days**。推荐：

- **立即保存完整 milestone/provenance 后停止，不补 seed。** 理由是从今天起主线起点、RSI 参数化和 estimand 都变了；继续消耗 5.13 GPU-days会直接挤占新 FT 和主 matched controls。
- 若新 FT 代码尚需 >2 天才能通过 parity，可让 **E2b 先自然跑完**（仅余约 1.03 GPU-days），作为 appendix 的“E1-head official source”参考；E2 余量约 4.10 GPU-days，不建议仅为 sunk cost 跑完。
- 只有在 PI 决定不采用 RSI-free 新拓扑时，才继续当前 E2/E2b 并补三 seeds。

### 2.7 新拓扑的最小 arm set；E1c 是否保留

| 新 ID（建议） | 起点 | RSI source | Support | 训练 | 主用途 |
|---|---|---|---|---|---|
| F0-RSIFREE-FT-A-S3407 | official P1 minus RSI | 无 | 无 | 100k，FT 配方 | 新 baseline |
| F1-OFFRSI-S{seed} | F0 | official `Ec(R[0])` | off | 80k | 新增 RSI 容量/官方 source control |
| F2-DELTARSI-S{seed} | F0 | Δ | off | 80k | 与 F1 matched，Δ effect |
| F3-JOINT-DS-S{seed} | F0 | Δ | on，drop=.20 | 80k（或统一 100k） | 最终联合系统；与 F2 matched |

所有 F1/F2/F3 必须共享 identity-safe RSI 参数化、ordered-R、n-shot 9-token、cache、source-drop、batch/noise/RNG、steps/lr/schedule；F1 vs F2 只改 source，F2 vs F3 只改 support treatment。若 F3 定 100k，F2 也必须 100k 才能做主 matched comparison。

**E1c 不再存活为新主线 arm。** 它从 E1@100k 的带 RSI 头出发，回答的是旧拓扑下 frozen-encoder 1-shot official-RSI continuation；新主线已有 E1 anchor、F0 和 F1。可把未启动的 E1c 删除出运行计划，在文档中标记 `superseded-before-launch`，不制造空结果。当前 E1c 本来也尚未启动（`reports/WEEKLY_20260905.md:60-62`）。

### 2.8 registry / provenance

- 新建 variant/model fingerprint，不能复用 `cn2west_ft_v2` 或 E2 run ID；记录 `parent_ckpt=official P1`、保留键 SHA、丢弃 RSI/DCN 键清单、new-key init manifest、up-block topology、代码 commit。
- cache SHA 必须绑定 F0 的 frozen Es/Ec；现规范已有 encoder SHA fail-closed（`.cursor/rules/hrfont-execution-spec.mdc:21-24,38-40`）。如果 F0 全训 Es/Ec，必须在 F0 完成后重建 cache，不能复用 E1 cache。
- 为 F0/F1/F2/F3 分别保存 canonical config SHA、dataset/ink table SHA、parent-child lineage、trainable/frozen parameter name+SHA、parity-test artifact。
- 旧 E2/E2b registry 状态写 `stopped_topology_superseded` 或 `completed_appendix_legacy_head`，不得覆盖目录或改写成新 arm（`.cursor/rules/hrfont-execution-spec.mdc:98-104`）。

## 3. Q2：截止日算术

### 3.1 旧三臂计划

按 PI 给定观测速度：E2（Δ path）约 5.0 s/step，E2b（official path）约 1.7 s/step；E1c 暂按 official path 1.7 s/step。`GPU-day = steps × sec/step / 86400`。

| 工作 | 算式 | GPU-days |
|---|---:|---:|
| E2，80k×3 seeds | 240k×5/86400 | 13.89 |
| E2b，80k×3 seeds | 240k×1.7/86400 | 4.72 |
| E1c，80k×1 seed | 80k×1.7/86400 | 1.57 |
| **合计** |  | **20.19** |

3 GPU 完美并行的理论墙钟为 **6.73 天**。若“E1c/E2/E2b ×80k”也意味着 E1c 三 seeds，则 E1c=4.72、总计 **23.33 GPU-days / 7.78 天**。观测上同一墙钟 E2b 27.8k、E2 9.2k，步速比约 3.02，与 5/1.7=2.94 一致，不能用 E2b 的速度外推 Δ 臂。

### 3.2 新融合计划

这里给出**因果上最小而不是展示上最小**的预算：F0 100k×1 seed；F1/F2/F3 各 3 seeds。F0 无 RSI 的速度尚未实测，保守暂以 1.7 s/step；F1 official 同 1.7；F2/F3 至少按 E2 的 5.0，F3 加 support 后可能更慢，因此数字是下界。

| endpoint | F0 | F1 official×3 | F2 Δ×3 | F3 joint×3 | 合计 | 3 GPU 理想墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 80k 后续臂 | 1.97 | 4.72 | 13.89 | ≥13.89 | **≥34.47 GPU-days** | **≥11.49 天** |
| 100k 后续臂 | 1.97 | 5.90 | 17.36 | ≥17.36 | **≥42.59 GPU-days** | **≥14.20 天** |

如果砍掉 F2，只跑 F1+F3，80k 下约 20.58 GPU-days，但将无法隔离 Support，联合系统的增益也无法拆为 Δ 与 support；这不推荐用于主文机制结论。

**9/25 风险判断：高。** 9/5 到 9/25 只有 20 个自然日。11.5–14.2 天是 GPU 100% 利用、无工程/预检/cache/val/故障的数学下界，只剩 5.8–8.5 天做实现、parity、生成 test16、统计、人评/图表和全文。推荐固定 80k 后续臂、F0 设置预注册 60k early-stop、先跑 seed3407 的 F1/F2/F3；只有三个方向和实现门都通过才补 3408/3409。9/12 前若 seed3407 未形成可信趋势，应回退原冻结 Stage-B 设计或缩为 F0/F1/F2，不应在 9/20 后再改 QKV 主方法。

---

## 4. Q3：实际 Q/K/V 与计算语义

### 4.1 代码事实

`OffsetRefStrucInter` 在 `unet_blocks.py` 被导入并实例化（`code/official/FontDiffuser/src/modules/unet_blocks.py:1-7,460-467`），定义实际位于 `attention.py`。它构造 `CrossAttention(query_dim=style_feat_channels, context_dim=res_skip_channels)`（`code/official/FontDiffuser/src/modules/attention.py:288-292`），forward 调用为：

```python
hidden_states = self.cross_attention(style_content_hidden_states,
                                     context=res_hidden_states)
```

（`code/official/FontDiffuser/src/modules/attention.py:301-320`）。通用 attention 再明确执行 `Q=to_q(hidden_states)`、`K=to_k(context)`、`V=to_v(context)`（同文件 `:189-193,209-215`），并计算 softmax(QKᵀ)V（同文件 `:234-242`）。

因此：

- **Q = RSI 结构条件**：官方是参考图的多尺度 `Ec(style)`；HR-Font E2 是同目标字符的 Δ；
- **K = UNet down-path skip feature**；
- **V = 同一个 UNet skip feature**；
- attention 输出的 token 数跟 Q 网格一致，再投影为 18-channel 3×3 DCN offset（`code/official/FontDiffuser/src/modules/attention.py:294-330`）。

`StyleRSIUpBlock2D` 取指定结构尺度，逐层以 `(res_hidden_states, style_content_feat)` 调 offset head，再用 offset 形变 skip（`code/official/FontDiffuser/src/modules/unet_blocks.py:545-563`）。

**审计文档没有写反。** `RSI_CORRECTNESS_REVIEW_20260905.md:83-104` 与 `WEEKLY_20260905.md:9-13` 都正确写为参考结构 Q、skip K/V。本次没有 QKV 方向上的文档—代码冲突。

### 4.2 对 Δ-RSI 是否合理

#### attention 实际表达什么

令每个 Δ 空间位置为查询，skip 的所有位置为键和值：

\[
A_{ij}=\operatorname{softmax}_j(q(\Delta_i)^Tk(r_j)/\sqrt d),\quad
h_i=\sum_j A_{ij}v(r_j),\quad offset_i=P(h_i).
\]

Q 决定“在哪个 Δ 位置提出什么匹配问题”；K 决定 skip 的哪些位置与问题相关；V 提供被读出的 noisy denoising-state 内容。它不是直接把 Δ 当 offset，也不保证局部/同坐标对应。优点是 Δ 与目标字符同字，Q 网格可表达相对中性字的局部变化；缺点是 offset 的语义位置由 Q 网格定义，而被形变的是 skip 网格，二者只靠训练学会对齐。

#### 零模式并不天然优雅

Δ=0 时，不能简单推出 Q=0：Δ 先经过 affine GroupNorm、带 bias 的 Conv、LayerNorm（`code/official/FontDiffuser/src/modules/attention.py:278-286,301-316`）。即使人为保证 Q=0，QK 分数全相等只会让 attention 对所有 skip token **均匀平均**；V 的全局均值、attention output bias、FFN 与 `proj_out` 仍可能产生非零 offset。即使 offset 被严格置零，DCN 仍不是 raw-skip identity（§2.3）。因此正确的零语义必须由 identity-safe 结构保证，不能靠 attention 的数学退化猜测。

#### 尺度问题：已有规范化，但仍有风险

“Δ 较小，所以 head 必须隐式学一个幅度放大”不完全准确：结构和 skip 在 attention 前都已经分别经过 GroupNorm、1×1 Conv、LayerNorm（`code/official/FontDiffuser/src/modules/attention.py:278-286,301-316`），绝对 RMS 大多被消除。这会让 Q=Δ 对幅度缩小较不敏感，**也可能把极小残差/缓存量化噪声归一化成显著方向**。真正要测的是归一化前后 RMS、通道方差、Q/K logit 标准差和 attention entropy，而不是盲目再叠一层 LN。

与继承 E1 非零 head 相比，Q=Δ 的分布从绝对 Ec 变成相对残差，domain shift 仍大；但 identity-safe zero-init 可让系统从 FT 行为平滑离开。没有 identity-safe 参数化时，“无 zero-init gate”使首次前向就经过随机/非匹配 DCN，转换更差。

#### 跨语系边界

官方 source 为汉字 Ec 时，Q 的笔画/部件位置与拉丁目标 skip 的局部对应可能弱；全局 attention 可以学习“某类横/竖/闭合区域与某类 skip 区域相关”，但无法凭 Q/K 恢复输入中不存在的信息，例如目标字体未由 ref8、近邻库或 support 约束的独特 Latin serif 形状。Attention 只重组 V 中已有的 skip 表征，并由 Q 调制 offset；它不能信息论上识别多个与同一汉字 ref8 完全一致、但 Latin 字形不同的字体。Δ 的同字构造改善字符坐标与候选先验，却仍受 top-10 库凸组合/编码器表示能力限制。

## 5. Q3：优化方案评估与排序

评分：质量潜力 1–5；因果风险 1–5（越高越危险）；成本 1–5（越高越贵）。所有改变 RSI 内部 QKV/offset 参数化的方案都必须让 official-source 与 Δ-source matched arms 共用同一实现，否则需重跑 matched pair。

| 排名 | 方案 | 质量潜力 | 因果风险 | 成本 | 代码改动与判断 | 放置 |
|---:|---|---:|---:|---:|---|---|
| 1 | **identity-safe offset residual / bypass** | 4 | 1 | 2 | 改 `StyleRSIUpBlock2D.forward` 的 DCN 输出组合，并统一 new-head init；解决新 FT step0 硬问题，不改变 Δ vs official 的 treatment，只要两臂共用 | 主方法基础设施；F1/F2/F3 全重跑 |
| 2 | **Q=skip，K/V=Δ** | 4 | 3 | 2 | 在 `OffsetRefStrucInter.forward` 对调 query/context，并处理输出网格：Q 长度变为 skip 网格，offset 天然落在被形变位置。语义变为“当前 denoising state 哪里需要校正、从 Δ 读取什么” | 20k/S3407 筛选；若胜出，作为主方法并重跑 F1/F2 matched |
| 3 | **Δ additive modulation of K/V** | 4 | 2 | 3 | 保留 Q/K/V 主路，增加尺度对齐投影 `K'=K+P_k(Δ), V'=V+P_v(Δ)`，投影 zero-init；或在 role-swap 后调制 skip Q。比完全改 attention 更连续 | 主方法候选或强消融；若启用主方法需重跑 matched pair |
| 4 | **归一化/温度校准** | 3 | 1 | 1 | 不先加重复 LN；先记录 pre/post norm RMS、logit std、entropy。若异常，再用 RMS clamp、learned positive temperature 或 Δ norm floor。修改 `attention.py:301-320` 前处理 | 可独立短跑消融；最终启用需 matched pair |
| 5 | **K=Δ-layout / V=Δ-content split** | 4 | 4 | 4 | 合理版本应是 Q=skip、K=Δ spatial layout、V=独立 Δ content/style readout；若 K/V 只是同一 Δ 的两个线性投影，和普通 cross-attn区别有限。需新增两路 feature contract/cache/projection | 附录研究，不进 9/25 主线 |
| 6 | **增加 RSI 尺度** | 3 | 3 | 4 | 当前官方 build 仅两个 RSI up blocks（`code/official/FontDiffuser/src/build.py:19-22`），对应现审计的 24²/48² 两尺度；加低分辨率 block 会新增多组 head/DCN、算力和 matched 变量 | 先做无训练/20k定位；附录 |
| 7 | **坐标/位置编码** | 3 | 3 | 2 | 给 24²/48² Δ 与 skip token加共享 normalized 2D coordinates 或相对位置 bias；注意 3×3 是 Es style token，不是 RSI Δ 网格，二者不可混称。可减轻全局注意力失去坐标的问题，也可能压制必要的跨位置匹配 | 附录 20k；启用需 matched pair |
| 8 | **post-hoc learned gate** | 2 | 3 | 1 | 若训练后才加 gate 而不重训，只能缩放既有 offset，校准范围有限；若训练中学习，就是新参数并改变优化。固定 scalar sweep 可作诊断，不应偷渡为主方法 | 附录诊断 |

### 5.1 对各方案的具体答复

**(a) 交换 Q/K/V。** 这是最值得短跑的架构反事实。Q=skip 使每个被形变位置主动查询 Δ，offset 输出网格与 skip 天然同长；解释更接近“denoising state 哪里需要修正”。风险是 skip 随 timestep/noise 变化，query 分布更动态，K/V=Δ 的信息容量又受近邻残差限制。需要改 `attention.py:318-320` 的调用方向及构造时 query/context dims（`:288-292`），并验证通道、空间尺寸。

**(b) K=layout/V=content。** 只有 K、V 来自语义不同的特征才有真正增益；把同一 Δ 过两个 Linear 本来就是当前 attention 的标准行为。建议 layout 取归一化 Δ/edge-like feature，V 取未减或分解后的同字 Ec 变化，但这会改变“只用 Δ residual”的叙事并增加 cache，截止日前不宜主推。

**(c) additive modulation。** 可让 Δ 不仅提出 query，也直接改变被检索的键/值；zero-init projection 可保持初始 attention 与 control 一致。它仍是架构创新，必须在 official source control 中使用同构 projection，或明确只作为 Δ package ablation。

**(d) scale normalization。** 已有 GN+LN，所以先测后改。推荐记录每层 `RMS(Δ)`、`RMS(skip)`、norm 后 RMS、QK logit std、entropy、offset RMS；若 Δ 近零导致 norm 放大噪声，加 `x / max(rms(x), floor)` 或 norm bypass-on-zero，而非再堆 LN。

**(e) 多尺度。** 24²与48²覆盖局部笔画/边缘，作为第一版足够且与 FD 官方两 RSI blocks 一致；更低分辨率可能帮助全局字宽、倾斜、比例，但也可能让 DCN 在粗尺度过度移动身份结构。先根据 offset 与 `|GT-B0|` 定位、按尺度增益判断，不在主线盲加。

**(f) gate。** PI 先前拒绝 zero-init gate 的归因担忧，在“从 RSI-free FT 新增结构”后需要重审：gate 的主要价值现在是**定义可验证的 identity 起点**，不是制造额外能力。建议使用固定结构性的 residual zero-init，并让所有 RSI arms 共用；不建议仅 Δ 臂拥有 learned gate。训练后 scalar gate sweep可以独立做诊断，但不能挽救未经训练的错误 QKV。

**(g) 位置编码。** RSI Q/K 当前没有显式坐标，cross-attention 可全局匹配。共享 2D Fourier/relative bias 能编码“同坐标优先但允许错位”；需在两个尺度分别生成，不能误把 Es 的 3×3 style tokens当 Δ tokens。它改变匹配先验，放附录。

**(h) 论文先验。** 项目现有审计中可安全依赖的先验是：DG-Font 用与目标字符相关的低层特征做 deformable skip，并对 offset/mask predictor zero-init；这支持“目标字符条件形变”和稳定初始化，但不直接证明某个 QKV 排列（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:11-29,41-49`）。FontDiffuser 明确支持 reference-as-Q 的 RSI 与 cross-attention 处理错位，且 Phase 2 只是增加 SCR 监督，不是改变 QKV（同文件 `:29-39,129-137`；官方 README `code/official/FontDiffuser/README.md:143-152`）。CF-Font 的 component/prototype 思路可支持局部/部件先验或 medoid 检索，但不能作为 role-swap 的直接证据；FontStudio 的多参考/扩散先验同样不能直接证明 QKV 方向。故主文应把 role-swap、位置编码等写成受控设计选择，不写成“已有论文已证明”。

网络检索在本次审查中连接失败，因此上述 known-paper 边界仅采用仓库内已记录的一手论文/官方代码审计，不新增未经核对的具体 paper claim。

### 5.2 主方法 vs 附录

**主方法必须有：**

1. identity-safe RSI 参数化与 step0 parity gate；
2. 两尺度 Δ（保持官方 RSI 插入位置）；
3. 现有 GN/Conv/LN，先加诊断而非重复 norm；
4. F1 official-source vs F2 Δ-source matched pair；
5. 若融合 support，则 F2 vs F3 同预算 matched pair。

**只在 20k seed3407 胜出后才进入主方法：** Q=skip/KV=Δ role-swap 或 additive K/V modulation，二选一，避免同时改变多个因素。胜者一旦进入主方法，F1/F2/F3 必须从共同 F0 重跑；不能拿新 QKV 的单臂与旧 QKV control 比。

**附录：** K/V split、额外低分辨率尺度、2D position bias、post-hoc gate sweep、K=3 vs 10、wrong-char/wrong-style/spatial-shuffle/RMS-matched Δ。纯诊断（attention entropy、offset localization、固定 checkpoint 的 gate sweep）可 standalone；任何可学习的 QKV、尺度、位置或 gate 改动都需要 matched retraining。

## 6. 文档与代码冲突/口径纠正

1. **没有 QKV 方向冲突：** 审计/周报的“结构 Q、skip K/V”与代码一致（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:83-104`；`code/official/FontDiffuser/src/modules/attention.py:301-320`）。
2. **“Δ=0 就是无形变/FT 行为”若出现在旧 rationale 中，应作废。** 现审计已经正确警告零 Δ 不等于零 offset（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:156-162`）；本报告进一步指出，即使零 offset，当前 DCN 也不等于 plain skip。
3. **编号冲突：** 现 E3 是评测，Stage B 是 E5。PI 的“融合 E2+E3”必须在下一版决策文件中明确改写为“融合 Stage A+Stage B（旧 E2+E5）”或重新统一编号。
4. **“RSI 是 standalone offset head”表述不完整：** 实际 treatment 包含 offset interpreter、DCN 及被替换的 skip pathway（`code/official/FontDiffuser/src/modules/unet_blocks.py:423-477,545-563`）。

## 7. 必须 PI 决策清单

- **D-F1｜融合对象**：A. 现编号 E2+E3 流水线；B. Stage A+Stage B（现 E2+E5）。**推荐 B**，并立即统一命名。
- **D-F2｜融合策略**：A. 原冻结两阶段；B. joint + no-support matched control；C. 只跑 joint。**推荐 B**；若时间进一步恶化，回退 A，不选 C。
- **D-F3｜RSI-free 定义**：A. 新建 `StyleUpBlockNoRSI`，保留 resnet/style attention、只旁路 offset+DCN；B. 直接换现成 `UpBlock2D`；C. 只把 structure 置零。**推荐 A**；B 额外删 style attention，C 仍在运行 RSI/DCN。
- **D-F4｜step0 契约**：A. 显式 identity-safe residual/bypass；B. identity-DCN init；C. 仅 zero-init offset head。**推荐 A**；C 不满足精确等价。
- **D-F5｜F0 预算**：A. 固定100k；B. 60k预注册 early-stop、最多100k；C. 固定短训。**推荐 B**。
- **D-F6｜后续 endpoint**：A. 80k；B. 100k。**推荐 A**，且所有 matched arms 完全同 endpoint。
- **D-F7｜旧 E2/E2b**：A. 立即保存并停；B. E2b跑完、E2停；C. 全跑完并补 seeds。**推荐 A**；若新代码预计阻塞>2天，选 B。
- **D-F8｜E1c**：A. 保留主臂；B. 改 appendix；C. superseded-before-launch。**推荐 C**，E1@100k 本身保留主表 anchor。
- **D-F9｜最小新臂**：A. F0+F1+F2+F3；B. F0+F1+F3；C. F0+F3。**推荐 A**；B/C 无法拆 Δ/support。
- **D-F10｜QKV 主线**：A. 保留 Q=Δ、K/V=skip；B. 先做 role-swap 20k 筛选；C. 直接切 role-swap。**推荐 B**；预注册选择指标，禁止看 test16。
- **D-F11｜Δ 调制**：A. 暂不加；B. additive K/V 作为20k第二候选；C. 与 role-swap同时加入。**推荐 B**，且只允许一个候选胜出进主方法，不选 C。
- **D-F12｜归一化**：A. 直接再加 LN；B. 先记录 RMS/logit/entropy，再决定 RMS floor/temperature。**推荐 B**。
- **D-F13｜gate**：A. 所有 arms 共用 identity-safe residual init；B. 仅 Δ 臂 learned gate；C. 无 identity 机制。**推荐 A**，它是初始化契约而非 Δ 特权。
- **D-F14｜9/25 范围**：A. 80k三主臂、先3407后补 seeds；B. 100k全臂+全消融。**推荐 A**；B 的数学下界已占 14.2 天 GPU 墙钟。

## 8. 最终建议

批准“融合”但改成下列决策句：**从 official P1 构建 RSI-free、A/train228 的 F0；用 identity-safe RSI 从 F0 同起点训练 F1 official-source、F2 Δ-source、F3 Δ+Support joint 三臂，F1/F2 与 F2/F3 分别 matched；后续统一 80k，F0 60k early-stop/100k cap；旧 E2/E2b 保存后停止，E1c 不启动；QKV role-swap 只做 20k seed3407 预注册筛选。**

这保留了 PI 想要的两点——减少串行阶段、避免继承 E1 的非零 RSI 头——同时不牺牲“不是更多数据”和“Δ/Support 各自贡献”的最低因果证据。唯一不可让步的是：在当前 DCN 结构下，不能把“zero-init head”直接写成“step0 等价 RSI-free FT”；必须先把 identity 契约做成代码事实。
