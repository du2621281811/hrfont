# K4 同家族 donor 排除与重训

**已确认旧实现存在同家族条件泄露路径，已修复并从原定父权重重新开始 K4，顺序为 K4-A → K4-C → K4-B，每组 10,000 个成功更新。**

2026-09-18T01:32:38.900633+08:00 核验：新 K4-A 已到 **100/10,000**，8 张 V100、global batch 64、AMP 跳步 0、同步检查差值 0.0；TC beta 保持 0.8。当前进程观察到同家族 donor 违规次数为 **0**。这里只确认过滤和训练健康，尚无重训后的生成质量结论。

## 发现了什么

旧 `Library.select` 只排除当前 font，没有排除同家族的其他字重。虽然辅助配对与报告里已有 alias 分组，这份分组没有被应用到 alpha 候选过滤；其中 3 组旧字体字重还未合并。

| 审计范围 | 查询数 | 选中同家族 donor | 条件未 dropout 的查询 | 其中实际带同家族 donor |
|---|---:|---:|---:|---:|
| 旧 K4-C 实际前 40 步的确定性回放 | 2,560 | 510 | 2,358 | **471** |
| 旧 A 配方前 16 步的反事实采样；当时尚未运行 A | 1,024 | 252 | 945 | 231 |

旧 C 的问题涉及 80 个目标字体。471/2,358 约为 19.97%，这是审计窗口的比例，不能外推为完整 6,265 步的精确比例。回放额外核对了前两步、全部 8 卡的 font/char/refs/noise/timestep/CFG/source SHA，与实际日志 **16/16 一致**。

例如旧回放中 `FZLuoMXTJW-EB` 的 `u0059` 选中了 `FZLuoMXTJW-M`，alpha 约 0.1011；`FZLuoMTJW-L` 的 `u0044` 选中了 `FZLuoMTJW-R`，alpha 约 0.1019，且条件均未 dropout。这是实际参与 Delta 条件构造的正权重，并非只有候选表中存在这些字体。

旧 C 在 **6,265 步**安全保存并停止，保留完整恢复状态及证据，标记为 `SUPERSEDED_FAMILY_LEAKAGE`；它不再作为修复版继续训练。旧 B/A 尚未启动。旧队列曾在子进程安全退出后先写阶段标记再检查 DONE，因此留下一个不能代表完成 10k 的阶段标记；STOPPED.json 和退役说明记录真实步数。新队列已改为核验正式 DONE 后才写完成标记。

## 修复规则

对于目标字体 q、目标字符 c、中文参考集合 R，先构造合法候选：

`D(q,c,R) = {d ∈ train : d 有合法 GT(c)，d 拥有 R 中全部参考字，family(d) ≠ family(q)}`

再仅在 D 内计算 alpha 的归一化、top-k/截断以及 Delta。保留原相似度、温度、top-k、损失、采样和学习率；不会先选同家族字体再把其权重置零。val/test 查询即使不在训练库中，也会排除训练库里该查询的所有已知家族成员。未知家族、过滤后空库或最终选中同家族 donor 都直接报错，不降级到自体或缺字替代。

家族清单覆盖 **492 个字体、398 个已知家族**；64 个多成员家族包含 158 个字体。依据包括既有别名/数据分组、manifest family、明确字重后缀和 TTF typographic family name 16，记录了每条合并依据。补齐的三组是：

- `FZSiNTJW-H / -UB / -UL`
- `FZXinZYHJW_EB / _UL`
- `FZYouSJJW_507R / _508R / _509R`

规则统一覆盖 K4 主训练、K3 辅助完整生成、Val192 监控和新版 K 族最终推理。辅助配对和 lineage 统计也使用同一份更新后的家族映射。family manifest 的 SHA 进入数据/运行/推理身份，修复前状态不能混入新 run 续训。训练和推理日志记录家族过滤计数与违规次数。

[完整家族表](family_membership.tsv) · [带来源的家族清单](family_groups.json) · [旧回放明细](OLD_ALPHA_SUMMARY.json)

## 验证结果

| 检查 | 结果 |
|---|---|
| v2 train/val/test 全量合法 pair × 1/2/4/8-shot 候选覆盖 | 441,580 条检查通过，最少 164 个合法候选 |
| 0917 train 全量合法 pair × 1/2/4/8-shot 候选覆盖 | 158,188 条检查通过，最少 42 个合法候选 |
| 实际 alpha 计算，按 font/script/shot 覆盖 | v2 4,172 条＋0917 1,468 条，同家族总权重均为 0 |
| 任意训练参考集合的保守检查 | 所有训练 pair 排除家族后仍有完整 338 字参考候选；v2 最少 164，0917 最少 42 |
| 单元回归 | train 字体、库外 holdout 字体、缺参考/GT、未知家族、空候选均按规定处理 |
| 过滤时机 | 同家族相似度最高且 top-k/截断预算为 1 时，topk/soft/threshold 都返回合法异家族 donor |
| 旧适配器对齐 | K0 通过；K1 在新家族规则不应改变候选的查询上通过；重叠家族查询本来就应改变 |
| 完整续训 | 新代码 C 连续 120 步，与第 100 步恢复到 120 步比较；首两步 loss 差 {'101': 0.0, '102': 0.0}，最终状态相对 L2 0.000142 |
| B/C 主采样 | 8 卡前 17 步 episode/noise/drop 身份一致；辅助 rollout 完整梯度通过 |
| A 起点与目标 | 完整 K1 最终 EMA 一致；成熟 TC beta=0.8；新 optimizer/scaler/LR 周期 |
| 正式训练 | 新 A 已完成上述 100 步，8 卡健康、同家族违规 0 |

数据清单、旧特征缓存 SHA、10k 采样配额均重检通过。0913 与 0917 的 train/val/test 划分没有改变；旧缓存只读复用。此次修改了训练路径，所以八卡、续训和 Val192 预检都用新代码重新执行，未沿用旧训练预检来代替。

## 重训与后续评估

| 顺序 | run | 起点 | 训练数据 | 预算 |
|---|---|---|---|---:|
| 1 | K4-A-K1FT-0917-FAMILY-S3407 | 原正式 K1 最终完整 EMA | 0917 train；同家族排除 | 新增 10k |
| 2 | K4-C-K1RECIPE-V2-FAMILY-S3407 | 原共同 K0 | v2 train；同家族排除 | 10k |
| 3 | K4-B-K3RECIPE-V2-FAMILY-S3407 | 同 C 的 K0 | 同 C，加原 K3 辅助监督 | 10k |

**A 的解释边界：它保留用户原定的 K1 起点。新过滤阻止本轮增训/推理继续读取同家族 donor，不能清除原 K1 历史训练可能已吸收的信息。** 因而 A 是“原 K1 的过滤后增训”；C/B 从无 Delta 的 K0 开始，是本轮较干净的过滤后对照。旧 K1/K3 只作为历史 baseline，不重新包装成从训练开始就使用家族过滤的模型。

0913 固定划分中已有的跨 split 家族仍保留。新的 donor 过滤解决直接检索同家族 GT 的条件路径，但不能把原有 train/test 亲缘关系宣称为消失；最终继续给出剔除已知重叠家族的敏感性表。元数据映射也不等于证明所有视觉相关字体都已识别。

训练完成后自动执行 v2 全量 val/test，K0 仅 1-shot；K1/K3/新 K4-A/C/B 各 1/2/4/8-shot，共 302,799 张结果。K0 的 2,336 张严格匹配图仍可复用；带 Delta 模型全部按新家族规则计算，旧预测不混用。继续按来源、shot、script、难度统计，E12-c 仍为辅助相似度。

## 运行位置与版本

- 执行 commit：`de79498f7e59ac6c8e6d5f724853c7b5f07a034e`；独立冻结目录 `/root/projects/hrfont_k4_family_20260918`。
- 新队列 PID：`3860274`；正式 A torchrun PID：`3863030`。
- 队列状态与日志：`/root/data1/hrfont_k4_family_20260918/control/status.json`、`queue.log`。
- checkpoint / 推理 / 指标：`/root/data1/hrfont_k4_family_20260918/`。
- 本次快照 data1 可用 38.06 GiB，根盘 8.21 GiB。旧实验、数据与权重保留。
- 队列 `control/STOP` 在阶段边界停止；当前 run 的 `STOP` 在成功更新边界保存后停止。任何异常会停队列并留下日志。

[正式运行凭据](evidence/LIVE_RESTART.json) · [家族过滤检查](evidence/FAMILY_POLICY_PASSED.json) · [八卡与续训](evidence/TRAINING_CHECKS_PASSED.json) · [回放身份核验](evidence/REPLAY_IDENTITY_PASSED.json) · [top-1 回归](evidence/FAMILY_TOP1_PASSED.json)
