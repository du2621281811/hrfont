# K4 字重排除规则修订与重启

快照：2026-09-18T02:03:34.969402+08:00。源码提交：`779de670cf460f44e6b3e938dd8af22b27178755`。

## 排除边界

alpha 归一化、top-k 之前排除目标自身，以及同一设计的显式不同字重；不根据别名、编码版本、斜体或视觉相似度扩展合并。查询字体不在训练 donor bank 中时也按字重组排除。GT 字符及参考字符的可用性掩码仍然生效；未知字体或无可用 donor 时中止，不回退到泄露候选。

覆盖 492 个字体、402 个组，其中 62 个多成员组包含 152 个字体。这不是删除 152 个字体，而是逐查询排除对应字重。`weight_membership.tsv` 列出每个字体的排除对象。

以下过宽关系已从 alpha 排除中解除：FZLTHProGBK_H/FZLTHProJW_H、FZSongYJW/FZSongYuanK、fzsj_1275635/fzsj_2017088，以及 FZYouHJW/FZYouHK 两套版本之间的合并；各版本内部的不同字重仍排除。字重表与辅助任务的 aliases/lineage 表分开，后者未改。0913/0917/v2 划分未改。

## 重启与配方

旧宽泛规则 K4-A 在 714 步安全保存并停止，旧 C/B 未启动；旧结果保留，不能混入新规则结果。所有新运行使用新目录、重新从约定 parent 初始化优化器与调度，不接续旧 K4 状态。

| 顺序 | 实验 | 数据 | 初始化 | 配方 | 更新数 |
|---|---|---|---|---|---|
| 1 | K4-A-K1FT-0917-WEIGHT-S3407 | 0917 train | 原 K1 final EMA | K1 增训 | 10000 |
| 2 | K4-C-K1RECIPE-V2-WEIGHT-S3407 | v2 train | K0 | K1 原配方 | 10000 |
| 3 | K4-B-K3RECIPE-V2-WEIGHT-S3407 | v2 train | K0 | K3 风格监督配方 | 10000 |

8 V100，global batch 64，seed 3407。原 K1 历史训练没有此过滤，因此 A 仅能确保本次新增阶段的 donor 不含已识别字重；不宣称抹去父模型历史信息。

## 验证

- 分组正反例通过：不同字重被屏蔽，版本/别名/斜体独立设计仍可选。
- 强制 top-k=1，最高相似度为被禁字重时，仍正确返回合法 donor；topk/soft/threshold 三种模式通过。
- v2 全部 train/val/test、0917 train 的合法查询按 1/2/4/8 shot 检查无空候选。
- v2 实际 alpha 4172 次，0917 1468 次，禁止字重 alpha 总质量均为 0。
- K0/K1 条件路径数值检查、A 完整 EMA 热启动检查通过。
- C 120 步与 100→120 全状态恢复通过；首两次恢复 loss 差为 {'101': 0.0, '102': 5.811452865600586e-07}，最终状态相对 L2 为 0.00010980。
- B/C 前 17 步主训练采样一致，B 辅助 rollout 梯度及第 17 步路由梯度通过；8 rank 无 AMP skip。

## 当前状态

K4-A 已正式运行至 100/10000；8 rank global batch 64、AMP skips=0、DDP spread=0、禁止 donor 违规数=0。K4-C、K4-B 排队。预检通过表示训练实现健康，不等于已验证生成质量。

训练完成后继续 v2 全量 val/test 推理和分 shot/script/difficulty 指标；K0 仅 1shot，其他 K 族 1/2/4/8shot。Delta 条件输出按新规则重算；只复用协议与来源匹配的资产。

运行代码：`/root/projects/hrfont_k4_weight_20260918`；队列：`/root/data1/hrfont_k4_weight_20260918/control`。磁盘快照余量 30.78 GiB；训练原有低空间保护继续生效。
