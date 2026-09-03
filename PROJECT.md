# HR-Font / ICLR 2027 项目台账

> 唯一内部入口。计划、状态、决策和结果只更新本文件；已有报告视为历史证据，不再新增同类计划文档。
> 工作区：`/root/projects/hrfont`（从 `font_crosslingual` 抽出的中→西专用仓）。
> 实验/结果速查与防误用：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。
> 官方 vs 我们：[`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md)。
> 合作者了解官方设置、数据、渲染、Loss和历史FT来源时，统一阅读 [`COLLABORATOR_GUIDE.md`](COLLABORATOR_GUIDE.md)。

## 当前状态

- 阶段：Stage A 最小因果验证双臂10k训练、全量评测、配对统计与定性图集均已完成。
- 结论：Delta有统计显著的正向信号，但未达到预设的0.002效应阈值，判定 `INCONCLUSIVE`；不进入Stage B。
- 核心问题：在严格同数据、同初始化、同预算下，`RSI ← α特征Δ` 是否优于 `RSI ← Ec(永)`。
- 暂停项：Stage B、RS-Gap、Support、4/8-shot、256 分辨率、完整 Plan v2。
- 历史结果仅作背景：ft=0.0810；pilot A=0.0870；pilot B=0.0964。pilot 协议不能证明新方案有效。

## 已锁定的最小实验

1. 复用 `data/fontdiffuser/train/`，不重渲数据。
2. 严格 1-shot：Style、α 和对照均只使用“永”。
3. 两臂均从 `ft_cnstyle@25k` 初始化；encoder 冻结；RSI 同样重初始化；UNet 同配置训练 10k。
4. Control：`RSI ← Ec(永)`。
5. Delta：`RSI ← Σ top3 α·Ec(其他训练字体同字 c) − Ec(DejaVu c)`；当前字体 leave-one-out；禁止像素混。
6. 两臂共用数据顺序、seed、loss、dropout、优化器和统一评测。

## 通过标准

- Delta 相对 Control 的 L1 至少降低 0.002；
- 配对 bootstrap 95% CI 不跨 0；
- write accuracy 下降不超过 1 个百分点；
- 976 个配对样本中 Delta 胜率超过 50%。

通过后才运行多 seed / 长训练；未通过则停止 Stage B，先检查 α 与 Δ。

## Stage A MVP 结果

- 全量976对：Control L1=0.08255，Delta L1=0.08099，改善0.00155。
- 配对95% CI（Delta-Control）=[-0.00235, -0.00079]；Delta胜率54.71%。
- 3项数值标准通过2项：CI与胜率通过，绝对改善未达到0.002；write accuracy尚未测。
- 字体间异质性明显：FZJing、FZLing、FZCuan改善0.0040–0.0057；FZChuang退化0.0052，说明当前α/Δ并非稳定增益。
- 定性图显示多数差异是局部轮廓和粗细调整，没有形成一致、显著的结构跃迁；部分异常字形两臂均未解决。
- 扩展人眼分析页：`reports/hrfont_stagea_mvp_visual/index.html`，含168组配对结果、筛选排序和本地人工标注导出。
- 决策：保留核心假设，但先诊断top3 α检索及字体退化原因；不直接扩展Stage B或多seed。

## 解释边界

- 当前是最小因果早筛：P1、batch=1、AMP、恒定 LR、两臂同为 25% structure dropout；它能比较 RSI 条件源，但不是最终“官方训练配置复现”。
- 若数值通过，投稿前须用官方 batch/LR scheduler/全训练字符配置复验，并补统一的拉丁 write accuracy；当前比较脚本不会把缺失的 write 指标伪装成已通过。
- 若运行中断，先核验两臂共同进度与 RNG/数据序列；不得将不对称续跑直接写成严格配对结论。

## 实现边界

- 新实验使用新入口，不修改已绑定历史结果的训练/评测脚本。
- 官方 `FontDiffuser` 的 UNet、MCA、RSI、扩散损失保持不变。
- 不覆盖已有数据、checkpoint、metrics 或报告。
- 未收到明确开训指令前，不启动长训练或扩展实验。

## 实验登记

- `A-MVP-CONTROL`：10k completed；全量L1=0.08255，seed=20260902。
- `A-MVP-DELTA`：10k completed；全量L1=0.08099，seed=20260902。

每次实验只在对应run目录保留配置、状态、指标和checkpoint；结论回填本文件，不新建总结文档。

## 版本与恢复

- 根项目Git管理自研代码、规则、台账和机器来源清单；旧报告已在首个基线提交中归档，之后不跟踪动态状态文件。
- `code/ours/FontDiffuser` 独立Git分支 `hrfont/local-patches-20260903` 保存本地补丁，当前恢复基线为 `99e42b5`。
- 数据指纹：`provenance/datasets/fontdiffuser42-cnstyle-v1.json`；实验来源：`provenance/runs/`。
- `FT-CNSTYLE-25K` 标记为 `retro_partial`：数据、命令、配置和权重可验证，但2026-08-07未提交的源码状态无法精确恢复。
- Stage A训练源码可恢复；评测脚本是在根Git建立前修改的，仅保留当时SHA256，因此两臂来源清单也诚实标记为 `retro_partial`。
- 今后正式实验只允许从两个干净Git工作树启动，并自动生成一个来源清单；不再人工维护逐文件哈希。

## 精简变更记录

- 2026-09-02：新增独立 MVP 训练/评测入口；双臂 2-step + 2-sample 冒烟通过。修复 DPM CFG batch=2 时 Δ 未按样本扩展/清零的问题。
- 2026-09-02：启动本地无网络 watchdog、5分钟状态检查和完成后自动评测/配对统计；锁定实验代码 SHA256。
- 2026-09-02：将完整 Plan v2 收缩为 Stage A 两臂最小验证；建立单一台账与变更护栏。
- 2026-09-03：双臂训练均完成 10k。Control 全量评测完成；Delta 评测发现空格无 donor，明确采用“无候选则 Δ=0”后恢复评测。该修复仅影响评测中 8/976 个空格样本，不改模型权重。
- 2026-09-03：完成976对全量统计与168对代表字符定性图集；Delta改善0.00155且CI显著，但未达预设效应阈值，结论为 `INCONCLUSIVE`。
- 2026-09-03：建立根项目Git与FontDiffuser独立补丁提交；冻结数据集指纹，补建ft25k和Stage A来源清单。旧实验缺失信息明确标为部分可追溯，不再用推测补齐。
- 2026-09-03：迁入 `/root/projects/hrfont`；后续中→西 / Stage A·B 仅在此目录开展。
