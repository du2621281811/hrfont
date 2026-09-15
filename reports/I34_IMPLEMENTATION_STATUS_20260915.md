# I3/I4/E12-c 实施状态

更新：2026-09-16。本页替代上一版“组件已测、入口未实现”的状态。正式启动回执随后独立记录，不以计划或巡检代替真实派发。

## 已完成

- 独立 I3/I4 trainer、实际144-token墨迹监督、输出细节loss、冻结校准难例采样，以及两协议推理/归档入口均已实现并部署至 `/root/projects/hrfont_i34_20260915.bJikwQ`；旧I快照未修改。
- 223个有效训练字体、131,803张native96 PNG全量解码/SHA检查；541个字体—语种组分桶与36组面板实际复核完成，清洗表不变。
- I3真实8卡80步固定batch loss约0.165→0.108；80→84恢复成功，无重放，全部rank AMP skips=0、DDP spread=0，reader/encoder/router/token projection/ink readout梯度非零。
- I4真实8卡10步中文辅助路径通过，主/辅loss有限，全部rank无跳步或参数分歧。
- 7项I34组件测试；E12-c集合/掩码/多正例/BN及AUC测试已落地。预检发现执行环境无sklearn，已改为等价average-rank AUC并补并列分数测试，不修改评测定义。
- E12-c按178/22/23字体实例划分，主生成val/test排除；已知家族接口保留，本次未知家族明确记录，不阻塞已批准实验。
- I2 matched4888与I0 formal4096完成并取回本地；I2 6395登记文件SHA通过，I0 4352预测/GT PNG解码、4372文件登记。
- PI授权新checkpoint存储于 `/root/data1/hrfont_i34_20260915`，仅链接新产物，不移动/删除旧模型；根盘与checkpoint盘均保留10GiB安全线。

## 正在收尾的准入项

E12-c A20+B20短流程和实际DPM/重复ref测试仍在执行；通过后产生绑定代码/数据SHA的 `PREFLIGHT_PASSED.json`。随后Git同步旧结果、生成回执、dry-run，再启动可执行队列并检查I3真实更新。此处不把尚在运行的预检写成全部通过。

详细实现、路径、命令和恢复约束见 [执行手册](I34_EXECUTION_RUNBOOK_20260916.md)。正式执行顺序仍是 I3 → E12-c → I4；两生成器各10k，不继承预检或I1/I2权重。论文及Overleaf本次未修改。
