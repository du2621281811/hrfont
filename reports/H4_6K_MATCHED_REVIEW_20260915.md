# H3/H4 6k消融：补全监督有小幅收益，仍未超过H0

巡检时间：北京时间2026-09-15约10:25。H4已核实到6500个成功optimizer updates；本报告的质量比较全部使用6000步EMA，不能混为最终10k结果。

## 训练与执行核验

- 队列PID3219789、H4 launcher PID3317140正常；八个rank均记录step6500/attempt6500，样本分配不同。
- 八卡均有实际GPU工作负载。rank峰值分配显存约4.8GiB；nvidia-smi进程/上下文占用约7.5GiB，二者统计口径不同。
- reader梯度0.0091153、全局梯度范数0.34746、DDP参数指纹差0、AMP skips0。
- 6500步继承参数LR1.6290067e-5，新参数LR8.1450336e-5，符合500warmup、5k后cosine的批准配方；gate=1，H4 comp_coef=0。
- H4记录的completion约0.296是未加权诊断值，其权重为0，不代表训练误用了补全监督。
- 4个运行代码文件、cache manifest、共同父unet权重及clean-map聚合SHA均与config.json逐一匹配；未热改运行快照。
- H3/H0均已核实DONE：10000步且inference_complete=true。H4的6k推理DONE为192图，用时50.69秒。

## 同步数配对结果

H0/H3/H4的192个episode逐一核对font、cp、group、k，以及refs和noise seed完全一致。统一EMA、DPM++20、CFG7.5。

| shot | H0 L1↓ | H3 L1↓ | H4 L1↓ | H3 SSIM↑ | H4 SSIM↑ | H3 edge L1↓ | H4 edge L1↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .060356 | .061090 | .061897 | .797961 | .794434 | .037890 | .038398 |
| 4 | .058406 | .059775 | .061318 | .802981 | .798036 | .037718 | .038078 |
| 8 | .058430 | .059584 | .060604 | .802837 | .799517 | .037915 | .037972 |

- H3相对H4平均L1降低1.30%/2.52%/1.68%；112/192个episode的L1较低。
- H3的SSIM和edge L1在三个shot均略优于H4，但H0的平均L1/SSIM/edge仍全部领先。
- 该结果支持当前网络中补全监督带来小幅收益，不支持完整H3已经优于原global9路线，也不能将像素指标解释成已解决笔触/特效。

## 图像检查

实际查看H3与H4各两张拼图，覆盖固定16个字体的A/g、GT和k=1/4/8；并非声称逐张视觉检查了全部192张。

- FZBangSKKXJW空心描边A/g仍被生成普通实线字形。
- FZKuaiHTJW_Te在4/8-shot出现更接近目标的方形g；FZJunYTJW-H等普通g也出现错误方化，局部几何响应并非总是正确。
- FZSuHJW的装饰切分没有恢复；FZZanMTJW_Te的g在H3中较完整，但装饰A仍不匹配。
- 未见整体崩坏。继续原批准H4至10k及4096张最终推理，之后按原队列进入H2；不在本轮修改loss、加步数或加入新研究方案。

## 存储维护

清理前磁盘约24GiB可用。先运行白名单脚本预览，再使用已批准的`--apply`删除：

- `/root/projects/hrfont/runs/H3-V0915-S3407/state_step_9500`
- `/root/projects/hrfont/runs/H0-V0915-S3407/state_step_9500`

两目录共2,539,474,222字节，均属于训练和推理已DONE的H臂。删除不可直接撤销；保留各臂最终完整恢复状态、全部推理milestone、图像、日志与指标。清理后df显示约27GiB可用，未删除旧G、数据集或合作者文件。后续仍需关注整轮存储余量。

## 资产与协作状态

- [6k完整验证页面](h_20260915/H4_step6000/review.html)
- [A/g对照1](h_20260915/H4_step6000/contact_Ag_1.png)
- [A/g对照2](h_20260915/H4_step6000/contact_Ag_2.png)
- [原始指标](h_20260915/H4_step6000/metrics.json)
- [上一阶段4k记录](H4_4K_MATCHED_REVIEW_20260915.md)

本报告及验证资产已保存到本地仓库。main推送仍待批准，未重试之前被拒绝的推送。中文留字辅助任务与可学习局部记忆TC仍是待review设计，没有加入当前队列。

使用ml-training-recipes的DDP、混合精度、EMA与checkpoint检查项核验；不据此改变已批准的H超参数。
