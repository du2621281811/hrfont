> Execution update: User explicitly authorized applying this revision and launching K6. R2 queue is running; the historical pending-review text below is superseded. A resumed 2 to 18 successfully; B preflight in progress.

# K6 新监督计算路径修订 — 待批准

原计划已实施并真实执行：K6-0生成192图；train-only校准4096样本；原sampler100批一致；8rank相对监督梯度等价；K6-A 2步+完整状态恢复至4步，0跳步、rank一致。但32条rank/update记录、52个通过门控的样本，ranking loss均为0。正式10k未启动，B预检尚未启动。

原因证据：GT-noised clean在本次初始化下已经明显比neutral更接近GT，margin被提前满足。对FZBangSKLTJW字符0，在t=50/250/500/750都为0；独立纯噪声8步DDIM测试同一ranking与GT-distance门控得到loss0.158276、所检UNet参数梯度范数0.643636。这只证明一个样例上监督可起效，不证明最终视觉改善。

建议修订（尚未实施正式训练）：

1. base K1训练global64、t/noise、配额、loss、optimizer全部不变。
2. 每16个成功更新计算一次额外监督；重用该批原有8个same-content pairs，共16样本，每rank一对。独立RNG，pair共享初始纯噪声；不重新抽字/字体，不增加K3重建/差分loss。
3. 当前模型从纯噪声做8步可微DDIM，新增A ranking、B relative loss施加到其输出；base loss仍原单步训练。用activation checkpoint保留全路径梯度，不能detach前7步。
4. 保留冻结VGG、train校准尺度和GT差异门控、margin0.2*delta、lambda0.05与1000更新ramp。纯噪声生成没有原单步alpha_bar，因此仅新增项去掉alpha_bar>=.5及sqrt(alpha_bar)权重，全部target-conditioned；base dropout不变。
5. A按16样本平均，B按8pair平均；不乘16放大稀疏辅助项。记录损失、hinge活跃比例、有效GT门控比例与新增梯度量级。若新增梯度压过base，先报告再改系数。
6. A/B仍各独立K0，从0正式开始、10000成功更新；旧预检不用于正式初始化；C仍不启动。
7. A/B都做新路径真实多rank前反向、同batch梯度/数值、完整恢复预检，且至少观察到有效的新项反向后才启动正式训练。训练时间在前200成功更新后重估，不能沿用原4–6小时承诺。

本修订改变新增监督路径和频率，属于科学配方改变，不是可自动决定的数值修复；原TRAINING_PLAN_REVIEW.md第8节约定此类变化需review。因此control/STOP保留，等待用户明确批准后再续跑。其他实验旧STOP不动。
