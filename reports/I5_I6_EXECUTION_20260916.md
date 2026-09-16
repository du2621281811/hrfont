# I5执行与I6预留：真实细节覆盖 + 区域均衡监督

当前状态（2026-09-16 11:54 +08）：**REVIEW_REQUIRED。I4训练与formal4096完成，但matched4888启动前被新增GPU上下文拦截，前序队列退出，I5等待进程随之退出；尚无GPU预检、正式训练或效果结果。** 额外任务已结束，队列仍需经确认恢复，见[阻塞记录](I45_RESOURCE_BLOCK_20260916.md)。下文派发PID与WAITING状态为原始历史回执。I6仅保留公共代码接口，本次不启动。

## 授权与命名

PI要求“实现并训练I5”，并明确回复采用“I5不加中文，I6再加”。因此：

|模型|初始化|主任务|中文辅助|本次执行|
|---|---|---|---|---|
|I5|同一I0/G0b@10k独立初始化|0913中→西文/假名/注音，global64|无|已派发，GPU预检通过才正式训练|
|I6|将从同一I0独立初始化，不继承I5|与I5相同|沿用I2/I4的同字体中文留一目标，global32，系数0.5|仅预留，不派发|

这替代此前本地草稿“I5继承I4中文支线”的命名。Delta、在线ref编码、144个TC tokens、注入位置和三贡献叙事均不变。不加decoder/codebook/GAN，不把E12c当训练loss。

## 已实现的数据与监督

训练数据仍为0913、223有效字体、56,429目标图、75,374中文参考图。RGB/native96/抗锯齿与清洗映射不变。固定Es检索困难不再决定I5复杂风格加权。

已扩展并逐图审阅36个训练字体—语种面板，每组8中文+8目标字，确认22组有可见的细笔端、断笔、空心或几何装饰：西文8组、假名7组、注音7组。它是保守确认子集，**不是541组的完整风格普查**；其余组没有被判作简单字体或删除。

证据：[标签与理由](experiments/I/preflight_i56/detail_labels.json)、[采样清单](experiments/I/preflight_i56/detail_manifest.json)、[审核面板及字符](experiments/I/preflight_i56/panels/panels.json)。实际审核人为Codex视觉检查，尚不标为PI逐组人工批准。

采样先按语种50/38/12，再混合70%基础分布与最多30%确认细节分布。细节分布按字体均匀再按字符均匀；单字体最终概率上限基础3倍，超额退回基础分布；前1k渐进切换。没有叠加旧Es难桶权重。

|语种|确认组数|基础采样中确认组比例|稳定阶段实际比例|
|---|---:|---:|---:|
|Latin|8|3.59%|10.76%|
|Kana|7|3.72%|11.17%|
|Bopomofo|7|5.38%|16.15%|

30%是额外采样分支预算，不是声称30%目标样本都来自确认组；3倍上限优先。主任务每个成功更新的字体/语种曝光计数进入日志与checkpoint，恢复时使用checkpoint内计数，避免把未持久化步骤重复累计。

新detail距离：墨迹/近邻白色细节/远背景分别归一后按0.5/0.4/0.1合并，再加0.5边缘误差；96/48尺度权重1/0.5。近邻白区除2像素膨胀带，还包括被墨迹水平与垂直包围的内部白区，以覆盖较大空心描边内部。这只是GT mask，不修改GT。空mask跳过并重新归一；FP32计算。

`L_task = L_eps + .01 L_VGG + .25 L_offset + ramp[.02 D_region(TC,y) + lambda_render alpha_bar D_region(x0_raw,y)]`

保留conditional mask、固定任务batch分母、原时间采样和alpha_bar。lambda_render将从0.1/0.2/0.5中用32个训练批次实测梯度后冻结；还未完成校准，不提前填写最终数值。测量参数为同一组UNet输出卷积与up-block cross-attention投影，不声称是全网络梯度范数；使用1024缩放后还原，避免FP16未缩放反传低估梯度。

选择目标为加权render/base梯度比中位数接近0.3；工程安全区中位数0.02–0.8、p95≤2。没有候选通过时暂停review，不自动扩大搜索或更改方法。

## 预检、训练与推理顺序

1. 不干预现有I4。等I4训练10k、formal4096、matched4888及服务器归档全部完成，原统一queue.lock释放。
2. 冻结指纹、磁盘与GPU准入。保留已有不可见host命名空间的空闲CUDA上下文，不终止外部进程。
3. 单卡无更新梯度校准；I0初始模型导出固定训练例的ref/GT/TC/DPM20及local-off诊断。
4. **8卡300步固定复杂训练批次**，然后恢复300→304，验证成功步、RNG/优化器/EMA恢复、零跳步/DDP一致及梯度。
5. 从304步EMA再次导出相同16个训练例。TC平均区域误差降低至少5%；最终生成平均区域误差下降且至少8/16例改善；local-off干预中位像素差>1e-5；图像有限且非全白/全黑。通过才启动正式I5。该门槛只证明小样本可学习性和通路使用，**不是泛化结果**。
6. I5从全新I0独立训练，**不继承预检权重**。8卡global64、最多10k成功步；原500warmup、5k后cosine、继承LR2e-5/新模块1e-4、FP16、clip1、EMA。
7. 每2k固定val192。4k按预先固定规则对比I3@4k：区域细节误差改善，L1退化≤2%，SSIM下降≤0.005才继续；否则保留4k checkpoint并暂停review。阈值是低成本早期复核触发器，不是统计显著性结论；不自动加步或调参。
8. 若继续到10k，完成原formal4096和同款matched4888，归档PNG/GT/refs/content/metrics/协议/哈希/离线HTML，随后由每小时巡检同步Git。

4k门槛只使用val，不用test挑checkpoint。E12c的人评一致性尚未完成，不作为唯一成功标准。若短预检或4k触发review，任务状态会明确为REVIEW_REQUIRED，不把暂停写成训练完成。

## 实际部署与状态

- 执行机：既有 `sitonholy`。
- 独立代码快照：`/root/projects/hrfont_i56_20260916.xUFfHO`；未修改I4快照。
- 新checkpoint：`/root/data1/hrfont_i56_20260916/`，经原runs路径链接；只写本次新状态。两文件系统10GiB停止线。
- 队列：`/root/projects/hrfont/reports/i56_20260916`。
- 后台入口：`scripts/queue_i56_20260916.py --execute`；派发PID **3513740**。
- 2026-09-16派发实测：`WAITING_FOR_I4_ARCHIVE`，进程存活、无失败、无I5 GPU子进程；前序I4为9700步、skips0/DDP0。
- [真实派发回执](experiments/I/preflight_i56/dispatch_state.json)、[CPU准入与415个源码指纹](experiments/I/preflight_i56/ARMED.json)。状态是CPU_READY_GPU_PENDING，**不是GPU预检已通过**。
- 8项新单元测试通过，训练/推理/校准/诊断四个实际环境CLI入口通过；真实GPU数值检查仍排在前序之后。
- 本地与执行快照任务相关脚本及variant源码哈希一致；快照保留的5个无关旧工具与当前Git不同（eval_f03_test16_strat、launch_cn2west_f123、prepare_i_cn、scan_f0_val_loss、serve_19000_board），本次未覆盖或调用它们。

时间暂按I3实测估I5净训练与保存约3–4小时，含两协议推理约4–5小时；另加前序等待和约15–30分钟GPU预检。以正式首100更新吞吐校正，4k若需review不继续估完成时间。

## 一小时低上下文巡检

读工作区游标，再调用：

```sh
ssh sitonholy '/root/miniforge3/envs/boogu/bin/python /root/projects/hrfont_i56_20260916.xUFfHO/scripts/probe_i56_compact.py'
```

普通巡检不扫描全仓/全部图片/权重。预检结束、4k复核、最终推理、故障才读取必要的新增资产。不得重复派发I5，不得启动I6，不重跑I3/E12c/H/G，不热改冻结代码。I4归档完成后先同步它，但不能因I4完成而暂停I5巡检。I5完成归档与Git同步后再暂停；若REVIEW_REQUIRED，报告原因并等待明确的新决定。
