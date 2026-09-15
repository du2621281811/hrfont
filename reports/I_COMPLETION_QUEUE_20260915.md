# I 系列完成队列：当前推理 → I2 续训 → 同款推理

2026-09-15，PI 明确要求：“等推理结束了，把I系列的模型都跑完，然后跑同款的推理。”本记录是清理后新的恢复授权，取代旧文档中“尚待恢复批准”的状态；不覆盖旧暂停/清理证据。

## 唯一执行顺序

1. 等待现有 `reports/i_20260915/run_i01_k1248.sh` 完成 I0→I1，两者各 4,888 图。不能仅因 I0 结束就启动训练。确认 `k1248_infer_status.json` COMPLETE、两个模型各四个 shot 的 1,222 张文件逐个可解码，且现有协调/worker 进程全部退出。
2. I1 已10k训练及原4096图完成，不重复训练；I0为指定G0b@10k，不新增训练。I2从同run的 `state_step_5146` 恢复到总计10000成功更新，剩余4854步。不得重新初始化或再增加10k。
3. 保留原训练内6k/8k的192图和10k最终4096图评测。完成后先检查DONE、全部rank最终步数、AMP/梯度/DDP与checkpoint。
4. I2@10k EMA使用新增的独立适配入口 `scripts/eval_i2_k1248.py`，复用正在执行的 `eval_i_k1248.py` 的任务构造、图像处理、采样和输出路径。当前I0/I1脚本不热改。I2输出 `reports/g_v0913_shot_k1248/preds/I2_s{k}/`，随后按I系列检索入口归档。
5. 原定I协议的I0最终4096图若仍缺失，另以原 `eval_i_checkpoint.py --baseline-i0` 补齐。4888展示板与4096正式评测是不同协议，分别核对、比较、归档，不能交叉汇总。

## 同款展示协议

26字体（test16、train5、val5）×47字符×k={1,2,4,8}=4888图/模型。Ref8为“永和书风骨韵天地”，各shot取前缀；96×96、DPM-Solver++20、CFG7.5、seed3407，沿用同一PROTOCOL.json和数据路径。I1/I2均用10k EMA，I0保留指定基线加载方式。训练配方和数据版本不变。训练/验证字体仅用于展示诊断，不混入test正式指标。

## 恢复检查与命令

- 已有磁盘清理后约18GiB，允许在本次授权下恢复，不再将旧建议20GiB当作硬门槛；实际启动前须复查至少10GiB，并保留训练中低于10GiB的安全STOP。
- 确认没有其他I训练/推理或调度进程占用同队列；使用现有queue.lock互斥，不重跑原queue（其SMOKE目录已被并发移除，且不支持直接接续未完成run）。
- 对照原config中的四个代码SHA，确认训练文件未变化；核验CN COMPLETE及缓存，CPU加载trainer，确认step=attempt=5146、8rank RNG及optimizer/scaler齐全。
- 在移除生效STOP前，将队列STOP、run STOP、STOPPED.json、原config和provenance复制归档至独立恢复审计目录。仅归档本次已授权解除的两个低磁盘STOP，不解除后来新增的不同原因STOP。
- 记录恢复批准和命令，保留旧STOPPED作为历史。下一成功更新应5147，学习率按既有绝对步数接续，不能重置warmup。训练自身沿用最近两个完整state及里程碑权重保留策略，不额外删除其他模型/缓存。

工作目录 `/root/projects/hrfont_i_20260915.zvbB1H`，解释器 `/root/miniforge3/envs/boogu/bin/python`；环境 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1`。

```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29573 scripts/train_i.py --arm I2 --run-id I2-V0915-S3407 --limit 10000 --microbatch 8 --resume /root/projects/hrfont/runs/I2-V0915-S3407/state_step_5146
python -m torch.distributed.run --nproc_per_node=8 --master_port=29583 scripts/eval_i2_k1248.py --shots 1,2,4,8
```

以上两行须顺序执行并核验前置结果，不能并发。独立适配器在I2正式DONE及10k完整EMA存在后才允许加载模型。

## 巡检、预算与交付

使用现有10分钟巡检衔接，不新增重复训练队列。18:47现场I0每rank约260–280/611；按当前吞吐，I0剩余约10分钟，随后I1约20–30分钟，仅为估计。I2剩余净训练按历史约2.21秒/步估计约3小时；加初始化、保存、阶段评测和末次推理，预计当前队列结束后约3.5–4.5小时，其他负载/磁盘变化可能延长。

启动后核验首个周期5200日志八rank一致、梯度非零、AMP有限及LR接续；低磁盘按现有STOP安全保存，不强杀其他作业或自动扩大删除范围。失败推理只补缺失任务，不重训已完成模型。最终逐模型归档PNG/GT/协议/指标/哈希/离线展示页，更新EXPERIMENT_TRACKER与Git；全部批准工作真正完成后通知并暂停巡检。E12-c及G/H不在本次派发范围。
