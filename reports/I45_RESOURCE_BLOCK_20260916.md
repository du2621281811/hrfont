# I4 / I5 资源阻塞记录

现场核验：2026-09-16 11:54 +08。本轮只读诊断；没有终止其他任务、修改数值代码、放宽准入或重启队列。

## 已完成与未完成

- I4 `DONE.json`：completed，step/attempt=10000/10000，inference_complete=true，smoke=false；训练及首套推理进程退出码0。
- 最后训练记录：loss=0.030105719342827797，skips=0，ddp_spread=0，grad_norm=0.1578942984342575。
- `eval_step_10000/DONE.json`：completed，images=4096，step=10000，seconds=1266.5765581130981。此次只核对完成回执与文件存在，没有重复扫描所有图片或宣称已完成Git图像归档。
- I4 `K1248_DONE.json` 与 `I4_ARCHIVE_READY.json` 尚不存在：matched4888未启动。
- I5 state=REVIEW_REQUIRED，coordinator_alive=false，child_alive=false；calibration/preflight为空。未开训，不是效果不达标。

## 触发原因与调查

11:10，I4_TRAIN成功返回后，I4_K1248启动前的GPU准入抛出 `AssertionError('New GPU context: {539026, 539317}')`。I5随后记录 `I4 predecessor failed; review rather than steal resources` 并退出。

11:52调查当前新增上下文时，发现容器内进程3525478/3525552运行 `evaluate_act_validation.py`，目录为 `/root/data2/lush/space-worktrees/dataset-training-pipeline`；调度脚本 `validate_act_formal_ddp_stage2_r02.sh` 分别指定GPU1和GPU3，评测ACT抓取策略5500到10000步的checkpoint，数据集为 `floating_grasp_validation_v001`。NVML宿主PID与容器PID不同，不能凭 `/proc/<NVML_PID>` 不存在判断任务不在此容器。没有据目录名推断操作者身份。

以上直接定位的是11:52当时运行的验证任务；11:10触发故障的两个旧PID已退出，没有将其身份未经核验地等同于当前进程。

11:53，9500/10000验证日志均显示succeeded，相关进程退出；GPU恢复原8个1832MiB基线上下文，8卡利用率0%。未停止这些基线上下文。11:54根盘余13.53GiB、新checkpoint盘余117.11GiB，均高于10GiB线。

## 下一步边界

等待PI确认恢复。恢复仅需补跑I4 matched4888及归档，再衔接I5预检和主任务训练；不重训I4、不重跑formal4096、不启动I6。恢复前重新验证进程归属、GPU容量、原队列锁及冻结指纹，保留原失败记录。原 `resume_i34_archive_20260916.py` 是针对更早I3归档故障的专用恢复入口，不能直接重复执行。

本次Git同步为进度/阻塞证据，不是I4完整图像结果归档或I5效果报告。
