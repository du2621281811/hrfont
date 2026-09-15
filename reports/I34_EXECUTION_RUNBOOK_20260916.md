# I3 → E12-c → I4：执行与恢复手册

2026-09-16。代码、真实数据分桶、E12-c实例划分与端到端预检已完成；正式队列已派发I3，PID3457564。以 [派发回执](I34_DISPATCH_20260916.json) 和执行机 `state.json` 核验最新状态。

## 本次实现

- `i34_runtime.py`、`train_i34.py`：独立I0初始化，复用I1局部TC/动态Delta架构；旧VGG TC teacher读出替换为实际144×1024 tokens到96×96墨迹的训练期读出，添加已批准两尺度细节loss。原生成图VGG项不变；推理不执行墨迹读出。旧I1/I2快照没有修改。
- `new_data_inventory.py`、`prepare_i34_difficulty.py`、`i34_sampling.py`：223个0913有效训练字体、131,803张native96 PNG解码和SHA检查；同字符排名校准、目标语种留一字符可观察性、中文/目标字符半样本稳定性。541个字体—语种组，易180/中181/难180。只在训练池打分，不变更清洗表。
- 36组图像面板已实际查看，见 [分桶复核](experiments/I/preflight_i34/difficulty_manifest.json) 和同目录3张PNG。装饰强的字体常更易被Es区分，普通黑体/宋体组可能更难；这仍是静态相对可区分性代理，不是模型学习难度已被验证。严格使用已批准的1/1.5/2权重、最多50%难例分布混合，前1k不改变分布。
- `e12c_data.py`：清洗训练池内部178/22/23字体实例划分；已知家族接口可用，本次映射为空并明确记录223个unknown，不冒称完整谱系隔离。主生成val/test均排除。固定中文ref8和三语种×四shot验证单元。
- `train_e12c.py`、`e12c_model.py`、`score_e12c.py`：独立ImageNet ResNet18；A阶段跨文字多正例对比，B阶段冻结编码器并学习mean/max集合匹配头。A按验证R@1选模，B按12单元macro-AUC选模；内部test只在锁定后使用。主分数为logit、cosine仅诊断。生成图打分只需query和实际中文refs，不读GT或字体标签。
- `eval_i34_k1248.py`：沿用固定DPM++20/CFG7.5/3407、26字体×47字符×4shot的4888图协议。每2k保留val192，最后formal4096与matched4888分别归档。

## 资源与唯一身份

| 内容 | 路径/约束 |
|---|---|
| 执行代码快照 | `/root/projects/hrfont_i34_20260915.bJikwQ` |
| 正式run | `I3-V0916-S3407` → `E12C-V0916-S3407` → `I4-V0916-S3407` |
| 预检run | `I3-PREFLIGHT-V0916-S3407`、`I4-PREFLIGHT-V0916-S3407`、`E12C-PREFLIGHT-V0916-S3407`；绝不作为正式初始化 |
| 生成器父权重 | `G0b-F0-V0913-BS256-A-S3407/global_step_10000`，两臂独立从同一I0开始 |
| 数据身份 | 主数据始终0913；V0916是本次run日期，不是更换数据版本 |
| 元数据 | `/root/projects/hrfont/artifacts/i34_20260915` |
| 新checkpoint | PI已授权 `/root/data1/hrfont_i34_20260915/<run>/`；原runs下checkpoint/last_state/global_step链接可用 |
| 训练日志/预测 | 仍在 `/root/projects/hrfont/runs/<run>/`；没有向data1复制字体数据或移动旧模型 |
| 锁 | `/root/projects/hrfont/reports/i_20260915/queue.lock`；同一把锁覆盖两生成器和E12-c |
| 队列记录 | `/root/projects/hrfont/reports/i34_20260916/` |

根盘原余量约16.4GiB；data1约129GiB。本轮生成器每臂峰值预留6GiB、E12-c与预检合计预留4GiB，NFS最低10GiB余量；根盘新图像/元数据/特征预算3GiB，不把旧模型列为自动清理对象。生成器每500步保完整model/EMA/optimizer/scaler/8rank RNG/attempt，滚动保留最近两份；每2k EMA硬链接保留。仅回收本次runner明确创建的过期滚动文件。

## 准入、启动与巡检

1. 7项I34组件测试和4项E12c单元测试；真实8卡I3固定batch80步下降、80→84恢复且不重放；I4中文路径10步；E12-c A20+B20全流程；实际模型重复ref与DPM推理。全部通过并写 `PREFLIGHT_PASSED.json`，绑定数据/代码SHA。小样本测试不作为论文效果。
2. I2 matched4888、I0 formal4096取回本地：I2 6395个登记文件SHA通过；I0 4352张预测/GT逐图解码、4372文件登记。Git push后生成 `OLD_ARCHIVES_SYNCED.json` 保存提交号，作为队列准入证据。
3. `python scripts/queue_i34_20260916.py` 默认只预检；只有显式 `--execute` 才派发。后台启动需保存真实PID并验证I3前几个update，不把heartbeat等同于启动。
4. 每小时运行 `python scripts/probe_i34_compact.py`，先读小摘要。阶段切换再读相关日志、权重身份和归档清单。队列每10秒检查STOP和两个文件系统余量。数值错误/缺文件/代码或manifest漂移即停止后继，不静默换权重或跳过E12-c。
5. I3/I4每臂8×V100、10k成功更新、主global64；I4另global32中文、系数0.5。E12-c单卡A2000+B1000，不能与I4重叠。
6. 每臂自动生成 `*_ARCHIVE_READY.json`；巡检将对应报告同步Git。所有计算结束为 `COMPUTE_COMPLETE_ARCHIVE_PENDING`，全部归档推送验收后才暂停hourly heartbeat。

## 异常恢复

先检查队列、torchrun及8个worker是否仍活着；不要第二次启动同一队列。STOP优先、不得因为旧5146状态重启I2。

生成器支持原run加 `--resume <run>/last_state`，恢复前验证arm/父权重/代码/采样SHA/批量/模式，成功步与尝试步继续计数，仍以10k为上限；不能把预检权重转为正式run。E12-c支持 `--resume`，A_last/B_last保存优化器/RNG，读取配置一致性、已选A模型和缓存hash；不允许对已有DONE重新选模或重复内部test。若最后状态只剩已完成训练而推理未完，单独补对应推理，不重复10k训练。队列故障不自动无界重试，先修复具体原因、核对已完成阶段再恢复剩余阶段。

论文/Overleaf本次不修改。遵循训练预检技能的FP16数值、梯度与恢复检查，同时沿用本项目优化器和字形渲染约定。
