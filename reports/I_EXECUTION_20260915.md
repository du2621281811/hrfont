# I 系列：在线局部外观补全 + 动态结构集合

更新：2026-09-15。本文件是新训练的执行规格；跨系列现状见 [EXPERIMENT_TRACKER](EXPERIMENT_TRACKER.md)，原图索引见 [实验归档](experiments/README.md)。历史 H 排程已经被本文件替代，不再执行其剩余臂。

**2026-09-15 13:39 +08：PI批准中文扩展，I2恢复排队。** 已在新目录重新构建全部338张PNG及Ec特征；338图哈希、1690个特征张量与审核版全部一致。旧中文扩展完整备份，正式路径切换到重建版，特定渲染review STOP已归档并解除。I1继续，I2在I1训练及推理完成后先执行20步预检，再启动正式10k；不是现在已开始I2。历史295字环境像素复现仍作为独立记录保留，主任务原图和cache不改。见 [批准证据](review_20260915/I2_RENDER_APPROVAL_20260915.json)、[构建审核](review_20260915/I2_RENDER_REVIEW.md)。E12-c只制定计划，不抢占I队列，见 [E12-c计划](E12C_PLAN_20260915.md)。

## 目标与命名

先获得明显更好的跨语种字体生成，再补消融。三项贡献保持为：**跨语种补全任务及评测、结构变化空间 Delta、目标字外观补全 TC**。I 将后两项升级为动态集合路由和在线多尺度局部记忆，不叠加旧 TC 模块，不加入矢量/CGE/GAN。

| 名称 | 起点 | 修改 | 更新预算 | 排程 |
|---|---|---|---:|---|
| I0 | G0b-F0-V0913-BS256-A-S3407/global_step_10000 | 仅是同一权重的别名；不是 G0/G0c，也不是 G2 | 0 | 保留参照、补相同协议推理 |
| I1 | I0 | 在线局部 ref/TC + 动态 set-delta；原 global9 保留 | 10,000 | 第一优先，独占本轮8卡 |
| I2 | 同一个 I0 | I1 相同架构及初始化，额外中文留一字辅助任务 | 10,000 | I1 训练及最终推理完成后运行 |

I2 **不继承 I1 训练权重**；I1/I2 都不继承 H/G2/CONT/旧 TC。I0 是预训练底座，后接一次联合训练。I2 比 I1 每更新多32个辅助目标，比较时明确额外计算量；本轮不先铺消融矩阵。

## 架构的实际实现

入口：`scripts/hrfont_i.py`、`scripts/i_runtime.py`、`scripts/train_i.py`、`scripts/i_eval.py`。

### 外观主线

1. 原 Es 冻结，继续提供 global9 和 Alpha 检索特征；固定检索库与 query 不漂移。
2. 复制 Es 前三块作为**独立可训练**的局部编码器，在线读取96px参考图，不缓存其输出。取24×24与12×12特征，浅层投影到256维，共720个局部 token/ref。
3. 目标字 Ec 的12×12网格与144个可学习 slot 相加形成 query，四头注意力联合读取全部有效参考的多尺度 token；不先逐参考平均。保留144个目标对齐外观 token 和训练用128维 VGG readout。
4. 上采样 cross-attention 保留旧 global9 结果，加上局部注意力的残差。可训练 gain 初始0.01，前1000更新线性开放，保护 I0 的初始生成能力。
5. 学习目标为目标训练字 VGG enc2 的12×12标准化特征。推理不输入目标风格 GT；同一个 TC 表征也直接供生成器使用。

### Delta 支线

同一组 ref 召回最多10个合法训练 donor，排除目标字体及不支持该字符脚本的 donor。保留每个尺度的 `d_i(c)=Ec(donor_i,c)-Ec(neutral,c)`，不在数据接口提前压成 mean。

每一个 RSI offset 单元根据当前生成 hidden、目标中性内容、参考全局特征和 timestep 计算位置相关权重：

`w_i(l,t,p) = softmax_i(log alpha_i + strength_l * cosine(q_l(t,p), key_l(d_i)(p)))`。

key 使用3×3局部卷积，混合仍位于候选结构残差的凸组合空间；权重随层、位置、生成步变化，无 donor-ID embedding。混合结果进入原 offset/deform/identity-safe residual。Alpha 是初始先验，不是被学习权重替代掉的固定均值。计算量加速用批量 cache 读取；仅冻结 Ec/donor 的取值处于 no_grad，路由网络和局部编码器正常反传。

CFG 10%：content/global/local/Delta 联合无条件化；source drop 25%：关闭 Delta offset 的实际输出，避免 affine bias 泄漏。训推共享同一实现。I2 中文支线 Delta 输出为零。

## 数据与损失

主训练始终是 **V0913**：

- PNG：`data/fontdiffuser-p253-t295-s338-cn2west-v2`，原生96×96。
- split：`manifests/split_v3_228_16_16.json`；清洗：`manifests/v0913_clean/`。
- train56429 / val4123 / test3880；训练按既有脚本组50/38/12采样，1–8 shot。
- Es/Ec：`artifacts/g0/es_spatial`、`artifacts/g0/ec_multiscale`，绑定 I0 编码器哈希。

`L_main = L_epsilon + 0.01 L_VGG + 0.25 L_offset + lambda_TC L_completion`；`lambda_TC` 前1000更新由0线性升至0.01。TC teacher 通道统计复用训练集校准，冻结。

I2：`L = mean(L_main over64) + 0.5 mean(L_CN over32)`。中文任务是同字体、不同 content 的重建，ref 必须排除目标字；使用相同生成器和 TC。目标/参考仅来自清洗后有效训练字体的338汉字池，不加入 val/test 字体。

中文中性图及其冻结 Ec 特征由 `prepare_i_cn.py` 单独生成到 `artifacts/i_20260915/cn_content/`。定义为新扩展 **v0913_clean+I2_CN_Noto83_Pillow12.2**：当前Noto字体文件固定哈希、83px、96px画布、居中及6px边距，Pillow12.2/FreeType2.14.3独立安装到快照`render_deps/`，不替换训练环境的Pillow。准备在CPU运行，不占用I1的GPU。

重绘了295个既有西文ContentImage做审计：86字逐像素相同，少数字符差异超出抗锯齿（最大平均灰度误差8.17/255、最低二值IoU约.769），即使固定Pillow/FreeType版本也存在，**原因尚未确定，不能声称复现历史渲染**。原“重绘完全匹配”检查没有通过，保留此发现；它不用于替换任何主任务资产。由于338汉字中性图原本不存在，I2明确使用上述新版本中性锚点，而非冒称从原cache续接。验收对象相应是新汉字图本身：338唯一字符、非空、前景留白不越界、完整Ec键和字体/PNG/特征哈希；记录全部旧西文审计误差供review。**原V0913 PNG、Ec/Es缓存、训练主任务和val/test均不修改。** 这不是主数据清洗版本变更，也不证明原训练数据存在错误。

准备命令（仅准备进程使用隔离Pillow）：

```bash
PYTHONPATH=/root/projects/hrfont_i_20260915.zvbB1H/render_deps \
 /root/miniforge3/envs/boogu/bin/python scripts/prepare_i_cn.py
```

## 配方、执行和验收

- 8×V100；主任务每卡8、累积1，global64。I2 每卡另4个中文目标，主任务曝光不减少。
- FP16 + GradScaler1024；注意力 logits/loss/优化器 FP32；无 BF16/TF32 假设。
- AdamW：继承主干2e-5，新局部模块/路由1e-4；betas(.9,.999)，eps1e-8，矩阵 decay0.01，bias/norm无decay；clip1。
- 500更新 warmup，至5k保持峰值，后 cosine 至10k的10%。步数只计成功更新；AMP累计跳步超过10且比例>1%则停止。
- EMA≤0.999；每500保存可恢复状态，保留滚动两个及评测里程碑；不删除其他系列模型。
- 每2k固定 val192（16字体×4字×1/4/8-shot、嵌套旧ref）；10k val4096（16字体×16有效字×4组ref×1/2/4/8-shot）。所有输出用 EMA、DPM++20、CFG7.5；原始PNG/GT/逐项指标/protocol/HTML一并归档。
- 先检查重复ref、candidate排列、CFG无泄漏、真实梯度、DPM20；再8卡100更新及恢复至120；I2另20更新验证两任务。Smoke不是效果结论，也不充当正式10k预算。
- 正式运行在独立快照 `/root/projects/hrfont_i_20260915.zvbB1H`；不热改 H/G 工作区。`queue_i_20260915.py --execute` 只允许已通过预检的 I1→I2；单队列锁；任何失败停止等待诊断，不无限重启。
- 2026-09-15 I1 100-update预检：100/100成功、无跳步、DDP参数差0、router/online encoder梯度非零；step100约1.06s，峰值8686MiB。正式耗时以实际日志校准，先按每臂数小时而非之前15–26h粗估安排。初始化、验证推理和中文分支另计。

正式 I1 已启动，run=`I1-V0915-S3407`，队列PID3360111/训练协调PID3360112，现场已核验100次更新。I1初步训练净耗时约2.7h/10k，考虑存档/中间与最终推理先预留3.5–5h；I2尚未实测，先预留5–7h，实际以20步预检与100步正式日志更新。I0相同4096协议推理排两项训练之后，不阻塞新训练。

当前运行步数与预检最终状态见 `experiments/I/QUEUE.json`、各run的`status.json`和 [总表](EXPERIMENT_TRACKER.md)。训练完成与 Git 已上传是两个独立状态，只有成功 push 后才能声称合作者已获得更新。

## 效果决策

每2k检查笔触/描边/端点/可读性和1/2/4/8-shot稳定性，同时查看逐字体 matched L1/SSIM/edge；不能仅凭训练loss降就宣称风格恢复。I1/I2均完成后优先选整体最好者，再决定是否补无动态路由、无局部多尺度、无TC监督等消融。明显非有限值、梯度丢失、checkpoint损坏立即停；普通阶段波动不提前判死。尚未得到I最终结果，不预先宣称效果提升。
