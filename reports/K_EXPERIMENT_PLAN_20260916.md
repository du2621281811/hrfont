# K 系列实验执行规格：效果优先的目标字体保真度训练

日期：2026-09-16  
状态：**待实现、待派发**。本文件取代 I7 作为下一阶段实验设计；原 `HRFONT_I7_PROPOSAL_20260916.md` 与 `REVIEW_I7_PROPOSAL_20260916.md` 保留为历史证据，不追溯修改。  
目标：优先获得肉眼明显的目标字体保真度提升，不在本轮铺设大规模判断树、验证树或消融矩阵。

## 0. 核心判断

当前主要风险不是训练不收敛，而是生成器存在低风险 shortcut：neutral content 已经提供很强的字符身份和粗结构，global style 足以完成字重、宽窄、倾斜等容易优化的属性；真正决定字体身份的笔端、断笔、空心、装饰和局部连接只占少量像素。I5 冻结权重诊断已经表明 TC/局部 reference 分支有信息，但实际进入最终生成的影响非常弱，因此模型可以把训练 loss 做下来，同时继续依赖 content/global path，避免承担复杂局部设计的风险。

K1 的目标不是发明第四个方法模块，而是：

> **让生成器不能轻易绕开 target-conditioned appearance，并显著提高 GT 相对 neutral content 真正发生变化的位置的优化优先级。**

---

## 1. 命名与起点

### K0

K0 不训练，只作为共同 baseline 和 parent：

`K0 ≡ I0 ≡ G0b-F0-V0913-BS256-A-S3407/global_step_10000`

禁止把 G0c、G2、I1、I5 或重新训练的“等价模型”称为 K0。

### K1

建议 run id：

`K1-FORCED-STYLE-COMPLETION-V0916-S3407`

K1 从 K0 独立初始化，只做一次正式 10k 主实验。本轮不排 K1a/K1b/K1c，也不预先排 K2；K2+ 仅保留编号，等 K1 最终效果后再决定。

K1 同时包含三类协同修改：

1. 强制、不可自由缩零的 TC 生成消费；
2. change-focused GT 输出监督；
3. 高信息量 complex/pair sampling 与更低 condition drop。

Dynamic Delta 本轮保持现有架构，不重构结构消费接口。

---

## 2. K1 必须保持不变的部分

以下全部沿用 I 主线，Cursor 不得顺手改动：

- 主数据：V0913 clean；split 228/16/16；native 96×96 RGB；
- neutral content：当前 Noto ContentImage；
- 冻结 Es/Ec 与现有 cache；
- Delta donor bank、Alpha、top-K、动态 set router、现有 identity-safe structural path；
- 训练 ref 数量 1–8；语言主比例约 50/38/12；
- UNet 主体、VGG perceptual、TC LocalMemory、144 target-aligned tokens、VGG enc2 teacher/readout；
- 不加中文 auxiliary、不加 E12 training feedback、不加 GAN/discriminator、不加 vector/CGE、不训练新 StyleEncoder。

K1 **禁止从 I5@4k 续训**；必须从 K0 重新构造模型。

---

## 3. 修改一：TC 必须真正进入生成器

### 3.1 删除自由 local gain

当前 I 实现形如：

```python
A = A_global + local_gain * ramp * A_TC
```

K1 中删除可训练 `local_gain`，禁止出现任何可学习标量把 TC 整条分支压到接近 0。

### 3.2 K1 注入公式

每个 up-path `attn2` 分别计算：

```python
A_global = attention_once(attn, hidden, global_context)
A_tc     = attention_once(attn, hidden, tc_context)
```

按 sample 计算 RMS，均值维度为 sequence 和 channel，保留 batch 维：

```python
global_rms = sqrt(mean(stopgrad(A_global) ** 2) + 1e-6)
tc_rms     = sqrt(mean(stopgrad(A_tc) ** 2) + 1e-6)
tc_scale   = clamp(global_rms / tc_rms, 0.25, 4.0)
```

TC 强度：

```python
beta = 0.8 * min(1.0, successful_update / 1000)
```

最终：

```python
A = A_global + beta * tc_scale * A_tc * conditional_active
conditional_active = ~cfg_mask
```

硬约束：

- `A_global` 系数始终为 1；禁止 `(1-beta) * A_global`；
- `tc_scale` 不反传、不是 parameter；beta 不是 parameter；
- 1000 successful updates 后 beta 永久固定 0.8；
- CFG unconditional sample 的 TC contribution 必须严格为 0；
- source-drop 只关闭 Delta，不关闭 TC。

设计目的：强迫 TC 保持有意义的生成影响，但不破坏成熟的 global style 主路。

---

## 4. 修改二：Change-Focused 输出监督

定义：

- `C`：neutral content image；
- `Y`：目标字体 GT；
- `X`：当前 epsilon prediction 反推得到的 raw x0 prediction。

所有距离在灰度空间计算；`Y/C` 为 `[0,1]`。`X` 映射到同一坐标，但 K1 detail loss 前**不得 clamp 到 `[0,1]`**。VGG perceptual 可继续当前安全处理。所有由 `C/Y` 构造的 mask 必须 detach；推理不使用 GT。

### 4.1 `D_region`

保留 I5 `region_detail_distance` 的思想：ink、near/enclosed white、distant background、edge，以及 96/48 两尺度。建议复制为 K 专用实现，避免修改历史 I 行为。

### 4.2 `D_change`

固定阈值：

```python
change_threshold = 0.10
```

黑色墨迹值更低，因此：

```python
M_add    = (C - Y) > 0.10   # GT 比 content 更黑：需要新增墨迹
M_remove = (Y - C) > 0.10   # GT 比 content 更白：需要删除墨迹
```

两者分别 3×3 dilation。距离使用：

```python
charb(a, b) = sqrt((a-b)**2 + 1e-6)
```

分别在 mask 内归一得到 `D_add` / `D_remove`：

- 两个都有效：`D_change = 0.5*D_add + 0.5*D_remove`；
- 只有一个有效：直接使用该项；
- 两个都空：`D_change = 0`。

禁止把 add/remove 合并成一个大 mask 后统一平均，避免大面积加粗掩盖断笔、空心和缺口。

### 4.3 `D_high`

```python
H(z) = z - avg_pool_3x3(z)
M_ink  = (Y < 0.95) OR (C < 0.95)
M_high = dilate(M_ink, kernel=5)
```

在 96×96 与 avg-pool 后的 48×48 上分别计算 `abs(H(X)-H(Y))`，只在 `M_high` 内归一：

`D_high = D_high_96 + 0.5 * D_high_48`

### 4.4 总输出距离

```python
D_out = D_region + 1.0 * D_change + 0.25 * D_high
```

K1 明确把 `D_change` 提到与 `D_region` 同等级，不允许“大部分普通像素做对就够了”。

---

## 5. 时间步加权

I5 detail supervision 的 `alpha_bar` 改为：

```python
sqrt(alpha_bar)
```

```python
L_detail = mean(conditional_mask * sqrt(alpha_bar_t) * D_out)
conditional_mask = ~cfg_mask
```

分母固定为完整 batch size，不按 conditional 样本数重新缩放。

禁止：

- 完全去掉时间权重；
- 对 raw x0 使用无界二次 detail loss；
- 使用 `-D(X,C)`；
- 奖励模型“离 content 越远越好”。

---

## 6. TC completion supervision

现有 TC teacher 保留：target GT → VGG enc2 → pool 12×12 → frozen teacher channel statistics → SmoothL1 appearance prediction，记作 `L_comp`。

即使样本本次 CFG drop，TC reader 仍允许接受 `L_comp` 监督，但其 TC token 不得进入 unconditional denoiser。

---

## 7. K1 总损失

```text
ramp = min(1, successful_update / 1000)

L = L_epsilon
  + 0.01 * L_VGG
  + 0.25 * L_offset
  + ramp * (0.01 * L_comp + 0.05 * L_detail)
```

保持原 Dynamic Delta offset regularization；本轮不通过减弱 offset regularization 来制造 Delta 响应。正式训练启动后不得中途改系数。

---

## 8. 修改三：高信息量 sampler

每 successful update：global batch=64。每个 update **硬配额**：

| script group | 总数 | confirmed complex | other |
|---|---:|---:|---:|
| western | 32 | 19 | 13 |
| kana | 24 | 14 | 10 |
| bopomofo | 8 | 5 | 3 |
| total | 64 | 38 | 26 |

即 complex exposure 38/64=59.375%，视为 60%。

使用现有 I5 已冻结、人工确认的 complex/detail manifest；不得根据 K1 结果重新挑字体。

`other = all legal clean training pairs - confirmed complex pool`，其中包含未人工确认复杂度的字体，不称“简单字体”。

complex/other 池均采用：先均匀选 font，再在该 font 对应 script 的 clean target chars 中均匀选 char，避免 pair 数量多的字体垄断。

### 8.1 Same-content / different-font pairing

global64 中固定 16 个 sample 组成 8 对：

| script | pairs | samples |
|---|---:|---:|
| western | 4 | 8 |
| kana | 3 | 6 |
| bopomofo | 1 | 2 |

每对要求：target character 完全相同、font A != font B、GT 各自正确、refs 各自来自对应目标字体、neutral content 完全相同。

默认 font A 从 complex pool，font B 从 other pool；两字体都必须有该 target clean pair。若找不到合法 B，重新抽 target character。

禁止错误 ref、交换 GT、增加额外 forward 或添加 contrastive loss。pairing 只改变 batch 构成。

### 8.2 Reference sampling

每 episode：`n_ref ~ UniformInteger(1,8)`。refs 必须来自当前 target font 的 clean 合法 reference pool；禁止 target GT 泄漏。

---

## 9. Dropout

K1 固定：

```text
joint CFG condition drop = 0.02
Delta/source drop        = 0.05
```

两个 RNG 独立。

joint CFG drop 同步关闭 denoiser 的 content、global style、TC appearance、Delta；`L_comp` 仍可监督 Reader。

source drop 只关闭 Dynamic Delta 实际结构输出；content/global/TC 保持正常。

每100 update记录实际 drop 比例。

---

## 10. Dynamic Delta

K1 **不修改 Dynamic Delta 架构**，继续使用当前 donor candidate set、Alpha、per-layer/per-position/per-timestep router 和 identity-safe deformation path。

禁止实现历史 I7 提案中的差分 deformation：

```text
Offset(h,d)-Offset(h,0)
DCN(h,o)-DCN(h,0)
rho
```

K1 只通过 `source_drop .25 → .05` 提高 Delta 的实际训练使用频率。

---

## 11. 正式训练配置

```text
GPUs              = 8 × V100
microbatch/rank   = 8
gradient accum    = 1
global batch      = 64
successful steps  = 10,000
seed              = 3407
precision         = FP16
GradScaler init   = 1024
grad clip         = 1.0
```

AdamW：betas(.9,.999)，eps=1e-8；matrix weight decay=.01，bias/norm no-decay。

LR：

- inherited UNet/existing backbone = `2e-5`；
- TC Reader / Dynamic router / K new code = `1e-4`。

schedule：1–500 warmup；501–5000 peak；5001–10000 cosine decay；10k 为 peak 的10%。successful update 才增加 step。EMA 沿用现有 I runner 语义，不另改 decay。

---

## 12. 代码隔离

禁止热改历史 I/G/H 执行行为。建议新增：

```text
scripts/hrfont_k.py
scripts/k_components.py
scripts/k_runtime.py
scripts/train_k.py
scripts/k_eval.py
scripts/queue_k_20260916.py
```

允许复制 I 实现后修改。历史 I/G/H 文件、checkpoint、reports 不追溯覆盖。

运行目录：

`runs/K1-FORCED-STYLE-COMPLETION-V0916-S3407/`

归档目录：

`experiments/K/K1/`

---

## 13. 训练日志最低字段

每100 successful updates 至少记录：

- step / attempt / total_loss / epsilon_loss / VGG_loss / completion_loss / detail_loss；
- `D_region / D_change / D_add / D_remove / D_high / offset_loss`；
- beta；每层 TC scale median/p10/p90；
- reader_grad / router_grad / online_encoder_grad / grad_norm；
- AMP scale/skips、DDP spread、LR、update_seconds、peak_memory；
- script counts / complex counts / other counts / paired counts；
- joint CFG/source-drop counts；font exposure histogram。

1000/2000/5000/10000 输出 exposure snapshot。

---

## 14. 必须通过的实现单元测试

### A. beta=0 parity

beta=0 时 KModel TC 注入必须严格退化为 `A_global`。

### B. global path 不被缩放

任意 beta 下 global coefficient 恒为 1；禁止 `(1-beta)`。

### C. CFG

cfg sample：TC contribution=0；Delta output=0；content/global 使用现有 unconditional 定义。

### D. source drop

source-drop sample：Delta=0，但 TC/global/content 均保持 active。

### E. change mask

人工构造“只新增一笔 / 只删除一笔 / 同时 add-remove / C=Y”四类图，确认 `M_add/M_remove/D_change` 方向无误。

### F. sampler

模拟至少1000 update，确认**每个 update 本身**满足 32/24/8 script quota、19/14/5 complex quota、8 个 paired pairs，而非仅长期均值接近。

### G. leakage

assert K1 sampler 仅 train split；target font 从 donor 排除；val/test 不进入训练。

---

## 15. 正式运行前 preflight

仅做工程正确性，不做长效果诊断：

```text
20-step single-process functional smoke
→ 8 GPU 100 successful updates
→ 保存
→ exact resume 至 120
```

检查：无 NaN/Inf；8 rank 一致；AMP 正常；Reader/TC/router 有梯度；TC scale 有限；checkpoint exact resume；exposure quota 正确。通过后直接启动正式10k，不再安排长 overfit 诊断链。

---

## 16. 中途检查

只设一个正式中检：`step=2000`。

目标：确认没有 identity collapse、系统性 artifact、数值/工程故障，并查看三语种实际图像。

2k 不因 1% 左右 L1/SSIM/局部字体波动自动停止。除非出现 identity collapse、大范围黑/白屏、TC injection 系统性破图、NaN、DDP/AMP/checkpoint 故障，否则继续跑满10k。

### K_VAL192

不要复用历史仅四个西文字的 val192。创建冻结 manifest：

`experiments/K/K_VAL192.json`

从 clean validation pairs 中按 seed3407 确定性选择：western64、kana64、bopomofo64，共192 pairs；不依据 K1 结果。2k 每 pair 只生成4-shot，共192张。提前生成完全 matched 的 K0 输出。

---

## 17. 正式最终推理

K1@10k 使用 EMA。

K 系列 primary inference 固定：

```text
DPM-Solver++ multistep order2
20 steps
CFG = 1.0
```

K0 必须使用相同 sampler、CFG1、noise、refs 重新生成 matched baseline；禁止拿旧 CFG7.5 图片直接比较。

### 最终 clean test

现有16 test fonts × 47-character stratified panel，经 clean pair filter 后约704 font-character pairs；每 pair 生成 1/2/4/8-shot，总约2816 images/model。K0/K1 完全 matched。

---

## 18. 最终指标与 Review 页面

统计：L1、SSIM、edge、D_region、D_change、D_add、D_remove、D_high；按 all/script/font/change-magnitude 分层。E12c 只作辅助 family-compatibility statistic，不用于选 checkpoint。

必须生成：

`reports/k1_final_review/index.html`

每行固定显示：

`Reference set | Neutral Content | K0 | K1 | GT`

同 noise、同 ref budget。支持 script/font/character/shot filter、D_change magnitude sort、K0→K1 improvement sort，并保留全量页，禁止只挑 K1 好看的样本。

冻结 qualitative panel 至少覆盖：ordinary stroke、heavy/light、rounded/sharp terminals、serif、hollow、broken stroke、decorative、unusual local connection、kana、bopomofo。

---

## 19. K1 成功定义

K1 不以某个小数点指标作为唯一裁决。核心要求是：与 K0 并排时，在三个 script 中都能看到目标字体特征明显增强，尤其是 content 与 GT 明显不同的位置。

最终人工 review 应确认：

1. 字符 identity 无系统性下降；
2. 提升不是只做全局加粗/变窄；
3. terminals / hollow / broken / decorative / local connection 出现明显且与 GT 一致的变化；
4. `D_change` 在总体和高-change子集优于 K0；
5. 改善不是只集中在少数 complex fonts；
6. reference 改变时，输出变化方向与 reference 字体一致。

只有数值/工程故障设置自动 gate；视觉效果由最终人工 review 决定。

---

## 20. 正式启动后禁止事项

- 中途改 beta、loss coefficient、threshold、complex manifest；
- 从 I5 checkpoint 续训；
- 加中文 auxiliary；
- 热修 Delta architecture；
- 加新评测网络作为 training loss；
- 把 loss/sampling 包装成第四个论文贡献；
- 2k 因 1% 左右指标波动提前终止；
- 旧 CFG7.5 K0 与新 CFG1 K1 直接比较。

若发现实现错误，旧 run 标 failed，使用新 run ID 从 K0 重新开始。

---

## 21. Cursor 最终交付物

Cursor 实现阶段最终应提交：

1. K1 implementation；
2. resolved config；
3. K sampler manifest/tests；
4. code/data/parent hashes；
5. preflight report；
6. K1@2k snapshot；
7. K1@10k final report；
8. K0/K1 matched final HTML；
9. metrics JSON；
10. exposure JSON。

代码 commit 必须发生在正式训练启动前；训练日志和配置保存 commit SHA；大型 checkpoint 仍按项目现有规则留执行机。

---

## 22. 一句话定义

**K1 = K0 + 不可绕开的 target-conditioned appearance consumption + content→GT change-focused supervision + 高信息量 complex/pair sampling。**

它不是新方法族，而是让现有 HR-Font 的 Delta + TC 真正优化到目标字体设计质量。