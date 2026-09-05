# Stage A cache-only 实现独立审查（2026-09-05）

审查对象：仓库 HEAD `827046dfb588a3f9f0b0d0a97396e585b7ff9fa4`；核心实现提交 `447db01`。本报告只读审查代码、配置和现有状态材料；本 checkout 不含服务器 run 目录、cache 实体或 GPU 日志，因此不能复核服务器上的实际 resolved CLI、draw log、吞吐时间戳与 cache manifest。现有状态材料称运行 SHA 为 `d382247`，但该对象不在当前 Git 对象库中，且 registry 仍把 E2/E2b 记为 `planned`，故“同提交、同 seed、同日起跑”目前只是项目材料中的陈述，不是本地可独立验证的 provenance（`PROJECT.md:19-20,38-40`；`provenance/REGISTRY.md:30-33`；`reports/WEEKLY_20260905.md:60-62`）。

## 1. 总评

**结论：不批准把当前 E2/E2b/E1c 训练结果直接作为论文因果证据；当前 E2/E2b 并非冻结规范所定义的 matched pair。** cache-only 训练主路径、9-token style、top-10 接线、encoder SHA 启动校验已经落地，但至少有四个实质性阻断：

1. **D-P2 source-drop 未统一。** E2 的 Δ source 会按 `.25` 置零，official 分支在 mask 应用前直接返回，E2b/E1c 从不 source-drop；因此 E2 vs E2b 同时改变 source 类型和 source-drop，不能单变量归因（`train.py:139-170,380-385`；`hrfont-execution-spec.mdc:43-46,76`）。
2. **D-P6 parity/storage 未过门。** Ec 对 target/style/content 三种 role 都保存五个尺度，实算约 93.39 GiB，而 RSI 对 target/style 只取两个尺度；Es pooled 也以 fp16 落盘，不是要求的 fp32。builder 只有抽样逐尺度 cache-vs-online allclose，没有端到端 noise/offset parity gate（`hrfont_feature_cache.py:11-19,109-125`；`hrfont_build_e1_caches.py:122-134,223-236`；`hrfont-execution-spec.mdc:49-50`）。
3. **D-P7 仅训练成立、推理不成立。** 训练传入 cached style/content/structure 并用 hook 禁止 encoder forward；但 DPM model 和 sample 仍在线运行 Es/Ec，且只接受单张 style 图（`train.py:52-56,243-254,326-327`；`src/model.py:98-116`；`sample.py:63-90,143-161`）。
4. **恢复不 exact-ish，heartbeat 也不是 60 秒。** checkpoint 未保存 DataLoader generator/sampler cursor/epoch/batch cursor/best；resume 从新 epoch 的 loader 起点继续，batch/R/n-shot 序列会漂移。heartbeat 每 100 optimizer step 才写一次，E2 慢速下不能保证 60 秒（`train.py:173-222,347-370,416-426`；`hrfont-execution-spec.mdc:19,100-103`）。

**对正在跑的 seed 3407：建议立即由 PI 决定。** 若目标是论文 matched 因果主证据，推荐在最近安全 checkpoint 通过 STOP 停止两臂、修 D-P2 后从共同 E1@100k 重新开新 run ID；继续当前 E2/E2b 只能降级为工程预跑/非 matched 诊断。不要原地 patch 后 resume，因为这会在同一轨迹中改变训练分布，且当前 resume 本身也不保持样本序列（`train.py:359-370,424-446`；`hrfont-execution-spec.mdc:75-76,100-104`）。

## 2. 逐项结论

### 2.1 `train.py` 重写（任务 1）— **高风险**

#### E2 每步数据路径

- Dataset 对每个样本用 Python RNG 先均匀抽 `n∈[1,8]`，再从排序后的 338 字池无放回抽 R；val 则固定配置中的 ordered R（`dataset/font_dataset.py:114-131`）。这满足 n-shot 的边际定义，n=1 仍返回一张 `[1024,3,3]` spatial map（`train.py:113-121`）。
- Es cache 是只读 `np.memmap`，每个 ref 分别把 spatial 与 pooled row 做 `np.array(..., copy=True)`、转 fp32 Torch；spatial 对 n 张逐元素均值为 `[B,1024,3,3]`，model 再转为 `[B,9,1024]`，所以 n=1 也是 9 token，不走 pooled 1-token（`hrfont_feature_cache.py:87-106`；`train.py:113-121`；`src/model.py:42-49`）。
- α 库的 train228×338 pooled 表在启动时整体复制到 RAM fp32；每样本把对应 R 的 `[228,n,1024]` 再 `.to(device)`，计算逐字 cosine 均值、leave-one-out、top-10 softmax（`train.py:78-93,150-163`；`hrfont_delta_v2.py:79-136`）。
- 每个 E2 样本随后读取 10 个 target Ec row + 1 个 neutral Ec row；每 row 含 5 个尺度，每个尺度均由 memmap 复制到新 NumPy 数组、转 fp32 Torch，再逐 tensor `.to(device)`。之后又读取 1 个 MCA content row。因此是 **12 次 `EcCache.features`、60 个尺度 row copy/CPU→GPU tensor transfer/样本/步**；batch=8 时是 96 次 feature lookup、480 个尺度 tensor（`train.py:129-136,156-170`；`hrfont_feature_cache.py:109-125`）。
- 存在多层 Python per-sample/per-neighbor/per-scale loop（`train.py:113-120,129-136,156-170`；`hrfont_feature_cache.py:128-137`）。显式 GPU 同步点至少包括：`compute_alpha` 排序对 227 个候选逐个 `float(valid_cos[i])`，entropy/max 转 Python float；每样本 `bool(dropped)`；混合时 10×5 次 `float(weight)`；每步 `loss.item()`；前 64 步 mask `.cpu()`（`hrfont_delta_v2.py:109-125`；`hrfont_feature_cache.py:133-136`；`train.py:166,400-414`）。尤其排序的约 `8×227=1816` 次 device-scalar→Python 转换/step 足以严重串行化 GPU。
- source-drop 后置得太晚：即使样本被 drop，代码仍先完成 α、10 邻居+neutral I/O 和 GPU transfer，之后才置零，因此 `.25` drop 不节省 E2 工作（`train.py:156-169`）。

#### E2b/E1c 每步数据路径与差异

- official arm 每样本只读取 `Ec(style,font,R[0])` 的 5 个尺度，再读取 MCA content 的 5 个尺度，即 **2 次 `EcCache.features`、10 个尺度 row/样本/步**；没有 α、228-font prototype GPU copy、10-neighbor mix（`train.py:129-147`）。
- E2 与 E2b 的共享主干（图像 DataLoader、Es n-shot fetch/mean、UNet forward/backward、loss）相同；额外差异明确集中在 E2 的 α 构造、11 个 structure Ec row 和 Python 标量同步。观察到的约 `27800/9200=3.02×` step 吞吐差与该实现完全一致；不是在线 Ec/Es，因为训练 hook 会禁止它们（`train.py:52-56,326-327`）。仅凭仓库不能把 3.02× 精确拆成 I/O 与同步百分比，必须用服务器 profiler 验证。

#### D-P2、P2、P3、CFG

- 三份 YAML 都声明 `drop: 0.25`，launcher 也传 `--delta_drop`，且所有臂在相同位置先抽 `delta_draw`（`configs/e1c_ft_continue_s3407.yaml:31-45`；`configs/e2_stage_a_s3407.yaml:30-44`；`configs/e2b_ft_continue_s3407.yaml:30-44`；`launch_e2.py:76-80`；`train.py:380-384`）。但 official 分支完全忽略 draw，故 **D-P2 FAIL**（`train.py:139-147`）。
- k_top 已从三份 YAML 的 10，经 launcher `--delta_k_top`、parser，进入 `DeltaConfig(k_top=...)`；topk 分支确实取 K=10 后 softmax，且 leave-one-out 在选 top-K 前应用。**P2 主路径 PASS**；但 `eps_alpha` 对 topk 实际不参与数值 floor，只是 dead metadata，且 `k_max`/`k_top` 双字段仍易漂移（`launch_e2.py:76-78`；`configs/fontdiffuser.py:33-42`；`train.py:150-152`；`hrfont_delta_v2.py:99-136`）。
- 9-token 与 element-wise n-shot mean **PASS**（`train.py:113-121`；`src/model.py:42-49`）。
- joint CFG 使用同一 `cfg_mask` 清零整个 style map 与所有 MCA content scales，并保留 structure，符合冻结语义；source mask 独立先抽，但仅 E2 消费（`train.py:381-387`；`hrfont-execution-spec.mdc:45,85`）。

#### RNG、resume、heartbeat、STOP

- 保存了 Python、NumPy、Torch CPU、全部 CUDA RNG，以及 optimizer/scheduler；AMP scaler 参数存在但调用保存/恢复时从未传 scaler，因此未显式持久化 scaler（`train.py:173-221,424-431`）。Accelerate 的 mixed precision scaler 也未通过 `accelerator.save_state` 保存。
- 未保存 DataLoader generator、sampler permutation/cursor、epoch、当前 batch、gradient accumulation partial state和 `best`。resume 后 `global_step` 恢复，但外层 epoch 从 0、loader 从新 permutation 开始；因此 batch/font/char/R、Python RNG消费位置与 uninterrupted run 不等价（`train.py:347-370`）。
- 对未中断、同 seed 的 E2/E2b，draw 调用顺序是相同的：DataLoader/R sampling → source draw → CFG draw → diffusion noise/timestep；Δ 分支本身没有随机抽样。因此运行速度不同不会改变 step-indexed RNG 顺序。但本 checkout 没有两臂 `draw_log.jsonl`，不能验证实际 launch CLI/初始状态；任一侧 resume 后即不能再称 exact matched（`train.py:378-389,405-415`；`PROJECT.md:19`）。
- STOP 在每个 `accelerator.sync_gradients` 后检查；当前 accum=1，等价每 optimizer step，**PASS**（`train.py:391-398,442-446`）。heartbeat 是 step-based（前20步或每100步），不是 60s timer，**FAIL**（`train.py:416-423`）。

#### validation

- 每 5k milestone 后运行 `val_set` 的全部 `16×295=4720` 样本，batch=8 即 590 batch；它为每个 batch 构造固定 R、无 drop 的 cache condition，然后只做一次随机 timestep/noise 的 DDPM diffusion MSE forward。**没有生成字符、没有 DPM-Solver、没有 20 steps/CFG7.5/order2**（`train.py:266-289,352-355,427-441`；`configs/e2_stage_a_s3407.yaml:71-78`）。
- 因此 YAML 的 `eval.inference_steps/guidance_scale/solver/order` 全是 launcher 未消费的 dead config。按 forward 数估算，一次 val 相当于 590 个无 backward batch，即训练 5000 batch 周期的 11.8% forward 数；若 backward 约为 forward 的 2 倍，纯模型计算约增加 4%，但 E2 val 仍重复昂贵的 Δ cache/α 前处理，实际可更高。应从服务器日志单独量测每次 5k pause（`launch_e2.py:65-95`；`train.py:267-289`）。

### 2.2 Cache builders（任务 2）— **高风险**

#### 布局、格式与消费尺度

- 格式是每个命名尺度一个连续 `.dat` 的 fp16 `np.memmap`，另有 `keys.txt`、`manifest.json`、`progress.json`；训练启动一次打开只读 handle，逐 lookup 复制 row，不会 `torch.load` 整表或每步重开文件（`hrfont_feature_cache.py:39-84,87-125`）。这是正确的 mmap 方向，但 lookup API 完全标量化。
- Ec 定义并保存 `[input RGB, 64×48²,128×24²,256×12² residual,256×12² final]` 五尺度（`hrfont_feature_cache.py:11-17`；`hrfont_build_e1_caches.py:122-134`）。MCA 确实按 down/mid index 消费 content pyramid 的这些层（`src/modules/unet_blocks.py:194-217,307-339`；`src/modules/unet.py:243-263`）。
- RSI 两个 up block只分别索引 `structure_features[-3]` 与 `[-4]`，即 s2=`128×24²`、s1=`64×48²`；target/style role 的 s0/s3/s4 从不被 RSI 使用（`src/modules/unet_blocks.py:538-554`；`src/modules/unet.py:267-286`）。builder 却为三种 role 一律保存五尺度，违反 D-P6“only consumed scales”。
- 实算每个五尺度 row 为 322,560 fp16 元素=0.615 MiB。content 295≈0.18 GiB、target 228×295≈40.41 GiB、style 260×338≈52.80 GiB，总约 **93.39 GiB**；若 target/style 只存 RSI 实际消费的 s1+s2，而 content 保留五尺度，约 **63.9 GiB**，与冻结的约64 GiB吻合（`hrfont_feature_cache.py:11-17`；`hrfont-execution-spec.mdc:49`）。

#### SHA、精度、parity、build 成本

- builder 从指定 `--ckpt-dir` 严格载入 encoder，并把该权重文件 SHA 写进 manifest；但它没有独立的“init checkpoint expected SHA”输入可交叉比较，也没记录/校验 cache data 文件 SHA 或 dataset/split 内容 SHA（`hrfont_build_e1_caches.py:54-67,186-201,239-276`）。训练启动则同时计算 init Es/Ec SHA 并与两份 manifest fail-closed 比较，**D-A1 训练入口 PASS**（`train.py:59-75,305-309`）。launcher 自己只查文件/完成计数，不查 SHA（`launch_e2.py:36-49`）。
- Es spatial 与 normalized pooled 都 `.half()` 落盘；训练再转 fp32并在 query/compute_alpha 中重复 normalize。空间 mean 与 Δ 混合/减法是 fp32，方向合理，但 **pooled 不是 D-P6 要求的 fp32**，top-10 第10/11名边界可能因 fp16 量化换邻居（`hrfont_build_e1_caches.py:114-119`；`train.py:113-120`；`hrfont_delta_v2.py:99-125`）。
- `verify_es/verify_ec` 默认抽8条，把刚编码并已转 fp16的数组与 cache 做 `allclose(2e-3)`；这主要验证写读一致，未量化 fp32-online→fp16-cache 的误差，也没有逐 RSI 层 offset/noise_pred/end-to-end generation parity，**D-P6 parity gate FAIL**（`hrfont_build_e1_caches.py:205-236,251-276`）。
- job 数约 Es 87,880，Ec 155,435（295 content + 67,260 target + 87,880 style），默认 batch16，分别约5,493与9,715 encoder batches；实际耗时取决于 GPU与PNG解码。计划只写“约数小时”，仓库无本次 build 时间日志，不能给伪精确值（`hrfont_build_e1_caches.py:70-111,239-283`；`reports/EXECUTION_PLAN_STAGE_A_20260905.md:56-64`）。

### 2.3 launcher、model、三臂配置（任务 3）— **高风险**

- 单一 launcher 能读取 E1c/E2/E2b YAML，传 cache、RSI source、top-10、n-shot、seed、80k、bs/lr/warmup/drop/loss/checkpoint，并自动从 `last_state` resume（`launch_e2.py:28-100`）。`model.py` 的9行变化只是新增可注入 `content_features`，从而让训练绕过在线 Ec；style/structure cache 入口原已存在（`src/model.py:26-67`）。
- E2/E2b 文本 diff 只含冻结 allowlist 的 `experiment.id/init.new_layer/model.rsi_source/model.delta.enabled`（注释除外），静态配置匹配；但 launcher 没有 structural allowlist validator，也不消费/验证 `init.new_layer`、`model.style_condition/style_tokens/train.freeze_encoders/eval.*`，未知/dead字段不会 fail closed（`configs/e2_stage_a_s3407.yaml:1-78`；`configs/e2b_ft_continue_s3407.yaml:1-78`；`hrfont-execution-spec.mdc:71-76`）。
- `model.freeze_encoders` 没从 YAML 显式传 CLI，只是依赖 parser 默认 `True`；这是脆弱的配置假匹配（`configs/fontdiffuser.py:49`；`launch_e2.py:65-95`）。optimizer 从所有 `requires_grad` 参数收集，因此在当前默认值下确实只训练 UNet；若默认漂移则会静默改变训练（`train.py:323-333`）。
- E1c/E2b official RSI 都取 `refs[0]` 的 Ec style cache，故 E1c 固定 `永`、E2b取同一随机/固定 R 的首项；二者走同一代码（`train.py:139-147`；`configs/e1c_ft_continue_s3407.yaml:41-45`；`configs/e2b_ft_continue_s3407.yaml:40-44`）。但没有独立 ordered-R manifest 文件，只有 YAML list；训练 R 的首项来自 `random.sample` 顺序（`dataset/font_dataset.py:122-131`），因此 D-P3 只部分满足。
- 输出保护允许已有目录内只有 watchdog logs，也允许有 checkpoint 时 resume，最近修复不会把 watchdog log 当失败 run；但它未校验 resume config SHA，且 `last_state.exists()` 即被视作 checkpoint，即使内容残缺（`launch_e2.py:50-60,92-94`；`hrfont-execution-spec.mdc:75,102`）。
- **推理 wiring FAIL**：DPM wrapper的 cond 只有 raw content/style，DPM model每 solver step在线跑 Es/Ec，既不消费 cached content/structure，也不实现 n-shot/Δ/official source equivalence（`pipeline_dpm_solver.py:42-83`；`src/model.py:90-124`）。

### 2.4 `e2_smoke_test.py`（任务 4）— **高风险**

- 本次实际执行 `python scripts/e2_smoke_test.py` 在 import 阶段失败：`ModuleNotFoundError: No module named 'numpy'`，所以本机结果是 **FAIL/未运行**，不是 PASS。
- 测试覆盖：独立 synthetic n-shot采样可复现/无重复、`compute_alpha` top-10/LOO/finite/sum、两邻居 cache delta 算术、源码 `compile()`、以及一个随机 tensor 的 9-token cached forward与 encoder零调用（`e2_smoke_test.py:22-57,105-155`）。
- 它仅在环境缺少 diffusers 时注入 stub；若 diffusers 已安装则用真实包。因此“仍 stub diffusers”的答案是 **条件性 stub**（`e2_smoke_test.py:60-102`）。
- 未覆盖：launcher→parser resolved k_top=10；E2/E2b/E1c 同 mask及 checksum；official source-drop；真实 memmap/cache key/shape/SHA/finite；fp32-online parity、逐层/end-to-end parity；n张 spatial element-wise mean（现 forward test直接造最终 map）；n=1的9-token数据链；CFG保留 structure；resume连续性；heartbeat/STOP；20-step cache-only DPM sample；5k完整 val；matched YAML allowlist（`e2_smoke_test.py:158-168`；`hrfont-execution-spec.mdc:43-53,100-104`）。

### 2.5 结果与训练状态（任务 5）— **高风险**

- 周报与 PROJECT 对快照数字一致：09-05 13:55 E2=9,200/80,000（11.5%，四舍五入12%）、E2b=27,800/80,000（34.75%，约35%），比值3.02；E1c因退出进程占显存未启动（`reports/WEEKLY_20260905.md:58-64`；`PROJECT.md:10,38-40`）。
- 但无法验证“same-commit”：周报未写 SHA；PROJECT写 `d382247`，当前仓库不存在该对象，而实现提交实际为 `447db01`，HEAD又含两个 launcher修复和文档提交。也无法验证实际 resolved seed、启动时刻与未 resume；registry仍 planned且 provenance=TBD（`PROJECT.md:19-20,38-40,149`；`provenance/REGISTRY.md:30-33`）。
- 更关键的是，即便同 seed/同 commit，source-drop bug也使对照不 matched。未中断时两臂 RNG draw顺序在代码上相同、运行速度不影响序列；但 official arm丢弃该 draw，输入分布仍不同，且任何 resume 会因 sampler cursor缺失破坏后续配对（`train.py:139-170,347-385`）。
- E1c晚开本身不必破坏统计因果，只要代码/cache/config/hardware环境冻结并完整记录；风险包括：代码或 cache漂移、GPU型号/驱动/库差异、晚期开跑导致运营选择性停止、未与另两臂共享 wall-clock环境、以及三 seed在deadline前不完整。已有缓解是相同 E1@100k/cache/seed/YAML预算与 D-A1 SHA；缺少的是可验证 commit对象、canonical config SHA、dataset/cache文件SHA、环境锁、registry provenance和真正 exact resume（`configs/e1c_ft_continue_s3407.yaml:8-78`；`train.py:59-75,225-240`；`hrfont-execution-spec.mdc:71-76,100-104`）。
- 周报对方法机制谨慎地写为待验证假设，这一点不过claim（`reports/WEEKLY_20260905.md:13-17,29-32`）。但“二者同日、同提交启动，保证对照可比”是过claim：同日不是充分条件，当前实现已知不 matched且 provenance不可复核；“科学设定已冻结，无待拍板的方法问题”也不成立（`reports/WEEKLY_20260905.md:60-64`）。“E1c 原FD前向再训”也应改称 frozen-encoder 1-shot official-RSI arm，因为它冻结Es/Ec、重启warmup，不是原E1优化轨迹的严格续训（`hrfont-execution-spec.mdc:43-45`；`configs/e1c_ft_continue_s3407.yaml:24-64`）。

#### RSI Q/K/V 审核

RSI audit 的原文是：“**以参考结构为 Query、skip feature 为 Key/Value 做 cross-attention**”（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:33-37`），并写成 `Q=Φq(Fs), K=Φk(ri), V=Φv(ri)`（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:83-94`）。这与代码一致：`CrossAttention(query_dim=style_feat..., context_dim=res...)`，forward 调用 `cross_attention(style_content_hidden_states, context=res_hidden_states)`（`code/official/FontDiffuser/src/modules/attention.py:288-320`）。随后 attention/FFN输出投影为18通道offset，`DeformConv2d`用它重采样的是 UNet skip `res_hidden_states`，再与decoder hidden拼接（`code/official/FontDiffuser/src/modules/attention.py:321-332`；`code/official/FontDiffuser/src/modules/unet_blocks.py:545-563`）。**Q/K/V角色描述正确，无需反转。** 需要保留的细节是 DeformConv2d 不在 attention 内、也不形变 Ec reference；它形变 UNet skip，audit 已正确说明（`reports/RSI_CORRECTNESS_REVIEW_20260905.md:43-47,96-104`）。

### 2.6 PI 必须阻断/决定的事项（任务 6）— **严重风险**

- 当前 pair **不 matched**，原因不是3×速度本身，而是 official source-drop未应用。速度差只反映实现工作量不同；只要无resume且draw序列相同，异步wall-clock不会单独破坏step-based配对（`train.py:139-170,378-389`）。
- deadline粗投影只能给区间。若两臂在09-05 00:00之后启动，则截至13:55的平均速率下界约 E2≥661 step/h、E2b≥1,998 step/h；剩余时间上界约 E2≤107 h（约09-10 01:00前）、E2b≤26.1 h（约09-06 16:00前）。若E1c吞吐近E2b，单seed约40 h。两卡并行、E2种子串行且速率不恶化时，3407/3408/3409约可在09-19前完成，理论上早于09-25；但重跑、5k val pause、共享I/O、故障/resume都会侵蚀余量。上述是用单个状态点的保守算术，不是服务器实测ETA（状态证据：`reports/WEEKLY_20260905.md:60-62`）。
- **推荐修速度，但不要原地优化当前正式轨。** 先修语义阻断并新开matched run；随后把 α 批量GPU计算、topk替换Python sort、role-specific两尺度Ec读取、按batch聚合唯一key/连续row、避免scalar `.item()/float/bool`，再做短benchmark。若PI坚持保留当前轨，应保持代码不变跑完并明确标为非matched诊断，不能把它升级成主证据。

## 3. E2 vs E2b 速度差的定量归因

以下 I/O 字节按五尺度 fp16 payload计算；当前 loader读出后立即扩成fp32，实际内存带宽和PCIe传输更高。耗时是从操作规模和观察到3.02×总吞吐作的区间估计，非服务器 profiler 数值。

| 每步操作（bs=8） | E2 | E2b | 估计额外耗时/影响 | 证据 |
|---|---:|---:|---|---|
| Es spatial+pooled ref fetch | 同一n，期望4.5 refs/样本 | 同一 | 近似相同，非差速源 | `train.py:113-121`；`dataset/font_dataset.py:122-127` |
| α prototype H2D | 8×`[228,n,1024]`，期望约32 MiB fp32 | 0 | 中；且按样本小传输，难合并 | `train.py:91-93,156-159` |
| α cosine/top-K | 8次；每次228×n cosine | 0 | **高**；约1816次/step GPU scalar→Python排序同步，另有meta同步 | `hrfont_delta_v2.py:99-125` |
| structure Ec lookups | 11/样本=88/step；440尺度row | 1/样本=8/step；40尺度row | **高**；E2比E2b多80 lookup/400尺度row | `train.py:143-169`；`hrfont_feature_cache.py:120-125` |
| 总Ec payload（含MCA content） | 约59.1 MiB fp16/step（7.38 MiB/样本） | 约9.84 MiB/step（1.23 MiB/样本） | **高**；约6×磁盘/page-cache row copy，转fp32后CPU内存流量约翻倍 | `hrfont_feature_cache.py:11-17,120-125`；`train.py:129-170` |
| Δ混合 | 8×10×5=400次加权add；每次`float(weight)` | 0 | **高**；400次潜在device scalar同步+大tensor fp32加法 | `hrfont_feature_cache.py:128-137` |
| dropped样本短路 | 0；drop后才判断 | 不适用 | E2浪费约25%的α/I/O/mix前置工作；可在语义允许处早短路 | `train.py:156-169` |
| UNet forward/backward/loss | 相同结构与batch | 相同 | 共同固定成本；会把6×I/O差稀释为观察到约3.02×总差 | `train.py:243-263,388-395` |
| 每5k val | 590 batch，同样昂贵Δ预处理 | 590 batch official | 周期性扩大E2 wall-clock差；不是20-step DPM | `train.py:266-289,427-441` |

**归因排序：** (1) Python scalar同步式 top-K + 权重转换；(2) 10-neighbor五尺度的标量memmap复制/H2D；(3) fp32多尺度mix；(4) 共同UNet。Δ数学构造本身的张量加减不是唯一主因，真正的问题是其当前逐样本、逐邻居、逐尺度、逐scalar实现。需用 `torch.profiler`/NVTX在服务器分别测 `style/alpha/ec_fetch/mix/forward/backward/val` 才能给论文级精确百分比。

## 4. 必须 PI 决策清单

### D-S1：当前 seed3407 如何定性

- A：继续跑完并作为主matched证据。
- B：在安全checkpoint用STOP停两臂，修D-P2和resume/provenance gate，从共同E1@100k以新run ID重跑。
- C：保持当前代码跑完，但降级为非matched工程诊断；同时另开修复后的正式pair。

**推荐 B；若GPU时限不允许则 C。反对 A。** source-drop混杂是方法分布变化，不是可在统计分析中轻易补救的吞吐差（`train.py:139-170`）。

### D-S2：是否先做性能修复

- A：语义修复后先做20–100 step分段profiling/benchmark并批量化，再开正式80k。
- B：只修source-drop，保留慢路径直接重跑。

**推荐 A**，但性能patch必须在两臂共同commit、通过draw checksum/cache parity后从step0新开；不得只优化E2后原轨resume（`hrfont-execution-spec.mdc:76,100-104`）。

### D-S3：旧轨与deadline

- A：保留旧轨占卡至80k，再补三seed。
- B：现在止损重跑正式3407，优先确保09-25前完成修复版主pair；E1c与额外seeds按预注册优先级排队。

**推荐 B。** 当前速率理论上仍有余量，但把约4–5天GPU继续投入已知非matched E2会挤压3408/3409和修复验证。

### D-S4：E1c晚开是否可接受

- A：允许晚开，要求锁定同一可解析commit、cache/config/dataset SHA、容器/驱动/GPU记录和停止规则。
- B：要求三臂同日重开。

**推荐 A。** 同日不是因果识别条件；可复现环境和预注册停止规则更关键。E1c应明确命名为 frozen-encoder 1-shot official-RSI arm（`hrfont-execution-spec.mdc:43-46`）。

### D-S5：论文端点与validation定义

- A：维持当前随机单timestep val loss作为checkpoint诊断，另实现固定noise/manifest的20-step DPM生成评测；80k为主端点。
- B：把当前val loss称为YAML中DPM eval。

**推荐 A；反对 B。** 当前代码完全没有运行DPM配置（`train.py:266-289`；`configs/e2_stage_a_s3407.yaml:71-78`）。

## 5. 必须修的缺陷清单

### 继续训练前必须修

1. official与Δ source在共同core中消费同一个独立 `.25` mask；添加三臂mask checksum与tensor断言，修复D-P2（`train.py:139-170`）。
2. 修复exact-ish resume：持久化/恢复DataLoader generator、sampler permutation+cursor、epoch/batch cursor、best与AMP scaler；做连续跑vs中断恢复的batch/R/mask/loss/state对比（`train.py:173-222,347-370`）。
3. 建立真实matched preflight：structural YAML allowlist、resolved canonical config SHA、可解析code SHA、dataset/cache文件SHA、环境与实际CLI；registry/provenance在启动时落档（`launch_e2.py:28-100`；`provenance/REGISTRY.md:30-33`）。
4. 通过真实cache的逐尺度fp32-online误差、top-10 agreement、逐RSI层offset与end-to-end noise_pred parity；Es pooled改fp32（`hrfont_build_e1_caches.py:205-236`）。
5. smoke新增launcher/parser k=10、official/Δ同mask、n=1/n=8九token均值、CFG、resume、STOP/60s heartbeat、真实memmap与matched diff；当前smoke依赖环境也必须能实际执行（`e2_smoke_test.py:158-168`）。
6. heartbeat改为wall-clock≤60s；checkpoint保存采用原子替换，避免 watchdog 在半写 `last_state` 上resume（`train.py:124-126,192-202,416-431`）。

### 可训练结束后修（但正式推理/论文出数前必须修）

1. DPM/sample改为cache-only的9-token+n-shot+cached MCA+official/Δ structure；加20-step/order2/CFG7.5 paired-noise smoke。当前推理不是E2/E2b（`sample.py:78-105,143-161`；`src/model.py:98-116`）。
2. role-specific Ec layout：content保留五尺度，target/style只存RSI消费的s1/s2，将约93.39 GiB降至约63.9 GiB；训练改batched/连续row fetch（`hrfont_feature_cache.py:11-17,109-125`）。
3. α用torch批量cosine+`torch.topk`/稳定tie策略，批量唯一neighbor fetch与mix，消除Python scalar同步；先固定数值parity再测速（`hrfont_delta_v2.py:99-125`；`hrfont_feature_cache.py:128-137`）。
4. ordered-R用独立manifest及SHA，而不是仅YAML list；固定val noise/timestep，明确区分train-loss diagnostic与DPM generation eval（`dataset/font_dataset.py:122-131`；`train.py:266-289`）。
5. builder manifest补全dataset/split/key/file SHA、shape/finite/norm检查与原子发布；build入口显式接受expected init SHA（`hrfont_build_e1_caches.py:137-201`）。
6. 修正文档过claim和状态：删除“同日同提交即保证可比”“无待拍板问题”，将registry从planned更新为有可验证provenance的真实状态；运行SHA必须对应可解析Git对象（`reports/WEEKLY_20260905.md:60-64`；`provenance/REGISTRY.md:30-33`）。

## 6. 审查执行记录

- `git rev-parse HEAD`：`827046dfb588a3f9f0b0d0a97396e585b7ff9fa4`；审查开始时工作树无改动。
- `python scripts/e2_smoke_test.py`：**FAIL before tests**，`ModuleNotFoundError: No module named 'numpy'`。
- `python3 -m py_compile ...`：首次因系统Python试图写受sandbox禁止的用户cache目录而报 `PermissionError`；设置 `PYTHONPYCACHEPREFIX=/private/tmp/hrfont-review-pyc` 后，对本次审查的六个Python实现文件全部通过。smoke自身的 `compile()`覆盖六个源文件，但因前述numpy缺失未执行到该阶段（`e2_smoke_test.py:144-155`）。
- `git cat-file -t d382247`：`fatal: Not a valid object name d382247`。
- 本地没有 `artifacts/e2/`、服务器run目录与训练日志；所有吞吐/ETA结论均明确基于09-05 13:55单点状态，不冒充实时验证。
