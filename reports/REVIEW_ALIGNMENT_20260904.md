# HR-Font 对齐决策实现审查（2026-09-04）

审查对象：`HEAD 6eca165eb7be0b07b40f974717c5e6ca71f936e3`。范围仅限只读核验 D1–D6、E2b、E2c/E5 及回复稿所述主张；本报告不修改任何代码、配置、计划、provenance 或 manifest。

## 1. 总评

**结论：核心数学调用链基本对齐，但当前状态不应直接启动正式 E2/E2b，也还不能声称“全链路已对齐”。**

已经核实成立的部分：

- D1 的活跃训练路径确实以同一张 Noto `ContentImage/{cp}.png` 同时作为 MCA/Identity 输入和 Δ 减数；FZKTJW/FZFXKTJW runtime mapping 已从当前 Stage-A 训练代码删除（`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:96-106`，`code/variants/cn2west_stage_a/FontDiffuser/train.py:101-113`）。
- D2 的 `compute_alpha` 已统一为 query `[n,D]`、candidate `[N,n,D]`，逐字符 cosine 后平均；训练、V5、V6 和 delta smoke 的调用 shape 都一致（`scripts/hrfont_delta_v2.py:80-137,305-313`，`code/variants/cn2west_stage_a/FontDiffuser/train.py:101-107`，`scripts/hrfont_validate_e1_encoders.py:187-220`）。
- D5 的 gate/test 数据结构已隔离：gate 只取 val16 与 train228 的 V6，`test_report` 不进入 `print_summary`（`scripts/hrfont_validate_e1_encoders.py:258-296`）。V2/V4 只消费传入的当前 split 字体，无 test→val 回流（同文件 `:145-185`）。
- D6 的同一 R 消费关系在训练中成立：dataset 先抽同一组 `ref_chars/ref_image_paths`，live Es 同时产出 style condition 与 per-char query，cache 只提供 candidate prototype（`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:122-149`，`code/variants/cn2west_stage_a/FontDiffuser/train.py:53-60,101-107,251-257`）。
- E2 与 E2b 在不中断、同 seed 的前提下，当前有效 forward 差异确实主要是 RSI structure source；E2b 也消费 `delta_draw`，并以 `R[0]` 进入原 offset head（`code/variants/cn2west_stage_a/FontDiffuser/train.py:93-107,124-140,251-258`）。

正式开跑前有四个阻断项：

1. 配置中的 E1@100k 路径名与 E1 实际 checkpoint 命名规则不一致，launcher 很可能直接失败。
2. Es cache 没有和 E1@100k 的 style-encoder SHA 建立可执行不变量；若现有 cache 来自 best@98k，α 是跨表征空间 cosine。
3. Stage-A 目前没有符合新协议的推理/val/best 路径；继承的 `sample.py` 实际不送 Δ/official structure，且仍 Resize。
4. checkpoint 不保存 RNG/sampler 状态；一旦 resume，E2/E2b 的 R、noise、CFG/Delta draw 无法维持 matched。

静态/冒烟状态：`scripts/e2_smoke_test.py` 的 n-shot sampler 与源码编译检查通过；模型 smoke 因本机 Python 无 `torch` 被跳过，但脚本仍打印总 `PASS`（`scripts/e2_smoke_test.py:168-180`）。`hrfont_delta_v2.py smoke` 与 encoder validation relaxed-gate smoke 均因 `ModuleNotFoundError: torch` 未执行，不能记为通过。

## 2. 发现的问题清单

### F01 — E1@100k 初始化目录名与产物命名不一致

- **位置：** `configs/e2_stage_a_s3407.yaml:7`；`configs/e2b_ft_continue_s3407.yaml:7`；`code/variants/cn2west_ft_v2/FontDiffuser/train.py:50-66,347-355,369-375`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:31-37`
- **问题：** 两份 E2 配置写 `runs/E1-FTV2-A-S3407/step_100000`，但 E1 训练器只创建 `global_step_{global_step}` 与 `last_state`。Stage-A launcher 又对配置目录下三个权重 fail-fast。本 checkout 不含 gitignored 权重，无法确认训练机是否人工建立了 `step_100000` alias；provenance 也只登记 run 根目录（`provenance/runs/E1-FTV2-A-S3407.json:46-49`）。
- **严重度：高**
- **建议动作：** **【执行者可直接修】** 开跑机先核实实体目录；默认改为不可变的 `global_step_100000`，不要依赖未登记 alias，也不要用会继续覆盖的 `last_state`。

### F02 — Es cache 与 E1@100k 未绑定；现状无法证明 α 在同一 Es 空间

- **位置：** `configs/e2_stage_a_s3407.yaml:20`；`configs/e2b_ft_continue_s3407.yaml:20`；`code/variants/cn2west_stage_a/FontDiffuser/configs/fontdiffuser.py:43`；`scripts/hrfont_es_cache.py:88-118`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:31-37`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:224-226`
- **问题：** cache 文件仍名为 `es_per_font_char_e1_best.pt`，而 init 已改为 100k。builder 会在旁侧 manifest 写 `es_checkpoint_sha256`，但 launcher 只检查 `.pt` 存在，训练器只裸加载 tensor，从不读取 manifest 或比对 init 的 `style_encoder.pth` SHA。若 cache 实际来自 best@98k，则 live query 用 100k Es、candidate prototype 用 98k Es，cosine 不再有共同坐标系。即便文件实际由 100k 生成，当前仓库也无法证明这一点。
- **严重度：高**
- **建议动作：** **【必须 PI 决策】** 正式冻结“α retrieval Es 必须等于模型内冻结 Es”这一不变量；推荐从 100k 重建并命名为 `es_per_font_char_e1_step100000.pt`，launcher fail-closed 校验 manifest Es SHA、cache SHA、split/stems/chars/dtype/shape。E2b 实际不读 cache，却被 launcher 强制要求同一路径（`train.py:93-96` 对比 `launch_e2.py:31-37`），可由执行者顺手解除无效依赖。

### F03 — cache 的 shape/dtype/归一化契约正确，但缺少运行时完整性验证

- **位置：** `scripts/hrfont_es_cache.py:80-105,108-118`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:53-60,76-80,101-107`；`scripts/hrfont_delta_v2.py:95-101`
- **问题：** builder 对每个 `(font,char)` 保存 L2-normalized pooled Es，落盘 fp16；训练取出后 `.float()`，live query 也逐字符转 fp32 并 L2 normalize，`compute_alpha` 再对双方归一化。shape 与归一化一致，余下仅 fp16 量化差异。问题在于训练启动时不验证 228×338 coverage、键集合、tensor ndim/D、finite、norm、manifest/SHA；错误 cache 可能直到某个字符被抽到才报 KeyError，NaN 还可能静默变成空邻域。
- **严重度：中**
- **建议动作：** **【执行者可直接修】** 在 launcher/preflight 一次性验证完整 cache contract；不需要改变 D2 数学。

### F04 — E2b 确实复用 E1 offset head，但 E2b step0 不等价 E1 step0

- **位置：** `code/variants/cn2west_stage_a/FontDiffuser/train.py:93-96,124-126,153-156`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/unet_blocks.py:545-561`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/attention.py:266-330`；`code/variants/cn2west_ft_v2/FontDiffuser/src/model.py:34-47`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:53-60`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:36-48`
- **问题：** Stage-A 整体加载 E1 UNet；`OffsetRefStrucInter/proj_out` 没有新增 gate 或新 head，因此 E2b 的 `Ec(R[0])` 进入的就是 E1 原 head。可是 E1 style cross-attention 接收单张图的 spatial `style_emd`，Stage-A 则把多张 R 的 **逐张 L2-normalized pooled vector** 求均值，再扩成 `1×1` map。即使 n=1，后者也不是 E1 spatial style map。因此“同一头”成立，“完整 forward 等价”不成立。
- **严重度：高（解释/归因）**
- **建议动作：** **【必须 PI 决策】** 推荐把 E2b 定义为“E2 的 matched official-RSI control”，不称“E1 forward-equivalent FT continuation”；E2/E2b 仍可作只改变 structure source 的 matched pair，因为它们共享新的 n-shot style 表示。另行保留一个只读 E1 step0 anchor sanity，量化 style 表示变化本身。

### F05 — D6 的 “mean-pool Es(R)” 还隐含了未明说的归一化/空间降维决定

- **位置：** `code/variants/cn2west_stage_a/FontDiffuser/train.py:53-60`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/style_encoder.py:428-442`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:36-48`；`.cursor/rules/hrfont-execution-spec.mdc:50`
- **问题：** 代码语义不是简单的 “mean-pool Es(R)”：它取 StyleEncoder 第二返回值（spatial average 后的 pooled vector），先逐张 L2-normalize，再跨 R 求均值，且不对最终均值再 normalize；随后只给 cross-attention 一个 token。exec-spec 没冻结这四个细节。它们对 E2/E2b 相同，所以不破坏二者 pair，但会形成相对 E1 的额外 architecture/scale transition。
- **严重度：中**
- **建议动作：** **【必须 PI 决策】** 推荐接受当前表示作为 D6 的精确定义并写入规范，同时停止声称 E2b 与 E1 前向等价；若 PI 原意是平均 spatial maps 或平均 raw pooled vectors，必须在开跑前改，不能跑后解释。

### F06 — 当前 Stage-A 推理/评测路径没有实现新条件流，5k eval/best 实际不可用

- **位置：** `configs/e2_stage_a_s3407.yaml:63-72`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:27-67`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:284-301`；`code/variants/cn2west_stage_a/FontDiffuser/sample.py:78-90,126-161`；`code/variants/cn2west_stage_a/FontDiffuser/src/dpm_solver/pipeline_dpm_solver.py:42-82`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:95-111`
- **问题：** YAML 声明每 5k eval，但 launcher 不读取 `eval`，trainer 只保存 checkpoint，没有 val/best。继承的 `sample.py` 只接受单张 style image、仍做 bilinear Resize；DPM pipeline 的 `cond` 只有 content/style 两项，不构造 n-shot style vector 或 Δ/Ec(R[0])；`FontDiffuserModelDPM` 因 `cond[3]` 不存在而把 structure 设为全零。用该路径评测将不是 E2，也不是 E2b。
- **严重度：高**
- **建议动作：** **【执行者可直接修；CFG 结构语义需 PI 确认】** 在正式 80k 前实现固定 ref8、per-char α、Content-neutral、E2/E2b structure source 和 paired-noise 的 val/inference；移除 A 协议 Resize。PI 需冻结 classifier-free inference 时 structure stream 在 cond/uncond 两支的处理，因为当前训练的 CFG mask 与 Delta-drop 是分离的（`train.py:255-260`）。

### F07 — 无 gate 后早期 head recalibration 存在，但当前甚至没有 best 选择器

- **位置：** `code/variants/cn2west_stage_a/FontDiffuser/src/modules/unet_blocks.py:545-561`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/attention.py:298-330`；`configs/e2_stage_a_s3407.yaml:63-68`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:284-301`
- **问题：** 已训练 offset head 从绝对 `Ec(S)` 切换到尺度/分布不同的 Δ，5k 很可能主要反映 head 重新校准。这不是 correctness bug，而是 milestone eligibility 的预注册问题；但现有 trainer 根本不执行 eval/best，无法落实任何规则。
- **严重度：中**
- **建议动作：** **【必须 PI 决策】** 推荐在开跑前冻结“0/5k 仅诊断，best 候选从 10k 起”，同时完整保存并报告 0/5k；E2/E2b 使用同一 eligibility。禁止观察曲线后再决定 burn-in。

### F08 — matched draws 只在无中断运行中成立；resume 不保存 RNG

- **位置：** `code/variants/cn2west_stage_a/FontDiffuser/train.py:143-165,251-265`；`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:122-125`；`.cursor/rules/hrfont-execution-spec.mdc:65-66`
- **问题：** Stage-A checkpoint 只保存 step/optimizer/scheduler，没有 Python、Torch CPU/CUDA RNG、DataLoader generator/sampler cursor 或 AMP scaler。R 用 Python `random`，Delta/CFG/noise/timestep 用 Torch RNG。任一 arm resume 后都不能保证继续消费与另一个 arm 相同的 R/draw，违反 matched spec。
- **严重度：高**
- **建议动作：** **【执行者可直接修】** 补 exact-resume state 与前 N batch/draw checksum；在修复前若发生中断，不得把恢复后的 E2/E2b 宣称为严格 matched。

### F09 — soft-α 可被 ε 全部截空，现有 V6 gate 又看不见这一失败模式

- **位置：** `scripts/hrfont_delta_v2.py:64-76,107-137`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:130-133`；`scripts/hrfont_validate_e1_encoders.py:211-223,280-296`
- **问题：** softmax 后只保留严格 `weight > .01`；若 227 候选分布较平，所有权重都可能被删除，训练把该样本结构源静默置零。V6 的 gate 仅使用截断前 `max_cosine`，不 gate `n_active/top1_mass`，所以即使所有 α 都为空仍可能通过 “alpha quality”。这是 D2 新规则带来的未决语义：空邻域究竟是允许 abstention、fail-fast，还是回退 top-1。
- **严重度：高**
- **建议动作：** **【必须 PI 决策】** 推荐正式 cache 上先跑全量 R 采样诊断，并冻结“空邻域率必须为 0；否则 fail-fast 并重新定 ε”，而不是静默改算法。若 PI 希望允许拒识，则明确为方法组成、报告空率，并定义 Δ=0 语义；若希望始终有邻居，再明确 top-1 fallback。V5/V6 增加 `empty_rate/n_active/top1_mass` gate。

### F10 — V1 的 rank 是自匹配；pairwise negatives 固定在单一字符且重复加权

- **位置：** `scripts/hrfont_validate_e1_encoders.py:121-143`
- **问题：** V1 query 和 gallery 中该 eval 字体的 prototype 都由同一组完整 ref8 feature 构成，self score 本质为同张量自匹配，rank/R@k 接近恒定；返回 note 却仍写 “queries use disjoint ref8 halves”。pairwise AUC 的 positive 遍历 even×odd，但 negative 永远只取其他字体的 `self.ref8[1]`（“和”）且被 odd-loop 重复加入，gate 会偏向单一字符。
- **严重度：中**
- **建议动作：** **【必须 PI 决策】** 冻结 V1 的统计目标：推荐 rank 只有在能构造独立 view/独立字符证据时才保留；否则移出 gate。pairwise negative 改为对称的跨字体 even×odd 配对，并修正 note。执行者可按冻结定义直接修。

### F11 — val/test 隔离正确，但 test V1 为 null、真实 CLI gate 失败仍返回成功

- **位置：** `scripts/hrfont_validate_e1_encoders.py:132-143,258-296,349-367`
- **问题：** `test_report.V1.pairwise_auc=None` 是当前代码的确定结果，因为 pairwise 只为 `font in val_fonts` 构造；JSON `null` 合法，也不进入 gate。`print_summary` 确实只读 gates，V2/V4 也没有 test leak。另一个实现缺口是非-smoke CLI 只调用 `print_summary(results)`，没有用布尔结果设置非零退出码；自动化若只看 exit code，会把 gate FAIL 当作命令成功。
- **严重度：中**
- **建议动作：** **【test null 是否保留：必须 PI 决策；退出码：执行者可直接修】** 推荐保留 test pairwise `null` 并在 schema 明确 nullable，避免为“填表”让 test 参与开发；真实 gate FAIL 应返回非零。若 PI 要 test pairwise AUC，则必须预先定义 test-only 统计，仍不得反馈调参。

### F12 — D4 活跃代码无 gate，但 YAML 与多份规格仍宣称 zero-init

- **位置：** `configs/e2_stage_a_s3407.yaml:7-8`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:27-67`；`.cursor/rules/hrfont-execution-spec.mdc:49,75`；`reports/E2_IMPLEMENTATION_READINESS.md:72-79`
- **问题：** runtime 的确没有新增 gate/adapter，整个 E1 UNet 被直接加载；但 E2 YAML 仍写 `init.new_layer: zero`，launcher 完全不读该字段。权威 spec 第 49 行说 no-gate，第 75 行又说 zero-init；readiness 仍要求 zero-init。配置/provenance 会虚假记录不存在的新层。
- **严重度：中**
- **建议动作：** **【执行者可直接修】** 移除 `new_layer` 或改为 `offset_head: reuse_e1_no_gate`，同步规格；不需要再次请 PI 决定 D4。

### F13 — top-3 ablation 的 `k_top` 配置字段不可执行；proto manifest 是未版本化 breaking change

- **位置：** `configs/e2_stage_a_s3407.yaml:31`；`code/variants/cn2west_stage_a/FontDiffuser/configs/fontdiffuser.py:33-39`；`code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:51-54`；`code/variants/cn2west_stage_a/FontDiffuser/train.py:98-100`；`scripts/hrfont_delta_v2.py:217-248,331-363`
- **问题：** YAML 有 `k_top: 3`，但 parser/launcher/training 都不传 `k_top`；当前只是碰巧使用 `DeltaConfig` 默认 3，改 YAML 不生效。`proto` CLI 的 `fonts[*].prototype` 已从 `[D]` 变为 `[n,D]`，全仓未找到内部 consumer，因此当前不会破坏仓内调用；但 manifest 没有 schema/version/shape/aggregation 字段，外部 consumer 可能静默误解。另外 CLI 把 `variant` 列为合法 encoder choice，却会主动拒绝（`scripts/hrfont_delta_v2.py:331-350`）。
- **严重度：中（k_top）/低（manifest/CLI）**
- **建议动作：** **【执行者可直接修】** 接通 `delta_k_top`；manifest 增加 schema、`prototype_shape`、`aggregation=per_char`，或把字段改名为 `per_char_features`。若已有未登记外部 consumer，**【必须 PI 决策】** 是否保留兼容字段。

### F14 — D1 同一 Content 被 Ec 编码两次：没有 correctness 问题，但有额外成本

- **位置：** `code/variants/cn2west_stage_a/FontDiffuser/train.py:109-139,243-257`；`code/variants/cn2west_stage_a/FontDiffuser/src/model.py:50-56`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/content_encoder.py:386-405,426-435`
- **问题：** joint Ec pass 为 Δ 编码 neutral，随后 model forward 为 MCA 再编码 batch content。两边输入对非-CFG样本相同，Ec 冻结且每步回到 eval；实际 ContentEncoder 的活跃 blocks 没有 batch-dependent BN，因此没有语义或随机性错误。CFG 样本则是有意让 MCA content 置零、Δ stream 保持独立（`train.py:255-260`）。代价只是额外 Ec compute/显存带宽。
- **严重度：低**
- **建议动作：** **【执行者可直接优化，但不建议首轮 matched 前改接口】** 先保留以降低改动风险；若后续复用 feature，必须证明逐层 allclose 且不改变 CFG stream。

### F15 — E2c 的 retrieval space 尚未定义；E5 当前未实现但没有被 Stage-A 代码阻塞

- **位置：** `.cursor/rules/hrfont-execution-spec.mdc:49,50,77,79`；`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:91-98,113`；`code/variants/cn2west_stage_a/FontDiffuser/dataset/font_dataset.py:140-149`
- **问题：** `configs/` 目前只有 E1/E2/E2b，没有 E2c。E2c 若从 official P1 冷启，却复用 E1@100k candidate cache，会再次形成 P1 live query 对 E1 prototype 的跨空间 cosine。若为 E2c 单建 P1 cache，retrieval metric 也随 init 改变，E2c 就是“整套 cold system”而非只改 generator init。当前 Stage-A 没有 `gap(c,R)`、SupportAdapter 或 support 数据流，这对 E2 是正确的；保留的 `ref_chars/ref_image_paths` 足够让 E5 后续按同一 R 扩展，但 E5 必须另建 gap/support artifact 与 SHA。
- **严重度：中**
- **建议动作：** **【必须 PI 决策】** 推荐 E2c 使用 P1-specific per-char cache，并明确它是 cold-system sensitivity，不称单变量 matched；若目标真是只比较 generator init，则需独立冻结同一个 retrieval encoder，并把 query 也送该独立 encoder。E5 暂不改 Stage-A，只在实现时强制 `gap(c,R)` 与实际 sampled R、B₀=Content、encoder/cache SHA 绑定。

### F16 — 重建/追溯产物本身已陈旧，可能重新引入被删除实现

- **位置：** `docs/patches/cn2west_stage_a.diff:27-29,621-648`；`code/variants/cn2west_stage_a/VARIANT.md:1-2`；`scripts/hrfont_validate_e1_encoders.py:4-8`
- **问题：** stage-A diff 仍包含 `b0_font=FZKTJW`、FZFXKTJW fallback、mean-pooled prototype，按它重建会恢复 D1/D2 的旧实现；Stage-A `VARIANT.md` 还错误声明 `VARIANT=cn2west_ft_v2`；validation 顶部 docstring 仍宣称已删除的 `--b0-dir`/FZKTJW 用法。
- **严重度：高（patch 可重建性）/中（metadata/docstring）**
- **建议动作：** **【执行者可直接修】** 在 PI 完成下列文档口径决策后重生成 patch、修正 variant marker 和 docstring；这些不是算法新决策。

## 3. 待决策清单

### D-A1（最高优先级）— cache/init 一致性

**选项：** A. 强制 E2 cache 的 Es SHA 等于 E1@100k init Es；B. 允许独立 retrieval Es 并将其作为另一模型组件。**推荐 A：重建/重命名 100k cache，并 fail-closed 校验。**

### D-A2 — 文档与重建产物是否一次性 mass-update

**选项：** A. 现在统一五份主文档、execution spec、patch、VARIANT/docstring；B. 仅在回复中加勘误。**推荐 A：否则 collaborator 会从 patch/计划重新引入旧 B₀、旧 α 与 zero-init。**

### D-A3 — E2b 的科学身份

**选项：** A. 定义为 E2 的 matched official-RSI control；B. 要求它同时与 E1 step0 前向等价并重做 n-shot style 接线。**推荐 A，并明确“同头不等于同 forward”。**

### D-A4 — D6 style condition 的精确定义

**选项：** A. 当前“逐张 normalized pooled vector→mean→1 token”；B. mean raw pooled；C. mean spatial style maps。**推荐 A（尊重已实现 D6），但必须将该额外 E1→E2 transition 写入方法与控制解释。**

### D-A5 — no-gate 后 best eligibility

**选项：** A. 0/5k 仅诊断、10k 起可选 best；B. 所有 5k milestone 均可选。**推荐 A，必须开跑前冻结且两臂一致；同时先补真实 val/best 实现。**

### D-A6 — soft-α 空邻域

**选项：** A. 空率必须为 0，否则 fail-fast/重定 ε；B. 至少回退 top-1；C. 允许 abstention 并报告。**推荐 A；它最少引入隐式算法分支。**

### D-A7 — V1 与 test nullable

**选项：** A. test `pairwise_auc=null` 保留、V1 rank 移出 gate并修正 pairwise 配对；B. 另定义预注册 test-only AUC。**推荐 A，避免为了填 test 指标反向开发。**

### D-A8 — E2c retrieval space

**选项：** A. P1 init 配 P1 query/cache，定义为 cold-system；B. 独立冻结 E1@100k retrieval encoder，只改 generator init。**推荐 A；B 才是更严格单变量 init 实验，但需要新增独立 encoder 通路。**

### D-A9 — Stage-A CFG inference 的 structure 语义

**选项：** A. 按训练的 streams-separated 语义，在 style/content CFG uncond 分支保留 structure，Delta-drop 单独控制；B. joint uncond 同时清零 structure。**推荐 A，与 `.cursor/rules/hrfont-execution-spec.mdc:50` 和训练 `train.py:255-260` 更一致；需在 DPM eval 实现前冻结。**

### D-A10 — proto manifest 向后兼容

**选项：** A. 若无外部 consumer，直接升 schema 并将二维字段显式命名；B. 同时保留旧 mean prototype。**推荐 A；若 PI 确认已有 collaborator 外部 consumer，再选 B 过渡一个版本。**

## 4. 文档陈旧引用清单（exact quotes）

以下只列与本轮 D1–D6 对齐直接相关的陈旧文本。引用内容逐字抄录；为控制长度，个别长段只摘录足以识别陈旧主张的完整句子或完整表格行。通用符号 `B₀` 本身不必删除；只要文档明确 `B₀ = Noto ContentImage`，公式写 `−Ec(B₀)` 仍可成立。

### 4.1 `reports/DELTA_RSI_DESIGN_RATIONALE.md`

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:71`

  > 不是：`Content` 是 **Noto Sans CJK Regular** 上目标字符 $c$ 的 A 渲染，送入 $E_c(C)$ 作为 MCA/Identity 条件；`B_0` 现已统一为 **Noto ContentImage**（合作者 2026-09-04 决策）：Δ 减数与 Content/Identity 输入是同一张 Noto 同字渲染，并统一 RS-gap 与 Support 的结构坐标（[`.cursor/rules/hrfont-execution-spec.mdc:15`](../.cursor/rules/hrfont-execution-spec.mdc#L15)、[`23`](../.cursor/rules/hrfont-execution-spec.mdc#L23)–[`26`](../.cursor/rules/hrfont-execution-spec.mdc#L26)）。两者都是“中性同字”，但职责不同：Content 沿 official/FT-v2 的 Noto 数据约定以保持身份输入和基线可比；B₀ 沿已冻结的 FZKTJW 结构域，使 Δ、gap、support 共用一个不随目标字体变化的原点（[`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:26`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L26)–[`30`](./EXPERIMENT_PLAN_CN2WEST_V2.md#L30)）。概念上可以令 `B₀=Content`，减法与零模式仍成立，并不存在理论禁忌；但这会同时更换 Δ 原点、RS-gap/support 坐标和现有 cache/SHA，不能在主实验中无痕替换，否则破坏 matched 比较。若要统一，应作为独立消融重建全链路产物，而不是把两者在论文符号或数据加载中混称为同一张图。

  同一行前半说已统一，后半仍说 FZKTJW 且“若要统一”，内部自相矛盾。

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:99`

  > | Content | Noto Sans CJK Regular 的目标字符 $c$，A 协议 96×96 RGB PNG | $E_c(C)$ → MCA/Identity；不是 Δ 减数 | 随 target 字符确定，不从字体池随机 | 同一固定 Content 字体与目标字符集 |

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:100`

  > | B₀ | FZKTJW/FZFXKTJW 的同一目标字符 $c$，A 协议渲染 | Δ/RS-gap/Support 共用的中性结构坐标；不替代 Content 身份输入 | 固定字体、固定同字，无随机采样 | 同一 B₀ 与同字规则 |

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:103`

  > | Δ 减数 | $E_c(A(B_0,c))$，即 B₀ 上与当前目标完全相同的字符 $c$ 的多尺度特征 | 从每层 `Σ_s α̃_s Ec(A(B_s,c))` 中减去，结果 Δ → RSI | 随 $c$ 确定；不随机、不取 Content 图 | 同一 B₀、同一 $c$、同一 Ec/checkpoint 契约 |

同文件还残留其他本轮已覆盖决策：

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:75`

  > 没有额外手工风格特征或另训检索器：α 只用冻结 $E_s$ 的 ref8 编码；具体实现把目标字体八个已归一化向量 `[8,D]` 先均值池化并再归一化，然后与各库字体同样均值池化、归一化的 prototype 做 cosine

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:89`

  > 准确的当前实现语义是“八个已归一化 Es 向量先 mean-pool、再归一化、再与库 prototype 做一次 cosine”

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:93`

  > 必须区分两次采样：生成模型的 style 条件输入从该字体目录用 `random.choice(images_related_style)` 在 338 池随机取一张，而 α 始终读取固定八字并满足 `ref8_feats=[8,D]` 契约。训练 α 与评测 α 使用同一 ref8 字集，因而 α 的输入定义训评一致；训练时当前字体若在 train228 中仍须 leave-one-out，评测字体则只作 query、绝不进库

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:25`

  > zero-init 的新 offset 接入使 step 0 也近似该基线。

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:42`

  > offset 接入/末端必须 zero-init，并允许经预注册的 scale adapter 校准

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:79`

  > E2 设计要求独立 `.25` Delta-drop 与 step-0 zero-init

- `reports/DELTA_RSI_DESIGN_RATIONALE.md:107`

  > 另一个较小的文档差异是旧 §4.1 写“逐字 cosine 再平均”，而当前代码是“八向量 mean-pool 后 cosine”；论文应以最终锁定实现为准。

  现在实际代码已经改成逐字 cosine 再平均，故这句的“当前代码”判断反向陈旧。

### 4.2 `reports/EXPERIMENT_PLAN_CN2WEST_V2.md`

- `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:29-30`

  > | α/Δ 库 | 仅 train237；训练目标字体 leave-one-out；val/test 永不入池。对同一字符 (c)，`Δ=Σα_s Ec(A(B_s,c))−Ec(A(B₀,c))`。 |
  > | B₀ | **FZKTJW 固定**，但重新用 A 渲染；它作为 Δ 减数、RS-gap、support neutral 的共同坐标系。沿用已定义的单一中性结构底，避免同时改变算法语义。 |

- `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:95`

  > | init | **E1 FT-v2 best**；Ec/Es 冻结；Δ→RSI 新接入层零初始化，使 step0 等价无Δ | 同一 E1 best；官方 RSI | official P1；Δ层同 E2 |

- `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:136-138`

  > data: {dataset_id: fontdiffuser-p261-t295-s338-cn2west-v2a-r1-HASH, dataset_sha256: HASH, protocol: A, canvas: 96, resize: false, content_font: NotoSansCJK-Regular, b0_font: FZKTJW, ref8: 永和书风骨韵天地}
  > model: {base: official_p1, rsi_source: delta, delta: {enabled: true, drop: 0.25, init: zero}, support: {enabled: false}, scr: {enabled: false, weight: 0.01}}
  > train: {steps: 80000, batch_size: 1, accumulation: 4, lr: 1.0e-5, scheduler: linear, warmup_steps: 2000, optimizer: adamw, betas: [0.9, 0.999], weight_decay: 0.01, eps: 1.0e-8, fp16: true, grad_clip: 1.0, cfg_joint_drop: 0.10, ema: false, seed: 3407}

- `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:146`

  > 配置 wrapper 还必须在调用它之前检查：schema、无 TBD/null、A/p261、261=237+16+8、dataset/config/model SHA、96无resize、Content=Noto、B₀=FZKTJW、matched diff allowlist、seed/RNG resume、输出目录不存在、R0 ink gate=pass。

- `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:168,175`

  > | D-FT | E1 FT-v2 从 official P1、A/train237 重训；旧 FT legacy | 消除旧42字体/旧渲染域。 |
  > | B₀ | FZKTJW，重新 A 渲染 | 保持 Δ/RS-gap/support 共用坐标语义。 |

### 4.3 `reports/ICLR2027_HRFONT.md`

该文档 `:118` 已将 B₀ 定义为 Noto/source，因此一般性的 `−Ec(I(B₀,c))` 符号不是 D1 陈旧引用；需要更新的是两处仍把 Δ 写成像素混合后再进 Ec：

- `reports/ICLR2027_HRFONT.md:76`

  > \Delta = E_c\!\big(\textstyle\sum_s\tilde\alpha_s I(B_s,c)\big) - E_c(I(B_0,c))

- `reports/ICLR2027_HRFONT.md:221-226`

  > I_\Delta=\sum_{s\in\mathrm{top}M}\tilde\alpha_s\,I(B_s,c),\quad
  > \Delta=E_c(I_\Delta)-E_c(I(B_0,c))
  >
  > \(\tilde\alpha\) 是 top-\(M\) 重新归一化后的系数。

另有 D2/数据口径陈旧：

- `reports/ICLR2027_HRFONT.md:122-135`

  > **池，不是再聚 8～12 套。** CF-Font 把 basis 收到 ~10，是因为他们要在小基底上插值内容。我们只要 top-\(M\) 近邻混 Δ，42 套 × 8 个参考的 \(E_s\) 一次算完可缓存。
  >
  > | 候选池 \(\mathcal{P}\) | 训练 42 套，不含 Demo-8 | 训练时 leave-one-out：当前 \(f\notin\mathcal{P}\) |
  > | 字体原型 | \(e_s=\frac1m\sum_k E_s(I(B_s,r_k))\)，\(r=\)`永和书风骨韵天地` | 测试时用户 \(R\) 与各 \(e_s\) 比；训练时用该字体同一 8 字 |
  > | \(M\) | **3** | 混进 Δ 的套数；top-3 重新归一化 |
  > | 是否设 α 下限 | **否** | 永远用 top-3，不因「不够像」拒识 |
  > 推理只做 8×42 次 style cosine + top-3 读图。

- `reports/ICLR2027_HRFONT.md:371`

  > 主协议：42 / Demo-8 / 中文 8 ref / P1。

### 4.4 `.cursor/rules/hrfont-execution-spec.mdc`

`:15` 与 `:49` 已正确写明 B₀=Content、per-char α、E1@100k 与 no-gate；D1 无需改一般符号。以下仍陈旧：

- `.cursor/rules/hrfont-execution-spec.mdc:47`

  > E2 不依赖 E0：α 原型与 Δ 的 Ec 特征用 E1 final 冻结 encoder 在线算。

  实际 α candidate 来自离线 Es cache，只有 live query 与 Δ 的 Ec feature 在线。

- `.cursor/rules/hrfont-execution-spec.mdc:50`

  > style 条件=mean-pool(Es(R))、α query=同 R 的 Es mean-pool、gap/support 同 R。

  α query 应改成“保留 per-char rows，与候选同字 cosine 后平均”；mean-pool 只适用于 style condition。

- `.cursor/rules/hrfont-execution-spec.mdc:75`

  > E2：E1 best 初始化，Ec/Es冻，Delta→RSI zero-init，训UNet+offset的Stage A。

### 4.5 `reports/E2_IMPLEMENTATION_READINESS.md`

该文档对 B₀ 只用抽象符号，没有显式 FZKTJW stale quote；但它整体仍是实现前快照，以下内容已被 D2–D4/当前代码覆盖：

- `reports/E2_IMPLEMENTATION_READINESS.md:10,24-30`

  > **进度：94,100/100,000。**
  >
  > 没有 `DONE.json` 或等价完成标记；没有 100k checkpoint、100k val、100k 图或 100k dashboard 条目。
  >
  > 结论：**当前规范下 E2 尚未实现。** `configs/` 无 E2 YAML，`code/variants/` 无 E2/Stage-A variant，也无面向 CN2West-v2 E2 的正式 launcher/preflight/smoke。

- `reports/E2_IMPLEMENTATION_READINESS.md:58`

  > E0 owner 在 E1 final best 冻结后，用该 best 的 **Es** 建 ref8 prototype 表并冻结 `alpha/prototype manifest + tensor SHA`

- `reports/E2_IMPLEMENTATION_READINESS.md:60,68`

  > α 默认值遵循 E0：`M=3,tau=.07`
  >
  > 权重 softmax/top-M、stable tie-break 参考 `alpha_topk()`

- `reports/E2_IMPLEMENTATION_READINESS.md:78`

  > **将新增 Δ adapter 或 offset 末端初始化为零**（weight/bias 全零），保证 E1 权重加载后 step0 的新 Δ 增量为零

- `reports/E2_IMPLEMENTATION_READINESS.md:83,95`

  > 加载 E1 best 后执行 `content_encoder.requires_grad_(False)`、`style_encoder.requires_grad_(False)` 并置 `eval()`
  >
  > | init | E1 FT-v2 按预注册规则选出的 best |

- `reports/E2_IMPLEMENTATION_READINESS.md:138-140`

  > **零点 smoke：**同一 E1 init、相同 `x_t/t/content/style/noise` 下，`Δ=0` 的 E2 输出与注册的 no-Δ 基线 `≈` 一致；zero-init 新层参数和输出为零。

- `reports/E2_IMPLEMENTATION_READINESS.md:149-155`

  > E0 改为 train228，生成 Es prototypes/α calibration、gap/cache | 依赖 E1 final best SHA
  >
  > 正式 E2 训练必须等待 E1 best 和正式 α prototype artifact。

- `reports/E2_IMPLEMENTATION_READINESS.md:168-174`

  > Git 快照 train 到 94,100/100,000、最新 val 到 93k，剩 5,900 steps；没有 DONE/100k/provenance，故尚未完成。
  >
  > **E2 现状：**没有符合当前 CN2West-v2 规范的 E2 config、variant 或正式 script
  >
  > 逐候选 Ec→逐尺度 α 混合→减 B0，替换 RSI 的 `Ec(style)` 结构源并 zero-init 新增 Δ 接入

### 4.6 指定五文档之外、但会影响重建的陈旧引用

- `scripts/hrfont_validate_e1_encoders.py:4-8`

  > B0 content prototypes may either use the normal FZKTJW TargetImage directory or a dedicated directory passed via ``--b0-dir`` containing ``FZKTJW+uXXXX.png``

- `docs/patches/cn2west_stage_a.diff:27-29,621-648`

  > `--b0_font`, default=`FZKTJW`
  >
  > `FZFXKTJW  # split_v3 physical stem for the FZKTJW font`
  >
  > `torch.stack(...).mean(0)`
  >
  > `neutral_path = dataset.target_path(b0_font, cp)`

- `code/variants/cn2west_stage_a/VARIANT.md:1-2`

  > VARIANT=cn2west_ft_v2
  > PARENT=code/official/FontDiffuser

## 5. 回复稿准确性核对结果

仓库中未找到一份可唯一识别的回复稿全文；以下按任务中概括的五项主张逐项核对。

| 主张 | 结论 | 可安全发送的边界 |
|---|---|---|
| “B₀ done” | **活跃 runtime 基本属实，但 repo-wide 表述过度。** | 可说“当前 Stage-A train/validation runtime 已统一为 Noto ContentImage”；不可说“FZKTJW 依赖已全仓删除”，因为 validation docstring、主文档和重建 patch 仍残留（`train.py:109-113`；`scripts/hrfont_validate_e1_encoders.py:4-8`；`docs/patches/cn2west_stage_a.diff:621-648`）。 |
| “α reconciliation done” | **主要调用链属实。** | 可说“v2 活跃实现与 train/V5/V6/smoke 已切成 per-char cosine mean”；需补充 rationale/exec-spec/论文仍有 mean-pool/top-3 旧口径，`k_top` YAML 尚未接线（`scripts/hrfont_delta_v2.py:80-137`；`.cursor/rules/hrfont-execution-spec.mdc:50`；`configs/e2_stage_a_s3407.yaml:31`）。 |
| “E1@100k init fixed” | **只完成了配置文字，不能说全链路可运行。** | `step_100000` 很可能不是实际目录名，且 cache 未绑定 100k Es SHA；修复 F01/F02 后才能说“operationally fixed”（`configs/e2_stage_a_s3407.yaml:7,20`；E1 `train.py:347-355`）。 |
| “validation fixes done” | **neutral 与 val/test gate 隔离属实；质量仍有保留。** | 可说“V3/V5 neutral 与 gate/test 分离已改”；不可说“validation battery fully validated”，因为本环境数值 smoke 未跑、V1 self-gallery/negative 偏置未修、真实 CLI gate FAIL 仍 exit 0（`scripts/hrfont_validate_e1_encoders.py:121-143,258-296,349-367`）。 |
| “no-gate” | **活跃代码属实，配置/文档未同步。** | 可说“runtime 无新增 gate，复用 E1 offset head”；同时应主动说明 E2 step0≠E1，YAML 的 `new_layer: zero` 与 spec/readiness 是陈旧文本（`configs/e2_stage_a_s3407.yaml:8`；`code/variants/cn2west_stage_a/FontDiffuser/src/modules/attention.py:298-330`）。 |

若回复稿含“E2b step0 与 E1 完全等价”，该句是**错误**；正确说法是“E2b 与 E2 使用同一 E1 权重和同一新 n-shot style 表示，只把 RSI structure source 换回 `Ec(R[0])`，并复用原 offset head”。

## CHAT SUMMARY（中文，紧凑）

**Top findings：**

1. **先停开跑：**配置指向 `step_100000`，E1 实际保存名是 `global_step_100000`；同时 `es_per_font_char_e1_best.pt` 未与 100k Es SHA 绑定。推荐核对实体目录、用 100k 重建/重命名 cache，并在 launcher fail-closed 比 SHA。
2. **文档/重建产物严重漂移：**五份主文档仍混有 FZKTJW、mean-pool α、top-3、E1 best、zero-init；`cn2west_stage_a.diff` 甚至会重新引入旧 mapping/旧 shape，`VARIANT.md` 名称也错。推荐一次性 mass-update。
3. **E2b 同头但不等价 E1：**它确实复用 E1 offset head；但 style condition 已由单张 spatial map 改为 normalized pooled n-shot 的 1-token 均值。E2/E2b 仍是 structure-source matched pair，E2b 应改称 matched control。
4. **目前没有有效 E2 推理/val/best：**继承的 sample path 不构造 Δ/Ec(R[0])、structure 默认为零且仍 Resize；5k eval YAML 没被执行。
5. **额外未决：**soft ε 可导致空 α 且 V6 仍通过；V1 self-gallery/negative 配对有偏；resume 不存 RNG 会破坏 matched；E2c 必须明确 P1/E1 retrieval space。

**待 PI 决策：**D-A1 cache/init SHA 强绑定；D-A2 文档/patch mass-update；D-A3 E2b 定义为 matched control；D-A4 冻结 n-shot style 精确表示；D-A5 best 从 10k 起；D-A6 空 α 处理；D-A7 V1/test-null 口径；D-A8 E2c retrieval space；D-A9 CFG inference structure 语义；D-A10 proto manifest 兼容策略。推荐项均见第 3 节。
