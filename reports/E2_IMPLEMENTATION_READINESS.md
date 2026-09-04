# HR-Font E2 实施就绪报告（2026-09-04）

> 范围：基于仓库 Git 快照做 E1 健康检查，并把当前 CN2West-v2 的 E2（Stage A：feature-mix Δ→RSI）拆成 Cursor 可直接执行的实现清单。本报告不把旧实验线的 `scripts/hrfont_e2_*` 当作当前 E2 实现。后续执行以 PI 2026-09-04 覆盖决定为准：E1 的 `bs=8`、seed3407-only、无训练期 ink filter 均已人工验收且不是 deviation；后续实验统一 effective batch `8×1`；`split_v3_228_16_16` 为最终无泄漏 split；协议 H 已弃用，只允许 A。

# 1. E1 最终状态健康检查（基于 Git 快照）

## 1.1 快照状态与曲线

- **快照而非实时状态。** `reports/e1_ft_v2_dashboard/SNAPSHOT.json:1-4` 标记来源为 `runs/E1-FTV2-A-S3407/viz`，发布时间为 `2026-09-04T03:16:21Z`；该目录最后一次 Git 提交为 `350b21f`。没有更新提交时，只能断言该快照状态，不能断言训练机此刻状态。
- **进度：94,100/100,000。** `loss_history.json` 有 2,078 条、严格递增到 step 94,100；`index.html:39-48` 显示约 94k/100k。按最新 train 点尚余 **5,900 optimizer steps（5.9%）**。只有到达 100,000、100k full state/权重可读、`DONE.json` 为 completed 且 100k val/publish 完成，才判 E1 完成；当前不能判 completed。
- **val：0.076268 → 0.029775。** 全量 val16 每次 `n=4,720=16×295`。P1 step0 为 `0.076268`，5k 为 `0.035595`，50k 为 `0.030321`，最新 93k 为 `0.029775`，总降幅约 **60.96%**。50k 后主要落在 `0.0298–0.0305`，55k/70k/85k 有轻微回升但无灾难尖峰，已经进入平台区（`reports/REVIEW_E1_R0_20260904.md:63-68`）。正式 5k milestone 中最低是 80k `0.0297948`；93k 是 `last_state` 观察点，不得越过预注册规则直接选为 best。
- **train：有噪声但无静态证据显示发散。** 全历史均值 `0.03541`，50k 后均值 `0.02780`，90k 后均值 `0.03004`；全序列范围 `0.0038–0.275`。尾部 93,650–94,100 在 `0.00767–0.0822` 间明显逐 batch 波动，最新为 `0.02120`。这与随机 timestep/样本导致的单 batch loss 噪声相容；结合平稳 val，可判“训练主干健康、已平台”，但不能用 train tail 单点选择 checkpoint。
- **已冻结解释。** E1 实际 YAML 是 `batch_size=8, accumulation=1, seed=3407`（`configs/e1_ft_v2_a_s3407.yaml:12-24`）。依据 PI 新决定，这个 single-seed、无训练期 ink filter 的 E1 **按原样验收，不是 deviation，也不要求重训**；旧 review §7 对这三点的待决/偏差措辞已被覆盖。A 是唯一协议，H 不再进入任何训练、评测或 QA 结论（`.cursor/rules/hrfont-execution-spec.mdc:12-17`）。

## 1.2 100k 收尾必须完成

1. **终点事务：**保存并校验 100k `unet.pth/style_encoder.pth/content_encoder.pth/trainer_state.pt`，写 `DONE.json`（step/max/status），跑完整 val16×295，发布含 100k 的 dashboard；各文件算 SHA-256，禁止只留下可覆盖的 `last_state`。
2. **best 选择：**执行已预注册判据 `0.5*ID_z + 0.3*style_z - 0.2*quality_error_z`，只比较正式 **5k milestones**，同分取较早 step（`reports/EXPERIMENT_PLAN_CN2WEST_V2.md:81-89`）。不得以 test16、视觉挑图或非 5k 的 93k loss 选择。若独立评测轴尚未就绪，先保留全部 5k，不得临时改成最小 val loss；生成机器可读 `best.json`/指针，记录候选、原始指标、z-score 参照、公式、tie-break 和所选权重 SHA。
3. **provenance JSON：**至少记录 run/experiment ID、parent official P1 artifact+SHA、Git commit/dirty 状态、variant/code SHA、input/resolved config 与 canonical JSON SHA、dataset ID/root/tree SHA、split manifest SHA 与 228/16/16 计数、excluded 字体、A/96/RGB-PNG/no-resize、Content/B0、seed3407、bs8×accum1、optimizer/scheduler/warmup/fp16/CFG/loss 权重、起止/完成 step、所有 resume 事件及“E1 resume 非 exact”的已知限制、milestone/best/DONE/评测产物 SHA，以及 PI 四项覆盖决定。ink filter 必须显式写 `enabled=false, pi_accepted=true`，不能伪造 table SHA。
4. **publish 与台账：**发布稀疏可审计 dashboard（至少 JSON、曲线、100k/最佳 contact sheet），验证发布快照 SHA；再回填 run registry/PROJECT 台账的 `completed_step=100000`、best、artifact 路径/SHA、provenance 路径和完成时间。发布、台账回填发生前不得声称“证据归档完成”。

## 1.3 当前 Git 快照缺失

- 没有 `DONE.json` 或等价完成标记；没有 100k checkpoint、100k val、100k 图或 100k dashboard 条目。
- 没有本 run 的最终 provenance JSON/canonical config SHA/best 指针。
- 最新完整 val 是 93k，最新正式 5k val 是 90k；因此 E1 状态应写为 **健康、接近完成、尚未完成**。

# 2. E2 代码现状

结论：**当前规范下 E2 尚未实现。** `configs/` 无 E2 YAML，`code/variants/` 无 E2/Stage-A variant，也无面向 CN2West-v2 E2 的正式 launcher/preflight/smoke。仓库中已有若干 `scripts/hrfont_e2_*` 是旧 96/legacy 数据线产物；例如 `scripts/hrfont_e1_rsi_delta_smoke.py:14-20,30-38` 硬编码 `/root`、旧 ckpt，并使用 `Resize`，其 forward 只是把一张 `delta_images` 再送入 `Ec`（`:65-88`），不等于当前要求的逐候选编码 feature-mix Δ，不能作为正式 E2 代码。

必须新建 `code/variants/cn2west_stage_a/`。**建议从 `code/variants/cn2west_ft_v2/FontDiffuser/` 派生，而不是从 official 直接复制**：前者已具备 A 原生 96、无 resize、PNG target/style、确定性目录排序以及 P1/E1 权重加载（`train.py:145-205`；`dataset/font_dataset.py:42-72`）。同时 `diff -qr code/official/FontDiffuser/src code/variants/cn2west_ft_v2/FontDiffuser/src` 当前无输出，说明其 `src/` 仍与 official 相同，E2 的 Δ 补丁可以在新 variant 的干净上游 `src/` 上审计。不要修改 official 或 E1 variant。

# 3. E2 必须实现的改动清单（Cursor 可执行）

## 3.1 数据侧：继承 E1 base，但补强边界

### A. split manifest + excluded fail-fast（开跑阻断项）

目标文件：新 variant 的 `dataset/font_dataset.py`，以 E1 base 的 `get_path()` 为落点（当前 `code/variants/cn2west_ft_v2/FontDiffuser/dataset/font_dataset.py:42-72`）。

1. 构造器接收 `data.split_manifest`、期望 split 名和 `excluded_fonts`；读取 `manifests/split_v3_228_16_16.json` 的 `ratio/stems/excluded_fonts`（manifest `:1-20`），校验 228/16/16、三集合两两不交、excluded 不在任一集合。
2. 扫描物理 `TargetImage`/`StyleImage` 后，要求当前 phase 的字体 stem **精确等于** manifest 对应集合；missing、extra、val/test 混入 train、excluded 命中均聚合报错并 fail closed。不能继续依赖当前“传什么 root 就扫什么”的行为。
3. 所有角色仅接收 `.png`，在打开前校验 suffix；删除 `_open_content()` 的 JPG/JPEG fallback（当前 `font_dataset.py:74-81`），保留打开后的 RGB 与变换前 native 96×96 断言。
4. 将 manifest 路径、SHA-256、实际/期望 stem、角色样本计数写入 preflight/provenance；E0/Delta 候选池必须只来自 train228，目标字体 leave-one-out，val16/test16 永不入池（exec-spec `:23-26,64-68`）。

### B. 训练期 per-(font,char) ink filter：建议默认关闭，预留 hook

这里有一项仍需 PI 明确，但**不能把旧 review 的要求自动施加给 E2**。E1 无 ink filter 已获验收；若 E2/E2b 增加过滤，E2 对 E1 初始化后的训练数据分布发生变化，也会削弱 E1→E2 因果可比性。默认建议：

- E2/E2b/E2c/E2d 全 matched 组与 E1 保持相同数据管线，配置显式写 `ink_filter.enabled=false`、`reason=matched_to_pi_accepted_e1`；所有 matched 臂必须一致。
- 代码仍实现只读 table hook 与统计接口：启用时只读预计算 per-(font,char,role) 表，输出 role/script/char 和各 split 前后计数，候选不足 fail closed；默认 off 时也输出零过滤统计。禁止训练中在线重算 ink。
- 利：保持 E1/E2 数据可比、无需在 E2 前冻结一个尚无 PI 阈值的过滤表。弊：保留极小合法/异常 pair 的训练噪声，且与 exec-spec `:28-35` 的通用过滤条款存在表面张力。因此开跑 provenance 必须引用本次 PI 对 E2 的明确决定；若 PI 决定开启，则 E1 不重训但 E2/E2b 全组共享同一表，且论文中披露数据管线差异。

### C. Delta 库、α 与 Support 数据流

1. **先定义角色：**Style/ref8 用目标字体 A-style338 经冻结 `Es` 得 query；α prototype 是 train228 各候选字体的冻结 `Es(ref8)`；Delta bank 是同一 train228 候选字体、目标字符 `c` 的 A-target295；neutral 是固定 B0 的同字符 A 图。目标字体属于 train 时必须 leave-one-out；val/test 永远只作 query，不进 prototype/bank。Stage A 不消费 Support 图，Support 规则与 cache 可由 E0 同时产出供 E5，但不要把 Support 接进 E2 batch。
2. **谁建、何时建：**E0 owner 在 E1 final best 冻结后，用该 best 的 **Es** 建 ref8 prototype 表并冻结 `alpha/prototype manifest + tensor SHA`；同时冻结 train228/calib16 的 retrieval calibration。E2 训练 owner 在开跑前校验 encoder SHA、split SHA、B0 SHA、prototype SHA。E1 best 未定之前可写代码/测试假 tensor，但不能冻结正式 α 表。
3. **推荐混合方式：**α/候选选择可读 E0 冻结的 Es prototype/cache；`Ec(A(B_s,c))` 与 `Ec(A(B0,c))` 推荐在训练时由冻结 Ec 在线编码，再逐尺度 feature-mix，保证精确匹配当前 encoder 与 augmentation-free A 图。不要读取旧 E0 glyph/feature cache，也不要先做像素混合。若为速度缓存 Ec，多尺度 cache 必须在 E1 best 冻结后重建，键含 `(encoder_sha, split_sha, font, char, role)`，float32 生成并记录 dtype/shape/SHA；在线与缓存路径需逐层 `allclose` smoke 后才能切换。
4. DataLoader batch 增加 `font_stem, char_cp, alpha_fonts, alpha_weights, bank_paths, neutral_path`（或预取 tensor），并验证每个 bank 字符等于 target 字符、候选属于 train228、leave-one-out 生效、权重有限且和为 1。α 默认值遵循 E0：`M=3,tau=.07`（plan `:73-79`），但必须读冻结 E0 artifact，不在 E2 内重新调参。

## 3.2 模型侧：真正的 feature-mix Δ→RSI

### A. Δ 纯函数与数据形状

目标：把 `scripts/hrfont_delta_feature.py:43-102` 的思想迁入新 variant 内可测试模块（建议 `src/modules/delta.py`），但修正旧脚本的 JPG/旧路径假设（`:16-20`）。

1. 对每个候选 `B_s` **分别**运行冻结 Ec，得到全部尺度 `[residual..., final]`；每尺度做 `Σ_s α_s Ec(A(B_s,c))`；再减同字符 neutral `Ec(A(B0,c))`，得到 `delta_features[l]`。权重 softmax/top-M、stable tie-break 参考 `alpha_topk()`（`:43-64`）；逐尺度混合/相减和 shape check 参考 `feature_delta()`（`:67-102`）。
2. 不允许 `Ec(Σα image)`，不允许把 style 汉字或单张“delta image”当 feature-mix Δ。每层记录 shape/dtype/RMS；NaN、scale 数量不一致、字符错配立即失败。
3. Delta-drop 使用独立、可恢复 RNG stream，以样本为粒度把整个多尺度 Δ 同时置零，概率 `.25`；E2b 即使 `delta.enabled=false` 也消费完全相同 draw（exec-spec `:42,64-65`）。CFG joint drop `.10` 与 Δ-drop 独立；记录二者 mask 以便 matched checksum。

### B. RSI 接线与 zero-init

需要修改新 variant 的三个关键点；不能只 monkey-patch train.py：

1. `src/model.py`：当前 `FontDiffuserModel.forward()` 在 `:34-47` 先以 `Es(style_images)` 提供 style cross-attention，又把同一 style 图送入 `Ec` 得 `style_content_res_features` 作为 RSI 结构源。E2 改为 forward 显式接收/构造 `delta_features`，保留 `Es(R)` 与 `Ec(content)`，**删除 style 图进入 content encoder 的路径**；传给 UNet 的第四项改成 Δ。DPM/eval 路径 `src/model.py:88-106` 必须同步，避免 train/eval 接线不一致。
2. `src/modules/unet.py`：当前 up path 在 `:264-285` 将 `encoder_hidden_states[3]` 作为 `style_structure_features` 传入 RSI。改为命名明确的 `delta_structure_features`（最好用结构化/keyword 参数替代 magic list index），逐 up-block 传对应尺度，并在入口断言层数/shape/device/dtype。down/MCA 仍使用正常 content features，style cross-attention 仍使用 Es 特征。
3. `src/modules/unet_blocks.py`：official/E1 的 `StyleRSIUpBlock2D` 位于 `:423-587`；结构源由 `:545` 选层，并在 `:553-561` 送入 `OffsetRefStrucInter` 后用于 DCN。改名为 neutral 的 `structure_features`/`delta_feature`，维持尺度索引语义并加入 shape 断言。`OffsetRefStrucInter` 定义实际位于 `src/modules/attention.py:266-332`：其最终 offset 投影是 `proj_out`（`:298-330`）。**将新增 Δ adapter 或 offset 末端初始化为零**（weight/bias 全零），保证 E1 权重加载后 step0 的新 Δ 增量为零；不要把整个已训练 E1 RSI 无条件清零。推荐残差形式 `offset = offset_e1_or_base + zero_init(delta_adapter(...))`，并用 E2b/off 路径验证 step0 等价。若设计为完全替换 official structure 源，则需明确 base offset 的定义，smoke 必须证明 `Δ=0` 与注册的 no-Δ 基线在容差内一致。
4. `src/build.py:8-35` 增加显式 `rsi_source={delta,official}`、delta adapter channel/scale 配置；checkpoint loader 以 strict allowlist 处理唯一新增参数，并输出 missing/unexpected keys。E2b 保持 official RSI；E2 使用 delta；两者从同一 E1 best 初始化并保存初始化参数哈希。

### C. freeze 与优化器审计

- 加载 E1 best 后执行 `content_encoder.requires_grad_(False)`、`style_encoder.requires_grad_(False)` 并置 `eval()`；训练循环不得因 `model.train()` 把冻结 encoder 的 BN/状态切回训练模式，建议覆写/分模块设 mode。Δ 在线编码放在 `torch.no_grad()` 下。
- optimizer 只接收 `requires_grad=True` 的 UNet 参数及新增/offset 头；Ec/Es 不进 optimizer。开跑前和每个 5k milestone 输出 trainable/frozen 参数名、数量与 SHA；反向后断言 Ec/Es grad 全为 None，5k 时断言 Ec/Es state SHA 与 E1 init 完全相同。
- E2b 也按同一 freeze 计划训练 UNet+offset；E2/E2b 的差异只能是白名单字段。

## 3.3 训练侧：冻结值与基础设施

### A. 当前有效配置（覆盖 plan 中旧 1×4 文案）

新建 schema/base YAML 与每 seed 配置（建议 `configs/e2_stage_a_a_s340{7,8,9}.yaml`，E2b 由结构化 diff 派生），必须解析为：

| 字段 | E2 冻结/建议值 |
|---|---|
| init | E1 FT-v2 按预注册规则选出的 best |
| protocol/split | A only；p260；train/val/test=228/16/16；H forbidden |
| steps | 80,000 固定终点；5k val 可诊断/早停，但 matched 主比较仍用 80k |
| batch | **batch_size=8, accumulation=1, effective_batch=8**；全部后续实验统一 |
| optimizer | AdamW β=(.9,.999), wd=.01, eps=1e-8；clip=1；scale_lr=false |
| lr | `1e-5`，linear，warmup 2,000 |
| precision | fp16；metrics/loss float32；gradient checkpointing=false；EMA=false |
| loss | diffusion MSE + `.01` perceptual + `.5` offset；E2 SCR=false |
| drops | CFG content+style joint `.10`；Delta `.25`，独立 RNG |
| freeze | Ec/Es frozen+eval；train UNet + offset/new Δ adapter |
| seeds | **建议仍执行 3407/3408/3409**；PI 的新决定只把 batch 锁为 8×1，未取消 exec-spec 的后续三-seed政策（`.cursor/rules/hrfont-execution-spec.mdc:16-17,49-50`）。先跑 3407 smoke/早筛可行，但主 matched 结论默认补齐三 seed；若要 single-seed，需 PI 另行覆盖。 |

计划 §4 的 E2 超参来源是 `reports/EXPERIMENT_PLAN_CN2WEST_V2.md:91-102`，其中 `bs=1,accum=4` 已被 PI 2026-09-04 的 `8×1` 覆盖；其余 80k/lr/warmup/drop/freeze/seeds 保持。禁止 Cursor照抄旧行恢复 effective batch4。

### B. exact resume（E2 开跑阻断项）

当前 E1 checkpoint helper 已保存 model/optimizer/scheduler 和部分 RNG（`train.py:69-122`），但调用时没有传 scaler（`:230-239,347-355`），DataLoader 也没有 generator/worker init（`:196-205`），没有 sampler cursor。E2 必须：

1. 保存/恢复 Python、NumPy、Torch CPU、全部 CUDA RNG；单独保存 CFG、Delta、batch/sampler RNG stream。
2. 从 Accelerator/AMP 取得实际 scaler并传入 save/load；发生 overflow/skip 时记录。若框架隐藏 scaler，使用其官方 state API，不保留形同虚设的 `scaler=None`。
3. DataLoader 显式 `torch.Generator`；worker seed=`seed+1000*epoch+worker_id`；持久化 generator state、sampler permutation/epoch、**下一 batch cursor**、global sample index。resume 必须从下一未消费 batch 继续，不能只重置 epoch 后跳步并重放随机选择。
4. 保存 optimizer、scheduler、global optimizer step、microstep（虽 accum=1 仍显式）、所有 drop draws 状态、best tracker和 val queue。
5. 加载前比较 resolved canonical config SHA；canonical JSON UTF-8/sorted keys，排除时间戳/运行时路径；resume 只允许改 `resume_from/stop_file`。同时校验 dataset/split/E1-init/model/code SHA。
6. 新增确定性验收：同一 seed 跑 N step，一条不中断，一条在 K step 保存/恢复；比较后续 batch IDs/drop masks/loss，最终 trainable state、optimizer、scheduler、RNG checksum 必须逐项相同（fp16 若不能 bitwise，先预注册严格 atol/rtol并解释）。未通过不得启动 80k。

### C. STOP、heartbeat、milestone、best

- 每个 **optimizer step 完成后**检查 STOP；命中时原子保存 state、`stopped_step`、RNG/checksum 后正常退出，不写 completed DONE。当前循环只检查 max steps（`train.py:340-377`），需补。
- 主进程每 60s append JSONL heartbeat：run_id、pid/starttime、host/GPU、step、last_loss/lr、last_ckpt、timestamp；超过 5min 由外部监控告警，但不得盲目 kill/restart。
- 每 1k lightweight，滚动保留最近 5 个；每 5k full + 全量 val16，保留全部；写临时目录、校验后原子 rename。维护 `last.json` 和 `best.json` 指针而不是复制/覆盖唯一证据。最终 80k 是强制 full+val+DONE。
- val 的固定 pair/style/noise manifest 要落盘并哈希；选择公式与 E1/计划保持预注册，不看 test16。

### D. matched YAML diff（E2b 的因果防线）

在 config loader/preflight 中对 **resolved canonical objects** 做递归 structural diff，不做文本 diff。E2b 相对 E2 只允许：`experiment.id`、`model.rsi_source=official`、`model.delta.enabled=false`、`init.new_layer=null`（plan `:98`；exec-spec `:39-42`）；任何额外 path 都打印 expected/actual 后 fail。

还应校验 steps、8×1、lr/scheduler/warmup、seed、batch manifest/order、init SHA、optimizer、loss、CFG/Delta/SCR draw schedule、ink setting完全相同。E2b 虽禁用 Delta，仍须执行/消费相同 Δ draw；E2/E2b 每 seed开跑前生成前 N 个 batch IDs、timesteps、noise seeds、CFG/Δ masks checksum，必须相等。E2d 相对配对臂只允许 SCR 字段变化。

## 3.4 评测侧：smoke 与 val16

正式 80k 前新增 current-variant smoke，旧 `hrfont_e1_rsi_delta_smoke.py` 仅可参考调用结构（其 resize/旧路径/单图 Ec 实现不可复用）：

1. **数据 smoke：**manifest 精确 228/16/16、零泄漏、excluded fail-fast、全部角色 PNG/RGB/native96、bank 仅 train228、leave-one-out、同字符、B0 存在。
2. **feature-mix 单测：**人工小 tensor 验证逐候选编码→逐尺度加权→减 neutral；α one-hot、α sum=1、候选顺序置换不改变结果；在线与 cache（若启用）逐层一致。
3. **接线 smoke：**hook `OffsetRefStrucInter` 输入，证明收到的是 Δ 而非 `Ec(style)`；style 图只进入 Es；content identity 仍进入 MCA。记录每层 shape/RMS。
4. **零点 smoke：**同一 E1 init、相同 `x_t/t/content/style/noise` 下，`Δ=0` 的 E2 输出与注册的 no-Δ 基线 `≈` 一致；zero-init 新层参数和输出为零。容差在 fp32/fp16 分别预注册。
5. **响应 smoke：**给非零 Δ 与受控扰动（符号翻转/空间 shuffle）后，确认 offset/noise output 有有限、非零响应；只扰动 Δ 时 Es style feature 不变。该 smoke 证明接线活着，不作为效果结论。
6. **freeze/resume/matched smoke：**一次 backward 后 Ec/Es 无 grad且 SHA 不变；中断恢复 checksum 通过；E2/E2b YAML allowlist 和前 N random draws 相同。
7. **评测：**每 5k 在固定 val16×295 上评测，`n=4720`，固定 ref/pair/noise manifest；保存 loss 与预注册 identity/style/quality 指标、配置/模型/评测代码 SHA。test16 不参与 checkpoint 或超参选择。

# 4. 实施顺序建议（含可并行项与依赖）

| 阶段 | 工作 | 依赖/并行性 | 出口条件 |
|---|---|---|---|
| 0 | R0 正式发布 + E1 跑至100k并完成 §1.2 收尾 | 关键路径起点；R0 发布与 E1 尾段可并行 | p260/split/tree SHA；E1 DONE、100k val、best、provenance |
| 1 | E0 改为 train228，生成 Es prototypes/α calibration、gap/cache | 依赖 E1 final best SHA；不得读 legacy meta或 val/test | prototype/retrieval config/cache manifest + SHA，leave-one-out/preflight 通过 |
| 2 | 新建 `cn2west_stage_a`、schema/YAML、manifest防线、Δ/RSI、freeze、exact resume、运维 | **可与 E0 并行开发**：先用 synthetic α/feature fixtures；正式 artifact 接入等待阶段1 | 单测、lint/import、config/matched preflight 全过 |
| 3 | smoke（数据、feature-mix、零点、响应、freeze、resume、matched） | 需要候选 E1 best；涉及正式 α 的集成 smoke 需要 E0 | §3.4 全绿并生成机器可读报告/SHA |
| 4 | E2/E2b seed3407 早筛，再补 3408/3409，固定80k | 依赖阶段0–3；同 seed 的 E2/E2b 可在不同 GPU 并行但必须同 frozen manifests | 每 seed 80k full+val+DONE；matched audit；best/provenance |

执行上的关键区别：**E2 代码开发不必等待 E0；正式 E2 训练必须等待 E1 best 和正式 α prototype artifact。** 若采用 Ec 在线算，E0 不必预先生成多尺度 Ec cache，但仍必须先完成 Es prototype/α 校准与其 SHA；若 α 也在线临时重算而不冻结，则不具备 matched/reproducible 条件，不得开 80k。

# 5. 风险与待 PI 拍板（极短清单）

1. **E2 ink filter：**建议与已验收 E1 同管线，matched 全组 `enabled=false`，只预留默认关闭的 hook；请 PI 明确此豁免是否扩展到 E2/E2b。
2. **E2 seeds：**建议按仍有效的计划执行 3407/3408/3409；bs 决策不等于 seed 决策。请 PI 确认是否保留三 seed。
3. **variant 派生源：**建议从 E1 `cn2west_ft_v2` 派生；其 A loader/train 修复可继承，且 `src/` 与 official 当前一致，最利于审计。
4. **E0 前置范围：**建议 E2 正式训练前必须有冻结的 Es prototype/α artifact；Ec 多尺度特征推荐在线算，因此不要求 E0 先完成 Ec cache。若要求读 cache，则 cache 必须基于 E1 final Ec 重建并通过在线一致性 smoke。

# CHAT SUMMARY（中文，紧凑）

**E1 三行：**

1. Git 快照 train 到 94,100/100,000、最新 val 到 93k，剩 5,900 steps；没有 DONE/100k/provenance，故尚未完成。
2. val `0.076268→0.029775`，50k 后约 `0.0298–0.0305` 平台；train tail 有 batch 噪声但结合 val 无发散迹象。
3. E1 的 bs8、seed3407-only、无训练期 ink filter 已获 PI 验收，不是 deviation；100k 后仍须按组合指标选 5k best、补 provenance/publish/台账。

**E2 现状：**没有符合当前 CN2West-v2 规范的 E2 config、variant 或正式 script；现存 `hrfont_e2_*` 属旧线，不能直接复用。

**实现 top 要点：**从 E1 base 新建 `cn2west_stage_a`；loader 读取 final split 并对 extra/missing/excluded/泄漏 fail-fast；E0 用 E1 final Es 冻结 train228 α prototypes，Ec feature-mix 推荐在线算；逐候选 Ec→逐尺度 α 混合→减 B0，替换 RSI 的 `Ec(style)` 结构源并 zero-init 新增 Δ 接入；冻 Ec/Es、训 UNet+offset；配置固定 80k、8×1、1e-5、warmup2k、fp16、CFG .10、Δ-drop .25；开跑前补 scaler+sampler cursor 等 exact resume、STOP/heartbeat/milestone、matched YAML diff 和 Δ=0/扰动 smoke。

**待拍板：**E2 是否延续 no-ink-filter（建议是）、是否三 seed（建议是）、是否从 E1 base 派生（建议是）、E0 只冻结 α 还是同时强制 Ec cache（建议 α 必需、Ec 在线）。
