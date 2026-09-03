# FT / Stage A / Stage B 派生源代码审查

## 结论先行

这些脚本可以作为新 E1/E2/E5 variant 的**算法草图和局部函数来源**，不能直接作为 v2 正式训练入口。新代码必须进入 `code/variants/<id>/`。最严重的实质问题是：Stage A/B 当前都在像素域混合近邻字形，之后才编码，不是计划定义的 feature-mix；Stage B 没有加载 style/content encoder 权重；两阶段均无完整 seed/RNG/scheduler/accumulation/config-SHA 恢复，Stage B 还会把“adapter 没有梯度”的 batch 计为训练 step。

审查范围：`scripts/retrain_v2_finetune_fontdiffuser.py`、`scripts/hrfont_e2_stageA96_formal_train.py`、`scripts/hrfont_delta_feature.py`、`scripts/hrfont_e2_stageB96_train.py`、`scripts/hrfont_support_adapter.py`、`scripts/hrfont_support_utils.py`；`scripts/hrfont_e1_rsi_delta_smoke.py` 仅用于核对接线意图。

## 1. 逐文件审查

### 1.1 `scripts/retrain_v2_finetune_fontdiffuser.py`（FT 派生源）

**可复用**

- CLI 到官方训练器的薄封装思路可保留：batch、accumulation、steps、lr、warmup、初始化目录均已透传（`scripts/retrain_v2_finetune_fontdiffuser.py:31-47`、`:70-87`）。在 E1 variant 中应改为 schema 驱动的函数调用/入口，而不是拼接松散命令。
- 启动前检查 StyleImage 与 UNet 权重存在的 fail-fast 习惯可迁移（`:61-66`），扩展成全角色、三 encoder 权重、dataset/config/model SHA 和 A 契约检查。
- 通过 `CUDA_VISIBLE_DEVICES` 隔离设备的做法可作为 launcher 层可选实现（`:88-92`），GPU 不应成为实验 YAML 的硬编码语义。

**必须重写**

- 全部根路径、Python 环境、ours repo、输出和默认数据路径均绑定 `/root` 与旧目录（`:22-27`）；新 variant 必须以 repo root/config 解析，并只读 official P1。
- 默认仍是旧 30k/bs4/warmup10k、mixed precision `no`（`:33-47`、`:84`），与 E1 的 100k、fp16、warmup5k 及主/预热双轨不符。
- `--rebuild_data` 会从训练 launcher 修改数据（`:48-60`）；v2 应把 R0 数据发布和训练严格分离，训练只验证已冻结 p261 SHA。
- 输出目录 `exist_ok=True`（`:68-69`）允许复用/覆盖同名 run；必须改为目录存在即拒绝，resume 只能显式指定已验证 checkpoint。

**Bug / 风险核实**

- **encoder 加载风险存在**：wrapper 只验证 `unet.pth`（`:64-66`），却把目录交给下游作为 phase-1 checkpoint（`:83`）；未在本层验证 `style_encoder.pth/content_encoder.pth`、权重 SHA 或 strict load。
- **seed 缺失存在**：参数与下游命令均无 seed（`:31-50`、`:70-87`）；也无 DataLoader generator/worker seed/RNG checkpoint。
- **step/RNG/config 不可审计**：本脚本只透传 `max_train_steps`，没有 matched batch manifest、resume 等价性、config SHA、milestone/keepalive/STOP 约束。
- **硬编码路径/GPU 存在**：绝对路径见 `:22-27`，GPU CLI 默认0并直接写环境（`:32`、`:89`）。
- **rmtree/os.listdir 未发现**：本文件没有 `rmtree` 或 `os.listdir`；但 `mkdir(exist_ok=True)` 仍造成历史目录混写风险（`:68-69`）。

### 1.2 `scripts/hrfont_e2_stageA96_formal_train.py`（Stage A 主派生源）

**可复用**

- alpha 做目标字体 leave-one-out，并以余弦、top-M、softmax 形成权重的基本结构可提取（`scripts/hrfont_e2_stageA96_formal_train.py:173-183`）；需统一调用 E0 冻结 cache/API。
- content/style encoder 冻结、UNet 训练的阶段边界可迁移（`:296-308`），但要按参数名显式审计 trainable set。
- joint CFG drop 与独立 Delta drop 的概念已经分开（`:401-404`），适合改造成两个命名 RNG stream。
- diffusion/perceptual/offset 损失组合与系数符合旧目标（`:424-445`）；保存临时文件后原子替换 `last.pt` 的实现值得保留（`:342-347`）。
- 信号/STOP 保存退出框架可迁移（`:357-389`），改成 optimizer-step 后检查并保存全恢复状态。

**必须重写**

- **pixel-mix 改 feature-mix（明确位置）**：当前 `__getitem__` 在 `:191-200` 逐张读取像素并求 `delta=sum(w*image)`；训练再于 `:415` 只编码一次该混合图，最后在 `:421` 减 content feature。应改为 dataset 返回 `(bank_paths,weights,ch)` 或 E0 feature-cache keys；训练/缓存层对每个 `A(B_s,c)` 分别运行冻结 Ec，按每个尺度执行 `mixed_l=sum_s w_s*Ec_l(A(B_s,c))`，再减 `Ec_l(A(B0,c))`。不得减当前 CFG 后的 `content` 特征；B0 必须是 FZKTJW 的同字符 A 图。`scripts/hrfont_delta_feature.py:67-102` 已实现正确的“分别编码→逐层加权→减 base”雏形，可抽取为批量、cache-aware 实现。
- 旧域写死：DejaVu/JPG Content、旧 e0 bank、`ft_cnstyle@25k` 初始化与 `/root` 路径（`:29-35`、`:141-168`）全部不符合 A/p261/E1 best。
- loader 明确执行 Resize（`:94-102`），必须改为尺寸断言+ToTensor+Normalize；不得接受 JPG。
- RSI 重初始化当前对全部 Conv/Linear 用 Xavier（`:116-138`），不满足新层/offset 头 zero-init 与 step0 等价基线；必须只初始化新增接入层，并用单元测试证明 Delta=0 等价 E2b 起点。
- 代码主动开启 gradient checkpointing（`:309-314`），与冻结值 false 冲突；删除。
- 训练器缺 accumulation=4、linear scheduler/warmup2k、SCR 开关、val/milestone 选择和 strict matched 配置。

**Bug / 风险核实**

- **pixel-mix 存在**：证据如上 `:191-200`；虽 `:421` 得到形式上的 feature difference，非线性 Ec 意味着 `Ec(sum wI) != sum wEc(I)`。
- **encoder 加载本文件正确但来源过时**：UNet/SE/CE 三者均显式加载（`:296-301`）；风险是旧 FT checkpoint，且无 strict/hash/架构签名验证。故“漏加载 encoder”在 Stage A 当前代码中**不存在**，来源与完整性风险仍在。
- **seed 缺失存在**：使用 `random.choice/random.random`（`:188`、`:401-404`）、DataLoader shuffle（`:294`）、Torch noise/timestep（`:406-407`），但无任何 seed 设置或 RNG 保存。
- **step 计数不符合新 matched 语义**：每 batch `scaler.step` 后直接 `step+=1`（`:447-453`），无 accum=4、无 overflow 检测，GradScaler 因溢出跳过 optimizer update 时仍计 step；resume 只恢复 UNet/optimizer/整数 step（`:324-336`），不恢复 scaler/scheduler/RNG/sampler cursor。
- **错误吞并风险**：任意非 OOM `RuntimeError` 被记录后继续（`:477-487`），可能持续跳样本并改变 batch 序列，matched 实验失配；应 fail closed。
- **硬编码路径/GPU 存在**：`:29-35`；自动挑卡排除 GPU1（`:48-67`），实际 device 固定 `cuda:0`（`:279`）。
- **覆盖/删除风险存在但无 rmtree**：日志/OUT `exist_ok`（`:81-87`）；自动发现旧 checkpoint并 resume（`:209-214`、`:325-336`）；`last.pt` 被替换（`:345-347`）；`prune_ckpts` 会 unlink 非保留 checkpoint（`:217-230`）。未发现 `rmtree` 或 `os.listdir`。
- **里程碑策略不符**：保存函数每500 step落盘并删旧（`:348-352`），主循环又每100/500重复调用（`:464-468`），不是1k轻量/5k完整+val；状态 checkpoint 排序在 `write_status` 里是词法排序（`:242-247`），可能把 `step_9500` 当成比 `step_10000` 新。

### 1.3 `scripts/hrfont_delta_feature.py`（Delta 工具派生源）

**可复用**

- `alpha_topk` 已做分数降序、font 名升序的确定性 tie-break（`scripts/hrfont_delta_feature.py:43-64`），是 E0 API 的良好雏形。
- `feature_delta` 正确地分别编码每个库字、逐尺度按 alpha 混合，再减 base，且检查尺度数/shape（`:67-102`）；这是替换 Stage A/B pixel-mix 的首选摘取点。

**必须重写/风险**

- 路径契约仍是 JPG，Content 与其他角色命名不同（`:16-20`）；须换成 p261 manifest 驱动的 RGB PNG locator、96断言、ink表过滤和 SHA 校验。
- 函数参数名 `content` 没有强制其为 A 渲染的 FZKTJW B0（`:68-81`）；应改名 `b0_same_char` 并校验 font/char metadata，防止像 Stage A 那样错误减 CFG content。
- 每样本即时编码 top-M，未接 float32 E0 cache；正式版需批量化/缓存并保留输出 dtype/层顺序元数据。无 seed 问题（本文件算法确定性），无 rmtree/os.listdir/覆盖。

### 1.4 `scripts/hrfont_e2_stageB96_train.py`（Stage B 主派生源）

**可复用**

- “冻结 UNet+encoders，仅优化 SupportAdapter”的总体阶段边界表达清楚（`scripts/hrfont_e2_stageB96_train.py:246-273`）。
- `pool_ec` 与把 adapter token 追加到 style cross-attention context 的接线可作为原型（`:172-208`）；应补权重/位置编码/批处理语义验证。
- support drop 独立于 support 是否可用的概念可迁移（`:332-342`）。

**必须重写**

- 同 Stage A，`:133-140` 先 pixel-mix，`:190-192` 后编码并减 content，必须换为 `hrfont_delta_feature.feature_delta` 类的逐库字 feature-mix，并减 A/FZKTJW 同字符特征。
- **encoder 必须加载**：当前只从 Stage A blob 加载 UNet（`:246-250`）；新建的 SE/CE 从未加载 official/E1/StageA 权重便被 freeze（`:247-260`）。正式版需从父 milestone strict 加载 UNet、Ec、Es（或验证父 blob 中冻结 encoder SHA 并从该 SHA 对应权重加载）。
- SupportAdapter 训练目标与计划不一致：LR=1e-4、30k（`:33-35`），应为1e-5、25k、linear/warmup1k、accum4；当前只用 diffusion MSE（`:343-345`），缺计划指定的共同 objective、可配 SCR。
- support 的 alpha*cover 权重虽由工具返回，却在 dataset 只取 paths（`:146-154`），adapter 输入等权；必须显式应用冻结权重或在配置中定义 token weighting。
- 仅实现 support drop，没有官方 joint CFG=.10（`:332-345`）；注释把20% support drop 称为 CFG（`:333-334`）会混淆语义。

**Bug / 风险核实**

- **encoder 加载缺失是确定 bug**：见 `:246-260`；这会用随机 Ec/Es 生成 Delta、style 与 support token。
- **seed 缺失存在**：style ref `random.choice`（`:142-146`）、support drop（`:332`）、shuffle（`:244`）、Torch noise/timestep（`:336-337`）均未 seed/恢复。
- **step 计数 bug 存在**：support 为空或 drop 时 `loss.requires_grad` 很可能为 false，代码跳过 backward/optimizer（`:347-353`），却无条件 `step+=1`（`:355`）；即“无更新 batch”消耗训练预算。即使有梯度，AMP overflow 也没有检测。
- **batch size 假定为1**：`build_hidden` 只检查/使用 `support_imgs[0]`（`:195-206`）；扩 batch 或 accum 时应显式逐样本打包/mask。
- **支撑检索顺序依赖存在**：Stage B 的 train/font/char 顺序来自 meta（`:107-127`）；support 工具候选按原列表拼接，MMR 严格 `>` 且无稳定 tie-break（`scripts/hrfont_support_utils.py:50-55`、`:87-108`），alpha 排序也只按分数（`:124-136`）。同分结果依赖上游列表顺序。必须先按 stem/char 排序，并以 `(score desc, stem asc)` 打破并列。
- **硬编码路径/GPU/自动 resume/覆盖存在**：路径 `:23-35`，挑卡 `:38-57`，device `:225`；`last.pt` 存在即隐式恢复（`:275-287`），保存直接覆盖 `last.pt`（`:297-308`），OUT `exist_ok`（`:75-81`）。未发现 rmtree/os.listdir。
- **resume 不完整**：只恢复 adapter/optimizer/step且 optimizer load 失败静默忽略（`:275-287`）；无 scaler/scheduler/RNG/sampler/config SHA。STOP 只在 epoch外与batch前检查，不严格是每次有效 optimizer step 后（`:321-327`）。

### 1.5 `scripts/hrfont_support_adapter.py`

**可复用**

- `Linear→GELU→Linear→LayerNorm` 与末 Linear zero-init 正好对应 E5 计划（`scripts/hrfont_support_adapter.py:8-21`）；空 token 返回也安全（`:23-27`）。可原样摘入 variant 的模型模块，并补单元测试：初始化时非空输入输出为 LayerNorm(0)，梯度能到末层。

**必须补齐/风险**

- 输入只是 pooled Ec 向量，没有 alpha*cover 权重、q/font 元数据、mask 或可变 batch API；这些应在调用方定义。`hidden=max(in_dim,context_dim)`（`:11-14`）必须进入 YAML/模型 SHA，不能隐式变化。
- zero-init 末 Linear 后接 LayerNorm，在零输入时输出为0，符合无扰动起点；但接入到 style token 后“输出0 token”不必然等价“无 token”，需用 UNet 前向等价测试确认 attention mask/长度效应。

### 1.6 `scripts/hrfont_support_utils.py`

**可复用**

- gap gate、`q!=c`、cover、MMR、alpha top-M 与 `alpha*cover` 权重的数据结构可作为 E0/E5 检索模块骨架（`scripts/hrfont_support_utils.py:44-108`、`:111-165`）。

**必须重写/风险**

- 模块 import 时读取硬编码旧 bank/meta（`:11-17`），参数也为全局常量（`:19-24`）；改为纯函数+只读 calibration artifact/config SHA 注入。
- `residual_vec` 实际只是 `normalize(vc-v_best_ref)`（`:62-72`），不是计划定义的参考子空间投影残差 `vc-P_R vc`；需按 E0 冻结实现重写。
- 候选是 `P1 alnum + DONORS`（`:50-55`），不等于计划的 hybrid `ref8∪donors18`、不足再 `P1∪donors18`；还未排除当前 style ref。
- 顺序依赖如上；所有候选、并列分数和最终 `(q,font)` 输出必须 canonical sort。ink 表、目标字体排除、候选池不足统计与 cache SHA 均缺失。

### 1.7 `scripts/hrfont_e1_rsi_delta_smoke.py`（仅接线证据）

- 它验证的只是“把 RSI 第四路由 `Ec(style)` 换为 `Ec(delta_images)`”可前向（`scripts/hrfont_e1_rsi_delta_smoke.py:65-88`），并未实现特征差；真实图 fallback 甚至令 `delta=content`（`:92-109`）。因此不能作为 Delta 正确性测试。
- 它使用 Resize（`:30-38`）、旧 checkpoint（`:14-20`），bank 目录用未排序 `iterdir()` 后取第一个字体（`:94-102`）。这不是审查主训练代码中的 `os.listdir`，但确有同类无序文件系统依赖，正式 smoke 必须用 manifest 固定样本并断言 A/96。

## 2. 与 cn2west v2 计划的差距矩阵

| 计划要求 | FT wrapper | Stage A | Stage B | 主要缺口 |
|---|---|---|---|---|
| matched-set YAML structural diff | 否 | 否 | 否 | 无 schema、allowlist、batch/drop manifest；需统一 config loader/preflight。 |
| RNG 完整保存/恢复 | 否 | 否 | 否 | 三者均缺 Python/NumPy/Torch CPU+CUDA、DataLoader、sampler、scaler；A/B 只恢复部分状态。 |
| 训练期 ink 过滤挂钩 | 否 | 否 | 否 | 无 per-(font,char) 表、全角色一致过滤、split统计、threshold/table SHA。 |
| A 协议数据加载 | 否 | 否 | 否 | FT 指向旧数据；A/B 用旧bank、DejaVu/JPG且Resize。 |
| freeze 计划 | E1 交给下游，未审计 | 部分 | 表面支持但encoder随机 | 需 trainable-name allowlist、参数/梯度 hash；E1全训，E2冻Ec/Es，E5只adapter。 |
| SCR 开关 | 未显式 | 否 | 否 | 需统一 loss factory、E2d/E5b 只改SCR字段，确定性负样本。 |
| keepalive / STOP | 否 | 部分 | 部分 | A状态随20 step，不保证60s；B随100 step；STOP与有效optimizer step语义不严。 |
| milestone 保存 | 仅透传单interval | 不符合 | 部分5k但last覆盖 | 统一1k lightweight、5k full+val、best、保留策略；全状态原子保存。 |
| config SHA / resume 校验 | 否 | 否 | 否 | 保存input/resolved/canonical/SHA；resume只允许相同SHA及白名单路径字段。 |

## 3. Variant 派生建议

### 3.1 骨架模块清单

| 模块 | 来源 | 动作 |
|---|---|---|
| `config_schema.py`、`config_io.py` | 新写 | YAML验证、只读dot access、canonical JSON/SHA、matched allowlist diff、resume override。 |
| `data_contract.py`、`dataset.py` | 新写 | p261 manifest、A/RGB/96断言、无Resize、全角色SHA、ink表过滤与统计、确定性batch manifest。 |
| `rng_state.py` | 新写 | 命名RNG streams、worker seed、保存/恢复全部RNG/scaler/sampler cursor。 |
| `delta.py` | 摘取 `hrfont_delta_feature.py:43-102` | 批量/cache化 feature-mix；base 强制 A/FZKTJW/c；逐层dtype/RMS元数据。删除 Stage A/B pixel-mix。 |
| `retrieval.py` | 重写 `hrfont_support_utils.py` | 保留API形状，改正确投影残差、hybrid候选、stable tie-break、ink/leave-one-out与artifact SHA。 |
| `support_adapter.py` | 摘取 `hrfont_support_adapter.py:8-27` | hidden显式配置，加入mask/权重/批处理；测试zero-init等价性。 |
| `model_wiring.py` | 从 Stage A `:410-430` 与 smoke `:65-88` 抽意图 | 实现 official/Delta RSI source 开关；Delta drop=0语义；新增层zero-init；参数名审计。 |
| `losses.py` | 从 Stage A `:424-445` 摘 | diffusion/perceptual/offset + SCR 可配；所有matched臂同调用路径。 |
| `trainer.py` | 新写，参考 A 的信号/原子保存 | accumulation、linear scheduler、overflow-aware optimizer-step、60s keepalive、STOP、1k/5k/val、strict resume。 |
| `train_e1.py/train_e2.py/train_e5.py` | 新写薄入口 | 只装配配置与freeze plan；不得重建数据、自动挑GPU或隐式resume。 |
| `preflight.py`、tests | 新写/调用现有pm_preflight | 数据/config/model/code/ink SHA、matched diff、输出不存在、step0等价、feature-mix数值测试。 |

应删除而非迁移：`/root` 常量、旧 DejaVu/JPG locator、Resize、自动 GPU 扫描/排除、隐式 latest/last resume、checkpoint unlink、`RuntimeError: continue`、训练时 rebuild data、全局 mutable cache 与 import-time IO。

### 3.2 最小修改路径

1. 先建立共享 config/data/RNG/checkpoint trainer，不改模型；用同一 A batch manifest 跑 E1 与 official-RSI E2b 的短 deterministic resume test。
2. 把 `hrfont_delta_feature.feature_delta` 搬入 variant，接 E0 float32 cache；用合成非线性 encoder 测试明确拒绝 pixel-mix，并验证每层 `sum alpha Ec(bank)-Ec(B0)`。
3. 在模型层加入 `rsi_source={official,delta}` 和独立 `delta_drop` RNG；仅新增 adapter/offset 接入 zero-init，验证 step0/Delta-drop 与无Delta前向等价。
4. 完成 E2/E2b matched YAML allowlist、相同 batch/drop draw 消费和 trainable参数审计，再接 perceptual/offset/SCR loss。
5. 修正检索残差与稳定顺序后迁移 SupportAdapter；先加载并 hash 审计 E2 的 UNet/Ec/Es，再冻结，仅优化 adapter。对无 support/drop batch 不计 optimizer step，或重采样至产生有效 adapter gradient；具体策略必须在 matched 配置中冻结。
6. 最后接 ink per-(font,char) 表、全角色一致性、split报告、keepalive/STOP/1k+5k milestones，并以 preflight 阻断任何旧域、Resize、SHA不符或已存在输出目录。

这一路径保留旧实现中最有价值的 Delta feature 算法、loss 组合、SupportAdapter 和停止/原子保存框架，同时把数据域、复现性和破坏性文件行为一次性隔离在共享基础设施中。
