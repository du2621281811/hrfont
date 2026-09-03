# HR-Font 代码审查报告（2026-09-03）

## 1. 总评

现有代码足以解释为何历史 Stage A/B 结果不宜作为新主线依据：Stage A formal 实现偏离论文给定的 feature-mix 定义，Stage B 又同时存在随机编码器与 support 永不启用的致命问题。`cn2west_ft_v2` 应严格从 official 做最小补丁，只迁移经过独立测试的数据布局、初始化加载和必要的兼容性修复；历史 Stage A/B/SCR 实验代码应冻结为不可复用证据。

## 2. 严重问题（bug / 泄漏 / 不可复现）

### S1. Stage B 训练使用随机初始化的 StyleEncoder/ContentEncoder

- 位置：`scripts/hrfont_e2_stageB96_train.py:246-269`。
- 问题：脚本只从 Stage A checkpoint 加载 `unet`（第 249-250 行），从未从 `ft_cnstyle` 或 Stage A 产物加载 `se`、`ce`；随后立即冻结并用于构造 Δ 与 support token。训练得到的是“冻结 Stage A UNet + 随机冻结编码器 + adapter”，而评测却加载 `FT` 编码器（`scripts/hrfont_e2_stageB_official_eval.py:182-199`），构成根本性的 train/eval 模型不一致。历史 Stage B L1=0.0964 不可解释为所述方法的性能。
- 修复：新实现显式从同一、带哈希的初始化目录加载 UNet/SE/CE，checkpoint 中保存三者来源和哈希；加载后做固定输入特征 checksum。Stage B eval 必须读取 checkpoint 记录的 encoder 来源，而非另行硬编码 `FT`。

### S2. Support retrieval 首次调用必然返回空候选，Stage B 实际退化为无 support

- 位置：`scripts/hrfont_support_utils.py:50-55`（Stage B 的直接运行依赖，虽未列入主清单但必须追踪）。
- 问题：`candidate_q()` 第 52 行在 `_load_cache()` 之前用 `(_CONTENT_VEC or {})` 过滤；进程首次调用时 `_CONTENT_VEC is None`，所以 `qs=[]`，之后加载缓存也无法恢复候选。`pick_support()` 因此在第 150-152 行总返回空 support。训练和评测中的 `n_support` 很可能恒为 0，adapter 没有有效梯度。
- 修复：将 `_load_cache()` 移到函数首行，再构造候选；增加单测断言高 gap 字符至少产生一个 `q`，并在训练启动时 fail-fast 检查 support 覆盖率和 adapter 梯度非零。

### S3. Stage A formal 不是论文定义的 feature-mix Delta

- 位置：`scripts/hrfont_e2_stageA96_formal_train.py:191-200,410-422`；同样问题见 `scripts/hrfont_e2_dropout_ablation_train.py:164-173,282-291`、`scripts/hrfont_e2_stageB96_train.py:133-140,183-192` 和 `scripts/hrfont_e2_official_eval.py:193-197,163-170`。
- 问题：这些路径先在像素空间计算 `mix_img=sum(alpha*I_g,c)`，再做 `Ec(mix_img)-Ec(content)`。方法定义要求 `sum(alpha*Ec(I_g,c))-Ec(neutral_c)`；非线性编码器下两者不等。只有 MVP 公用实现 `scripts/hrfont_delta_feature.py:80-98` 真正执行了 feature-mix。因此 formal 80k、dropout ablation、Stage B 及对应 official eval 测的不是声明的方法。
- 修复：在 cn2west_ft_v2 中统一复用“逐候选编码、逐尺度加权、再减 neutral feature”的纯函数；训练和评测共享同一实现，并用非线性 mock encoder 单测防止退化成 pixel-mix。

### S4. 数据构建默认会递归删除固定历史数据目录，违反版本化与不可覆盖护栏

- 位置：`scripts/build_retrain_v2_fontdiffuser_data.py:139-151`。
- 问题：默认目标是固定 `data/fontdiffuser`，存在即 `shutil.rmtree(dst)`；既无目标范围保护，也无 manifest/hash 确认。这会原地覆盖已绑定历史 checkpoint/metrics 的数据，直接违反 `.cursor/rules/hrfont-project-guardrails.mdc:9-10` 的“新数据新版本目录、历史链只读”。
- 修复：新 builder 强制传入不存在的版本目录（如 `fontdiffuser-p251-ref8-cn2west-v2`），拒绝 workspace 根/既有目录；先写 staging 目录、验证后原子 rename，并生成字体清单、源字体 hash、字符集 hash、renderer 版本和逐文件 manifest。

### S5. 正式 Stage A/B/ablation 没有设 seed，也未保存 RNG 状态

- 位置：`scripts/hrfont_e2_stageA96_formal_train.py:18,188,294,401-407`；`scripts/hrfont_e2_stageB96_train.py:12,145,244,332-337`；`scripts/hrfont_e2_dropout_ablation_train.py:11,161,225,277-280`。
- 问题：Python `random`、DataLoader shuffle、Torch CPU/CUDA RNG 均未初始化；checkpoint 也不含 RNG/DataLoader generator 状态。resume 会改变样本顺序、ref8 选择、dropout、噪声和 timestep，无法复现实验或做公平消融。
- 修复：所有入口要求 `--seed`；设置 Python/NumPy/Torch/CUDA seed，DataLoader 使用显式 generator；保存/恢复各 RNG 和 sampler 状态，记录 deterministic/CuDNN 配置。正式对照实验固定完全相同的样本顺序与噪声流。

### S6. 数据字体集合未做跨源身份规范化，Demo-8 隔离仅靠字符串 stem

- 位置：`scripts/build_retrain_v2_dataset.py:97-110`；`scripts/hrfont_e0_build_bank.py:59-69,336-364`。
- 问题：Demo-8 排除只比较 `Path(...).stem`；字体来自多个目录时没有基于文件 hash、字体内部 family/PostScript name 或规范化 stem 去重。相同字体的改名副本可进入训练池，构成静默 test-font 泄漏。E0 又按 train/test 两组一起渲染到同一 bank 根目录，若 stem 冲突会写同一路径并因“已存在则跳过”而掩盖污染。
- 修复：在构建前按字体二进制 SHA-256 + name table 身份去重；train/test 任一身份交集即 fail；输出隔离的 `train_bank/`、`demo8_bank/`，manifest 明确 split，消费者只接受对应 split 路径。

### S7. FontDiffuser 训练数据选择依赖未排序的目录遍历，seed 相同也不能跨机器复现

- 位置：`code/ours/FontDiffuser/dataset/font_dataset.py:41-64,82`。
- 问题：target/style 列表来自多层 `os.listdir()` 且不排序，随后 `random.choice`。文件系统枚举顺序不同会改变 dataset index、shuffle 后样本和随机 style 的映射，即使 seed 相同也不同。
- 修复：所有目录列表排序；更好的是 builder 生成带 hash 的显式 manifest，dataset 只读 manifest。该修复应以最小 patch 移植到 variant，不直接沿用整个 historical fork。

## 3. 中等问题

### M1. Formal eval 的随机噪声不成对，且 `--skip-ft` 会改变 Stage A 结果

- 位置：`scripts/hrfont_e2_official_eval.py:321-323,378-404,425-435`。
- 问题：只在整个程序开始调用一次 `set_seed(123)`。先跑 FT 会消耗 RNG，再跑 Stage A；使用 `--skip-ft` 时 Stage A 初始噪声又不同。两模型不是逐样本 paired comparison，L1 差可能混入采样噪声。
- 修复：为每个 `(font,char)` 从稳定 ID 派生 seed，或预生成共享初始噪声；每个方法在每个样本前重置同一 seed。输出逐样本 paired delta 与 bootstrap CI。

### M2. Stage A formal 的 style 条件训练为随机 Ref-8，评测固定单字“永”

- 位置：`scripts/hrfont_e2_stageA96_formal_train.py:187-200`；评测 `scripts/hrfont_e2_official_eval.py:39,408-421`。
- 问题：训练随机使用 8 个中文参考字之一，评测协议却是严格 1-shot “永”。这不是完全一致的条件分布，也使其和写明 1-shot 的 MVP/FT 对比含混。
- 修复：若主协议是 1-shot 永，训练也固定永；若训练要多参考字增强，需作为明确变量单独消融，报告中不得称“same protocol”。α 的 Ref-8 prototype 可保留，但应与生成条件区分命名。

### M3. 缺失 glyph 检测可能把 `.notdef` 方框当作有效字符

- 位置：`scripts/build_retrain_v2_dataset.py:29-34,75-84`。
- 问题：`ImageFont.getmask(ch).getbbox()` 非空并不证明 cmap 含该字符；FreeType fallback 的 `.notdef` 也可能有墨迹，因而生成大量“方框 GT/style”，形成静默标签污染。
- 修复：使用 fontTools cmap/glyph ID 检查并显式排除 `.notdef/.null`；再增加跨字符图像重复 hash/NCC 检测和覆盖率 fail threshold。

### M4. E0 对 cmap 解析异常采取 fail-open

- 位置：`scripts/hrfont_e0_build_bank.py:73-83`。
- 问题：字体解析出错时直接返回 `True`，把未知状态当成“有 glyph”，随后可能渲染 fallback 方框。
- 修复：正式数据构建必须 fail-closed，并把解析失败字体列入 manifest/quarantine；不可静默继续。

### M5. E0 cache 失败被吞掉，任务仍写“done”

- 位置：`scripts/hrfont_e0_build_bank.py:393-411`。
- 问题：每个 canvas 的 cache 异常只记日志，最后仍写 `e0_done.json`，下游可能误认 bank 完整；同时缺少数量/字体覆盖的硬断言。
- 修复：聚合失败并以非零退出；`done` 仅在所有 requested phase 验证通过后写入，附 manifest hash 和覆盖统计。

### M6. Stage A formal 吞掉任意 RuntimeError，可能带着系统性错误继续训练

- 位置：`scripts/hrfont_e2_stageA96_formal_train.py:477-487`。
- 问题：非 OOM 的 RuntimeError 被日志后 `continue`，会静默跳过坏样本/模型错误；若错误反复出现，训练可停滞或形成有偏数据分布而不失败。
- 修复：只捕获已知、可恢复且有计数上限的异常；其余立即保存诊断 checkpoint 并非零退出。记录 skipped sample ID 和比例，比例大于 0 即不允许作为正式 run。

### M7. Stage B 的 support 权重被计算但完全忽略，且数量截断含义错误

- 位置：`scripts/hrfont_support_utils.py:156-165`；`scripts/hrfont_e2_stageB96_train.py:146-148,195-206`；`scripts/hrfont_e2_stageB_official_eval.py:139-154,231-235`。
- 问题：retriever 返回 `aw*cover` 权重，但训练/评测仅加载图片并把所有 token 等权拼接。训练还截取 `TOP_M*3=9` 张，而方法写 K=3；实际是 K×M 的字体-字符图像，不是三个聚合 support，且 eval 不做同样截断。
- 修复：先明确单元是 K 个字符 support 还是 K×M 个字体实例；若按 α 重建 support，应按权重在 feature 空间聚合成 K 个 token。训练与评测共享同一函数并断言 token 数和权重归一化。

### M8. Stage B 将无梯度 batch 也计入训练 step，预算含大量空更新

- 位置：`scripts/hrfont_e2_stageB96_train.py:332-355`。
- 问题：低 gap、support bug 或 20% drop 时 loss 不依赖 adapter；脚本跳过 backward/step，却仍 `state["step"] += 1`。因此“25k”不是 25k 次参数更新，且在当前 support bug 下可能是 0 次更新。
- 修复：分别记录 `samples_seen` 与 `optimizer_steps`；正式预算按有效 optimizer step，启动前统计 support coverage，并对连续无梯度 batch 设置 fail-fast。

### M9. Style metric 的 `srr_pass_rate` 是自指阈值，几乎固定约 75%

- 位置：`scripts/hrfont_style_metrics.py:116-137`。
- 问题：阈值由同一批待评模型输出距离的 75 分位数定义，再计算 `< threshold` 的通过率；该比例按构造接近 75%，不能比较模型质量。所谓 FIC 实际只是生成与 GT 同字的 style-encoder 距离，也没有校准或可信区间。
- 修复：阈值必须在独立校准字体/真实样本上预先冻结，且校准字体不得进入训练或 Demo-8；报告连续指标、置信区间及对照模型，不使用自校准 pass rate。

### M10. checkpoint 自动选择会把未完成的 `last.pt` 当作正式结果

- 位置：`scripts/hrfont_e2_official_eval.py:236-250`；`scripts/hrfont_e2_stageB_official_eval.py:128-136`。
- 问题：指定里程碑不存在时回退到 `last.pt` 或任意最新 step，缺少完成状态、配置、数据 hash 和 run ID 校验；可能把中断或不同配置的 checkpoint 冒充 80k/25k。
- 修复：正式评测强制显式 checkpoint，校验 step/config/run provenance/hash；自动 fallback 仅供 smoke，输出必须标为非正式。

## 4. 轻微问题

### L1. 全项目大量硬编码 `/root`、GPU 编号和系统字体路径

- 位置：例如 `scripts/hrfont_stagea_mvp_train.py:40-45`、`scripts/hrfont_e2_official_eval.py:26-41`、`scripts/hrfont_e0_build_bank.py:19-32`、`code/ours/FontDiffuser/src/modules/stroke_scr.py:15-17`。
- 问题：当前 workspace 实际位于 `/Users/...`，脚本不可移植；自动排除 GPU 1、默认 GPU 2/3 也把机器拓扑写进实验逻辑。
- 修复：root、data、weights、font、device 全部 CLI/config 注入；配置写入 provenance，启动时打印并校验 resolved path，不做隐式 checkpoint fallback。

### L2. Historical fallback renderer 与原 pygame renderer 的版式不等价

- 位置：`code/ours/FontDiffuser/utils.py:98-132`。
- 问题：Pillow fallback 使用固定字号/居中，而 pygame 路径的 rasterization、metrics 和缩放不同；安装环境是否有 pygame 会静默改变输入分布。
- 修复：数据集固定单一 renderer 及版本，输出 renderer ID；不要把环境相关 fallback 带入新 variant。

### L3. E1 smoke 在 bank 缺失时用随机张量仍可报告 ok

- 位置：`scripts/hrfont_e1_rsi_delta_smoke.py:92-114,127-139`。
- 问题：只验证 shape/finite，不能证明真实 Δ 接线或 checkpoint 语义正确；随机 fallback 仍写 `ok=true`，容易被误读为功能验证。
- 修复：拆分 shape unit test 与 real-data integration test；正式集成测试缺依赖即失败，并断言 Δ=0/非零时输出或梯度按预期变化。

### L4. Stage A checkpoint 不保存数据/代码指纹；formal resume 还允许静默丢弃 optimizer state

- 位置：`scripts/hrfont_e2_stageA96_formal_train.py:325-345`；MVP 也仅保存有限 config：`scripts/hrfont_stagea_mvp_train.py:239-296`。
- 问题：无法确认 checkpoint 对应哪个字体池、cache、Git commit；optimizer load 异常只记录后继续，会把 resume 变成部分重启。
- 修复：保存 commit、dirty 状态、dataset/cache SHA-256、完整 args、依赖版本、RNG；正式 resume 任一关键项不匹配即失败。

## 5. 训练/评测一致性检查结果

| 检查项 | 结果 | 证据与结论 |
|---|---|---|
| Stage A MVP Δ 接线 | 基本一致 | 训练用 `feature_delta`（`scripts/hrfont_stagea_mvp_train.py:346-365`），评测复用同一函数（`scripts/hrfont_stagea_mvp_eval.py:223-247`）；逐样本 seed（第 248 行）使两臂采样可配对。 |
| Stage A formal Δ 定义 | **不一致** | 训练和 eval 均为 pixel-mix 后编码（`scripts/hrfont_e2_stageA96_formal_train.py:191-200,410-422`；`scripts/hrfont_e2_official_eval.py:193-197,163-170`），与方法给定 feature-mix 不符。二者彼此一致，但都不是论文方法。 |
| Stage A style shot | 不一致 | 训练随机 Ref-8（`scripts/hrfont_e2_stageA96_formal_train.py:187-199`），eval 固定“永”（`scripts/hrfont_e2_official_eval.py:39,408-421`）。 |
| Stage B encoder 权重 | **严重不一致** | train 随机 SE/CE（`scripts/hrfont_e2_stageB96_train.py:246-269`），eval 加载 FT SE/CE（`scripts/hrfont_e2_stageB_official_eval.py:182-199`）。 |
| Stage B support 数量/权重 | 不一致 | train 最多截 9 图（`scripts/hrfont_e2_stageB96_train.py:146-148`），eval 全取（`scripts/hrfont_e2_stageB_official_eval.py:139-154`）；两者均忽略 retriever 权重。 |
| CFG / Δ uncond | 部分一致 | formal train CFG blank content+style，但 Δ 独立以 25% drop（`scripts/hrfont_e2_stageA96_formal_train.py:401-421`）；eval 的 CFG uncond 自动把 Δ 清零（`scripts/hrfont_e2_official_eval.py:156-170`）。训练中 CFG drop 并不必然清 Δ，联合条件分布不同。 |
| 采样协议 | 名义一致、随机流不一致 | DPM++20、CFG7.5、96 和 DejaVu 一致（`scripts/hrfont_e2_official_eval.py:200-232,254-286`），但方法间未共享逐样本噪声。 |
| 历史 sanity 数值 | 可定位但不应继续背书 | 0.0810/0.0870/0.0964 的相对趋势与上述实现缺陷并不矛盾；尤其 Stage B 可能是零有效更新，不能用数值接近性证明实现正确。 |

## 6. 数据泄漏检查结果

### Leave-one-out 字体池

- MVP train 明确排除当前字体：`scripts/hrfont_stagea_mvp_train.py:346-352`，**通过**。
- Formal A/B 的 α 也排除 `g != font`：`scripts/hrfont_e2_stageA96_formal_train.py:160,189`、`scripts/hrfont_e2_stageB96_train.py:123,134`，**代码层通过**；但 identity 仅按 stem，比对不出改名副本，因此数据身份层仍未通过。
- Demo-8 eval 查询的 pool 为 train 列表：`scripts/hrfont_e2_official_eval.py:374-376,180-197`，按字符串列表看通过。

### Demo-8 隔离

- `build_retrain_v2_dataset.py` 按 stem 将 Demo-8 排除训练（`scripts/build_retrain_v2_dataset.py:97-118`），消费者也再次做字符串排除（`scripts/hrfont_e2_stageA96_formal_train.py:145-146`；`scripts/hrfont_e2_official_eval.py:374-376`）。
- **结论：列表级隔离有实现，身份级隔离不足。** 缺字体 hash/name-table 去重；E0 train/demo 共用输出命名空间（`scripts/hrfont_e0_build_bank.py:360-365`）会放大 stem 冲突风险。cn2west_ft_v2 上线前必须做 manifest 交集硬断言。

### Identity contamination（q == c）

- `candidate_q` 和 `cover` 均显式拒绝 `q == target_ch`：`scripts/hrfont_support_utils.py:50-55,75-84`，**逻辑规则通过**。
- 但当前首次加载 bug 让候选整体为空，因此不是“安全地工作”，而是 support 完全失效。修复加载顺序后应加测试：返回的每个 q 均不等于 c，且图片路径也不能通过别名映射回目标字符。

## 7. cn2west_ft_v2 迁移建议

### 值得带入（以重写/摘取最小 patch 的方式）

1. **CN `StyleImage` 独立池语义**：`code/ours/FontDiffuser/dataset/font_dataset.py:37-81`。这是 pivot 的核心已验证需求，但应改成 manifest 驱动、排序、严格覆盖检查。
2. **非 phase-2 也可显式加载官方初始化权重**：`code/ours/FontDiffuser/train.py:74-82`。保留语义，补充路径参数、hash 和 strict provenance。
3. **新版 diffusers 的 gradient-checkpointing 兼容修复**：`code/ours/FontDiffuser/src/modules/unet.py:199-211` 与 `src/modules/unet_blocks.py:190-232,325-371`。仅在针对当前依赖版本做 forward/backward 等价测试后摘取。
4. **MVP 的真正 feature-mix 工具思想**：`scripts/hrfont_delta_feature.py:43-64,67-102`。若未来重启 Stage A，可迁移为 variant 内受测模块；不应成为 cn2west 基线的必需复杂度。
5. **原子 JSON/checkpoint 写法与显式 paired seed 思路**：`scripts/hrfont_stagea_mvp_train.py:83-87,170-203,283-296` 和 `scripts/hrfont_stagea_mvp_eval.py:248-259`；需扩展到完整 RNG/provenance。

### 应丢弃或留在历史归档

1. Stage A formal / dropout / Stage B 中所有 pixel-mix Δ 实现：`scripts/hrfont_e2_stageA96_formal_train.py:191-200,410-422`、`scripts/hrfont_e2_dropout_ablation_train.py:164-173,282-291`、`scripts/hrfont_e2_stageB96_train.py:133-140,183-192`。
2. 当前 SupportAdapter/RS-Gap/Support retrieval 训练链：`scripts/hrfont_support_adapter.py:8-27`、`scripts/hrfont_support_utils.py:32-165`、`scripts/hrfont_e2_stageB96_train.py:106-208`。除非重新定义 token/权重并从单测开始验证，不带入 v2；PROJECT 也已明确暂停 Stage B。
3. StrokeSCR 整条 historical patch：`code/ours/FontDiffuser/configs/fontdiffuser.py:47-64`、`src/build.py:57-73`、`src/modules/stroke_scr.py:15-47`、`train.py:89-260`。它引入外部硬编码 repo、额外目标和复杂负样本语义，不属于最小 cn2west 基线。
4. 环境相关 pygame/Pillow fallback：`code/ours/FontDiffuser/utils.py:98-132`。
5. 所有自动找 GPU、自动 fallback checkpoint、覆盖固定输出目录的 orchestration：例如 `scripts/hrfont_e2_stageA96_formal_train.py:48-68,209-230` 与 `scripts/build_retrain_v2_fontdiffuser_data.py:139-151`。

## 8. 审查覆盖声明

本次完成了用户指定的全部范围，并额外阅读了 Stage B 的直接依赖 `scripts/hrfont_support_utils.py`，因为不读它无法验证 γ/θ/MMR/q≠c。实际阅读文件如下：

- 项目规则：`PROJECT.md`；`.cursor/rules/hrfont-project-guardrails.mdc`。
- 核心训练/接线：`scripts/hrfont_stagea_mvp_train.py`；`scripts/hrfont_e1_rsi_delta_smoke.py`；`scripts/hrfont_e2_stageA96_formal_train.py`；`scripts/hrfont_e2_stageB96_train.py`；`scripts/hrfont_e2_dropout_ablation_train.py`；`scripts/hrfont_delta_feature.py`；`scripts/hrfont_support_adapter.py`；以及直接依赖 `scripts/hrfont_support_utils.py`。
- 数据：`scripts/hrfont_e0_build_bank.py`；`scripts/build_retrain_v2_fontdiffuser_data.py`；`scripts/build_retrain_v2_dataset.py`。
- 评测：`scripts/hrfont_e2_official_eval.py`；`scripts/hrfont_e2_stageB_official_eval.py`；`scripts/hrfont_stagea_mvp_eval.py`；`scripts/hrfont_style_metrics.py`。
- 历史 diff：实际执行并逐项审阅 `diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude '*.txt'` 的全部 7 个差异文件：`configs/fontdiffuser.py`、`dataset/font_dataset.py`、`src/build.py`、`src/modules/unet.py`、`src/modules/unet_blocks.py`、`train.py`、`utils.py`；另读新增 `src/modules/stroke_scr.py`。

本审查未修改 `code/official/`、`code/ours/` 或 `scripts/`；仅新增本报告。
