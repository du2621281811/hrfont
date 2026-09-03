# HR-Font cn2west v2 完整实验执行计划

**版本：2026-09-04；供 Cursor 直接拆任务实现。** 本计划不启动训练、不占用现有 GPU、不覆盖任何历史或在跑产物。`PROJECT.md` 当前明确写“新基模尚未开训、Stage B/RS-Gap/Support 暂停” [PROJECT.md:8](/Users/xiaoweiliang/projects/hrfont/PROJECT.md:8)，与口头上下文“修复后重跑正在集群运行”矛盾；R0 前须由 PI/集群作业表确认真实状态并回填台账。

## 1. 依赖拓扑与资源假设

```text
状态核验 + B-r2/H 决策
  └─ R0 数据全链重建 ──┬─ E0 α/Ec/gap cache ──┬─ E2 Stage A ──┬─ E5 Stage B ── E6 主结果
                        │                      ├─ E2b FT-continue │              ├─ E7 support 消融
                        │                      ├─ E2c A-cold       │              └─ E5b B+SCR
                        │                      └─ E2d A+SCR        ├─ E3/E4
                        ├─ E12a 评测器训练/自测 ───────────────────┤
                        └─ E8 ref 干预（无需新训练）               └─ E9 Δ 接入/破坏/offset
生成结果 + E12 评测器 ── E10 中文→中文、E11 人评、E12b 汇总
```

4 卡只是上限，且现有任务优先。启动前保存 `nvidia-smi`/scheduler snapshot；新任务只使用确认空闲卡。建议：GPU0 保留现有重跑；GPU1 跑 R0/E0 或 E12；GPU2 跑 E2 主线；GPU3 跑 E2b/SCR。E2 达标后 GPU2 转 E5、GPU3 转 E9/E7。若只有一张空卡，严格按“E12→E2/E2b→E5→E6”主文关键路径排队。

## 2. 全链路数据契约（所有实验强制）

**建议协议目录：** 只有在 `RENDER_PROTOCOL_ANALYSIS.md` 的阻断项修复后，使用新建只写一次的 `data/fontdiffuser-p261-t295-s338-cn2west-v2b-r2-<fingerprint>/`；不要复用现有 `...p253...v2b-hfit`，虽然 review 显示实际是 261=237+16+8 [fonts.json:36](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:36)。若历史分布桥接失败，则正式目录改为严格复刻旧 128/JPG/Resize96 的 `...v2h-historical-r1-<fingerprint>/`。

每个 run 的 `data_contract.json` 必须逐角色写明：

| 角色 | 统一规则 |
|---|---|
| Content | Noto 或 DejaVu **只能选一个并冻结**；若选 B-r2，为与新 target 一致则用同一 B-r2 renderer；若强调与旧 FT 一致则用历史 DejaVu@128→JPG→Resize96。不能混写。 |
| Style / ref8 / Target / GT | 同一目标 TTF、同一协议版本和同一 96 网络预处理；ref8 固定“永和书风骨韵天地”。 |
| Δ 库同字与 B₀ | 同一个 c、同一协议、同画布/letterbox/normalize；`ΣαEc(B_s,c)−Ec(B₀,c)` 两项只允许字体不同。B₀ 固定 FZKTJW，Demo-8 永不入池。 |
| Support q | 库字体的 q 与 B₀ 的 q 同协议；`q≠c`、目标字体 leave-one-out、与当前 ref 去重。 |
| Eval | ref8、GT、生成输入与训练协议一致；DPM++20、CFG7.5；每 `(font,char,method,seed)` 配对同一初始噪声。 |
| 网络预处理 | 盘上已为 96 时只做 ToTensor+Normalize，不再次几何 resize；历史 128 盘才 BILINEAR→96。禁止任意拉伸；如 letterbox，四边等量、白底、参数入 fingerprint。 |

启动检查：读取 dataset fingerprint；随机抽 100 个 `(font,char)` 比对各角色 shape、ink centroid、预处理 hash；验证 cache 内 fingerprint 完全相等，否则 fail closed。旧计划要求 bank/target/style/content 与 FT 完全一致 [HRFONT_TRAIN_PLAN_V2.md:15](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:15)，新协议若偏离，必须在论文与 provenance 明示为 domain-adapted FT。

## 3. 通用训练与评测模板

- 默认训练：96²，AdamW，bs=1，gradient accumulation=4（effective bs=4；显存允许可改 physical bs，但同一 matched set 必须相同），lr=1e-5，mixed precision；content+style joint CFG drop=0.10。Stage A 另有 Δ-drop=0.25；Stage B support-drop=0.20，这与既有计划相符 [HRFONT_TRAIN_PLAN_V2.md:121](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:121) [HRFONT_TRAIN_PLAN_V2.md:226](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:226)。
- 正式主表至少 seeds `{3407, 3408, 3409}`；资源不足先用 3407 选配置，再对最终方法和关键对照补齐 3 seeds。显式 `--seed`；设置 Python/NumPy/Torch/CUDA、`PYTHONHASHSEED`；DataLoader 用 seeded `torch.Generator` 与 `worker_init_fn`。checkpoint 保存/恢复 Python、NumPy、CPU/CUDA RNG、sampler epoch/offset、optimizer/scheduler/scaler；resume 后以固定 batch/noise 做 bitwise 或容差验证。
- 里程碑：每 1k 保存可恢复 `latest`，每 5k 保存不可覆盖 milestone，val best 单独指针；Stage A 80k，Stage B 25k。磁盘紧张只清理由登记策略允许的非 milestone，绝不覆盖。
- 每 run 新 ID：`CN2W-V2-<EXP>-<PROTO>-S<seed>-YYYYMMDDThhmm`；先在 `provenance/REGISTRY.md` 查重并新增 JSON。preflight 会检查 Git clean、variant、run/provenance ID 唯一 [pm_preflight.py:41](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:41) [pm_preflight.py:84](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:84)。代码只进 `code/variants/<id>`，official/ours 只读 [docs/EXPERIMENTS.md:8](/Users/xiaoweiliang/projects/hrfont/docs/EXPERIMENTS.md:8)。
- 每 run 有 `STOP_<run_id>`：训练循环在 optimizer step 后检查，收到后保存 `stopped_step_x`、RNG 与原因再退出。keepalive 每 60 秒追加 JSONL（time/step/loss/lr/GPU/last_ckpt）；超过 5 分钟无心跳告警，不自动抢卡或重启未知进程。
- 统一评测：test8×295 为完整主表；另报 ASCII/Latin-ext/平假名/片假名/注音。三轴为身份（独立外部字体 ID-CLS 主、OCR ensemble 辅）、style（独立 φ_s2、episodic Rank@1、membership verifier）、quality（coverage 与 LPIPS-to-library）。所有指标按 calibration-font 冻结的 gap 低/中/高层报告；GT 是 **positive control**，不得称 ceiling。现有框架要求评测编码器与方法 Es 隔离 [EVAL_FRAMEWORK.md:24](/Users/xiaoweiliang/projects/hrfont/reports/EVAL_FRAMEWORK.md:24)，gap 边界只在校准字体冻结 [EVAL_FRAMEWORK.md:67](/Users/xiaoweiliang/projects/hrfont/reports/EVAL_FRAMEWORK.md:67)。

## 4. 实验卡

### R0（主文基础）数据重建

| 字段 | 计划 |
|---|---|
| 实验ID | `R0-DATA-BR2`（或状态门改为 `R0-DATA-HR1`） |
| 假设/目的 | 建立无缺字、可追溯、全角色一致的 261 字体 v2 盘。 |
| 依赖 | PI 冻结协议；确认在跑任务与数据目录；render audit 全过。 |
| 数据规格 | B-r2；train/val/test=237/16/8；295 target+338 style；完整角色契约见 §2。Content/ref/GT/库/B₀ 同 protocol；网络不二次 resize，白底等比 letterbox 仅在协议定义内。 |
| 模型/初始化 | 无。 |
| 配置 | CPU 16 workers；worker 无 RNG；排序 job manifest；异常聚合；临时目录→QA→原子发布。建 target/style/content、α pool、support bank、Ec cache、gap 表；从 train 中预注册 calibration fonts，绝不使用 val/test 调阈值。 |
| 评测 | cmap/tofu/空图/重复图；bbox/ink/coverage/touch/shrink；全盘 hash；抽样人工 review。 |
| 证实/证伪 | 所有阻断 QA=0、fingerprint 一致即通过；任一 silent fallback/角色 fingerprint 不同即失败。 |
| 产出 | 版本化 dataset、`manifest.json`、`render_audit.json`、bank/cache、`gap_raw.json`、dataset provenance ID。 |
| 时长 | 0.5–1 天 CPU + 0.5 天 QA。 |

### E0（主文基础）检索与缓存

| 字段 | 计划 |
|---|---|
| 实验ID | `E0-CACHE-BR2` |
| 假设/目的 | 冻结 top-3 α 原型、Ec 多尺度特征、B₀ neutral、RS-gap/coverage/MMR；检索可复现。 |
| 依赖 | R0；冻结 official Es/Ec ckpt。 |
| 数据规格 | R0 同盘；α pool 仅 train237、训练 leave-one-out、Demo-8/val16/calibration fonts 按角色排除；ref8 与 eval 同协议；库同字、B₀ 均 B-r2、相同 letterbox/normalize。 |
| 模型/初始化 | 冻结 FT 使用的 Ec/Es；α=Es ref8 prototype cosine、top3、τ=.07；gap≥.35 只是起点，最终由 calibration tertiles/开启率冻结。 |
| 配置 | 无训练；seed=3407 仅用于 tie-break 且稳定按 `(score,stem)` 排序；cache 写 dataset/model fingerprint。 |
| 评测 | 重跑 hash 一致；leave-one-out/Demo-8 isolation；α 权重和=1；QR 投影数值稳定；gap tertiles 仅 calibration。 |
| 证实/证伪 | cache 重跑完全一致，gap 非退化且 support 开启约 30–60%；否则修定义，不在 test 上调。 |
| 产出 | `alpha.json`, `ec_multiscale.pt`, `b0.pt`, `gap_strata.json`, `retrieval_audit.json`, provenance。 |
| 时长 | 0.5–1 GPU 日。 |

### E2 / E2b / E2c（主文关键、附录）Stage A 因果组

| 字段 | E2 Stage A（主文） | E2b FT-continue（主文关键） | E2c A-cold（附录） |
|---|---|---|---|
| 假设/目的 | Δ-RSI 比同训练量官方 RSI 更好 | 排除“只是多训了 steps” | 初始化敏感性 |
| 依赖 | R0+E0；新 variant 冒烟 | 同左 | official P1 ckpt |
| 数据规格 | 三者完全相同 R0 train237/val16、295/338、ref8；Content/Style/Target/Δ/B₀/eval 均 §2；E2/E2c 才用 Δ cache | 同数据但 RSI=Ec(style glyph) | 同 E2 |
| 初始化 | `ft_cnstyle@25k`，新 Δ/offset 层初始化 | 同一 `ft_cnstyle@25k`，官方 RSI | official P1；Δ/offset 新初始化 |
| 训练 | 80k、bs1×accum4、lr1e-5；冻 Ec/Es、训 UNet+offset；CFG .10、Δ-drop .25；3 seeds；1k/5k ckpt | **完全同 steps/bs/lr/seed/batch order/drop schedule**；冻/训范围尽量 matched，官方 RSI；若结构参数数目不同须报告 | 同 E2；3 seeds 可先 1 seed，80k或预注册 40k诊断+80k终点 |
| 评测 | test8×295；统一三轴、gap tertiles、paired noise、φ_s2/ID-CLS/OCR/membership/Rank@1；GT positive control | 同一生成 manifest | 同一生成 manifest |
| 证实/证伪 | E2 相对 E2b 在 identity/style membership 显著提升且非只高噪声 seed；offset 定位证据由 E9补。无差/更差则否定 Δ 接线收益 | 若追平 E2，则原 Stage A 增益属 extra steps | 若显著依赖 FT，只将方法定位为 adaptation，而非从 P1 普适训练 |
| 产出 | 各 seed ckpt、paired generations、metrics、run provenance | 同左 | appendix ckpt/metrics |
| 时长 | 各约 2–3 GPU 日；E2/E2b 可并行，E2c 后排 |

若预算允许，补完整 2×2 `{P1,FT}×{official-RSI,Δ-RSI}`：E2c 是 P1+Δ；再加 P1+official matched continuation。关键统计用 paired bootstrap（font 为 cluster）和 seed 间方差，不只报单点。

### E2d（主文至少一组匹配；其余附录）SCR 匹配消融

| 字段 | 计划 |
|---|---|
| 实验ID | `E2D-FT-SCR`, `E2D-A-SCR`, `E5B-B-SCR` |
| 假设/目的 | 排除“方法只是补回被删除的官方 Phase-2 style supervision”；检验 SCR 是否近似正交增益、A/B 排名不变。官方 Phase2 确有冻结 SCR+风格对比 [ICLR2027_HRFONT.md:300](/Users/xiaoweiliang/projects/hrfont/reports/ICLR2027_HRFONT.md:300)。 |
| 依赖 | R0/E0；E2；B+SCR 依赖 E5。 |
| 数据规格 | 与各自无 SCR 配对臂完全相同；负样本同 content、异字体，仍需 deterministic sampling；全部 B-r2。 |
| 模型/初始化 | FT+SCR 从 FT 起点；A+SCR 从同 FT 起点；B+SCR 从 E2 best，冻结策略与 B 对齐，仅额外 SCR loss。 |
| 训练 | 与无 SCR 臂 matched steps/seed/order；SCR 权重沿官方设定并预注册，不在 test 调。理想跑 3 seeds；最低要求 A vs A+SCR、B vs B+SCR 同 seed。 |
| 评测/判据 | 统一 E3/E6；若 SCR 全面增益但 A−FT、B−A 在高 gap 仍保持，则 ranking invariant；若 A/B 优势消失，主张需降级。 |
| 产出/时长 | 3 组 ckpt/metrics/provenance；每组 2–3 GPU 日，可与主线错峰并行。 |

### E3（主文）96²主对比

| 字段 | 计划 |
|---|---|
| 实验ID | `E3-MAIN96` |
| 假设/目的 | 比较零样本官方、旧 FT（零步）、FT-continue、Stage A、FT/A 的 SCR 变体，建立 Δ 的 matched causal evidence。 |
| 依赖 | E12 自测通过；E2/E2b；至少一对 SCR。 |
| 数据规格 | test8×295；所有方法的 content/ref8/GT 必须用 R0 protocol。旧 ckpt 若输入分布不同，另列 `legacy-domain`，不得混作 matched 行。 |
| 模型/初始化 | 只读各 ckpt。 |
| 配置 | 每方法×3 seeds；DPM++20、CFG7.5、同 `(font,char,seed)` noise；无训练。 |
| 评测 | ID-CLS primary+OCR ensemble；φ_s2 Rank@1+membership；coverage+LPIPS-library；总体/脚本/gap tertile；GT positive control。 |
| 证实/证伪 | E2 显著胜 E2b 才支持 Δ；只胜旧 FT 不足。SCR 下排名保持增强可信度。 |
| 产出/时长 | `paired_generation_manifest.json`, 三轴表、CI/效应量、provenance；0.5–1 GPU 日。 |

### E4（主文）coverage-failure 相关性

| 字段 | 计划 |
|---|---|
| 实验ID | `E4-GAP-FAILURE` |
| 假设/目的 | official/FT 的错误随 `gap(c,R)` 单调上升，支持“参考覆盖不足”。 |
| 依赖 | E0、E3 generation、E12。 |
| 数据规格 | 同 E3；gap 用冻结 calibration tertiles，不能根据 test error 重切。 |
| 模型/训练 | 无。 |
| 评测 | 每轴 error 对 continuous gap 做 mixed-effects regression（font/char random intercept），另报 tertile 趋势、Spearman、bootstrap CI。 |
| 证实/证伪 | official/FT identity/style error 随 gap 单调且斜率>0；无相关则弱化覆盖机制主张。 |
| 产出/时长 | 图、回归表、混淆矩阵；0.5 CPU/GPU 日。 |

### E5 / E5b（主文）Stage B 与 B+SCR

| 字段 | E5 Stage B | E5b B+SCR |
|---|---|---|
| 假设/目的 | Support 给出欠定区域的“what change”，收益集中高 gap | 检验 SCR 下 B−A 是否保持 |
| 依赖 | E2 best+E0；只有 E2 达标才开 | E2d A+SCR 或预注册的 matched SCR 起点 |
| 数据规格 | R0 全链；gap≥冻结门槛才取 q；residual coverage+MMR top3；库 q 与 B₀ q 同协议；低 gap 空 support；eval ref8 同协议 | 完全 matched |
| 初始化/冻结 | E2 best；冻结 A（UNet+Ec+Es），只训零初始化 SupportAdapter | matched SCR 版本；可训范围必须与无 SCR 对齐 |
| 训练 | 25k、bs1×accum4、lr1e-5、support-drop .20、3 seeds、1k/5k ckpt；除 adapter 外参数 hash 前后相同 | 同 steps/seeds/order，SCR weight 预注册 |
| 评测 | 同 E6；paired noise；低/中/高 gap；外部评测器 | 同左 |
| 证实/证伪 | 高 gap 显著改善，中 gap较小，低 gap约零（equivalence margin 预注册，如主指标标准差0.1）；若全层等幅或低 gap受损，机制不成立 | B−A 排名在 SCR 下保持；否则报告交互 |
| 产出/时长 | ckpt、adapter-only state、参数冻结 audit、metrics/provenance；约1–1.5 GPU日 | 同，约1–1.5 GPU日 |

### E6（主文核心）Base vs +Sup gap 主结果

| 字段 | 计划 |
|---|---|
| 实验ID | `E6-SUPPORT-GAP` |
| 假设/目的 | B−A 的增益随 gap 上升；低 gap 有统计等价性。 |
| 依赖 | E5、E12。 |
| 数据规格 | test8×295，R0 协议与 frozen tertiles；同 ref8/Content/GT/noise。 |
| 模型/训练 | 无。 |
| 评测 | 所有三轴总体/脚本/tertile；continuous gap×method mixed model；paired cluster bootstrap；GT positive control。 |
| 证实/证伪 | 交互项方向正确、high显著、low落入 equivalence bound；只报 high cherry-pick 不算证实。 |
| 产出/时长 | 主图、主表、sample sheet、provenance；0.5–1日。 |

### E7（主文精简、附录全量）Support 消融

同一 E5 ckpt 做推理时 intervention，必要时各训练一只 matched adapter：`random-q`、`whole-glyph-kNN`、`wrong-font-q`（故意选 style 最远但仍排除目标字体）、`no-neutral-subtraction`、`oracle-support`（仅作为机制上界，不进方法主表）。所有臂用 R0 同协议、同 q 候选预算/topK、同 noise/seed、test8×295 与 gap 层；random 重复 5 次。若主检索优于 random/wrong/no-neutral，且 oracle 更高，支持检索和减中性各有作用；oracle 不高说明 support 通道本身无效。产出为消融表、检索日志、provenance；0.5–2 GPU 日。完整字 support 的 neutral subtraction 公式来自既有设计 [HRFONT_TRAIN_PLAN_V2.md:206](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:206)。

### E8（主文机制实验）ref-set 干预

无需训练，使用 official RSI 与 FT-continue；在 test8 对 `{a,o,p,q}` 比较 ref8 与 `ref8+口+日`（可再加随机两个汉字的容量对照）。新增 ref 必须由同目标字体、R0 同协议渲染；Content/GT/noise 不变。检验闭合结构字符的 ID-CLS、membership、gap 是否定向改善，并用非闭合字符作负对照。只有 `口日` 比随机 refs 对 a/o/p/q 更有效，才支持“结构覆盖”而非“更多参考总会更好”。产出 paired intervention 表/图；0.5 GPU 日。

### E9（主文破坏证据；接入位置附录）Δ 接入与定位

- 接入位置：`Δ→RSI offset`（主）、`Δ→cross-attn`、`Δ→MCA`、`RSI+MCA`；从同 FT 起点、matched 20k 早筛，胜者补 80k，至少 1 seed，最终主/对照补 3 seeds。
- 破坏实验：推理时 spatial shuffle（每尺度固定 permutation）、channel mean（保留空间均值或广播，预注册具体定义）、magnitude-only `||Δ(x,y,:)||₂` 经 matched projection、wrong-char Δ、wrong-style-neighbor Δ。控制 Δ 的 RMS，避免“破坏后只是幅度变了”。
- offset 定量：保存多尺度 offset magnitude；由 `|GT−neutral|` 阈值/轮廓距离生成变化区域，在 calibration 冻结阈值；报告 pixel AUROC、AUPRC、IoU、point-biserial correlation，并按 gap tertile。可视化固定 test 字体/字符，不挑最好样本。
- 判据：正确 spatial Δ 和 RSI offset 优于结构破坏；offset magnitude 与真实 contour-change region 对齐。magnitude-only 若不降，不能声称空间结构必要；cross-attn 若同样好，叙事改为一般 change conditioning。产出各 ckpt/metrics/heatmaps/provenance；2–5 GPU 日。

这组实验把 Δ 限定为“主要决定 where to adapt 的空间线索”，Stage B 为“补足 what change 的局部实现”；不得声称 Δ 是 pure-where。

### E10（附录）中文→中文同管线

从未参与西文训练选择的中文字符集抽 held-out 字，Content/Style/Target/Δ/B₀ 全用 R0 同协议；比较 official、FT-continue、A、B，配对 noise 与三轴中适用于中文的 ID-CLS/style/quality，gap 同样分层。无训练或仅评测；目标是不显著伤害中文任务，而非证明主方法在中文 SOTA。产出 appendix table；0.5 GPU 日。

### E11（主文）人评问卷

在 E3/E6 冻结后生成题本：分层抽样 font×script×gap，随机左右与题序；包含 FT vs A、A vs B、official vs B 的 2AFC 成员资格题，MOS 三维（风格成员资格/可读身份/视觉质量），单独身份题；插入 GT positive control、wrong-font/错误字符负对照和重复题。每位至少覆盖预注册平衡区组，3–5 位字体设计相关评审。

统计用 mixed-effects logistic（选择~方法×gap，评审/font/char 随机效应）、2AFC CI；身份/MOS 用相应 cumulative-link mixed model；一致性报 Fleiss κ（类别）与 Krippendorff α（有序/缺失稳健）；自动指标对人评报 Kendall τ、Spearman 与预测 AUC，训练/选择阈值只用 calibration 问卷子集。现有框架的题型与 κ/τ/AUC 要求见 [EVAL_FRAMEWORK.md:117](/Users/xiaoweiliang/projects/hrfont/reports/EVAL_FRAMEWORK.md:117)。产出匿名题本、响应、分析脚本/报告；制题1天、收集2–4天、分析1天。

### E12（主文基础，可最早并行）评测框架建设

| 阶段 | 内容 |
|---|---|
| E12a（与 R0/E0/E2 并行） | 用项目池之外公开字体训练 φ_s2（跨脚本 InfoNCE）、专用 external-font single-glyph ID-CLS、membership verifier；划外部 train/held-out families，绝不碰 train237/val16/test8。训练 episodic prototypes/Rank@1 评测。 |
| 自测门 | T1 跨脚本同族一致性；T2 same/different family AUC≥.90；T3 GT ID-CLS 主分类器准确率预注册阈值（OCR ensemble 仅次级）；T4 换错 refs 分数显著下降。原框架列出 T1–T4 [EVAL_FRAMEWORK.md:86](/Users/xiaoweiliang/projects/hrfont/reports/EVAL_FRAMEWORK.md:86)。 |
| E12b（依赖生成结果） | 只读 E3/E6 PNG，运行 identity/style/quality/gap aggregate；输出 per-sample 与 aggregate JSON、版本化 dashboard。 |
| 配置 | φ_s2 ResNet18/ViT-S 起点、InfoNCE τ=.07、轻增强；3 seeds。ID-CLS 在外部字体多 renderer 增强，确保不是只认渲染器。membership verifier 以 held-out family 做 calibration。 |
| 判据 | 自测不过，自动 style/identity 指标不得进主表；优先修评测器，不能回看 test 输出挑模型。 |
| 时长/产出 | 1–2 GPU 日训练 + 0.5 日自测；`eval_models/<version>`、model cards、T1–T4、provenance。 |

## 5. 时间线与主文优先级

以 2026-09-04 起算、摘要 9/18、全文 9/25 倒排：

| 日期 | 必须完成 | 可并行/后补 |
|---|---|---|
| 9/4–9/6 | 状态核验、B-r2修复、R0、E12a启动 | 历史 H bridge、E2c |
| 9/7–9/11 | E0；E2与E2b并行；E12自测 | FT+SCR/A+SCR 至少一对 |
| 9/12–9/14 | E3/E4；E5；E8 | E9 接入位置早筛 |
| 9/15–9/17 | E6 主图/表；E9关键破坏；摘要数字冻结 | E7 全量、E10 |
| 9/18 | 摘要提交；冻结摘要所引数字和 run IDs | — |
| 9/19–9/22 | SCR matched set补齐；E7；E11收集；复现实验 | E2c、完整2×2 |
| 9/23–9/24 | 最终统计、图表、provenance audit、论文交叉引用 | E10 |
| 9/25 | 全文提交，产物只读归档 | — |

**必须进主文：** E2 vs E2b、E3、E4、E5/E6、至少一套 SCR matched ranking、E8、E9 的至少两项空间破坏+offset AUC/IoU、E11、E12 验证。**附录优先：** E2c/2×2、E7 全量、接入位置全量、E10、T1–T4细节。若关键路径延迟，先砍 256、A-cold 多 seed、非关键 support 臂，不能砍 FT-continue、SCR 风险控制或评测器验证。

## 6. 与当前/在跑重跑的接续

台账目前只登记 `FT-P251-REF8-CN2WEST-V2` 为 planned [REGISTRY.md:28](/Users/xiaoweiliang/projects/hrfont/provenance/REGISTRY.md:28)，并规定新入口、新 run ID、新数据版本且不覆盖历史 [PROJECT.md:24](/Users/xiaoweiliang/projects/hrfont/PROJECT.md:24)。因此执行者应：

1. 先读取 scheduler、`runs/`、provenance 与心跳，列出真实在跑 run ID/commit/data fingerprint；只读，不 kill、不 resume、不改 stop 文件。
2. 若修复后重跑确在跑：只复用经 hash 验证的 official/FT ckpt、冻结 encoder 权重和已通过 QA 的 renderer-independent metadata；**不得复用**旧像素 cache、gap、α、B₀、ref renders，因为协议变化会使它们失配。
3. 所有本计划任务生成新 ID、目录与 provenance；`pm_preflight` 后才 launch。已有任务占卡时排队，不把同一卡加入新 launcher。
4. R0/E0 完成后，PROJECT.md 回填真实协议 ID/fingerprint、状态、作业号和是否替代旧 planned 行；E2/E5 只在前置门通过后从 `planned` 改 `running`。
5. 复用任何产物都在 provenance 写 `parent_artifact_id + sha256 + compatibility_reason`；缺 hash 视为不可复用。
