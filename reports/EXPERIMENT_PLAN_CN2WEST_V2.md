# HR-Font cn2west v2 完整实验执行计划

**版本：** v2（2026-09-04）
**覆盖决策：** D-A / D-FT / D-CFG。本文件覆盖旧计划中 B-r2/H 决策树、从 `ft_cnstyle@25k` 热启及所有 TBD。所有新训练/评测统一使用 A 协议；旧 `ft_cnstyle@25k` 仅为 `legacy-domain` 参考行。

## 1. 依赖拓扑与论文优先级

```text
R0（A + ink 审查门） ─→ E0（cache / RS-gap） ─→ E1（FT-v2）
                                                   ├→ E2（Stage A） ─→ E3/E4 ─→ E5 ─→ E6
                                                   ├→ E2b（FT-continue matched）
                                                   ├→ E2c（A-cold）
                                                   └→ E2d（SCR matched）
E12a（外部评测器） ───────────────────────────────→ E3/E4/E6/E11/E12b
E3 + E0 ─→ E8；E2/E2b ─→ E9；E5 ─→ E7；E2/E5 ─→ E10；E3/E6 ─→ E11
```

主文必需：R0/E0/E1、E2 vs E2b、E3/E4、E5/E6、至少一套 E2d SCR matched 排名、E8、E9 两项破坏+定位、E11、E12 自测。附录优先：E2c、完整 SCR 组、E7 全量、E9 接入全量、E10。

## 2. 全链路数据契约（A 唯一版本）

| 角色 | 冻结值 |
|---|---|
| dataset | **PI 2026-09-04**：活跃 **260=228/16/16**（drop FZXianZTJW）；清单 `manifests/split_v3_228_16_16.json`；正式 ID 待发 `fontdiffuser-p260-…-v2a-r1-<manifest8>`。原计划 261=237/16/8 已覆盖。盘上目录暂仍名 p253。 |
| renderer | A：96×96、margin6、逐字体统一 fs、633 字符 `textbbox w,h≤84`、RGB PNG、无缩放 [build_cn2west_v2_proto_abc.py:4](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:4) [build_cn2west_v2_proto_abc.py:82](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:82)。 |
| Content | **Noto Sans CJK Regular**，用 A 在 295 target 上求固定 fs 并落盘。选择 Noto 是因为现成 A 数据就是 Noto [build_cn2west_v2_proto_abc.py:12](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:12) [build_cn2west_v2_proto_abc.py:236](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:236)；改回 DejaVu 会破坏“所有角色同一 A 盘”，旧 FT 一致性已不再是目标。 |
| Style / ref8 | 当前字体的 A-style 图；训练在 338 字符内确定性采样；评测固定 ref8=`永和书风骨韵天地`，manifest 已给出 [charset_cn2west_v2_planned.json:25](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:25)。 |
| Target / GT | 当前字体、同目标字符的 A-target 图；GT 只作 positive control。 |
| α/Δ 库 | 仅 train237；训练目标字体 leave-one-out；val/test 永不入池。对同一字符 (c)，`Δ=Σα_s Ec(A(B_s,c))−Ec(A(B₀,c))`。 |
| B₀ | **Noto ContentImage**（与 Content/Identity 同一张 A 渲染）；Δ 减数、RS-gap、support neutral 共用坐标系（合作者 2026-09-04 决策）。 |
| Support q | `q≠c`、q 不等于当前 style ref；图不来自目标字体。默认 hybrid 候选：先 `ref8∪donors18`，不足再 `P1∪donors18`；top-K=3。 |
| Eval | test8×295；同 ref8/Content/GT/noise；按脚本和冻结 gap tertile 报告。 |
| 预处理 | 读 RGB PNG，断言 96×96；**不 Resize**；`ToTensor()`；`Normalize([0.5],[0.5])`。官方当前会 resize 后归一化 [train.py:97](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:97)，variant 要移除 Resize。 |

任何角色 dataset SHA 不同、出现 JPG/128 输入或 loader resize，preflight 失败。旧渲染产物/cache/α/gap 不可复用。

## 3. 全局超参总表

官方基准：batch4 [fontdiffuser.py:46](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:46)、lr=1e-4、linear、warmup10k [fontdiffuser.py:59](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:59)、AdamW β=(.9,.999)/wd=.01/eps=1e-8/clip1 [fontdiffuser.py:73](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:73)、joint CFG drop=.1 [fontdiffuser.py:69](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:69)、mixed precision=`no` [fontdiffuser.py:79](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:79)、SCR=.01 [fontdiffuser.py:44](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:44)。训练实现确实 joint blank content/style [train.py:181](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:181)，SCR 乘该系数 [train.py:216](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:216)。

| 项 | 全局冻结值 | 理由/例外 |
|---|---|---|
| optimizer | AdamW β1=.9, β2=.999, wd=.01, eps=1e-8 | 跟官方。 |
| effective batch | 单 GPU `bs=1, accum=4`，effective=4；E1 可在显存允许时 `bs=4,accum=1`，但 E2 matched set 固定 1×4 | 官方有效 batch4；E2/E2b/E2c/E2d 同 batch manifest。 |
| lr/schedule | FT-v2 主轨 `1e-5`；预热轨 `5e-5`；Stage A/B `1e-5`；linear decay，warmup=5,000（E1）/2,000（E2）/1,000（E5） | 官方 1e-4/10k 对 440k；本计划是 P1 适配，降 LR 并按总步数约 5%。不使用 cosine。 |
| mixed precision | `fp16`，loss/metrics float32；溢出即跳 step 并记 scaler | 官方默认 no；此处为资源选择，所有 matched 臂一致。 |
| gradient checkpointing | `false` | 官方无开关；96²、bs1 不需要，避免引入实现差异。 |
| grad clip / scale_lr | 1.0 / false | 官方 clip 在同步梯度时执行 [train.py:234](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:234)。 |
| CFG joint drop | .10（content+style 同时 blank） | 与官方和既有计划一致 [HRFONT_TRAIN_PLAN_V2.md:129](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:129)。 |
| Δ / support drop | `.25 / .20` | 既有 Stage A/B 规格 [HRFONT_TRAIN_PLAN_V2.md:130](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:130) [HRFONT_TRAIN_PLAN_V2.md:227](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:227)。独立 RNG stream。 |
| loss | diffusion MSE + `.01 perceptual + .5 offset`；SCR 变体再 `+.01 SCR` | 官方系数 [fontdiffuser.py:49](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:49) [fontdiffuser.py:50](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:50)。E5 只训 adapter但仍用同 diffusion objective。 |
| EMA | `false` | 官方代码无 EMA；不为 matched 组增加状态。 |
| seeds | 3407/3408/3409；早筛=3407 | 保存/恢复 Python/NumPy/Torch CPU+CUDA RNG、AMP scaler、optimizer/scheduler、sampler cursor；DataLoader 显式 generator seed，worker seed=`seed+1000*epoch+worker_id`。 |
| sampling | DPM-Solver++ multistep order2，20 steps，CFG7.5 | 官方默认 [fontdiffuser.py:84](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:84)。paired noise manifest。 |
| checkpoints | 每1k lightweight；每5k full milestone+eval；保留所有5k和 best，1k 仅滚动最近5个 | 兼顾诊断与磁盘。 |
| keepalive/STOP | 60s JSONL；>5min 告警；每 optimizer step 后检查 STOP，保存 `stopped_step`+RNG 后退出 | 不自动 kill/restart。 |

matched set 的 steps、effective batch、lr、scheduler、warmup、seed、batch order、CFG/Δ/SCR drop random draws必须一致；对不适用的 Δ 仍消费同一 RNG draw。

## 4. 实验配置卡

以下“完整配置”列覆盖全部变化项；未列模型拓扑沿 official P1。每 run 同时保存 resolved YAML 与 SHA-256。

### R0 — A + ink 审查门（主文基础）

| 字段 | 冻结值 |
|---|---|
| ID/依赖 | `R0-DATA-A-INK-P261`；依赖 split/TTF inventory。 |
| 配置 | `canvas=96,margin=6,inner=84,format=RGB_PNG,resize=false,workers=16,probe=all633,ink_threshold=250`; 新目录、失败聚合、原子发布。 |
| 门 | 主指标 mean(ink-bbox-area/96²)，次级 ink-pixel ratio；初始 review=`calib P5 or <.20`，drop 起点 `<.12` 且零误剔才启用。完整规格见 [RENDER_PROTOCOL_ANALYSIS.md](./RENDER_PROTOCOL_ANALYSIS.md)。 |
| 通过 | bbox 后验、cmap/tofu/empty/repeat、touch、双人审查、树 hash 全过；产出 p261 manifest/dataset SHA。 |

### E0 — 检索/cache/gap（主文基础）

| 字段 | 冻结值 |
|---|---|
| 完整配置 | `M=3, tau_alpha=.07, K=3, theta=.25, gamma=.35, mmr_lambda=.70, eps=1e-8, tie_break=score_desc_then_stem, seed=3407`; frozen FT-v2 Ec/Es，float32 cache。 |
| 扫描 | 只用 train237 中按 ink-ratio/字族分层固定的 calib16：`M∈{1,3,5}`, `tau∈{.03,.05,.07,.10,.15}`, `K∈{1,3,5}`, `theta∈{.10,.20,.25,.30,.40}`, `gamma∈{.20,.25,.30,.35,.40,.45,.50}`, `lambda∈{.5,.6,.7,.8,.9}`。先锁 α（ref8 leave-one-out Rank@1；并列取小 M/大 τ），再锁检索（write_ok 主、ID-CLS 次；support 开启率30–60%；并列取 K=3/λ=.7），最后锁 γ。 |
| 判据 | 不看 val/test；3 seeds 均值，主指标提升且最差 seed不退化>1%；冻结 `retrieval_calibration.yaml+SHA`。 |

### E1 — FT-v2（新一等公民）

| 字段 | 冻结值 |
|---|---|
| 目的/初始化 | 官方 P1 ckpt；官方 RSI 接线；A 全量 train237。旧 FT 不初始化。 |
| 数据预算 | 69,915 target。effective batch4 时一轮=17,479 optimizer steps；**100,000 steps≈5.72 个采样覆盖轮**，故主终点 100k，既覆盖 5–6 轮又与后续 80k 量级相近。val16 每5k。 |
| 主轨 | `steps=100000,bs=1,accum=4,lr=1e-5,linear,warmup=5000,CFG=.10,SCR=false,fp16,clip=1,seed=3407/08/09`；UNet+Ec+Es 全训，perceptual=.01, offset=.5。 |
| LR 双轨 | 预热轨只跑 seed3407：`lr=5e-5`，其余完全相同；在 20k 比较预注册 val diffusion loss、ID-CLS、style membership。若无 NaN/灾难遗忘且综合 z-score 比主轨≥0.25，则补到100k并选轨；否则停止，主轨为默认。轨选择写入 provenance，不能看 test8。 |
| best | 以 val16 的 `0.5*ID_z+0.3*style_z-0.2*quality_error_z` 选 5k milestone；同分取更早。 |

### E2 / E2b / E2c — Stage A 因果组

| 项 | E2 Stage A | E2b FT-continue matched | E2c A-cold |
|---|---|---|---|
| init | **E1 FT-v2 @100k**；Ec/Es 冻结；复用 E1 offset 头、无新增层（无 gate，接受轻度过渡期） | 同一 E1@100k；官方 RSI | official P1；Δ层同 E2 |
| 完整配置 | `steps=80000,bs=1,accum=4,lr=1e-5,linear,warmup=2000,fp16,clip=1,CFG=.10,delta_drop=.25,SCR=false,seeds=3407/08/09`; 训 UNet+offset，冻 Ec/Es | **完全同左**，包括 seed、batch manifest、optimizer、drop draws；`delta.enabled=false` 但消费 draw | 同 E2；先 seed3407，资源允许补三 seed |
| 理由 | 80k≈4.58 轮；FT-v2 已完成5.72轮，Stage A 是结构适配，不随字体数再扩到100k；5k eval早停但80k固定终点用于 matched 比较 | 排除额外训练量 | 初始化敏感性 |
| YAML关系 | `e2b.yaml` 只允许相对 `e2.yaml` diff：`experiment.id,model.rsi_source=official,model.delta.enabled=false,init.new_layer=null`；自动 structural diff 否则 preflight fail | — | 仅 init 与 ID 不同 |

### E2d — SCR matched

`E2D-FT-SCR` 对 E2b、`E2D-A-SCR` 对 E2；完整配置等于配对臂，只改 `scr.enabled=true, scr.weight=.01, temperature=.07,num_neg=16,nce_layers=[0,1,2,3]`。官方 SCR weight=.01、num_neg=16 [fontdiffuser.py:38](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:38) [fontdiffuser.py:42](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/configs/fontdiffuser.py:42)。每个正样本负样本为**同 content、异字体**，候选按 stem 排序后用 `sha256(run_seed,font,char,global_sample_index)` 取 16 个；禁止目标字体，resume 后完全一致。三 seeds，80k。

### E3 / E4 — 主对比与 gap 机制

- **E3：** official P1、FT-v2、E2b、E2、E2d、旧 FT legacy 各用 test8×295；3 generation seeds、DPM++20/CFG7.5/paired noise。ID-CLS、OCR、φ_s2 Rank@1/membership、coverage/LPIPS；按脚本/gap。旧 FT 单列 `legacy-render`，不参与 matched 显著性。
- **E4：** 无训练；continuous gap 对各轴 error 做 mixed-effects（font/char 随机截距）、Spearman、font-cluster bootstrap 10,000 次、95% CI；tertile 由 calib16 冻结。斜率>0 且 CI 不跨0才支持机制。

### E5 / E5b / E6 — Stage B 与主结果

| 项 | 冻结值 |
|---|---|
| E5 | init=E2 best；A 全冻，只训 `Linear→GELU→Linear→LayerNorm` SupportAdapter，末层 zero-init。`steps=25000,bs=1,accum=4,lr=1e-5,linear,warmup=1000,fp16,clip=1,support_drop=.20,CFG=.10,seeds=3407/08/09`；M/K/θ/γ/λ 用 E0 冻结值。 |
| E5b | init=E2D-A-SCR best；与 E5 同 steps/order/drop；`SCR=.01`。若 SCR 只作损失、A 仍冻结，则仅 adapter 更新；参数 hash 审计。 |
| E6 | 无训练；E2 vs E5、E2d vs E5b；test8×295、3 paired seeds；gap×method mixed model、font-cluster bootstrap10k；low-gap equivalence margin=校准集主指标 SD×0.1，高 gap CI 须排除0。 |

### E7 / E8 / E9 / E10

- **E7 Support 消融：** 同 E5 ckpt/noise/topK，`random-q` 重复5次（seeds 4407–4411）、whole-glyph kNN、wrong-font-q、no-neutral-subtraction、oracle-support；每臂 test8×295。必要训练 adapter 时固定25k/1×4/1e-5/三 seed。
- **E8 ref-set 干预：** 无训练；E2b/E2 在 `{a,o,p,q}` 上比较 ref8、`ref8+口+日`、`ref8+随机2字`；随机集固定 seed3407，8 test fonts×3 generation seeds，paired noise；闭合字改善且非闭合负对照不变才支持覆盖主张。
- **E9 Δ 接入/破坏：** RSI-offset、cross-attn、MCA、RSI+MCA 从同 E1 起点做20k/seed3407，统一1×4、1e-5、warmup1k；胜者补80k×3 seeds。spatial shuffle、channel-mean、magnitude-only、wrong-char、wrong-style 均将张量重新缩放到正确 Δ 的 per-layer RMS（`rms=sqrt(mean(delta²))+1e-8`，scale clip `[.25,4]`）；超过 clip 的样本排除并报告。offset truth=`|GT−B₀|` 灰度差，在 calib16 扫阈值 `{.05,.10,.15,.20,.25}`，以 AUPRC 最大、并列取 .15；同时轮廓距离阈值 `{1,2,3}px`，并列取2，冻结后报告 AUROC/AUPRC/IoU/correlation。
- **E10 CN→CN：** 未参与选择的中文 held-out 128 字；official/FT-v2/E2/E5，无训练，A 全角色、ref8、3 paired seeds；报告适用三轴/gap，目标是 non-inferiority，margin=calib SD×0.1。

### E11 / E12

- **E11 人评：** 从 E3/E6 按 method×font×script×gap 分层抽 240 个 2AFC + 120 个 MOS，3–5 位字体设计相关评审；随机左右/题序，10% 重复题，GT 正控和 wrong-font/char 负控。mixed-effects logistic/cumulative-link；Fleiss κ、Krippendorff α；自动指标对人评 Kendall τ/Spearman/AUC。问卷 calibration 与 test 分离。
- **E12a 评测器：** 项目字体池外公开字体；φ_s2 ResNet18，InfoNCE τ=.07，AdamW lr3e-4、wd=.01、bs64、50 epochs、cosine+5 epoch warmup、seeds3407/08/09；ID-CLS 同配置。T1 跨脚本一致性、T2 family AUC≥.90、T3 GT ID-CLS≥.90、T4 wrong-ref 分数显著下降。E12b 只读生成 PNG，batch64、float32、固定 model SHA。

## 5. 配置系统设计

YAML 为唯一真源，加载后递归转 dot-access 只读对象；CLI 用 `--set train.lr=1e-5 --set model.delta.enabled=true`，类型按 schema 验证，未知键报错。建议 schema：

```yaml
schema_version: 2
experiment: {id: E2-STAGEA-A-S3407, parent: E1-FTV2-A-S3407}
data: {dataset_id: fontdiffuser-p261-t295-s338-cn2west-v2a-r1-HASH, dataset_sha256: HASH, protocol: A, canvas: 96, resize: false, content_font: NotoSansCJK-Regular, b0_font: Noto ContentImage, ref8: 永和书风骨韵天地}
model: {base: official_p1, rsi_source: delta, delta: {enabled: true, drop: 0.25, init: zero}, support: {enabled: false}, scr: {enabled: false, weight: 0.01}}
train: {steps: 80000, batch_size: 1, accumulation: 4, lr: 1.0e-5, scheduler: linear, warmup_steps: 2000, optimizer: adamw, betas: [0.9, 0.999], weight_decay: 0.01, eps: 1.0e-8, fp16: true, grad_clip: 1.0, cfg_joint_drop: 0.10, ema: false, seed: 3407}
eval: {every_steps: 5000, sampler: dpmsolver++, inference_steps: 20, guidance_scale: 7.5, paired_noise: true}
retrieval: {M: 3, tau: 0.07, K: 3, theta: 0.25, gamma: 0.35, lambda_mmr: 0.70}
provenance: {git_sha: AUTO, variant: REQUIRED, config_sha256: AUTO, dataset_sha256: HASH, resume_rng: true, keepalive_sec: 60, stop_file: AUTO}
```

优先级：base YAML < experiment YAML < CLI override；启动前输出 canonical JSON（UTF-8、键排序、无路径时间戳）并 SHA-256。每 run 保存 `config.input.yaml`、`config.resolved.yaml`、`config.canonical.json`、`config.sha256`；provenance 写 config/dataset/model/code SHA、完整 CLI 和环境。resume 必须 config SHA 相同，唯一允许 override 是 `resume_from/stop_file`。

`pm_preflight.py` 当前实际检查 root Git clean [pm_preflight.py:41](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:41)、official/ours tree 与 marker [pm_preflight.py:50](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:50)、必要文档 [pm_preflight.py:60](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:60)、variant [pm_preflight.py:73](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:73)、experiment ID/run/provenance 唯一 [pm_preflight.py:84](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:84)。配置 wrapper 还必须在调用它之前检查：schema、无 TBD/null、A/p261、261=237+16+8、dataset/config/model SHA、96无resize、Content=Noto、B₀=Noto ContentImage、matched diff allowlist、seed/RNG resume、输出目录不存在、R0 ink gate=pass。

## 6. 时间线（9/18 摘要、9/25 全文）

| 日期 | 关键路径 | GPU 天（约） | 主文/附录 |
|---|---|---:|---|
| 9/4–9/5 | R0 ink 审查、p261 发布；E12a 启动 | E12 1–2 | R0/E12 主文 |
| 9/5–9/10 | **E1 FT-v2 主轨三 seed**；5e-5 轨先20k；E0 在首个冻结 E1 encoder 后重算 | E1 6–9 + LR轨0.5 | E1 主文 |
| 9/8–9/13 | E2/E2b seed3407并行，门通过后补 seeds；E2d至少一对 | 8–12 | 主文关键 |
| 9/13–9/17 | E3/E4、E5 seed3407、E6；E8；E9 20k+两破坏；摘要数字冻结 | 5–8 | 主文 |
| 9/18 | 摘要提交；锁 run/config/data SHA | 0 | — |
| 9/19–9/22 | E5/E2d 补 seeds、E7、E11、E9 定位；E2c/E10 有余量再跑 | 6–10 | 主文补齐/附录 |
| 9/23–9/24 | 统计、图表、provenance audit、复现 | 1–2 | 全文 |
| 9/25 | 全文提交、产物只读归档 | 0 | — |

GPU 天按单卡串行估计；多卡只缩墙钟，不改变 matched 配置。若延期，先砍 E2c 多 seed、E7 非关键臂、E10；不得砍 E1、E2b、SCR 风险控制或 E12 验证。

## 7. 决策记录

| 决策 | 最终值 | 理由 |
|---|---|---|
| D-A | A + 全633 ink-bbox ratio 人工门 | PI 决定；已渲染，几何审查可执行。 |
| D-FT | E1 FT-v2 从 official P1、A/train237 重训；旧 FT legacy | 消除旧42字体/旧渲染域。 |
| D-CFG | YAML+CLI override+resolved config SHA；所有值冻结 | 可复现且 matched 可 diff。 |
| 字体数/命名 | **260=228/16/16**（PI）；新 ID p260；旧 261/237/16/8 作废 | 与 ink drop + 均衡 val/test 一致。 |
| ref8 | `永和书风骨韵天地` | manifest 已定义。 |
| 内部 val | val16；另从 train237 固定 calib16 | val 选 checkpoint，calib 只调阈值，test8 不调参。 |
| E1 步数/LR | 100k；1e-5 主，5e-5@20k 预热轨后按 val 规则决定是否补齐 | 约5.72轮；保留更快适配而不污染 test。 |
| Content | Noto Sans CJK Regular | 当前 A 盘事实，避免再造不一致数据域。 |
| B₀ | Noto ContentImage | 与 Content/Identity 同一张 A 渲染（合作者 2026-09-04 决策）。 |
| milestone | 1k轻量、5k完整+val | 兼顾早诊断、选择与存储。 |

## 8. 在跑任务、重跑与旧产物

1. 旧 `ft_cnstyle@25k` 只在 E3 加 `legacy-domain` 参考行，不初始化 E1/E2/E5，也不参与 matched 统计。
2. 启动前只读核验 scheduler、runs、心跳、registry；不 kill/resume/改 STOP。新实验用新 run ID。
3. 只复用 SHA-256 验证且契约兼容的 official P1、代码无关元数据和外部评测器；旧 render/cache/gap/α/B₀ 一律不复用。
4. 复用记录 `parent_artifact_id,sha256,compatibility_reason`；缺 hash 视为不可复用。已有目录不得覆盖。
5. `PROJECT.md` 当前状态/命名可能仍写旧 planned/p253；本轮按限制不改它，R0 发布后由项目负责人同步 p261、E1 和真实 run 状态。

## 9. 需 PI 拍板（极短）

仅一项：**E1 的 5e-5 预热轨若在 20k 达到预注册胜出条件，是否授权额外 GPU 预算补到100k×3 seeds？** 默认可执行方案是“不等拍板”：1e-5 主轨照常跑满；5e-5 只跑 seed3407@20k，未获额外预算不进入主表。其余协议、字体、Content、B₀、步数与超参均已有默认值，可直接执行。
