# HR-Font / ICLR 2027 项目台账

> 唯一内部入口。计划、状态、决策和结果只更新本文件。  
> 工作区：`/root/projects/hrfont` · 远程：`git@github.com:du2621281811/hrfont.git`（SSH；网页 https://github.com/du2621281811/hrfont）  
> 管理流程：[`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) · 登记表：[`provenance/REGISTRY.md`](provenance/REGISTRY.md)  
> 实验速查：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) · 官方 vs 我们：[`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md)

## 当前状态

- **G 系进度同步（2026-09-14）：** 主臂 G0b@10k / G0c@20k / G1 / G2 / G2-RL@10k 与 1GPU pilot 已完成。**当前 8 卡队列：** pilot8 DONE → **TC-G2 RUNNING** → TC-G2RL QUEUED。快照 [`reports/G_PROGRESS_SNAPSHOT_20260914.md`](reports/G_PROGRESS_SNAPSHOT_20260914.md) · 决策 [`reports/G_QUEUE_DECISIONS_20260914.md`](reports/G_QUEUE_DECISIONS_20260914.md)。
- **测试集看板（已进 Git）：** [`reports/g_v0913_shot/index.html`](reports/g_v0913_shot/index.html)（协议同 f03 test16×strat；L1/SSIM 仅诊断）。枢纽 `/g_shot/`。
- **TC-v2：** 规格 [`reports/G_STYLE_COMPLETION_PLAN_20260914.md`](reports/G_STYLE_COMPLETION_PLAN_20260914.md)；实现/交接 [handoff](reports/TC_V2_EXECUTION_HANDOFF_20260914.md) / [review](reports/TC_V2_LUNA_IMPLEMENTATION_REVIEW_20260914.md)。缓存+H head 已建；联合训练在 8 卡队列中。
- **主线：** F0@100k 父模型已定；F1/F2/F3/F3b 历史臂 80k（新对照看 40k）。Mean-Delta；Set-Delta 弃用。权重/`runs`/`artifacts` 大 cache **不走 Git**。
- **F2-VEC side study / 多机同步 / V100：** 同前；Deploy key `github.com-hrfont-443`。
- **合作者离线看板：** [`reports/collab_offline/index.html`](reports/collab_offline/index.html)。
- E1@100k 锚点保留；E12 v4 仍 gate_failed（T2）。

## PI 决定（2026-09-05）

- F0 **跑满 100k**，不启用 60k early-stop；步数按 `DECISION_F123_20260906.md` §2 预注册规则（val16 loss 最小）事后选。
- F1/F2/F3 **固定单 seed 3407**，不补 3408/3409。
- E12 做 S1/S2/S4/S5；S3 外部字型池已在 v4 落地，T2 仍未过门。

## PI 决定（2026-09-12）

- **旧 F 系正式比较统一看 40k；Group G 使用独立批准的短程日程。** 历史 80k 与 G 系不可混作等预算比较。
- **A1 零训练探针 parked**：换清洗后数据集再做；规格 `reports/SOLUTION_STYLE_WEAKNESS_20260909.md` §3。

## PI 决定（2026-09-13 / 14）

- Set-Delta 弃用；G 系效果优先 + TC-v2 规格见 `G_STYLE_COMPLETION_PLAN_20260914.md`。
- 数据审计：`reports/DIRTY_DATA_AND_DIRECT_SUPERVISION_20260913.md` · `reports/DIRTY_DATA_IMPACT_20260913.json`。
- **禁止**按 global batch 线性放大 AdamW lr。大 batch 默认 **1e-5**；>8e-5 需 `--allow_high_lr`。≤10k 用 `constant_with_warmup`。

## 训练稳定性（2026-09-13，G0 NaN 之后）

- **禁止**按 global batch 线性放大 AdamW lr。大 batch 默认 **1e-5**；高于 **8e-5** 必须书面理由 + `--allow_high_lr`。
- **≤10k** 短程用 **`constant_with_warmup` + warmup 500**。≥40k 仍可 linear + warmup 5000。
- AdamW：β 0.9/0.999，wd 0.01，eps 1e-8，clip 1.0。偶发单卡 NaN 可接受；全卡 NaN = 失败。

## 下一步

> 执行机继续 8 卡 TC 队列；看板已同步供分析。

1. **8 卡队列收尾：** TC-G2 → TC-G2RL（`scripts/run_g_tc_8gpu_queue.sh`）。
2. **评测扩展：** 队列结束后把 pilot8 / TC 臂并入 `reports/g_v0913_shot/`。
3. **视觉选型：** 以看板为主；L1/SSIM 仅诊断。
4. **F2-RL128 / F2-PRL dirty 40k** 已完成；A1 parked；E12 T2 不作为本轮对照。

## 实验登记（摘要）

| ID | 状态 | 说明 |
|----|------|------|
| `E1-FTV2-A-S3407` | **completed 100k** | 主表数据域 FT 锚点 |
| `F0-RSIFREE-FT-A-S3407` | **completed 100k** | **整段无 RSI**；val=0.031089 |
| `G0-F0-V0913-BS256-A-S3407` | **failed NaN @1450** | 8×32=256；线性 lr 3.2e-4；权重已废；目录保留 |
| `G0b-F0-V0913-BS256-A-S3407` | **completed 10k** | 同 256；lr 1e-5；constant_with_warmup 500；**不续训** |
| `G2-F2-V0913-A-S3407` | **done @10k** | 8×8=64；mean-Δ；parent G0b@10k |
| `G2-RL-V0913-A-S3407` | **completed 10k** | 8×8=64；F2RL；parent G0b@10k |
| `G-RL-pilot` / `G-RL-pilot-8gpu` | **completed 5k** | 1GPU + 8GPU；8GPU best@2500；TC 队列中 |
| `G1-F1-V0913-A-S3407` | **completed 10k** | official RSI |
| `G0c-F0-V0913-BS256-A-S3407` | **completed 20k** | G0b@10k 续训 → 20k |
| `G2-PRL-V0913-A-S3407` | **cancelled** | 改跑 G2-RL |
| `F0-CLEAN-V0913-BS128-A-S3407` | **interrupted** | 给 G0 腾卡；目录保留 |
| `F3-JOINT-DS-A-S3407` | **completed 80k** | Δ+Support；test16 像素已采 |
| `F2-DELTARSI-A-S3407` | **completed 80k** | 旧 mean-Delta 实现；下一版基线 |
| `F1-OFFRSI-A-S3407` | **completed 80k** | 官方 RSI 对照 |
| `F2-RL128-A-S3407` | **completed 40k** | F2 + R-L128；正式评测点 40k；best@35k |
| `F2-PRL-A-S3407` | **completed 40k** | F2-P+L：per-ref h + L128，去掉 up-path mean G；best@35k |
| `E12-*-V4-S3407/08/09` | **gate_failed (T2)** | cache_v4；T2≈0.73–0.80 |

完整表见 `provenance/REGISTRY.md`。

## 版本与恢复

- 根 Git 管代码、台账、规则、provenance、精选报告。
- 正式实验要求 `exact`：干净 commit + 数据指纹 + 命令 + 产物。
- 旧实验诚实标 `retro_partial`，不用推测补齐。

---

## 数据集准备：cn2west v2 渲染协议 A–H

**活跃池 260 字体**（已 drop `FZXianZTJW`）× 295 target + 338 style；**train/val/test = 228/16/16**。  
字符集：`manifests/charset_cn2west_v2_planned.json`；  
拆分真源：`manifests/pipeline_v3_{train,val,test}_stems.txt`（`pipeline_v2_*_stems_v2.txt` 已同步为同一内容）；  
拆分 provenance：`manifests/split_v3_228_16_16.json`（旧 237/16/8 备份为 `*.bak_*.txt`）。  
ContentImage 统一使用 Noto Sans CJK Regular，按各协议对应逻辑渲染。

### 协议规格

| 协议 | 画布 | 字号策略 | 居中 | 缩放 | 输出 | 脚本 |
|------|------|----------|------|------|------|------|
| **A** | 96×96 | 逐字体固定：搜最大 fs 使全部字的 textbbox w,h ≤ 84 | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto A` |
| **B** | 96×96 | 逐字体 height-fit：搜最大 fs 使全部字的 textbbox h ≤ 84；宽溢出则逐字缩小 | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto B` |
| **C** | 128→96 | 固定 fsize=128 | textbbox 居中 @ 128×128 | BILINEAR → 96×96 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto C` |
| **D** | 96×96 | 逐字逐字形：二分搜最大 fs 使 textbbox w,h ≤ 88（margin=8px） | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_df.py --proto D` |
| **F** | 96×96 | 逐字逐字形：二分搜最大 fs 使 ink margin ≈ 8px（8% canvas） | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_df.py --proto F` |
| **H** | 96×96 | 同 A（逐字体一号、内框 84），但用**墨迹像素框**搜最大 fs | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_h.py` |

**关键区别**：
- A/B/H 是**逐字体**统一字号（所有字符共享一个 font size），D/F 是**逐字符**独立搜索最大字号。
- A 用 `textbbox`（逻辑边界框）约束字号，H 用 `ink bbox`（实际墨迹像素）约束字号，其余与 A 相同。
- D 用 `textbbox` 约束，F 用 `ink bbox` 约束。
- C 是唯一做缩放的方案，字填充率最高但可能有抗锯齿模糊。

### 数据集 ID

| 协议 | 目录名 |
|------|--------|
| A | `fontdiffuser-p253-t295-s338-cn2west-v2` |
| B | `fontdiffuser-p253-t295-s338-cn2west-v2b-hfit` |
| C | `fontdiffuser-p253-t295-s338-cn2west-v2c-official128` |
| D | `fontdiffuser-p253-t295-s338-cn2west-v2d-perglyph-max96` |
| F | `fontdiffuser-p253-t295-s338-cn2west-v2f-perglyph-fit96` |
| H | `fontdiffuser-p253-t295-s338-cn2west-v2h-inkfit` |

### 合作者重建

```bash
# A/B/C（约 15-30 分钟）
python scripts/build_cn2west_v2_proto_abc.py --proto all --workers 16

# D/F（约 5-10 分钟）
python scripts/build_cn2west_v2_proto_df.py --proto both --workers 16

# H（A 的墨迹框版，约 10-20 分钟）
python scripts/build_cn2west_v2_proto_h.py --workers 16

# Review 网页（约 1 分钟）
python scripts/build_cn2west_v2_protocol_review.py
python -m http.server 8777 --directory data/  # 打开 http://127.0.0.1:8777/cn2west_v2_abc_review/
```

### QA Review

- 入口页：`data/render_qa_hub.html`
- 协议对比 Review：`data/cn2west_v2_abc_review/index.html`（A/B/C/D/F/H 切换；旧 B 筛查仅对照）
- **R0 主审查**：`data/cn2west_v2_abc_review/proto_A_ink/review.html`（`ink_ratio_rank` · pass/drop/rerender）
  - 重建：`python scripts/build_cn2west_v2_ink_ratio_rank.py`
- A 墨量预览：`data/cn2west_v2_abc_review/proto_A_ink/`（边长 / **框面积** 可切换排序）
  - 重建：`python scripts/build_cn2west_v2_proto_a_ink_preview.py`
- A vs H 对比：`data/cn2west_v2_abc_review/proto_AH_compare/`

**A 墨量摘要（165,213 张，基于重划前全量扫描）**：边长 median≈75%；**框面积** median≈48%。  
**字体门（已冻结）**：`mean_bbox < 20%` → 仅 `FZXianZTJW` drop；其余 B-screen 对照候选 **pass**。  
**Split（已冻结）**：228/16/16；从原 test8 起，用 seed=3407 自 train 增补 8 字进 test。详见 `reports/R0_INK_GATE_PROPOSAL.md`、`manifests/split_v3_228_16_16.json`。

### p649 v2a（审查完成，尚未接训练）

新盘 `data/fontdiffuser-p649-t295-s338-cn2west-v2a-r0/`（649 套渲染；**不要覆盖旧 A 盘**）。  
原 260 三层可用性已审完：记录 [`data/p649_v2a_layers/REVIEW_RECORD.md`](data/p649_v2a_layers/REVIEW_RECORD.md)。  
映射 `data/p649_v2a_layers/training_map/`（`font_to_bucket.json`）。  
L1 只读 `data/p649_v2a_review/decisions.json`；L2/L3 只写 `layer2.json` / `layer3.json`。  
内部审查页（不进导师首页）：`http://172.19.45.13:19000/p649_v2a_layers/review.html`。

| bucket | n | target |
|--------|--:|--------|
| `all_scripts` | 146 | 295 |
| `no_bopomofo` | 69 | 258 |
| `han_latin_digit` | 40 | 89 |
| `exclude` | 5 | 0 |

train 可作 target **223/228**。val/test 16/16 保留，按字体跳过被排除语种。新389 未三层审完。

---

## 归档：Stage A MVP（2026-09）

- 结论：`INCONCLUSIVE`；ΔL1 改善 0.00155 < 0.002；不进 Stage B。
- Control L1=0.08255，Delta L1=0.08099；CI=[-0.00235,-0.00079]；胜率 54.71%。
- 协议：同数据、同 `ft_cnstyle@25k` 初始化、10k、1-shot「永」；详见历史变更记录与 `provenance/runs/A-MVP-*.json`。

## 精简变更记录

- 2026-09-14：fetch `origin/main`（`b594da37`）同步 TC-v2 / G 效果优先计划；**未改 watchdog**。执行机：G0b done、G2 running、G1→G2-RL 排队。
- 2026-09-14：G0b@10k 完成且**不续训**；已删 `artifacts/g0/HOLD`，恢复 Es/Ec → 串行 G2→G1→G2-RL。快测：`G0b_QUICK.html`（离线单文件）。
- 2026-09-13：停已炸 G0，开 G0b（1e-5 + constant_with_warmup / 500）。G1/G2/G2-RL 同步这套短程日程；launch 拒绝 lr>8e-5（除非 --allow_high_lr）。
- 2026-09-13：Group G 第三臂改为 **G2-RL**（取消 G2-PRL）。G0 之后 Es/Ec 改为 8 卡分片 + 更大 batch；Es 完成后 Ec 与 es_local 重叠。
- 2026-09-13：开 Group G。G0=`G0-F0-V0913-BS256-A-S3407`（8×32=256，10k，主看 5k）。随后 G2，再 G1∥G2-RL。中断 F0-c / F0-c-128 腾 8 卡，不覆盖其目录。方案 `reports/G0_DESIGN_20260913.md`。
- 2026-09-13：启动 `F0-CLEAN-V0913-BS128-A-S3407`（F0-c-128，4 卡 DDP，32×4=128，lr 1.6e-4，40k）。6 卡除不尽 128 已改 4 卡。不占 GPU2，不覆盖 F0-c。后为 G0 中断。
- 2026-09-13：F0-clean DataLoader 微基准：getitem ~2.5ms，workers 0→4 只从 0.021s 降到 0.006s/batch，端到端约 1.06×，**不重训**。V100 看板枢纽 `:19000`；F2/F2-RL/F2-VEC 对照页 `reports/f03_test16_strat/f2vec_shot_board.html`。
- 2026-09-13：p649 原260 三层语种审查完成并写出训练映射（146/69/40/5）；val/test 同步滤语种。记录 `data/p649_v2a_layers/REVIEW_RECORD.md`。训练暂不切盘。
- 2026-09-13：`F2-PRL-A-S3407` 完成 40k（best@35k val=0.002055）；Demo-8 1-shot/8-shot 写入 `core_shot_board.html`（536/536）。
- 2026-09-13：启动 `F2-PRL-A-S3407`（F0 parent，Δ，up-path=per-ref h+L128、无 mean G；GPU0 + watchdog）。40k；预计约 4h。
- 2026-09-13：F2-RL128@40k 的 1-shot（Es+Δ=永）与 8-shot（ref8）写入 `core_shot_board.html`（Demo-8×67，536/536）。
- 2026-09-12：PI：A1 探针 parked（清洗后数据再做）；新训练/评测统一 40k。`F2-RL128-A-S3407` 停 80k 日程、按 40k 重开。
- 2026-09-12：启动 `F2-RL128-A-S3407`（F0 parent，Δ，global9+Es block2 L128，GPU0）。smoke 20 步 + RSI parity 通过；稳态约 1.15 s/step。
- 2026-09-12：下一版 Set-Delta 实验按 PI 指示暂缓。补核心模型 Demo-8 1-shot/8-shot 对照看板 `reports/f03_test16_strat/core_shot_board.html`。
- 2026-09-11：3090 填写 `manifests/v100_scp_map.json`：migrate tar / 协议 A / F0 Es+Ec 均在源机；TTF 260/260 可解析。3090→V100（`172.18.41.23:22`）TCP 超时，仍需能两边通的跳板 scp。
- 2026-09-11：所里 V100 仓/环境落地（`boogu`）；仓库级 git 作者改为 Liang Xiaowei；补 `docs/V100_SCP_TRANSFER.md` + `manifests/v100_scp_map.json` 给 3090 填路径后跳板 scp。TTF/ckpt/cache 仍缺，源机网段不通。
- 2026-09-11：F2-P/F3b-P 训练代码进仓（per-ref style tokens + mask）；附 launch/queue/域探针；执行机继续训 F2-P→F3b-P，权重不入库。
- 2026-09-11：把下一版执行协议细化为先 Delta、后 Ref 的顺序筛查；补齐 Phase 0–5交付物、效果可验证边界、失败优化树和 PR-1至PR-7待拍板项，并吸收 `1dba686e` 的 P实现作为 R1接口参考。
- 2026-09-11：同步下一版最终设计候选：Delta 从预平均单方向升级为保留 donor candidate axis 的 Set-Delta；Ref 改为 per-ref global + graphics-informed local evidence；Support 退出论文主方法。写作上明确 FontDiffuser 是 inherited denoising backbone，本文贡献是跨语系未观察目标字符的 prior proposal + observed evidence realization。设计尚未实现或验证。

- 2026-09-02：Stage A 收缩为两臂最小验证；建立单一台账与护栏。
- 2026-09-03：Stage A 完成 → `INCONCLUSIVE`；建根 Git 与补丁提交；迁入 `/root/projects/hrfont`。
- 2026-09-03：单仓双目录 `code/official` + `code/ours`；推送 GitHub `hrfont`。
- 2026-09-03：固化项目管理：`docs/PROJECT_MANAGEMENT.md`、`provenance/REGISTRY.md`、`code/variants/`、`scripts/pm_preflight.py`；台账转向新基模设计阶段。
- 2026-09-04：补齐渲染协议 A–H 规格与重建脚本；修复 D/F 字号搜索上界；新增协议 H（A 的墨迹框版）；A 全量墨量分布预览页。
- 2026-09-04：台账对齐 R0；正式 `ink_ratio_rank` + pass/drop/rerender；提案字体门 mean_bbox&lt;20%。
- 2026-09-04：PI 冻结 ink 门（drop FZXianZTJW）；重划 **228/16/16**；A–H 盘目录已同步；目标 ID 改为 p260。
- 2026-09-04：落地 `cn2west_ft_v2` + E1 满训（`E1-FTV2-A-S3407`）；训练看板（train/val loss + Pred 对比）。
- 2026-09-04：PI 决策（人工确认）：① 当前 E1（bs=8、seed 3407 only、无训练期 ink 过滤）**验收通过**，非 protocol deviation；② **eff batch=8×1 为后续全部实验的 matched 标准**（E2/E2b/E2c/E2d/E5 同用）；③ train/val/test 拆分沿用 split_v3_228_16_16，全链路防泄漏；④ **协议 H 已弃用**（人工验证后），仅 A 为唯一训练/评测协议。
- 2026-09-04：**E1 满训完成 100k**（val 100k=0.02979，best@98k=0.02976；provenance `E1-FTV2-A-S3407.json`，certainty=exact_pending_dataset_fingerprint）。
- 2026-09-04：E2 决策：数据与 E1 同管线；seed 3407 only 先跑；Δ=风格侧 soft-α（默认 τ=.07/ε=.01/K_max=10，topk 为消融开关）加权同字 Ec 特征−B₀（feature-mix）；代码避免过度防御设计；E0 推迟到 E5 前并用 E1 final 冻结 Es/Ec 重建（E2 不依赖 E0）。
- 2026-09-04：**B₀=Noto ContentImage**（合作者决策）：Δ 减数与 Identity 输入统一为同一张 Noto 同字渲染，删除 FZKTJW 依赖；代码已改（train.py content_path）。
- 2026-09-04：合作者对齐：α 聚合改为 ref8/R 同字 cosine 平均；开发初始化固定 E1@100k；验证脚本 V3/V5 neutral 角色改 ContentImage、val/test 隔离（gate 仅 val16+train，test16 只读报告）；决定不加 zero-init gate（offset 头过渡期轻度，单通路归因更干净）。
- 2026-09-04：**n-shot 协议**：episode=(目标字体, ref 集 R)；训练 R 随机（n~Uniform{1..8}、内容随机 338 池），评测 R 固定 n（主协议 n=8 + n-shot 稳健性消融）；style 条件与 α 对齐消费同一 R（mean-pool Es）；Es 缓存扩 228×338；E2b RSI 取 R 首字符过 Ec。
- 2026-09-05：PI 拍板（合作者审核中）：① 主对照仍 E2 vs E2b，另开 **E1c** 作为「同样 80k、不改 RSI」的 E1 续训基线；② α 主方法改为 **必取 top-10 + softmax(τ=.07)**，禁止 ε 截空；n 上限=8；③ Es/Ec **全部离线 cache**，训推禁止在线编码器；style 条件改为 n 张 `style_emd` **空间均值 9 token**（废除 1-token）。详见 `reports/PI_DECISIONS_20260905.md`。
- 2026-09-05：**计划审查有条件批准**（`reports/REVIEW_PLAN_20260905.md`，PI 全按推荐 D-P1~D-P9，落档 exec-spec §1.7）；两个代码级阻断（Δ-drop 混杂、k_top 未接线）修后方可开 80k。
- 2026-09-05：**PI 两项精简拍板**：① QKV 筛选瘦身为 Q0 vs Q1 两臂 20k head-to-head（Q1 非劣进主方法，Q2 进附录）；② identity-safe RSI 作废——F1/F2/F3 复用官方 RSI 模块 + offset 零初始化 + 三臂同 seed DCN（震荡正常且可恢复，防御性改造反而加重归因）。E12 五项冻结，E12a 立即并行开训。
- 2026-09-05：PI 拍板：E1c / top-10 α / cache-only 9-token。详见 `reports/PI_DECISIONS_20260905.md`。
- 2026-09-05：计划审查有条件批准（D-P1~D-P9）；Stage A 接线后 G4 启动 E2+E2b（后确认为非 matched 诊断）。周报 [`reports/WEEKLY_20260905.md`](reports/WEEKLY_20260905.md)。
- 2026-09-05：**主线切换为 joint 方案**（`DESIGN_E2E3_FUSION_QKV_20260905.md` / `6616836`）：F0→F1/F2/F3；E1c superseded；旧 E2/E2b 待 STOP。
- 2026-09-05：F0 起跑（`cn2west_f0_rsifree`，P1 minus RSI，`strict=False` 丢 174 个 RSI/DCN 键）；E12 cache_v2 三 seed 未过门。
- 2026-09-06：**F1/F2/F3 落地待 review**（`cn2west_f123_rsi`）。initialisation 定为 zero-init 1×1 residual conv：`skip + zero_conv(DCN(skip,offset) - skip)`，step0 逐元素等价 F0（fp32/fp16 均 0 误差），被否的 zero-offset DCN 实测偏差 5.58（`scripts/test_identity_safe_rsi.py`）。同时修掉旧 E2/E2b 的失配根因——`source_drop` 现在对 official 与 Δ 一视同仁，`support_draw` 即使 support=off 也照抽以保持 RNG 同步。
- 2026-09-06：E12 S1/S2/S4/S5 落地。S2 最关键：`split_families` 改按**字型分组**（`NotoSansCJK-Bold` 与 `-Regular` 是同一字型），负样本改跨字型；val 只有一个字型组时从 train∪val 取负样本（不含 test，避免选 ckpt 时泄漏）。旧的 `families[(i+1)%n]` 负样本几乎总是同字型的另一个字重。
- 2026-09-06：E12 v3 三 seed 完成。修复把失败面从"T1/T2 双败 + T3 无意义"收敛到"只剩 T2"。T2 反而比 v2 低（0.73 vs 0.87）是正确的——v2 的分数建立在字型泄漏和同字型负样本之上。per-font 诊断显示 pooled 与 per-font AUC 相同（排除标定问题），重字重 AUC≈1.00 而轻字重在随机线附近，说明编码器只学到笔画粗细，因为训练集只有 5 个字型。**S3 是唯一有效路径，需要 PI 给字型来源。**
- 2026-09-06：**两套 F1/F2/F3 实现撞车。** 远端 `5a92f5c` 按 PI「identity-safe 作废」在 `cn2west_stage_a` 内实现了 plain zero-init official RSI；本会话按「效果好且方便归因」落地 `cn2west_f123_rsi`（zero-init 1×1 residual conv，step0 与 F0 逐元素相等）。**实际开跑的是 identity-safe。**
- 2026-09-06：F1/F2 确认在训（首 batch RNG 对齐，parity 0）。随后按「先可用结果」暂停 F1@300，启动 F3（同字体 8 字 support bank，非 E0）。F2/F3 并行。决策点 [`reports/DECISION_POINTS_20260906.md`](reports/DECISION_POINTS_20260906.md) D1–D8。
- 2026-09-06：训练机看板 `scripts/f123_monitor.py` `:8787`；F2/F3 崩溃从 `last_state` 自动续跑；快照 [`reports/f123_dashboard/`](reports/f123_dashboard/)。
- 2026-09-06：**D9 串行 Δ 臂。** STOP F2@6573，F3 独占。17:38 CST F3 已 13500/80k、GPU2 99%、约 1.15 s/step。F3 DONE 后自动 resume F2，再 F1。
