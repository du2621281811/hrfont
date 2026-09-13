# HR-Font / ICLR 2027 项目台账

> 唯一内部入口。计划、状态、决策和结果只更新本文件。  
> 工作区：`/root/projects/hrfont` · 远程：`git@github.com:du2621281811/hrfont.git`（SSH；网页 https://github.com/du2621281811/hrfont）  
> 管理流程：[`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) · 登记表：[`provenance/REGISTRY.md`](provenance/REGISTRY.md)  
> 实验速查：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) · 官方 vs 我们：[`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md)

## 当前状态

- **主线：** F0@100k 父模型已定；F1/F2/F3/F3b 历史臂已跑满 80k（**新对照统一看 40k**）。Glyph Board Mode D（Es 一拍）已上 `main`（`682940f0`）。**F2-P / F3b-P 训练代码已进仓**（`cn2west_f123_rsi` + `launch_cn2west_f123.py`）；执行机报告 F2-P 正向40k训练，Git中可核验的最新 probe checkpoint 为25k，权重/大 cache **不走 Git**。训推：正式 `sample.py` 尚未接 `style_seq`；域探针用 `scripts/probe_style_domain_f2_f2p.py`。
- **Group G（执行中）：** `G0-F0-V0913-BS256-A-S3407` 8×32=**256**，lr `3.2e-4`，warmup 500，日程 **10k**，主看 **5k**。方案 `reports/G0_DESIGN_20260913.md`。之后 **G2 → G1∥G2-PRL**（均 10k，G0@10k parent）。看板 http://127.0.0.1:19000/g/ 。不覆盖 `F0-CLEAN-V0913-*`。
- **`v0913_clean` F0-c / F0-c-128：** 已中断给 G0 腾 8 卡。目录保留，不覆盖。v0913 watchdog 已停，避免把 F0-c 拉起来。
- **F2-VEC side study（2026-09-13 选择性迁入）：** 来源 `v100/f2-vec-mt-40k-eval@0aae4d1e`。分支报告 `F2-VEC-MT-A-S3407` 40k/fp16 完成（best@35k val=0.002105）；main 只接收矢量 variant、启动/评测脚本、设计文档、test16 few/one-shot 各 752 项预测及像素指标，不接收该旧分支对论文、主线看板或其他实验的删除/覆盖。权重和完整训练日志未随分支提供，结论仍按 side study 管理。
- **下一版方法设计：** Set-Delta Variation Prior + Graphics-Informed Local Reference Attention 已形成最终设计候选；Support 退出论文主方法。完整规格与合作者 review 点见 [`reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md`](reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md)，一页叙事见 [`reports/STORY_IDEA_20260907.md`](reports/STORY_IDEA_20260907.md)。状态是 **design-only，尚未实现/训练/验证**。
- **远端增量核查（2026-09-11，`74f92b7a`→`1dba686e`）：** `ce5f8079` 只更新 SSH 说明；`1dba686e` 带来实质增量：F2-P/F3b-P per-ref token与mask训练代码、launch/queue/probe脚本，以及 F2@40k vs F2-P@25k 的320张 train/val probe图。当前只能确认执行链路与可视化产物存在；probe没有量化指标或已填写人工结论，步数也不匹配，正式 `sample.py` 尚未接 `style_seq`，且没有对应 run provenance，因此不能据此声称效果改善。
- **多机同步：** 规则见 [`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) §3；新机步骤 [`docs/SETUP_COLLABORATOR.md`](docs/SETUP_COLLABORATOR.md)。扫描脚本 `scripts/pm_sync_scan.py`（约每 2h）。数据/cache 不走 Git；打包目录 `artifacts/migrate_v100/`（本机、不进仓）。V100 缺项与 scp 映射表：[`docs/V100_SCP_TRANSFER.md`](docs/V100_SCP_TRANSFER.md) + [`manifests/v100_scp_map.json`](manifests/v100_scp_map.json)（**3090 已填路径/体积/TTF；两机互不通，需第三方跳板 scp**）。
- **所里 V100（2026-09-11）：** 仓在 `/root/projects/hrfont`；conda `boogu` / torch `2.7.1+cu126` / V100 capability `(7,0)`。仓库级作者 `Liang Xiaowei <qr0w6666666@gmail.com>`；push 走 Deploy key `github.com-hrfont-443`（可写，禁止 force-push）。本机无 `/root/data/font_50`、无随体 TTF、无协议 B `summary.json`；Es/Ec/ckpt/support_bank 也不能从 TTF 生成。源机 `172.19.45.13:2222` 从本网段不可达。
- **旧状态备查：** [`reports/F123_STATUS_20260907.md`](reports/F123_STATUS_20260907.md)（日期已过，以本段为准）。
- **合作者离线看板：** [`reports/collab_offline/index.html`](reports/collab_offline/index.html)（分层评测 / 时间线 / Δ 检索 / 导师页）。
- **E12 v4：** 26 字型 cache_v4，三 seed **仍 gate_failed（T2）**；见 [`reports/E12_SELFTEST_V4_REVIEW_20260907.md`](reports/E12_SELFTEST_V4_REVIEW_20260907.md)。Mac 复训：[`reports/E12_MAC_COLLAB.md`](reports/E12_MAC_COLLAB.md)（`cache_v4` 已进 git，约 18 MB）。
- E1@100k 锚点保留。

## PI 决定（2026-09-05）

- F0 **跑满 100k**，不启用 60k early-stop；步数按 `DECISION_F123_20260906.md` §2 预注册规则（val16 loss 最小）事后选。
- F1/F2/F3 **固定单 seed 3407**，不补 3408/3409。
- E12 做 S1/S2/S4/S5；S3 外部字型池已在 v4 落地，T2 仍未过门。

## PI 决定（2026-09-12）

- **新训练默认 40k；正式比较统一看 40k。** 历史 F1/F2/F3 的 80k 跑完记录保留，新对照用 `global_step_40000`。
- **A1 零训练探针 parked**：换清洗后数据集再做；规格 `reports/SOLUTION_STYLE_WEAKNESS_20260909.md` §3。

## 下一步

> **下一版 Set-Delta / Round D–R 暂时停开**（2026-09-12 PI：先搁置，待新 idea）。新训练/正式评测统一 **40k**。

1. **Group G：** 先跑完 G0（约 2.4 h 到 10k；1.2 h 先看 5k）。watchdog 在 G0+cache 后开 G2，G2 完成后并行 G1 与 G2-PRL。`scripts/start_g_watchdog.sh`。
2. **F2-PRL：** `F2-PRL-A-S3407` **已完成 40k**（best@35k val=0.002055；40k val=0.002085；墙钟 3h55）。Demo-8 1-shot/8-shot 已进看板 `reports/f03_test16_strat/core_shot_board.html`。
3. **F2-RL128：** `F2-RL128-A-S3407` **已完成 40k**（best@35k val=0.002055；40k val=0.002080）。1-shot/8-shot 已进 Demo-8 看板。
4. **A1 零训练探针 parked：** 换清洗后数据集再做。题本/gate：`reports/SOLUTION_STYLE_WEAKNESS_20260909.md` §3。当前数据不跑、不作 A1 结论。
5. **F2 / F2-RL / F2-VEC 对照：** Demo-8；页 `reports/f03_test16_strat/f2vec_shot_board.html`（`http://127.0.0.1:19000/f2vec_shot_board.html`）。F2-VEC 只有 test16×47；1-shot 是 Mode D（Es=永、Δ 仍 ref8），不是 F2/F2-RL 的 true-1。完整核心对照仍是 `core_shot_board.html`。
6. 旧 F1/F2/F3 80k 证据冻结为历史；`hrfont_setdelta_graphicsref` 未实现前不要开新消融训练。
7. E12：T2 方案仍需改；不作为本轮对照。
8. p649：sampler/eval 接上 `training_map` 之前，训练继续走旧 A 盘 `fontdiffuser-p253-t295-s338-cn2west-v2`。

## 实验登记（摘要）

| ID | 状态 | 说明 |
|----|------|------|
| `E1-FTV2-A-S3407` | **completed 100k** | 主表数据域 FT 锚点 |
| `F0-RSIFREE-FT-A-S3407` | **completed 100k** | **整段无 RSI**；val=0.031089 |
| `G0-F0-V0913-BS256-A-S3407` | **starting 10k** | Group G parent；8×32=256；lr 3.2e-4；主看 5k |
| `G2-F2-V0913-A-S3407` | **queued after G0** | G0 parent；mean-Δ；10k |
| `G1-F1-V0913-A-S3407` | **queued after G2** | G0 parent；official RSI；10k |
| `G2-PRL-V0913-A-S3407` | **queued after G2** | G0 parent；F2-P+L；10k |
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

- 2026-09-13：开 Group G。G0=`G0-F0-V0913-BS256-A-S3407`（8×32=256，10k，主看 5k）。随后 G2，再 G1∥G2-PRL。中断 F0-c / F0-c-128 腾 8 卡，不覆盖其目录。方案 `reports/G0_DESIGN_20260913.md`。
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
