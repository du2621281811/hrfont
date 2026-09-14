# 实验一览（合作者速查）

> **一个仓即可：** `git clone git@github.com:du2621281811/hrfont.git`（网页 [`hrfont`](https://github.com/du2621281811/hrfont)）。  
> 项目管理：[`PROJECT_MANAGEMENT.md`](PROJECT_MANAGEMENT.md) · 登记表：[`../provenance/REGISTRY.md`](../provenance/REGISTRY.md)

## 防误用（最重要）

| 路径 | 含义 |
|------|------|
| `code/official/FontDiffuser/` | **官方干净**上游快照（只读） |
| `code/ours/FontDiffuser/` | **历史补丁**恢复版（旧实验线；默认只读） |
| `code/variants/<id>/` | **新实验**最小补丁（从 official 派生） |

- 看历史差异：`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`
- 说明：`code/README.md`、`docs/OFFICIAL_VS_OURS.md`

**禁止**把 `ours/` 称为官方；**禁止**在 `official/` / `ours/` 上堆新实验改动。

## 实验与结果

| 实验 | 做了什么 | 结果入口 | 代码树 |
|------|----------|----------|--------|
| `FT-CNSTYLE-25K` | 42 字体 CN→西，25k | `provenance/runs/…`；`reports/retrain_v2/FD_CNSTYLE_RESULTS.json` | `code/ours`（历史） |
| `FT-P253-CNSTYLE-12K` | 253 字体，12k | 同上；Style 池存疑 | `code/ours`（历史） |
| `A-MVP-CONTROL` / `DELTA` | Stage A，`INCONCLUSIVE` | `PROJECT.md`；`provenance/runs/A-MVP-*.json` | 脚本 + `ours` 权重树 |
| `E1-FTV2-A-S3407` | A 协议 FT-v2，228/16/16，seed3407，100k | `PROJECT.md`（已完成 100k）；`provenance/runs/E1-FTV2-A-S3407.json`；正式评测待补 | **`code/variants/cn2west_ft_v2`** |
| `E1C-FT-CONTINUE-S3407` | 原 E1 前向再训设想 | **superseded，未进入当前主线** | **`code/variants/cn2west_ft_v2`** |
| `E2-STAGE-A-S3407` / `E2B-…` | 旧 n-shot Stage-A 方案 | **STOP，不进当前结论** | **`code/variants/cn2west_stage_a`** |
| `F0-RSIFREE-FT-A-S3407` | RSI-free parent | **completed 100k；dirty 协议 A parent（勿覆盖）** | **`code/variants/cn2west_f0_rsifree`** |
| `F0-CLEAN-V0913-A-S3407` | 同一 F0，数据换成 `v0913_clean`（50/38/12） | **interrupted**（给 G0 腾卡；目录保留） | **`code/variants/cn2west_f0_rsifree`** |
| `F0-CLEAN-V0913-BS128-A-S3407` | F0-c-128：同数据 4 卡 DDP，32×4=128，lr×16 | **interrupted**（给 G0 腾卡；目录保留） | **`code/variants/cn2west_f0_rsifree`** |
| `G0-F0-V0913-BS256-A-S3407` | 旧 G0 parent：8×32=256，lr 3.2e-4 | **failed NaN**；目录保留 | **`code/variants/cn2west_f0_rsifree`** |
| `G0b-F0-V0913-BS256-A-S3407` | 同 batch；lr 1e-5，10k | **completed 10k** | **`code/variants/cn2west_f0_rsifree`** |
| `G2 → G2-RL → G-RL-pilot → G1 → G0c` | 串行；pilot 等 git READY；决策 `G_QUEUE_DECISIONS_20260914.md` | **G2 running**；其余排队 | **`cn2west_f123_rsi` + `launch_g0`** |
| `F1/F2/F2-RL/F2-PRL-CLEAN-V0913` | F0-clean parent + 新 Es/Ec | **parked**；v0913 watchdog 已停 | **`code/variants/cn2west_f123_rsi`** |
| `F1-OFFRSI-A-S3407` | official-reference RSI 对照 | **completed 80k；正式比较仍待统一汇总** | **`code/variants/cn2west_f123_rsi`** |
| `F2-DELTARSI-A-S3407` | 旧 mean-Delta | **completed 80k；作为下一版 mean-Delta baseline** | **`code/variants/cn2west_f123_rsi`** |
| `F3-JOINT-DS-A-S3407` | 旧 mean-Delta + own-font Support | **completed 80k；legacy，不作为相同信息预算主方法** | **`code/variants/cn2west_f123_rsi`** |
| `F2-RL128-A-S3407` | F2 + R-L128（global9 + Es block2 4×4, Linear 256→1024） | **completed 40k**；best@35k val=0.002055；1-shot/8-shot 看板 `reports/f03_test16_strat/core_shot_board.html`；权重 `runs/F2-RL128-A-S3407`（不进 Git） | **`code/variants/cn2west_f123_rsi`** |
| `F2-PRL-A-S3407` | F2-P+L（per-ref pooled h + Es block2 L128，去掉 up-path mean global9） | **completed 40k**；best@35k val=0.002055；1-shot/8-shot 看板 `reports/f03_test16_strat/core_shot_board.html`；权重不进 Git | **`code/variants/cn2west_f123_rsi`** |
| `F2-VEC-MT-A-S3407` | F2 mean-Δ + 可微矢量 decoder multi-task，40k/fp16 | **远端分支报告 completed**；best@35k val=0.002105；test16 few/one-shot 均 752 项。预测和像素指标已迁入 main；训练日志/权重未随分支提供，故作为 side study，不覆盖 F2 | **`code/variants/cn2west_f2_vec`** |
| `f2_pattn_s3407`（F2-P） | F2 + per-ref pooled h（drop mean global9） | **completed 40k**；best@35k val=0.002042；看板 F2-P 列 | **`code/variants/cn2west_f123_rsi`** |
| `f3b_pattn_s3407`（F3b-P） | F3b + per-ref pooled h | **completed 40k**；best@35k val=0.002057 | **`code/variants/cn2west_f123_rsi`** |
| `FT-P251-REF8-CN2WEST-V2` | 旧 planned 名（已由 E1/p260 取代） | TBD | **`code/variants/cn2west_ft_v2`** |

官方超参 / Loss / 渲染：`COLLABORATOR_GUIDE.md`。

## 方法方向（PI 2026-09-13 更新）

- **2026-09-14 方案与执行更新：** G 系效果优先 + TC-v2 规格见 [`G_STYLE_COMPLETION_PLAN_20260914.md`](../reports/G_STYLE_COMPLETION_PLAN_20260914.md)。Luna TC 实现/交接见 [handoff](../reports/TC_V2_EXECUTION_HANDOFF_20260914.md) / [review](../reports/TC_V2_LUNA_IMPLEMENTATION_REVIEW_20260914.md)。执行机已完成主臂 G0b→G2→G2-RL→G1→G0c，并在跑 8 卡队列 pilot→TC-G2→TC-G2RL（见 [`G_QUEUE_DECISIONS_20260914.md`](../reports/G_QUEUE_DECISIONS_20260914.md) / [`G_PROGRESS_SNAPSHOT_20260914.md`](../reports/G_PROGRESS_SNAPSHOT_20260914.md)）。test16 看板：`reports/g_v0913_shot/`。
- 规格：[`../reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md`](../reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md)
- 当前主线：Mean-Delta。Set-Delta 因时间预算弃用，以上规格作为历史设计保留。
- Support：不进入论文主方法；own-font support 只作额外观测工程模式或 oracle upper bound。
- 实验 ID、variant ID 与 provenance 在实现 review 通过后再登记；当前没有对应 checkpoint 或结果。
- 旧 Set-Delta / absolute-set / Round D–R 不再排期。当前快验为 Group G（G0b/G1/G2/G2-RL + 8卡 TC）。CGE × vector 与直接监督候选见 `../reports/DIRTY_DATA_AND_DIRECT_SUPERVISION_20260913.md`。

## 数据与权重

训练 JPG / 渲染 PNG / `.pt` **不进 Git**。见 `docs/DATA_AND_WEIGHTS.md`。  
渲染协议 A–H、重建命令、QA 入口见 [`PROJECT.md`](../PROJECT.md)「数据集准备」章节。

| 工具 | 入口 |
|------|------|
| QA Hub | `data/render_qa_hub.html`（`python -m http.server 8777 --directory data/`） |
| A–H 协议 Review | `data/cn2west_v2_abc_review/` |
| A 墨量预览 | `data/cn2west_v2_abc_review/proto_A_ink/` |
| R0 ink 审查 | `data/cn2west_v2_abc_review/proto_A_ink/review.html` |
| Ink 阈值 | `reports/R0_INK_GATE_PROPOSAL.md`（**已冻结**） |
| Split v3 | `manifests/split_v3_228_16_16.json`（228/16/16） |
| `v0913_clean` | [`INDEX`](../manifests/v0913_clean/INDEX.json) · [`SYNC`](../manifests/v0913_clean/SYNC.md)（frozen pair 表；F0 已接 50/38/12） |
| E1 训练看板（Git 快照） | `reports/e1_ft_v2_dashboard/` · `python -m http.server 8777 --directory reports` → http://127.0.0.1:8777/e1_ft_v2_dashboard/ |
| E1 训练看板（训练机实时） | 仅训练机：`data/e1_ft_v2_dashboard` → `runs/.../viz`（不进 Git） |
| F0–F3 训练看板（实时） | 训练机 `:8787`（`scripts/f123_monitor.py`，崩溃自动从 last_state 续跑 F2/F3） |
| F0–F3 训练看板（快照） | `reports/f123_dashboard/` · 已有 `:8765` reports 服务 → `/f123_dashboard/` |
| 字库语种浏览 | `reports/charset_picker/`（`python scripts/serve_charset_picker.py` → :8766） |
