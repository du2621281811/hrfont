# 实验一览（合作者速查）

> **一个仓即可：** clone [`hrfont`](https://github.com/du2621281811/hrfont)。  
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
| `E1C-FT-CONTINUE-S3407` | 原 E1 前向再训 80k（不改 RSI） | 计划中；口径 `reports/PI_DECISIONS_20260905.md` | **`code/variants/cn2west_ft_v2`** |
| `E2-STAGE-A-S3407` / `E2B-…` | n-shot 9-token；E2=Δ，E2b=官方 RSI | 计划中；YAML `configs/e2*.yaml` | **`code/variants/cn2west_stage_a`** |
| `FT-P251-REF8-CN2WEST-V2` | 旧 planned 名（已由 E1/p260 取代） | TBD | **`code/variants/cn2west_ft_v2`** |

官方超参 / Loss / 渲染：`COLLABORATOR_GUIDE.md`。

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
| E1 训练看板（Git 快照） | `reports/e1_ft_v2_dashboard/` · `python -m http.server 8777 --directory reports` → http://127.0.0.1:8777/e1_ft_v2_dashboard/ |
| E1 训练看板（训练机实时） | 仅训练机：`data/e1_ft_v2_dashboard` → `runs/.../viz`（不进 Git） |
| F0–F3 训练看板（实时） | 训练机 `:8787`（`scripts/f123_monitor.py`，崩溃自动从 last_state 续跑 F2/F3） |
| F0–F3 训练看板（快照） | `reports/f123_dashboard/` · 已有 `:8765` reports 服务 → `/f123_dashboard/` |
| 字库语种浏览 | `reports/charset_picker/`（`python scripts/serve_charset_picker.py` → :8766） |
