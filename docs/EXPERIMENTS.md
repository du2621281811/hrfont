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
| `F0-RSIFREE-FT-A-S3407` | RSI-free parent | **completed 100k；下一版唯一 parent** | **`code/variants/cn2west_f0_rsifree`** |
| `F1-OFFRSI-A-S3407` | official-reference RSI 对照 | **completed 80k；正式比较仍待统一汇总** | **`code/variants/cn2west_f123_rsi`** |
| `F2-DELTARSI-A-S3407` | 旧 mean-Delta | **completed 80k；作为下一版 mean-Delta baseline** | **`code/variants/cn2west_f123_rsi`** |
| `F3-JOINT-DS-A-S3407` | 旧 mean-Delta + own-font Support | **completed 80k；legacy，不作为相同信息预算主方法** | **`code/variants/cn2west_f123_rsi`** |
| F2-P / F3b-P | per-ref h tokens（去 ref 均值）；规格 `reports/DESIGN_F2P_F3BP_20260910.md` | **训练代码已进仓**；跑 `scripts/launch_cn2west_f123.py --arm F2P|F3bP`；队列 `scripts/queue_f2p_f3bp.py`；现有 probe 是 F2@40k vs F2-P@25k、仅图像无结论。正式 `sample.py` 与 run provenance待补；**不作为下一版 Set-Delta parent** | **`code/variants/cn2west_f123_rsi`** |
| `F2-VEC-MT-A-S3407` | F2 mean-Δ + 可微渲染矢量头 multi-task；40k / fp16 | side study，不进主表、不覆盖 F2。设计 [`../reports/F2_VEC_MULTITASK_DESIGN_20260913.md`](../reports/F2_VEC_MULTITASK_DESIGN_20260913.md)；test16 oneshot/fewshot：`F2VEC_40000` / `F2VEC_40000_s1` | **`code/variants/cn2west_f2_vec`** |
| `FT-P251-REF8-CN2WEST-V2` | 旧 planned 名（已由 E1/p260 取代） | TBD | **`code/variants/cn2west_ft_v2`** |

官方超参 / Loss / 渲染：`COLLABORATOR_GUIDE.md`。

## 下一版方法（设计阶段，不得当作已完成实验）

- 规格：[`../reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md`](../reports/HRFONT_DELTA_REF_FINAL_DESIGN_20260911.md)
- 主线：Set-Delta Variation Prior + Graphics-Informed Local Reference Attention。
- Support：不进入论文主方法；own-font support 只作额外观测工程模式或 oracle upper bound。
- 实验 ID、variant ID 与 provenance 在实现 review 通过后再登记；当前没有对应 checkpoint 或结果。
- 最小正文对照：FD/ref-only、no-Delta、old mean-Delta、geometry mean、Set-Delta、absolute set、wrong-character，以及 global/per-ref/learned-local/graphics-local Ref。

### 执行顺序（设计 arm，不是已运行 ID）

| Round | Arm | Delta | Ref | 20k裁决问题 |
|---|---|---|---|---|
| D | DS-A | D0 no-Delta | R1 per-ref global | 空分支基线 |
| D | DS-B | D1 old Ec mean | R1 | 旧 Delta 总效应 |
| D | DS-C | D2 geometry mean | R1 | 去 donor appearance 是否有效 |
| D | DS-D | D3 geometry Set | R1 | 候选轴是否优于预平均 |
| R | RS-A | 固定 D* | R1 per-ref global | global-only基线 |
| R | RS-B | 固定 D* | R2 learned local Es12 | 局部特征是否有效 |
| R | RS-C | 固定 D* | R3 graphics-informed local | graphics Key 是否有独立增量 |

- 所有筛查 arm 以 F0@100k为唯一 parent，seed3407、数据顺序、drop draws、batch和40k scheduler horizon匹配；20k决定主张是否保留，每个保留主张的 treatment/control 对由原 run继续到40k，不重启。
- D*与R*先满足 Identity/伪影非劣约束，再按冻结主指标选择；低于最小效应阈值时选择较简单 arm。
- Round D 前先完成共用 R1底座；Round R 的 RS-A与 Round D 的 D*+R1为同一配置，直接复用 run/checkpoint，不重复训练。
- 40k固定保留：D3主张对应D2+D3，D2主张对应D1+D2；R3主张对应R2+R3，R2主张对应R1+R2；另保留D0总效应基线。条件性必要对照为 residual→D4、Delta→D6、graphics Key→R4与FSFont-style baseline、dual path→D7；未入选路线保留20k screening negative。
- 40k机制矩阵保持单 seed3407；正式主比较子集仅含完整方法和最强必要基线，并运行至少3个预注册 seeds。旧F1/F2/F3的单seed决定不约束下一版方法。
- 具体 margin、效应阈值、正式 seed编号、双路径和 graphics Key 主配置仍为 PI review 项；详见设计稿 §12。

### 启动前同步门

已入库P训练实现只作为R1接口参考；正式 `sample.py` 的 `style_seq`接线、mask/CFG回归测试仍须补齐。下一版实现进入 `code/variants/hrfont_setdelta_graphicsref/`；当前run状态、step、checkpoint路径、启动命令和数据/代码指纹进入 `provenance/runs/<RUN_ID>.json`，摘要更新 `PROJECT.md`。无活跃run或checkpoint时显式填 `NONE`。

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
