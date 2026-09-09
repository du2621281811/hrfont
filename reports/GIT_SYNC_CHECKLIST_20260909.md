# Git 同步清单与检查 — 2026-09-09

## 结论（检查后）

| 项 | 状态 |
|----|------|
| 必要代码/配置 | 可提交，无密钥 |
| 报告 HTML/指标 | 可提交 |
| F3/F3b support bank | 原在 `artifacts/` 被 ignore → **已拷到** `reports/artifacts_sync/` |
| E1 预测图 | 曾是指向 `runs/` 的符号链接 → **已换成实拷贝**（否则合作者拉不下来） |
| E12 数字 | `runs/` 不可提交 → **已写** `reports/e12_v51_summary.json` |
| F2@75k preds | 尚未进 git，应一并 `git add` |
| 明显密钥 | 未发现 |

## A. 必要：代码与配置（必交）

```
code/variants/cn2west_f123_rsi/FontDiffuser/configs/fontdiffuser.py
code/variants/cn2west_f123_rsi/FontDiffuser/train.py
scripts/launch_cn2west_f123.py
scripts/hrfont_support_adapter.py
scripts/eval_f03_test16_strat.py
scripts/build_f3b_support_bank_stroke.py
scripts/build_e12_cache_train228.py
scripts/build_e12_eye_probe.py
configs/e12_phi_s2_v51_train228_s3407.yaml
configs/e12_membership_v51_train228_s3407.yaml
```

## B. 必要：实验结果与说明（必交）

```
reports/PI_BRIEFING_20260909.html
reports/METHOD_FIGURE_BRIEF_FOR_GPT.html
reports/PI_EXEC_DECISIONS_20260909.md
reports/AGENT_DECISIONS_F1_F3B_E12.md
reports/TRAINING_STATUS_F1_F3B_E12.md
reports/EVAL_INDEX_AUDIT_20260909.md
reports/COMPARE_PORTAL_DESIGN_20260909.md
reports/RESULTS_COMPARE_20260907.html
reports/e12_v51_summary.json
reports/artifacts_sync/          # support banks + README
reports/compare_portal/         # index.html manifest.json（可省略重复的 briefing 拷贝）
reports/collab_offline/index.html
reports/e12_eye_probe_v4_s3407/ # 眼测页 + 图
```

### 主评测板 `reports/f03_test16_strat/`

**必交元数据：** `PROTOCOL.json` `browse_index.json` `metrics_*.json` `index.html` `status.json` `timeline.html` `timeline_f2.html`

**必交预测图（五列主对比）：**

```
preds/P1/          # 多半已在仓
preds/E1_100k/     # 本次改为实文件后需新 add
preds/F0_100k/     # 多半已在仓
preds/F2_75000/    # 未跟踪，需 add（约 8.6MB）
preds/F3_80k/      # 多半已在仓
```

**建议交：** `delta_retrieve/`（若已部分在仓则保持）  
**不要交：** `logs/`、`*.pid`、`f03` 下重复的 PI_BRIEFING 拷贝（根目录已有）

## C. 可选：本机运维脚本

```
scripts/watchdog_f1_f3b_e12.py
scripts/start_watchdog_f1_f3b_e12.sh
scripts/launch_f2_mid_eval.sh
scripts/launch_f2_mid_eval_parallel.sh
scripts/watch_f2_mid_until_done.sh
reports/watchdog_f1_f3b_e12/   # 仅 status/incidents 文本；不要 pid 也行
```

## D. 明确不要提交

- `runs/**`（权重与完整训练目录）
- `data/fontdiffuser*`、大 cache
- `artifacts/f0/es_*` `ec_*` 等大文件（用 `reports/artifacts_sync` 代替小 json）
- `**/*.pth` `**/*.pt`、`logs/`

## E. 推荐 `git add` 命令（核对后执行）

```bash
cd /root/projects/hrfont

git add \
  code/variants/cn2west_f123_rsi/FontDiffuser/configs/fontdiffuser.py \
  code/variants/cn2west_f123_rsi/FontDiffuser/train.py \
  scripts/launch_cn2west_f123.py \
  scripts/hrfont_support_adapter.py \
  scripts/eval_f03_test16_strat.py \
  scripts/build_f3b_support_bank_stroke.py \
  scripts/build_e12_cache_train228.py \
  scripts/build_e12_eye_probe.py \
  configs/e12_phi_s2_v51_train228_s3407.yaml \
  configs/e12_membership_v51_train228_s3407.yaml \
  reports/PI_BRIEFING_20260909.html \
  reports/METHOD_FIGURE_BRIEF_FOR_GPT.html \
  reports/PI_EXEC_DECISIONS_20260909.md \
  reports/AGENT_DECISIONS_F1_F3B_E12.md \
  reports/TRAINING_STATUS_F1_F3B_E12.md \
  reports/EVAL_INDEX_AUDIT_20260909.md \
  reports/COMPARE_PORTAL_DESIGN_20260909.md \
  reports/RESULTS_COMPARE_20260907.html \
  reports/e12_v51_summary.json \
  reports/artifacts_sync \
  reports/compare_portal/index.html \
  reports/compare_portal/manifest.json \
  reports/collab_offline/index.html \
  reports/e12_eye_probe_v4_s3407 \
  reports/f03_test16_strat/PROTOCOL.json \
  reports/f03_test16_strat/browse_index.json \
  reports/f03_test16_strat/metrics_items.json \
  reports/f03_test16_strat/metrics_summary.json \
  reports/f03_test16_strat/index.html \
  reports/f03_test16_strat/status.json \
  reports/f03_test16_strat/timeline.html \
  reports/f03_test16_strat/timeline_f2.html \
  reports/f03_test16_strat/preds/E1_100k \
  reports/f03_test16_strat/preds/F2_75000 \
  reports/GIT_SYNC_CHECKLIST_20260909.md

# 可选运维
# git add scripts/watchdog_f1_f3b_e12.py scripts/start_watchdog_f1_f3b_e12.sh
```

提交前再跑：`git status` / `git diff --cached --stat`，确认无 `.pth`、无 `runs/`。
