# 今晚执行清单（2026-09-14）

**报告总入口**：http://127.0.0.1:8780/hub/ · 目录源 `reports/REPORT_CATALOG.json` · 约定见 `reports/REPORT_MANAGEMENT.md`

主线（GPU0，串行）：
1. F0-CLEAN smoke → 100k 训  
2. scan best → 编 Es/Ec → F2-CLEAN smoke → 80k（F0 完成后）

并行（GPU1 或 CPU）：
3. E12-b 建 cache → 训 φ → membership（不挡主线）

看板：
- `reports/f0f2_clean_v0913/STATUS.md`
- `reports/e12_b/STATUS.md`
- **通宵监督入口**：`reports/watchdog_tonight_20260914/MORNING.md`
- 事故日志：`reports/watchdog_tonight_20260914/incidents.jsonl`
- 监督器状态：`reports/watchdog_tonight_20260914/status.json`

不做：改 eval_f03（等 F2-CLEAN 出图前）；不动脏臂；不用 rebuild_g0。

## 自动监督（通宵）
# 明早看板 · F0/F2-CLEAN + E12-b
- 更新：`2026-09-14T07:24:19.084718+00:00`
- 监督器：`reports/watchdog_tonight_20260914/`（`status.json` / `incidents.jsonl`）

## 阶段快照
- **F0**: `{"job": "F0", "state": "done", "step": 100000}`
- **scan**: `{"job": "scan", "state": "done"}`
- **cache**: `{"job": "cache", "state": "done"}`
- **smoke_f2**: `{"job": "smoke_f2", "state": "done"}`
- **F2**: `{"job": "F2", "state": "done", "step": 80000}`
- **E12_membership**: `{"job": "E12_membership", "state": "done"}`

## 入口文件
- `reports/f0f2_clean_v0913/STATUS.md`
- `reports/e12_b/STATUS.md`
- F0 log: `reports/f0f2_clean_v0913/logs/train_f0.console.log`
- F2 log: `reports/f0f2_clean_v0913/logs/train_f2.console.log`
- membership: `reports/e12_b/membership.console.log`
- 事故：`reports/watchdog_tonight_20260914/incidents.jsonl`

## 完成判据
- F0: `runs/F0-CLEAN-V0913-A-S3407/DONE.json` + `best/`
- Cache: `artifacts/f0_clean_v0913/{es_spatial,ec_multiscale}/manifest.json`
- F2: `runs/F2-CLEAN-V0913-A-S3407/DONE.json`（80k）
- E12-b: `runs/e12_membership_b_s3407/{best.pt,val_metrics.json,test_metrics.json}`

## 注意
- 脏臂未动；eval_f03 仍硬编码脏 Es/Ec（干净对比图等 F2 后再接）
- 网络/会话断开不影响本监督器（setsid）

