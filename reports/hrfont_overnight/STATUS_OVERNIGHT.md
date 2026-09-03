# HR-Font 过夜运行状态

更新时间：2026-09-03T04:18:15+00:00

## 进度

| 任务 | 状态 | 说明 |
|------|------|------|
| Formal Stage A (GPU0) | step **80000** / 80000 | `runs/e2_stageA96_formal/` |
| Supervisor | **stage_b** | A 完成后 **自动 Stage B** |
| Stage B SupportAdapter | step **25000** | `runs/e2_stageB96/` |
| Dropout 消融 (GPU2/3) | **complete** | `dropout_ablation/VERDICT.md` |

## 回来后先看

```bash
cat reports/hrfont_overnight/STATUS_OVERNIGHT.md
cat reports/hrfont_overnight/SUPERVISOR.json
cat reports/hrfont_overnight/NEXT_STEPS_COMPLETE.md    # A+B 全完成
cat reports/hrfont_overnight/STAGE_B_PLAN.json           # Stage B 决策
cat reports/hrfont_overnight/dropout_ablation/VERDICT.md
```

## 日志

- `reports/hrfont_overnight/night_watch.log`
- `reports/hrfont_overnight/supervisor.log`
- `reports/hrfont_overnight/e2_stageA_formal.stdout`
- `reports/hrfont_overnight/dropout_ablation/orchestrator.log`

## 停止全部

```bash
touch reports/hrfont_overnight/STOP_ALL
touch reports/hrfont_overnight/STOP_FORMAL
```
