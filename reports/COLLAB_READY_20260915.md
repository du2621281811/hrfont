# 合作者明日可用检查 · 2026-09-15

检查时间：本机已与 `origin/main`（`33684b879`）对齐；tracked 工作区干净。

## 已通过

| 项 | 结果 |
|----|------|
| git `main` ↔ `origin/main` | 同步（ahead/behind = 0） |
| E12-b 配置/脚本/STATUS/handoff | 关键路径均在 origin |
| Paper6 OOD 6 套说明 + 看板警告 | 在 origin |
| F0C / F2C preds | 已跟踪 |
| E12-b Release | https://github.com/du2621281811/hrfont/releases/tag/e12-b-v0913-20260914 |
| Release 内 `best.pt` SHA256 | 与本机 **MATCH** |

## 合作者最小步骤（E12-b 打分）

```bash
git pull
gh release download e12-b-v0913-20260914 -p e12-b-v0913-best-20260914.tar.gz
mkdir -p runs && tar -xzf e12-b-v0913-best-20260914.tar.gz -C runs/
# 核 sha：见 reports/e12_b/WEIGHTS_HANDOFF.md
python scripts/score_preds_e12_b.py --device cuda:0
```

## 故意未进 git（勿当漏交）

- `runs/**` 全量权重 / 中间 `step_*.pt`
- `artifacts/e12/cache_*`（打分可不下；重训再重建）
- `artifacts/f0_clean_v0913`、F0/F2-CLEAN 全量 ckpt（仅重生成需要）
- `logs/*.pid` / `*.stdout`

## Paper6 提醒

6 套补充测试集：**未查中西文风格一致性** → 可肉眼，指标须再确认。  
见 `reports/PAPER6_OOD_SUPPLEMENT_20260914.md`。
