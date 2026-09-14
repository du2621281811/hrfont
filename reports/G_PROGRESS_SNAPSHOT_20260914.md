# G 系列进度快照 — 2026-09-14

- 生成/更新：2026-09-14T15:49:51.944320+08:00
- 机器：V100 Docker / hrfont

## 8 卡队列

| 阶段 | 状态 |
|------|------|
| TC cache + H head | DONE |
| G-RL-pilot-8gpu | DONE 5k · best@2500 |
| G-TC-G2-8gpu | DONE 5k · best@2500 |
| G-TC-G2RL-8gpu | RUNNING（以 runs/.../heartbeat.json 为准） |

主臂 G0b/G0c/G1/G2/G2-RL 已完成。

## 评测计划（已确认 Ref）

见 [`G_MULTI_SHOT_EVAL_PLAN_20260914.md`](G_MULTI_SHOT_EVAL_PLAN_20260914.md) 与 [`g_v0913_shot/ref8_proposal/`](g_v0913_shot/ref8_proposal/)。

## 看板

- 现板（旧 Ref）：`reports/g_v0913_shot/`
- 新 Ref 确认后将按多 split / 1·2·4·8 + batch 重建。
