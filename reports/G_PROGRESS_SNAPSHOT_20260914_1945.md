# G 进度快照 · 2026-09-14 19:45 +08

## 训练队列（matched continue）

- 脚本：`scripts/queue_g_matched_continue_20260914.py --execute`
- **P0a `G-CONT-G2-8gpu-V0914-A-S3407`：DONE**  
  - parent `G2-F2-V0913-A-S3407/global_step_10000` · +5k · TC off · 8×V100  
  - `DONE.json`: completed · **best@2500** · val≈0.002079
- **P0b `G-CONT-G2RL-8gpu-V0914-A-S3407`：RUNNING**  
  - parent `G2-RL-V0913-A-S3407/global_step_10000` · +5k · TC off  
  - heartbeat ~**step 1400/5000** · loss≈0.013（19:45）

状态文件：`reports/g_matched_continue_20260914/status.json`

## 评测 / 看板（old Ref8 k=1/2/4/8）

- 推理：`reports/g_v0913_shot_k1248/` · `DONE.json` rc=0 · 30 臂齐  
- Ref8=`永和书风骨韵天地` · DPM++20 CFG7.5 seed3407  
- 看板：`index.html` + `metrics.json`（test L1/SSIM 诊断）  
- Hub：`http://127.0.0.1:19000/g_shot_k1248/`（`scripts/serve_v100_hub.py`）

## Git 本提交范围

- 评测/看板脚本与 hub 路由  
- k1248 preds + 看板 HTML/metrics/PROTOCOL  
- 本进度快照与 K1248 计划稿  

**未进仓**：`runs/` 权重（gitignore）；G-CONT-G2RL 训练仍在进行。
