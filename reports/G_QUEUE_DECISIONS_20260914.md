# Group G 队列决策点（执行机，供事后 review）

日期：2026-09-14。本文件只记执行机自主决策与依据；不替代 `PROJECT.md` / `G_STYLE_COMPLETION_PLAN_20260914.md`。

## PI / 用户指令（2026-09-14）

1. G2 跑完后顺序：**G2-RL → G-RL-pilot → G1 → G0 续训**；卡不要空着。
2. 挂 watchdog：**每 10 分钟**拉取最新 git，直到相关代码就绪，并校验可执行、符合设计与计划。
3. **不得中断**当前训练；问题尽量自解；决策点写入本文件。

## 决策记录

### D1 · 队列重排（2026-09-14）

- **决定：** `watchdog_g.py` 阶段改为  
  `G0b(done) → caches(done) → G2 → G2-RL → G-RL-pilot → G1 → G0c`。  
  `G2-PRL` 仍 cancelled。
- **依据：** 用户明确顺序；与 TC-v2 计划「先 RL 再 G1」一致，且保留 G1 对照。
- **未做：** 未杀 G2 进程；仅重启调度 watchdog。

### D2 · G-RL-pilot 代码未到时的占卡策略

- **决定：** 若 G2-RL 已完成但 `artifacts/g_queue/G_RL_PILOT_READY.json` 仍不存在，**不空等**；先跑 **G1**，pilot 标记为 `deferred`；G1 结束后若 READY 则插入 pilot，再跑 G0c；若仍未 READY 则先跑 G0c，pilot 留在队尾待代码。
- **依据：** 「卡不要空着」优先于严格空等；顺序偏差会记入 incidents / 本文件。
- **风险：** 实际顺序可能变成 G2-RL → G1 → pilot → G0c；属有意折中。

### D3 · G0 续训 run_id

- **决定：** 新目录 `G0c-F0-V0913-BS256-A-S3407`，从 `G0b.../global_step_10000` **weight+trainer resume**，`max_train_steps=20000`（再训到 20k），lr 仍 1e-5 + `constant_with_warmup`/500。不覆盖 G0b / 失败 G0 / F0-CLEAN。
- **依据：** 「续训」= 同配方继续；新 id 防覆盖。

### D4 · Git 拉取范围

- **决定：** 每 10 分钟 `git fetch`；仅对 **allowlist 路径** `checkout origin/main -- <path>`（pilot/TC/warm-start/计划文档相关）。不做整树 `git pull`/reset，避免搅乱本地训练改动与运行中作业。
- **依据：** 本地有大量未提交训练脚本/状态；全量 merge 风险高。

### D5 · G-RL-pilot READY 门槛

- **决定：** 存在可执行入口且通过校验才写 READY：  
  - 优先 `scripts/launch_g_rl_pilot.py`（`--help` / `py_compile` 通过）；或  
  - `artifacts/g_queue/pilot_launch.json`（含 argv，且指向存在的脚本）。  
  并抽样核对：从 **G2** warm-start、臂为 **F2RL / local**、步数预算落在 **2k–5k** 量级（与计划 §3/§4 一致）。不通过则不写 READY，记录原因，继续拉取。
- **依据：** 计划写明需 weight-only warm-start；未经验证不得开训。

### D6 · 当前执行机状态快照（部署时）

- G2-F2 ~3.8k/10k running；G0b@10k done；cache READY。
- Git HEAD 落后 origin；计划文档已部分同步；训练代码稍后由远端上传。

## 后续自动追加

调度 / git watchdog 会在 `reports/watchdog_g/decisions.jsonl` 追加机器可读条目；重要人工可读结论同步补到本节下方。

### D7 · 不自动覆盖状态文档

- **决定：** git watchdog allowlist **不再** checkout `PROJECT.md` / `docs/EXPERIMENTS.md` / provenance（曾把执行机状态冲回远端旧表）。
- **依据：** 执行机进度以本机台账为准；计划/脚本仍可拉取。

### D8 · 磁盘上同步 warm-start / TC 代码（不杀 G2）

- **决定：** 在 G2 仍跑时，仅 `git checkout origin/main --` allowlist 内的  
  `train.py` / `configs/fontdiffuser.py` / `src/tc_v2.py` 与 TC 脚本；**不**重启 G2。
- **依据：** 运行中进程已把旧模块载入内存；磁盘更新只影响后续作业（G2-RL / pilot）。
- **风险：** 若 G2 崩溃并由 watchdog 用新 `train.py` 重拉，应仍兼容原 F2 argv（已核对 warm_start 为可选）。

### D9 · 本地补齐 `launch_g_rl_pilot.py` 并写 READY

- **决定：** 远端尚无独立 launch 脚本时，按 handoff §4 在本机写  
  `scripts/launch_g_rl_pilot.py`：`--arm F2RL --warm_start_from <G2 ckpt> --no-tc_enabled --local_learning_rate 1e-4`，`max_steps=5000`，`constant_with_warmup`/200，单卡（默认 GPU0），run_id `G-RL-pilot-V0913-A-S3407`。校验通过后写 `artifacts/g_queue/G_RL_PILOT_READY.json`。
- **依据：** 计划 stage 1a；用户队列要求 G2-RL 后接 pilot。
- **占卡折中：** pilot 按计划单卡；短窗口内其余卡可能空闲，优先配方保真而非强行 8 卡改实验。

### D10 · Git allowlist 扩到 f123 train / TC

- **决定：** `watchdog_g_git_pull.py` allowlist 增加 TC 脚本前缀与 `train.py`/`fontdiffuser.py`/`tc_v2.py` 精确路径；READY 校验额外要求 `train.py` 含 `warm_start_from`+`local_learning_rate`，且 launch 关闭 TC。
- **依据：** 仅拉计划文档不够，pilot 需要可执行训练入口。

### D11 · 不完整 TC checkout 导致 G2-RL 启动失败（已自愈）

- **现象：** G2@10k 完成后 watchdog 连开 G2-RL，但先是 `ImportError: TCV2Cache`，修好后又 `TypeError: forward() got unexpected keyword argument 'tc_global_residual'`，空转 relaunch。
- **原因：** 只部分同步了 TC 训练入口，`src/__init__.py` / `src/model.py` 未齐套。
- **处置：** 从 `origin/main` 补齐 `__init__.py` + `model.py`；allowlist 同步扩大；**未杀**调度 watchdog，等其自动重拉。
- **教训：** 今后 TC/warm-start 相关 checkout 必须成套（train + configs + src/{__init__,model,tc_v2}），再写 READY / 开训。
