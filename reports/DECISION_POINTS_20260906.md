# 决策点记录（2026-09-06，F1/F2 已开跑之后）

既定方案：F0 100k → 预注册 val 选 milestone → 从该 ckpt 重建 Es/Ec → F1/F2/F3 单 seed 3407、80k、identity-safe RSI、F1 vs F2 只换 source。本文件只记**我做的、可能被当成改方案的选择**；不开新实验、不改损失、不改数据。

## 正在跑的事实（确认过，未改配置）

| | F1 | F2 |
|---|---|---|
| 进程 | 2712976 GPU2 | 2712975 GPU3 |
| 到 step 200 | loss 0.25936 | loss 0.25888 |
| `rsi_gain` @200 | 3.31e-6 / 3.46e-6 | 3.31e-6 / 3.38e-6 |
| parent | `F0-RSIFREE-FT-A-S3407/best` → `global_step_100000` | 同 |
| cache | `artifacts/f0/es_spatial_f0` + `ec_multiscale_f0`（SHA 对齐 F0 encoder） | 同 |
| 首 batch | 字体/字/source_draw/cfg **完全相同**（matched RNG 成立） | 同 |
| parity | `max_abs_diff=0` | 同 |

无 Traceback、无 non-finite、无 OOM killer。F1 一度 `D` 在 `wait_on_page_bit`（94G Ec mmap 冷启动），随后恢复 `R`，step 继续涨。

## 决策

### D1 — 用 identity-safe RSI，不用远端「官方 RSI + offset 零初始化」

- **方案依据**：本会话 PI 要求「效果好且方便归因」；parity 证明 step0 与 F0 逐元素相等。
- **冲突**：`origin/main` `5a92f5c` 写过「identity-safe 作废」。
- **我做了什么**：F1/F2 只从 `cn2west_f123_rsi` 启动。另一套代码仍在仓库，未开跑。
- **影响**：主表 F1/F2 的 RSI 头是 zero-conv residual，不是官方 DCN 的零 offset。若事后要切回官方头，这两条 80k 不能当 matched 对照，必须重跑。

### D2 — F0 选 100k，不启用 YAML 里的 60k early-stop

- **方案依据**：口头「都跑完，后面再挑步数」+ 预注册规则（5k 网格、val16 loss 最小）。
- **结果**：100k = 0.031089，全程近似单调，选中点就是终点。
- **影响**：F1/F2 的 parent 是 100k 不是 60k。E1@100k 是 0.029787，F0 高 0.00130 是去 RSI 的代价，保留。

### D3 — 不把 cosine 表 / 批量 α 加速打进正在跑的 F1/F2

- **现象**：F2 约 4–6 s/step，GPU util 经常接近 0，CPU 1600%+，Ec 随机 mmap。另一条线的 `SPEEDUP_IMPL` 声称约 3×。
- **为什么不动**：加速会改 `train.py` 的取数路径，必须停跑重开。既定 F123 配方没有锁这一项；step 200 时重开会丢掉 matched 的前 200 步 RNG 轨迹（可忽略），但会把「是否与当前代码逐行一致」变成一笔新账。
- **影响**：F2 墙钟大约 3–4.5 天/80k，F1 大约 1.5–2.5 天。不改科学结论，只影响什么时候能评测。

### D4 — F1 与 F2 继续并行，不改成串行

- **现象**：Ec cache 94G，机器 RAM 125G，swap 8G 已经用满；两进程 VSZ 各约 130G（mmap）。
- **为什么不动**：同一文件的 mmap 共享 page cache，不是两份 94G 常驻。step 已过 200，串行化要 STOP 其中一条，属于操作改动且拖后 matched 对照的完成时间。
- **影响**：前几 k step 可能偏慢（页缓存变热）；若之后 OOM killer 杀掉进程，用 `last_state`（每 1000 step）resume。**1000 step 之前没有 last_state**，那时被杀只能从 0 重开。

### D5 — F3 不开

- **方案依据**：F3 需要 support bank；`data/hrfont/e0_bank` 不存在；launcher fail closed。
- **影响**：本轮没有 F2 vs F3 的 Support 对照。F1 vs F2 的 Δ 对照不受影响。

### D6 — warmup=5000，跟 F0/E1，不跟旧 E2 的 2000

- **方案依据**：F1/F2 从 F0 续训，FT 配方与 F0 对齐（lr 1e-5、linear、warmup 5000、eff batch 8）。
- **影响**：与旧 Stage-A YAML（warmup 2000）数字不同；与 E1/F0 相同。不是新发明，是跟主线 FT 而不是跟已 STOP 的 E2。

### D7 — 暂停 F1，把 GPU2 给 F3（本轮，按「先可用结果」）

- **方案依据**：用户 2026-09-06 12:31：「确保 F3/F1/F2 开始训练，目标是快速拿到可用的结果，消融可以慢一些。」F3 是联合系统，F2 是 Δ 方法，F1 是官方 RSI 对照（消融）。
- **我做了什么**：
  1. 按 F123 已写死的约定建 `artifacts/f0/support_bank.json`：每个 target 字 → 8 个**同字体** style-pool 字符（优先「永和书风骨韵天地」），查 `ec|style|{font}|{cp}`。**不是 E0 跨字检索**，因为 e0_bank 不存在，且现有 `train.py` 只吃这张表。
  2. `STOP` F1（当时 ~300 step，权重在 `stopped_step/`），GPU2 转给 F3 80k。
  3. F2 不停。
- **影响**：F1 消融延后，可用 `stopped_step` resume，不丢已有步。F3 与 F2 并行，才能做 Support 的 matched 对照。Support 证据是「同字体多字」，不是跨字 glyph bank；若以后要 E0 检索，F3 这条不能直接当 E0 对照。


- 没改损失、batch、seed、source_drop、数据、split。
- 没把 F1/F2 切到 `cn2west_stage_a` 的官方 RSI 头。
- 没为了加速停跑。
- 没动 GPU0/1 上的僵尸显存（不是本任务进程；杀了可能影响别人）。

### D8 — 上 F123 监控看板 + F2/F3 崩溃自动续跑（2026-09-06 14:28）

- **触发**：用户要求确认不会因网络/硬件/代码 bug 拿不到结果，并要在线看板。
- **我做了什么**：`scripts/f123_monitor.py` 听 `:8787`，每 10s 备份 `train_log.jsonl`；F2/F3 进程消失则从 `last_state` 续跑（`--resume_from`，RNG 一并恢复）。F1 有 STOP，不自动续。试图加 16G swap，容器 overlay `swapon` 失败，未改训练进程。
- **明确没做**：`nvidia-smi -r` 清 GPU0/1 僵尸（可能带崩正在跑的 GPU2/3）。
- **影响**：最多丢未写入 last_state 的 <1000 step；不改配方。看板不依赖外网 CDN。
