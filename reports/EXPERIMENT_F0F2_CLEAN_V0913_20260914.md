# F0 / F2 @ v0913_clean — matched 对照实验合同

日期：2026-09-14  
状态：**计划冻结；未授权前不覆盖任何旧 run。**  
目的：量化 **清洗协议包**（`v0913_clean`）相对脏协议 A 的影响，不是测 Group G / 大 batch。

## 0. 一句话

同一套代码与超参，只把数据合同换成 `v0913_clean`；F0 仍走到 **100k**，F2 仍从 **F0 best(=100k 规则)** 起步训 **80k**；主读数对齐历史看板的 **F0@100k / F2@40k**，并保留 @80k。

## 1. 脏臂基线（只读，禁止覆盖）

| 角色 | 路径 | 关键事实 |
|------|------|----------|
| F0 | `runs/F0-RSIFREE-FT-A-S3407` | 100k 完成；`best` → `global_step_100000` |
| Es/Ec | `artifacts/f0/es_spatial_f0` · `artifacts/f0/ec_multiscale_f0` | 均从 F0@100k 编码 |
| F2 | `runs/F2-DELTARSI-A-S3407` | parent=`F0/.../best`；80k 完成；best@80k |
| 看板常用 | `F0_100k` · `F2_40000` | 评测产物已在 `reports/f03_test16_strat/` |

脏臂资源量级（历史）：F0 ≈ **9h / 1×3090**；Es+Ec 数小时 + **~95G**；F2 纯算约 **1–1.5 天 / 1×3090**（日历更长因曾让卡）。

## 2. 干净臂 ID（新目录）

| 阶段 | run / 路径 | 说明 |
|------|------------|------|
| F0 | `F0-CLEAN-V0913-A-S3407` | 若目录已存在则改用 `F0-CLEAN-V0913-100k-A-S3407` |
| cache | `artifacts/f0_clean_v0913/{es_spatial,ec_multiscale}/` | **禁止**写回 `artifacts/f0/` |
| F2 | `F2-CLEAN-V0913-A-S3407` | parent = 干净 F0 的 best |
| 状态页 | `reports/f0f2_clean_v0913/STATUS.md` | 由脚本刷新 |
| 合同快照 | 本文件 | 变更须改日期 |

启动器已有保护：`runs/<id>` 存在则拒绝；禁止把 dirty id 用在 `v0913_clean` 上。

## 3. 锁死变量（两侧相同）

| 项 | 值 |
|----|----|
| F0 variant | `cn2west_f0_rsifree`，offset=0，P1 初始化 |
| F2 variant | `cn2west_f123_rsi`，`rsi_source=delta`，support=off |
| seed | 3407 |
| batch / accum | 8 / 1 |
| lr / warmup / schedule | 1e-5 / 5000 / linear |
| fp16 · 96px · CFG drop | on · 96 · 0.1 |
| F2 source_drop · offset | 0.25 · 0.5 |
| F0 步数 | **100000**（与脏臂同） |
| F2 parent | **该侧 F0 `best`**（脏侧=100k；净侧用同 val 规则选出） |
| F2 步数 | **80000** |
| 选 F0 best | val16 diffusion loss 最小；平局取更小 step（同 `scan_f0_val_loss.py` / `F0_MILESTONE`） |

## 4. 唯一允许不同的因素

`dataset_id=v0913_clean` 协议包：

- 去掉 5 套 exclude；按字体语种可用性过滤 `(font, char)`
- 训练采样 拉丁 50 / 假名 38 / 注音 12；`cn2cn_p=0`
- PNG 仍在旧协议 A 盘上，不重渲

**对外表述：**「清洗协议包 vs 脏协议 A」，不要写成单一「去脏字」。

**禁止混入：** Group G（bs256 / 10k / 大 lr）、脏 Es/Ec、用 F0@40k 开干净 F2 却与脏 F2（从 100k 起步）比单因子。

## 5. 阶段与命令（执行时照抄）

### P0 — 登记

```bash
mkdir -p reports/f0f2_clean_v0913
python3 scripts/status_f0f2_clean_v0913.py
```

### P1 — F0 smoke → 100k

```bash
python scripts/launch_cn2west_f0_rsifree.py --smoke --dataset_id v0913_clean --gpu 0
# smoke OK 后：
python scripts/launch_cn2west_f0_rsifree.py --yes \
  --dataset_id v0913_clean \
  --run_id F0-CLEAN-V0913-A-S3407 \
  --max_steps 100000 --ckpt_interval 5000 --warmup 5000 \
  --gpu 0
```

期间每 5k 有 `global_step_*`；训完跑：

```bash
python scripts/scan_f0_val_loss.py \
  --run runs/F0-CLEAN-V0913-A-S3407 \
  --out reports/training_logs/F0-CLEAN-V0913-A-S3407 \
  --device cuda:0
# 确认 best 指向选中里程碑（预期多为 100k，以扫描为准）
```

### P2 — 编干净 Es/Ec

从 **选中的 F0 best 目录** 编码到 `artifacts/f0_clean_v0913/`（勿覆盖 dirty cache）。  
完成后 `status` 脚本应能读到两侧 cache 的 `ckpt_dir`。

### P3 — F2 smoke → 80k

```bash
python scripts/launch_cn2west_f123.py --arm F2 --smoke \
  --dataset_id v0913_clean \
  --parent runs/F0-CLEAN-V0913-A-S3407/best \
  --es_cache artifacts/f0_clean_v0913/es_spatial \
  --ec_cache artifacts/f0_clean_v0913/ec_multiscale \
  --gpu 0

python scripts/launch_cn2west_f123.py --arm F2 --yes \
  --dataset_id v0913_clean \
  --run_id F2-CLEAN-V0913-A-S3407 \
  --parent runs/F0-CLEAN-V0913-A-S3407/best \
  --es_cache artifacts/f0_clean_v0913/es_spatial \
  --ec_cache artifacts/f0_clean_v0913/ec_multiscale \
  --max_steps 80000 --ckpt_interval 5000 --warmup 5000 \
  --gpu 0
```

### P4 — 评测（主表）

- 生成：干净 F0@100k、干净 F2@40000（及可选 @80000）
- 指标：只在 `pairs_eval47 keep=1`（704），按语种可用性汇总
- 对照列：脏 `F0_100k`、脏 `F2_40000`（已有则复用，不重训）

## 6. 中间记录（明早看收敛）

唯一入口：

```bash
python3 scripts/status_f0f2_clean_v0913.py
# → reports/f0f2_clean_v0913/STATUS.md
# → reports/f0f2_clean_v0913/STATUS.json
```

建议：`watch -n 300` 或 crontab 每 5–10 分钟跑一次；训练 log 本身也写 `train_log.jsonl` / `val_log.jsonl`（F2）与 F0 训练 log。

STATUS 至少包含：

1. 阶段（smoke / F0 / cache / F2 / eval）与是否在跑  
2. 最新 step、墙钟、最近 train loss  
3. 已有 milestone 的 **val_loss 曲线摘要**（净 vs 脏同 step）  
4. 当前 best（step + val）  
5. 与脏臂同 step 的 Δval（干净 − 脏；负=干净更好）  
6. 磁盘路径与「勿覆盖」清单  

**选模规则（预注册，不许看 test 后改）：**

- F0 best：val16 最小（扫描脚本）  
- F2 主报告点：**40000**（对齐历史看板）；附录 **80000** 与 run 内 best  
- 禁止用 test16 / 生成图挑 checkpoint  

## 7. 资源预算（本机 3090）

| 阶段 | 卡 | 时间粗估 | 盘 |
|------|----|----------|----|
| F0 100k | 1 | ~9 h | ~23G |
| Es+Ec | 1/CPU | 数小时 | ~95G 新目录 |
| F2 80k | 1 | 纯算 ~1–1.5 天 | ~19G |
| 合计 | 串行 1 卡 | ~2–3 GPU 日 | 勿写旧目录 |

E12-b 可并行另一张空卡，目录隔离。

## 8. 入口脚本（已实现）

```bash
python3 scripts/run_f0f2_clean_v0913.py check      # 静态正确性
python3 scripts/run_f0f2_clean_v0913.py status
python3 scripts/run_f0f2_clean_v0913.py smoke-f0 --gpu 0
python3 scripts/run_f0f2_clean_v0913.py train-f0 --gpu 0   # 100k
python3 scripts/run_f0f2_clean_v0913.py scan-f0 --gpu 0    # 定 best（默认全量 val16）
python3 scripts/run_f0f2_clean_v0913.py cache --gpu 0      # 新 Es/Ec，不删脏 cache
python3 scripts/run_f0f2_clean_v0913.py smoke-f2 --gpu 0
python3 scripts/run_f0f2_clean_v0913.py train-f2 --gpu 0   # 80k
```

相关实现：
- `scripts/rebuild_f0_clean_v0913_caches.py`（**禁止**用 `rebuild_g0_caches.py`，后者可能删脏 Ec）
- `scripts/status_f0f2_clean_v0913.py`
- `launch_cn2west_f123.py`：clean 臂拒脏 F0 parent / 脏 Es·Ec
- `scan_f0_val_loss.py`：可选 `--v0913_clean_map`；默认仍用**全量脏 val16**以便与历史 F0 选模同表面

## 9. 正确性审计（实现检查结果）

| 项 | 结论 |
|----|------|
| clean 训练集 | 56429 pair / 223 字体；权重和≈1；exclude 不在 donor |
| dirty 训练集 | 67260 pair / 228 字体（对照基线） |
| F2 parent 历史 | 脏 F2 ← F0@**100k**；Es/Ec ← 同里程碑 |
| launcher 默认 F2 steps | **40k**；本实验 orchestrator 强制 **80k** |
| 选 F0 best 的 val | 默认**全量 val**（与脏臂扫描一致）。若改用 clean val，不可与脏 F0_MILESTONE 数值直接比 |
| 脏 cache 串味 | launch 在 `v0913_clean` 下拒绝 `artifacts/f0/es_spatial_f0` 与 `ec_multiscale_f0` |
| `rebuild_g0_caches.py` | 磁盘紧时会删脏 Ec → **本实验禁用** |
| Group G | bs/lr/steps 不同 → 不作本对照 |
| 评测 | 主表 keep=704 + 语种可用性（生成脚本需显式过滤；勿默认 752） |
| **评测脚本 Es/Ec** | `eval_f03_test16_strat.py` **仍写死** `artifacts/f0/es_spatial_f0` 与 `ec_multiscale_f0`。干净 F2 出图前必须加 method 指向 `artifacts/f0_clean_v0913/*` 与 clean donor，否则推理 Δ 与训练栈不一致 |

## 10. Go / No-go

- **Go：** `check` 通过；smoke 过；STATUS 能刷新；F0/F2 新 id；cache 新路径；评测 704。  
- **No-go：** 覆盖脏 run；F2 接脏 Es/Ec 或脏 F0；F0 只训 40k 却声称与脏 F2 单因子对比；用 G0/G2 数字冒充本实验；用 `rebuild_g0_caches.py` 编本实验 cache。
