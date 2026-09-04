# Stage A 执行计划（2026-09-05）

口径真源：[`PI_DECISIONS_20260905.md`](./PI_DECISIONS_20260905.md)、执行规格 §1.6。  
本计划把拍板落成可跑步骤。当前代码仍是 1-token / 在线 Ec / soft-ε；**未过 G0–G3 不得开 80k**。

## 0. 现状（开跑前快照）

| 项 | 状态 |
|---|---|
| E1@100k 权重 | 有：`runs/E1-FTV2-A-S3407/global_step_100000/{unet,style_encoder,content_encoder}.pth` |
| Es/Ec cache | **无** `artifacts/e2/` |
| YAML 口径 | 已冻：E1c / E2 / E2b，top-10，9-token，cache-only |
| Stage-A 代码 | 未对齐：live Es pooled、live Ec、未传 `k_top`、无 val/推理、checkpoint 不存 RNG |
| E1c 训练器 | `cn2west_ft_v2` **无 freeze / 无 cache lookup** |
| 盘 | `/root` 余约 4.4 T，cache 体积可承受 |
| GPU | 四卡均有残留显存、利用率 0%；优先用更空的卡，不杀进程除非 PI 点头 |

体积（fp16，约）：Es spatial 260×338 ≈ 1.6 GiB；Ec target 228×295 ≈ 41 GiB；Ec style 260×338 ≈ 54 GiB；Content×295 可忽略。合计盘上 ~100 GiB，mmap，禁止整表进 GPU。

## 1. 原则

- 一次只改白名单变量；E2/E2b 必须 matched（同 seed、同 R 序列、同 CFG/Δ draw）。
- 编码器只在建 cache 时跑；训练/val/sample **零次** Es/Ec forward。
- E1c 用原 E1 前向（1-shot 9-token + `Ec(style)`），不要塞进 n-shot 均值逻辑冒充。
- 实现建议：**三臂共用 Stage-A `train.py` 的 cache 加载与 freeze**；E1c 用 `nshot=1/1` + `rsi_source=official` + `style_condition=spatial_3x3`（n=1 时均值即自身）。YAML 里 `variant: cn2west_ft_v2` 改为执行时统一 `cn2west_stage_a`（协议仍是 E1 前向，不改科学定义）。
- 先 seed 3407；3408/3409 等 3407 过门再补。

## 2. 阶段与门

```text
G0 盘点/预检
 ├─ 并行 A：Es/Ec cache 构建（GPU）
 ├─ 并行 B：P1 vs E1 encoder 验证 val-only（GPU）
 └─ 并行 C：接线（CPU）9-token + top-10 + cache lookup + val + resume + E1c
      ↓
G1 cache SHA + 覆盖率
G2 代码 smoke（含 cache-only 断言）
G3 短步 matched（E2/E2b 前 N batch checksum）
      ↓
G4 三臂 80k：E1c ‖ E2 ‖ E2b（三张卡）
      ↓
G5 80k val + best≥10k + provenance
```

E12a、E1 正式评测、p260 SHA 全程可与 G0–G3 并行，**不进** matched 训练卡。

---

### G0 — 预检（约 0.5h，CPU）

1. 确认 E1@100k 三件套 SHA，写入 provenance 草稿。  
2. `split_v3` 228/16/16 与盘上 stem 一致；ContentImage 295 张齐。  
3. 建 `artifacts/e2/`；记录 GPU 占用 PID（只记录，默认不 kill）。  
4. 出口：检查清单打勾。

### 并行 A — 离线 cache（约数小时，1–2 卡）

新脚本（现有 `hrfont_es_cache.py` 只存 pooled，不够）：

| 产物 | 键 | 内容 | 脚本 |
|---|---|---|---|
| `artifacts/e2/es_spatial_e1_100k.pt` | `(split,font,style_cp)` 260×338 | `style_emd` `[1024,3,3]` fp16 + 可由其平均的 pooled | 扩写 `scripts/hrfont_es_cache.py` |
| `artifacts/e2/ec_multiscale_e1_100k/` | mmap 分片：target 228×295；style 260×338；content 295 | residuals+final，fp16 | **新建** `scripts/hrfont_ec_cache.py` |
| `*.manifest.json` | encoder SHA、split SHA、shape、dtype、条目数 | 启动 fail-closed | 两脚本都写 |

约束：

- `--es-ckpt` / `--ec-ckpt` = `global_step_100000`；manifest SHA 必须等于该 pth。  
- 覆盖率：style 338、target 295、缺键即失败。  
- 先 dummy 小集，再全量。Es 与 Ec 可两卡并行。  
- **G1 出口：** SHA 对齐；抽 16 个 (font,char) 与在线编码器 `allclose`（仅此一次在线，记入报告后冻结）。

### 并行 B — encoder 验证（1 卡，不挡接线）

`scripts/hrfont_validate_e1_encoders.py`：E1@100k，**gate 只看 val16**。  
修：V5 邻居走 TargetImage；CLI gate FAIL 非零退出。  
P1 对照同协议另跑一份，只报告不选参。  
**不作为 G4 硬门**，但 V3/V5 崩了要停下来改 cache/α，不能闷头训。

### 并行 C — 接线（CPU，可与 A/B 同时写，合入须等 G1）

**C1 α**（`scripts/hrfont_delta_v2.py` + `train.py`）

- 默认 `mode=topk`，`k_top=10`，`tau=0.07`。  
- launcher 必须传 `--delta_k_top`。  
- top-K 永远返回 `min(K, n_valid)`；`n_valid=0` 才 fail-closed（cache/split 坏）。  
- 删掉「ε 截空 → 重定 ε」路径。

**C2 style 9-token**（`train.py` `_style_conditions` + `model.py`）

- 从 Es spatial cache 取 n 张 `[1024,3,3]`，逐元素平均，permute 成 `[B,9,1024]`。  
- 禁止 pooled 1-token。  
- E1c：n=1，不平均（或平均 1 张，等价）。

**C3 结构源 cache-only**（`_structure_features`）

- E2：α 用 Es pooled cache；邻居/Content 用 Ec mmap；`feature_delta` 只做加权减，**不调用** `content_encoder`。  
- E2b：`Ec(R[0])` 查 Ec style cache。  
- E1c：`Ec(同一张 style)` 查 Ec style cache。  
- MCA Identity：查 Ec content cache；CFG 置零走 mask，不重新编码。  
- 断言：一步之内 `style_encoder`/`content_encoder` 调用次数 = 0。

**C4 val / sample**

- 固定 eval ref8（E1c 除外：1-shot 需预注册一张或沿 E1 规则，写入 val manifest）。  
- DPM `cond` 带 9-token style + structure（Δ 或 Ec(R[0])）。  
- 去掉 `sample.py` Resize。  
- CFG：uncond 清 style/content，**保留 structure**；Δ-drop 独立。  
- 每 5k 全量 val16；best 仅 step≥10000 的 5k milestone。

**C5 exact resume**

- 存 Python/NumPy/Torch CPU+CUDA RNG、AMP scaler、DataLoader generator、sampler cursor。  
- E2/E2b 前 N=64 batch：font/char/R/timestep/noise seed/CFG mask/Δ mask checksum 必须相同。

**C6 launcher**

- `launch_e2.py`：校验 Es+Ec cache、两份 SHA、`k_top`、`encoder_runtime=cache_only`。  
- 新建 `launch_e1c.py` 或同一入口 `--config configs/e1c_ft_continue_s3407.yaml`。  
- 输出目录存在则拒绝。

**G2 出口（smoke，必须全绿）：**

1. n-shot 采样可复现；n∈[1,8]。  
2. top-10：人工 227 维分数 → 永远 10 个权、和为 1。  
3. 9-token shape `[B,9,1024]`；n=1 与 n=8 长度相同。  
4. cache-only：hook 编码器 forward 计数 = 0。  
5. E2 structure 是 Δ 不是 `Ec(style)`；E2b 是 `Ec(R[0])`；E1c 是 `Ec(style)`。  
6. freeze：backward 后 Es/Ec grad 为 None。  
7. resume：中断后 batch id / mask 一致。  
8. E2 vs E2b 前 64 draws 一致。  
9. val 一步能出图且无 Resize。

**G3 出口：** 各臂 20 step 真数据不 NaN；E2/E2b 前 64 checksum 落盘。

---

### G4 — 满训（三卡并行）

| GPU | 实验 | 入口 |
|---|---|---|
| 空闲卡 A | `E1C-FT-CONTINUE-S3407` | `configs/e1c_ft_continue_s3407.yaml` |
| 空闲卡 B | `E2-STAGE-A-S3407` | `configs/e2_stage_a_s3407.yaml` |
| 空闲卡 C | `E2B-FT-CONTINUE-S3407` | `configs/e2b_ft_continue_s3407.yaml` |

启动前：`--yes` 仅在 G2/G3 绿、cache SHA 打印进 `run_note.txt` 之后。  
中途 STOP 必须能 exact resume；一侧 resume 后不得再称 matched，除非 checksum 续上。  
不要开 E2c/E2d/E5。不要第四张卡插匹配臂。

### G5 — 收尾

- 80k full + val16 + `DONE.json`。  
- `best.json`：≥10k 的 5k milestone，公式预注册，三臂同一规则。  
- provenance：git SHA、config SHA、dataset/split SHA、Es/Ec cache SHA、init SHA。  
- 未完成 E12 T1–T4 前，主结论只用诊断指标（val loss / 覆盖 / LPIPS），不把 φ_s2 写进论文表。

## 3. 建议日历（墙钟，可重叠）

| 日 | 工作 |
|---|---|
| D0 | G0 + 开写 C1–C3 + 启动 Es cache |
| D0–D1 | Ec cache（最长）；encoder 验证；写完 C4–C6 |
| D1 | G1 allclose；合入 cache 路径；G2 smoke |
| D1 晚 | G3 20-step；三臂启动 G4 |
| D1–D4 | 80k（约按 E1 速度外推）；并行 E12a / E1 评测 / p260 SHA |
| 80k 后 | G5 |

## 4. 明确不做

- 不改 RSI 内部 Q/K/V 或加 gate。  
- 不建 P1 cache（除非日后真开 E2c）。  
- 不把 val/test 字体写入 Δ 库。  
- 不用 98k best 初始化。  
- 不在观察曲线后改 best 规则或 K。  
- 不 kill 占卡进程，除非 PI 书面同意。

## 5. 合作者审核时只需确认

1. E1c 训练器并入 Stage-A（n=1 + official RSI）是否可接受（科学上仍是 E1 前向）。  
2. Ec style 全量 260×338 是否同意（~54 GiB）；若要缩，只能改为「按需编码一次再写入」，与 cache-only 冲突。  
3. G4 三卡并行是否同意（E2/E2b 必须同日同提交启动，保证 matched 代码 SHA）。
