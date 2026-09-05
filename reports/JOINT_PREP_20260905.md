# Joint 主线准备清单（2026-09-05）

真源：[`DESIGN_E2E3_FUSION_QKV_20260905.md`](./DESIGN_E2E3_FUSION_QKV_20260905.md)（`6616836`）。  
台账：[`PROJECT.md`](../PROJECT.md)。

## 旧臂（已 STOP）

| Run | 停步（train_log） | 状态 |
|---|---:|---|
| `E2-STAGE-A-S3407` | 10500 | `stopped_topology_superseded` |
| `E2B-FT-CONTINUE-S3407` | 31400 | `stopped_topology_superseded` |
| `E1C-FT-CONTINUE-S3407` | — | `superseded-before-launch` |

产物：各 run 下 `STOP`、`stopped_step/`、`STOP_PROVENANCE.json`。目录不覆盖、不改写成新 arm。

## 新臂顺序

1. **F0 代码门（已落地）：** `code/variants/cn2west_f0_rsifree/` · `StyleUpBlockNoRSI` · P1 `strict=False` 丢 RSI/DCN（allowlist 174 keys）· `offset_coefficient=0`。identity-safe RSI residual 留给 F1 接入，不阻塞 F0。
2. **训 F0（running）：** `F0-RSIFREE-FT-A-S3407` · 协议 A 复用 · 8×1 · lr1e-5 · warmup5k · 100k · GPU2 · log `logs/f0/`。
3. **重建 Es/Ec cache**（绑定 F0 encoder SHA；禁止复用 E1 cache）— F0 完成后。
4. **并行开 F1 / F2 / F3**（先 seed 3407）：
   - F1：official `Ec(R[0])`，Support off
   - F2：Δ，Support off（与 F1 matched）
   - F3：Δ + Support joint drop=.20（与 F2 matched）
5. **QKV**：仅 20k/S3407 预注册筛选 role-swap；胜出才进主方法并重跑 matched。

## 并行：E12a

- 代码：`scripts/eval_framework/`；launcher `scripts/launch_e12_formal_v2.sh`
- 数据：`artifacts/e12/fonts_pool_v2`（**20** 外部 CJK+Latin 面；Noto 多字重 + arphic/wqy 等；非项目 FZ 池）
- cache：`artifacts/e12/cache_v2`（1200 glyphs，SHA `18c7db18…`）
- **formal running：** φ_s2 / ID-CLS × seeds 3407/08/09；`max_steps=5000`、bs32（小池下纯 50 epoch 仅 ~100 step，故改 step budget）
- 日志：`logs/e12/`
- **仍缺：** 更多独立族字体 → 再发 cache_v3；T1–T4 未跑前指标不进主结论

## 阻塞

- GPU0–1 僵尸显存仍重；F0@GPU2、E12@GPU3。
- F1–F3 代码尚未落地；identity-safe RSI 未实现。
- E12 族数仍偏少（Noto 字重不完全算独立族）。
