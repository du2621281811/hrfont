# Provenance 登记表

> 只登记 ID 与指针；详情在对应 JSON。新增实验前先查重，禁止复用已存在的 run/dataset ID。

## Datasets

| ID | Manifest | 状态 | 备注 |
|----|----------|------|------|
| `fontdiffuser42-cnstyle-v1` | `provenance/datasets/fontdiffuser42-cnstyle-v1.json` | frozen | 42 字体历史盘；勿覆盖 |
| `fontdiffuser_p253`（磁盘名） | **缺失** | legacy_untracked | summary 与磁盘 Style 计数不一致；不得直接当新基模数据 |
| `fontdiffuser-p253-…-cn2west-v2`（磁盘名） | `manifests/split_v3_228_16_16.json` | **active_transition** | 盘上已是 **228/16/16** + excluded；正式 ID 待发 **p260** |
| `fontdiffuser-p260-…-v2a-r1-*` | TBD | **planned** | 正式发布名；勿与旧 237/16/8 / p261 混用 |

## Code variants

| ID | Path | 相对 official | 状态 |
|----|------|---------------|------|
| `official` | `code/official/FontDiffuser/` | — | frozen 只读 |
| `ours-legacy` | `code/ours/FontDiffuser/` | `docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff` | frozen 历史恢复 |
| `cn2west_stage_a` | `code/variants/cn2west_stage_a/` | `docs/patches/cn2west_stage_a.diff` | legacy Stage-A（已 STOP） |
| `cn2west_f0_rsifree` | `code/variants/cn2west_f0_rsifree/` | StyleUpBlockNoRSI；P1 drop RSI/DCN | **active** F0 |
| `cn2west_f123_rsi` | `code/variants/cn2west_f123_rsi/` | StyleRSIUpBlockIdentitySafe（zero-init 1×1 residual conv）；`--arm` 三臂共用一条 code path | **active** F1/F2/F3 |

## Experiments / Runs

| ID | Provenance | 状态 | 结论入口 |
|----|------------|------|----------|
| `FT-CNSTYLE-25K` | `provenance/runs/FT-CNSTYLE-25K.json` | `retro_partial` | `reports/retrain_v2/FD_CNSTYLE_RESULTS.json` |
| `FT-P253-CNSTYLE-12K` | **缺失** | legacy | 同上；Style 池当时版本存疑 |
| `A-MVP-CONTROL` | `provenance/runs/A-MVP-CONTROL.json` | `retro_partial` | `PROJECT.md` Stage A |
| `A-MVP-DELTA` | `provenance/runs/A-MVP-DELTA.json` | `retro_partial` | `PROJECT.md` Stage A |
| `E1-FTV2-A-S3407` | `provenance/runs/E1-FTV2-A-S3407.json` | **completed** 100000/100k | `PROJECT.md`；看板 `reports/e1_ft_v2_dashboard/` |
| `E1C-FT-CONTINUE-S3407` | TBD | **superseded-before-launch** | joint 方案；见 `DESIGN_E2E3_FUSION_QKV_20260905.md` |
| `E2-STAGE-A-S3407` | `runs/.../STOP_PROVENANCE.json` | **stopped @10500** | `stopped_topology_superseded` |
| `E2B-FT-CONTINUE-S3407` | `runs/.../STOP_PROVENANCE.json` | **stopped @31400** | 同上 |
| `E12-PHI-S2-S3407` | TBD | **bootstrap done** | cache_v1 |
| `E12-IDCLS-S3407` | TBD | **bootstrap done** | cache_v1 |
| `E12-*-V2-S3407/08/09` | `reports/training_logs/e12_*_v2_*` | **gate_failed** | cache_v2；字型泄漏 + 同字型负样本，`reports/E12_SELFTEST_V2_REVIEW_20260905.md` |
| `E12-*-V3-S3407/08/09` | `reports/training_logs/e12_*_v3_*` | **gate_failed (T2)** | cache_v3；`reports/E12_SELFTEST_V3_REVIEW_20260906.md` |
| `F0-RSIFREE-FT-A-S3407` | `reports/training_logs/F0-RSIFREE-FT-A-S3407/` | **completed 100k** | milestone=100k val=0.031089；`reports/F0_MILESTONE_20260906.md` |
| `F1-OFFRSI-A-S3407` | `reports/training_logs/F1-OFFRSI-A-S3407/` | **paused ~300** | 对照；resume `stopped_step`；`DECISION_POINTS` D7 |
| `F2-DELTARSI-A-S3407` | `reports/training_logs/F2-DELTARSI-A-S3407/` | **paused 6573** | D9 串行；resume after F3 DONE |
| `F3-JOINT-DS-A-S3407` | `reports/training_logs/F3-JOINT-DS-A-S3407/` | **running 13500/80k exclusive** | GPU2 99%；val@10k=0.00217（trainer，无 perceptual）；D9 |
| `FT-P260-A-CN2WEST-V2` | alias / planned formal ID | 同 E1 数据协议 | 正式 p260 指纹待发 |
