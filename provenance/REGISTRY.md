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
| `cn2west_stage_a` | `code/variants/cn2west_stage_a/` | `docs/patches/cn2west_stage_a.diff` | **active**（E2/E2b；代码尚未对齐 2026-09-05 口径） |

## Experiments / Runs

| ID | Provenance | 状态 | 结论入口 |
|----|------------|------|----------|
| `FT-CNSTYLE-25K` | `provenance/runs/FT-CNSTYLE-25K.json` | `retro_partial` | `reports/retrain_v2/FD_CNSTYLE_RESULTS.json` |
| `FT-P253-CNSTYLE-12K` | **缺失** | legacy | 同上；Style 池当时版本存疑 |
| `A-MVP-CONTROL` | `provenance/runs/A-MVP-CONTROL.json` | `retro_partial` | `PROJECT.md` Stage A |
| `A-MVP-DELTA` | `provenance/runs/A-MVP-DELTA.json` | `retro_partial` | `PROJECT.md` Stage A |
| `E1-FTV2-A-S3407` | `provenance/runs/E1-FTV2-A-S3407.json` | **completed** 100000/100k | `PROJECT.md`；看板 `reports/e1_ft_v2_dashboard/` |
| `E1C-FT-CONTINUE-S3407` | TBD | **planned** | `reports/PI_DECISIONS_20260905.md`；`configs/e1c_ft_continue_s3407.yaml` |
| `E2-STAGE-A-S3407` | TBD | **planned** | `configs/e2_stage_a_s3407.yaml` |
| `E2B-FT-CONTINUE-S3407` | TBD | **planned** | `configs/e2b_ft_continue_s3407.yaml` |
| `FT-P260-A-CN2WEST-V2` | alias / planned formal ID | 同 E1 数据协议 | 正式 p260 指纹待发 |
