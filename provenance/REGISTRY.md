# Provenance 登记表

> 只登记 ID 与指针；详情在对应 JSON。新增实验前先查重，禁止复用已存在的 run/dataset ID。

## Datasets

| ID | Manifest | 状态 | 备注 |
|----|----------|------|------|
| `fontdiffuser42-cnstyle-v1` | `provenance/datasets/fontdiffuser42-cnstyle-v1.json` | frozen | 42 字体历史盘；勿覆盖 |
| `fontdiffuser_p253`（磁盘名） | **缺失** | legacy_untracked | summary 与磁盘 Style 计数不一致；不得直接当新基模数据 |

## Code variants

| ID | Path | 相对 official | 状态 |
|----|------|---------------|------|
| `official` | `code/official/FontDiffuser/` | — | frozen 只读 |
| `ours-legacy` | `code/ours/FontDiffuser/` | `docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff` | frozen 历史恢复 |
| `cn2west_ft_v2` | `code/variants/cn2west_ft_v2/` | TBD | **planned** 新基模最小补丁 |

## Experiments / Runs

| ID | Provenance | 状态 | 结论入口 |
|----|------------|------|----------|
| `FT-CNSTYLE-25K` | `provenance/runs/FT-CNSTYLE-25K.json` | `retro_partial` | `reports/retrain_v2/FD_CNSTYLE_RESULTS.json` |
| `FT-P253-CNSTYLE-12K` | **缺失** | legacy | 同上；Style 池当时版本存疑 |
| `A-MVP-CONTROL` | `provenance/runs/A-MVP-CONTROL.json` | `retro_partial` | `PROJECT.md` Stage A |
| `A-MVP-DELTA` | `provenance/runs/A-MVP-DELTA.json` | `retro_partial` | `PROJECT.md` Stage A |
| `FT-P251-REF8-CN2WEST-V2` | TBD | **planned** | 新 200+ 基模（待确认后开） |
