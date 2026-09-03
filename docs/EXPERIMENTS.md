# 实验一览（合作者速查）

> **一个仓即可：** clone [`hrfont`](https://github.com/du2621281811/hrfont) 后，官方源码与我们的改动都在 `code/` 下**两个不同文件夹**里，不会靠「切换 git 分支」区分。

## 防误用（最重要）

| 路径 | 含义 |
|------|------|
| `code/official/FontDiffuser/` | **官方干净**上游快照 |
| `code/ours/FontDiffuser/` | **我们改过**的版本（训练默认用这个） |

- 看差异：`diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'`
- 或：`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`
- 说明：`code/README.md`、`docs/OFFICIAL_VS_OURS.md`

**禁止**把 `code/ours/` 称为官方；**禁止**在 `code/official/` 里做实验改动。

## 实验与结果

| 实验 | 做了什么 | 结果入口 | 对官方 FD 的改动 |
|------|----------|----------|------------------|
| `FT-CNSTYLE-25K` | 42 字体 CN style→西文，25k | `provenance/runs/FT-CNSTYLE-25K.json`；`reports/retrain_v2/FD_CNSTYLE_RESULTS.json`、`CN2WEST_INDEX.*` | 使用 **`code/ours`** 补丁（StyleImage 等）；不是 Stroke-SCR |
| `FT-P253-CNSTYLE-12K` | 253 字体，12k | 同上结果表 | 同上 |
| `A-MVP-CONTROL` / `A-MVP-DELTA` | Stage A RSI 对照，10k，976 对 | `PROJECT.md`；`provenance/runs/A-MVP-*.json`；`reports/hrfont_stagea_mvp_visual/` | **不改** UNet/MCA/RSI 结构；逻辑在 `scripts/hrfont_stagea_mvp_*.py`；权重从 `ft_cnstyle` 初始化，跑在 **ours** 树上 |
| Overnight Stage A/B | 历史证据 / Mentor | `reports/hrfont_overnight/` | 自研脚本；B 非主结论 |

官方超参 / Loss / 渲染细节：`COLLABORATOR_GUIDE.md`。

## 数据与权重

训练 JPG 与 `.pt` **不进 Git**（本机 `data/`、`runs/` symlink）。见 `docs/DATA_AND_WEIGHTS.md`。
