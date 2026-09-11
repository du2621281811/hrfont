# 数据与权重（本机 / 不进 Git）

Git 只跟踪清单与指纹；大文件用 symlink 或外盘。

## 当前主线要用的（协议 A / F 臂）

新机迁移打包见 [`SETUP_COLLABORATOR.md`](SETUP_COLLABORATOR.md) §3。这些 **不进 Git**。

| 逻辑路径 | 约体积 | 说明 |
|----------|--------|------|
| `data/fontdiffuser-p253-t295-s338-cn2west-v2/` | 0.7GB | 协议 A；F0/F1/F2/F3 训练与 test16 |
| `artifacts/f0/es_spatial_f0/` | 1.7GB | **F0** Es cache |
| `artifacts/f0/ec_multiscale_f0/` | 94GB | **F0** Ec cache |
| `runs/F0-RSIFREE-FT-A-S3407/best/` | 1.1GB | F 臂 parent（= `global_step_100000`） |
| `code/official/FontDiffuser/ckpt/` | ~0.4GB | 官方 P1；本机是 symlink |

不要把 E1 的 Es/Ec 接到 F 臂上。

## 数据（历史盘）

| 逻辑路径 | 本机实际（symlink） | 说明 |
|----------|---------------------|------|
| `data/fontdiffuser/` | `.../font_crosslingual/data/retrain_v2/fontdiffuser` | 42 字体 CN-style 训练盘 |
| `data/fontdiffuser_p253/` | `.../fontdiffuser_p253` | 253 字体放大盘 |
| `data/font/` | `.../font` | 底层 PNG 源（重建用） |
| `data/hrfont_bank/` | `.../data/hrfont` | overnight bank（可选） |
| `data/meta_chars.json` 等 | 进 Git | 字符表 / coverage |

指纹：`provenance/datasets/fontdiffuser42-cnstyle-v1.json`（`tree_sha256`）。

## 权重

| 逻辑路径 | 说明 |
|----------|------|
| `code/ours/FontDiffuser/ckpt/` | 官方发布三件套（+ 可选 SCR）；symlink |
| `runs/ft_cnstyle/global_step_25000/` | `FT-CNSTYLE-25K` |
| `runs/ft_p253_cnstyle/global_step_12000/` | 253 放大有效点 |
| `runs/stagea_mvp/` | Stage A Control / Delta |
| `runs/e2_stageA96_formal/`、`runs/e2_stageB96/` | overnight formal（证据） |

合作者若无 symlink：自行下载/拷贝后改路径，或按 provenance 命令复现。

## 字体清单

- `manifests/pipeline_v2_train_stems_42.txt`
- `manifests/pipeline_v2_train_stems_253.txt`
