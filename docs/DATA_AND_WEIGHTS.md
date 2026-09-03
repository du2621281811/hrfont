# 数据与权重（本机 / 不进 Git）

Git 只跟踪清单与指纹；大文件用 symlink 或外盘。

## 数据

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
