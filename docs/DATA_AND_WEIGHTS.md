# 数据与权重（本机 / 不进 Git）

Git 只跟踪清单与指纹；大文件用 symlink 或外盘。

## 当前主线要用的（协议 A / F 臂）

新机迁移打包见 [`SETUP_COLLABORATOR.md`](SETUP_COLLABORATOR.md) §3。V100 跳板 scp 映射：[`V100_SCP_TRANSFER.md`](V100_SCP_TRANSFER.md)。这些 **不进 Git**。

| 逻辑路径 | 约体积 | 说明 |
|----------|--------|------|
| `data/fontdiffuser-p253-t295-s338-cn2west-v2/` | 0.7GB | **当前训练盘**；协议 A；F0/F1/F2/F3 与 test16。只读，勿覆盖 |
| `data/fontdiffuser-p649-t295-s338-cn2west-v2a-r0/` | — | p649 新盘；原260 语种表见 `data/p649_v2a_layers/training_map/`。**未接训练** |
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
- **`manifests/v0917_west_style/`** — 西文风格补充 **232**（正式；见上节）

## v0917 西文风格补充（2026-09-17）

**正式名单 232 套**（不是 225）。说明：[`manifests/v0917_west_style/README.md`](../manifests/v0917_west_style/README.md)。

| 项 | 内容 |
|----|------|
| 合同 | `manifests/v0917_west_style.json` |
| 正式清单 | `manifests/v0917_west_style/fonts_all_232.json`（含 `scripts` / `missing_chars`） |
| 缺字 | `manifests/v0917_west_style/MISSING_CHARS.md`（7 套 Ext 不全，**仍收录**） |
| tag | `data-v0917` |
| TTF | 不进 Git；`artifacts/v0917_west_style_ttf_232.zip`（见 `DOWNLOAD.md`） |
| 与 v0913 | 补充字体元数据；**尚未**并入 `v0913_clean` PNG pair |

训练时跳过各字体 `missing_chars`；审核探针 ≠ 295 全表。

## 0917独立划分与融合v2（2026-09-17，待用户review）

已按232正式名单构建PNG及独立200/16/16划分，再与冻结0913的同名split合并为 **v2**；没有修改`v0913_clean`本身。

- [0917独立清单](../manifests/v0917_split/README.md)：45,963个合法target GT。
- [v2清单](../manifests/v2/README.md)：名义428/32/32字体，110,395个target GT；旧train的5个排除字体仍排除。
- [完整报告与验证](../reports/dataset_v2_20260917/REPORT.md)：旧153,197个PNG及新124,033个PNG验证通过，旧split/pair不变。
- V100图像根：`/root/data1/hrfont_dataset_v2_20260917/{v0917,v2}`；图册：同目录`review/{index.html,v0917.html}`。

4个新增字体缺少部分简体中文参考，须使用`style_pool.json`；不能直接套原固定ref8。训练donor须按`donor_train_by_cp.json`逐字过滤。数据已构建供review，尚未启动v2训练，现有K族实验仍使用其冻结数据。
