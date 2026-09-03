# HR-Font（中→西 FontDiffuser）

ICLR 2027 方向：**中文 few-shot 风格 → 生成拉丁 / 假名**。  
本目录是从旧多方法仓 `font_crosslingual` 抽出的 **干净工作区**；后续实验只在这里继续。

## 合作者从这里开始

> **警告：** 只打开 `hrfont` 仓 ≠ 拥有 FontDiffuser 源码。官方与补丁在独立仓 [`fontdiffuser-hrfont`](https://github.com/du2621281811/fontdiffuser-hrfont)；该仓默认分支 **`main` = 官方干净**，补丁在 `hrfont/local-patches-20260903`。请先读 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

1. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — **实验清单 + 结果入口 + 防误用铁律**（必读）
2. [`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md) — 官方 / 补丁 / 自研三层边界
3. [`PROJECT.md`](PROJECT.md) — 当前状态、Stage A 结论、下一步
4. [`COLLABORATOR_GUIDE.md`](COLLABORATOR_GUIDE.md) — 官方训练设置、我们的 42/253 FT、渲染与 Loss
5. [`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md) — 数据 / 权重本机路径（默认不进 Git）

## 目录结构

```text
hrfont/
  PROJECT.md                 # 台账
  COLLABORATOR_GUIDE.md
  scripts/                   # 自研：建数据 / FT / 评测 / Stage A·B
  code/FontDiffuser/         # 上游 + 补丁分支（独立 git）
  docs/
    OFFICIAL_VS_OURS.md
    DATA_AND_WEIGHTS.md
    patches/*.diff           # 相对 main 的补丁全文
  provenance/                # 数据指纹与实验来源清单
  reports/                   # 协议、CN2WEST 索引、Stage A 可视、overnight 预览
  manifests/                 # 42 / 253 字体清单
  data/                      # 小 meta 进 Git；训练盘/字体盘为 symlink（不进 Git）
  runs/                      # symlink 到 FT 与 Stage 权重（不进 Git）
```

## FontDiffuser：如何看出「改了官方哪里」

```bash
cd code/FontDiffuser
git checkout hrfont/local-patches-20260903
git diff main...HEAD --stat
# 或阅读 docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff
```

- 官方干净：`main` @ `7b28ce9`（= `yeungchenwa/FontDiffuser`）
- 我们的补丁：`hrfont/local-patches-20260903` @ `4a47351`（与早期本地 `99e42b5` 同内容；以 GitHub fork 为准）

## 本机数据与权重

训练 JPG 与 `.pt` **不进 Git**。在本机通过 symlink 指向原路径；详见 `docs/DATA_AND_WEIGHTS.md`。


## GitHub（private）

- 实验仓：https://github.com/du2621281811/hrfont
- FontDiffuser fork：https://github.com/du2621281811/fontdiffuser-hrfont
  - 官方：`main`
  - 我们的补丁：`hrfont/local-patches-20260903`
  - Compare：https://github.com/du2621281811/fontdiffuser-hrfont/compare/main...hrfont/local-patches-20260903

本机 `code/FontDiffuser` 已 clone 该 fork；`ckpt/` 仍为本机 symlink，不进 Git。

## 环境

- Python：`/root/miniforge3/envs/boogu`
- 入口示例：`scripts/retrain_v2_finetune_fontdiffuser.py`、`scripts/hrfont_stagea_mvp_train.py`
