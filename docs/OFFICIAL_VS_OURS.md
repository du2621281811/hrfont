# 官方 FontDiffuser vs 我们的改动 / 实验

合作者请先读 [`EXPERIMENTS.md`](EXPERIMENTS.md) 与 [`../code/README.md`](../code/README.md)。

## 1. 三层边界（单仓内）

| 层 | 位置 | 是什么 |
|----|------|--------|
| **官方源码** | `code/official/FontDiffuser/` | 上游 `main` 快照（`UPSTREAM_PIN.txt`） |
| **我们对官方的补丁** | `code/ours/FontDiffuser/` | 同一上游 + 补丁；相对 official 仅 8 个文件级差异 |
| **自研实验代码** | `scripts/`、`reports/`、`provenance/` | 不在官方仓库里 |

**不会**再用「同一个文件夹靠切分支」区分官方/补丁，避免误检出错。

查看补丁：

```bash
diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'
# 或
less docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff
```

## 2. 补丁内容摘要

1. `dataset/font_dataset.py` — 有 `StyleImage/` 时只从中文池采 style  
2. `train.py` / 配置 — SCR 关闭时仍可加载官方发布权重微调  
3. `unet.py` / `unet_blocks.py` — 工程兼容  
4. `stroke_scr.py` 等 — **后加实验线**，不得追溯为 `ft_cnstyle@25k` 当天补丁全集  

## 3. 实验线

见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。Stage A 不改 UNet/MCA/RSI 结构，改动在 `scripts/hrfont_*`。

## 4. 不要称为「官方」的

- `code/ours/**`
- `scripts/**`
- 自建跨语数据、Stage A/B 结果、Stroke-SCR / overnight 等
