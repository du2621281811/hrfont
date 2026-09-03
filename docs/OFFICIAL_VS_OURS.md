# 官方 FontDiffuser vs 我们的改动 / 实验

合作者请先读本页，再读根目录 [`PROJECT.md`](../PROJECT.md) 与 [`COLLABORATOR_GUIDE.md`](../COLLABORATOR_GUIDE.md)。

## 1. 三层边界（不要混）

| 层 | 位置 | 是什么 |
|----|------|--------|
| **官方源码** | `code/FontDiffuser` 分支 `main` @ `7b28ce9` | 与上游 [yeungchenwa/FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) 一致 |
| **我们对官方仓的补丁** | 同仓分支 `hrfont/local-patches-20260903` @ `4a47351`（GitHub fork；旧本地标签 `99e42b5` 内容等价） | 相对 `main` 的 **1 个 commit**（8 文件） |
| **自研实验代码** | `scripts/`、`PROJECT.md`、`reports/`、`provenance/` | **不在**上游仓库里 |

查看补丁（任选）：

```bash
cd code/FontDiffuser
git fetch origin   # 若已配置上游
git log --oneline main..hrfont/local-patches-20260903
git diff main...hrfont/local-patches-20260903
```

或直接打开：[`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`](patches/fontdiffuser-hrfont-local-patches-20260903.diff)

## 2. 补丁里改了什么（摘要）

相对官方 `main` 的确定改动：

1. **`dataset/font_dataset.py`** — 若存在 `StyleImage/<font>/`，style **只从中文池**采样（跨语协议）；否则保持官方「从 Target 池随机」行为。
2. **`train.py` / 配置** — SCR 关闭时仍可从 `phase_1_ckpt_dir` 加载官方发布的三个权重做微调。
3. **`unet.py` / `unet_blocks.py`** — MCA/StyleRSI 导入与工程兼容（attention slice / gradient checkpointing 等）。
4. **`src/modules/stroke_scr.py` 等** — Stroke-SCR **实验线**（后加）；**不得**追溯为 `ft_cnstyle@25k` / `ft_p253_cnstyle@12k` 当时的训练代码。

MCA / RSI / 扩散损失主体未为两次历史 FT 做结构性替换。细节见 `COLLABORATOR_GUIDE.md` §7。

## 3. 我们的实验线（中→西）

| ID | 含义 | 结果入口 |
|----|------|----------|
| `FT-CNSTYLE-25K` | 42 字体，中文 style → 西文 target，25k | `provenance/runs/FT-CNSTYLE-25K.json`；权重 `runs/ft_cnstyle/global_step_25000/`（本机 symlink） |
| `FT-P253-CNSTYLE-12K` | 253 字体放大，12k | `runs/ft_p253_cnstyle/global_step_12000/`；指标见 `reports/retrain_v2/` |
| `A-MVP-CONTROL` / `A-MVP-DELTA` | Stage A：RSI←Ec(永) vs RSI←特征Δ | `PROJECT.md`；`provenance/runs/`；可视 `reports/hrfont_stagea_mvp_visual/` |
| Overnight Stage A/B formal | 历史证据 / Mentor 页 | `reports/hrfont_overnight/`（B 未作为主结论推进） |

数据盘：`data/fontdiffuser`（42）与 `data/fontdiffuser_p253`（253），默认 **symlink** 到原机器路径；指纹见 `provenance/datasets/`。

## 4. 不要称为「官方」的内容

- `scripts/hrfont_*.py`、`retrain_v2_finetune_fontdiffuser.py`、自建数据布局
- 跨语 CN-style 训练集（非官方 400 字体×800 字）
- Stage A/B、overnight、formal_preview 结果
- Stroke-SCR / 256 分辨率等后加实验

官方论文任务是 **中文 content + 中文 style → 中文 target**；我们是 **中文 style → 拉丁/假名 target**。
