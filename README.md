# HR-Font（中→西 FontDiffuser）

ICLR 2027：**中文 few-shot 风格 → 生成拉丁 / 假名**。  
工作区：`/root/projects/hrfont`。**合作者只需 clone 这一个仓库。**

## 合作者从这里开始

1. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 实验、结果、**防把补丁当官方**
2. [`code/README.md`](code/README.md) — **`official/` vs `ours/` 两个目录**
3. [`PROJECT.md`](PROJECT.md) — 当前状态与 Stage A 结论
4. [`COLLABORATOR_GUIDE.md`](COLLABORATOR_GUIDE.md) — 官方设置 / 我们的 FT / 渲染 / Loss
5. [`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md) — 本机数据与权重（不进 Git）

## 官方代码 vs 我们的改动（单仓内物理隔离）

```text
code/
  official/FontDiffuser/   ← 官方干净（不要改、不要当训练入口）
  ours/FontDiffuser/       ← 官方 + 我们的补丁（脚本默认指向这里）
```

```bash
diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'
```

补丁全文：[`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`](docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff)

## 目录结构

```text
hrfont/
  PROJECT.md / COLLABORATOR_GUIDE.md / README.md
  scripts/                 # 自研实验入口
  code/official/…          # 官方 FD
  code/ours/…              # 我们的 FD
  docs/                    # 说明 + patches
  provenance/              # 实验指纹
  reports/                 # 结果页
  manifests/ data/ meta    # 清单；大数据 symlink
  runs/                    # 权重视 symlink（不进 Git）
```

## GitHub

- 唯一需要拉的实验仓：https://github.com/du2621281811/hrfont （private）
- 历史对照用的独立 fork（可选）：https://github.com/du2621281811/fontdiffuser-hrfont  

## 环境

- Python：`/root/miniforge3/envs/boogu`
- 训练入口示例：`scripts/retrain_v2_finetune_fontdiffuser.py`、`scripts/hrfont_stagea_mvp_train.py`（均使用 `code/ours/FontDiffuser`）
