# HR-Font（中→西 FontDiffuser）

ICLR 2027：**中文 few-shot 风格 → 生成拉丁 / 假名**。  
工作区：`/root/projects/hrfont`。**合作者只需 clone 这一个仓库。**

## 合作者从这里开始

1. [`PROJECT.md`](PROJECT.md) — **当前状态**（唯一台账）
2. [`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) — 进度 / 版本 / Git / 追溯
3. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 实验 ID 与结果入口
4. [`code/README.md`](code/README.md) — `official` / `ours` / `variants`
5. [`COLLABORATOR_GUIDE.md`](COLLABORATOR_GUIDE.md) — 协议 / Loss / 历史 FT
6. [`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md) — 本机数据与权重

## 代码树（单仓物理隔离）

```text
code/
  official/FontDiffuser/     ← 官方干净（只读）
  ours/FontDiffuser/         ← 历史补丁（旧实验；只读）
  variants/<id>/FontDiffuser ← 新实验最小补丁（从 official 派生）
```

```bash
diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'
python scripts/pm_preflight.py   # 开训前检查
```

历史补丁：[`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`](docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff)

## 目录结构

```text
hrfont/
  PROJECT.md / COLLABORATOR_GUIDE.md / README.md
  scripts/                 # 自研入口 + pm_preflight / provenance
  code/official|ours|variants/
  docs/                    # 管理规则 + patches
  provenance/              # REGISTRY + datasets/runs 指纹
  reports/ manifests/
  data/ runs/              # 本机 symlink，不进 Git
```

## GitHub

- 唯一需要拉的实验仓：https://github.com/du2621281811/hrfont （private）
- 历史对照用的独立 fork（可选）：https://github.com/du2621281811/fontdiffuser-hrfont  

## 环境

- Python：`/root/miniforge3/envs/boogu`
- 训练入口示例：`scripts/retrain_v2_finetune_fontdiffuser.py`、`scripts/hrfont_stagea_mvp_train.py`（均使用 `code/ours/FontDiffuser`）
