# 合作者 / 新机器初始化

先 clone，再拷数据与权重。路径尽量保持 **`/root/projects/hrfont`**，Python 用 conda **`boogu`**（启动脚本写死了这两处）。

## 0. 先读这些（不要另建进度文档）

| 读什么 | 里面有什么 |
|--------|------------|
| [`PROJECT.md`](../PROJECT.md) | **唯一**状态、决策、实验登记、下一步 |
| [`PROJECT_MANAGEMENT.md`](PROJECT_MANAGEMENT.md) | Git / 报告 / 目录命名 / 新机拷什么 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 实验 ID ↔ 结果入口 ↔ 代码树 |
| [`OFFICIAL_VS_OURS.md`](OFFICIAL_VS_OURS.md) | `official/` vs `ours/` vs `variants/` |
| [`DATA_AND_WEIGHTS.md`](DATA_AND_WEIGHTS.md) | 本机数据/权重路径（大文件不进 Git） |
| [`COLLABORATOR_GUIDE.md`](../COLLABORATOR_GUIDE.md) | 官方 Loss / 历史 FT 事实 |

**禁止**再新建 plan / status / handoff / NEXT_STEPS。要改结论就改 `PROJECT.md` 或 `provenance/`。

## 1. Git（代码与报告页）

```bash
# 公钥加到 GitHub 账号 SSH keys，或该仓 Deploy keys（可写）
git clone git@github.com:du2621281811/hrfont.git /root/projects/hrfont
cd /root/projects/hrfont
```

- 唯一仓：`git@github.com:du2621281811/hrfont.git`。网页才用 `https://github.com/du2621281811/hrfont`。
- **`fetch` / `push` 走 SSH**，不要 HTTPS。
- 禁止 force-push `main`。落后时 `git pull --rebase origin main`（不要 `rebase -i`）。
- **进 Git：** 脚本、台账、docs、provenance、`code/`（无 `.pth`）、精选 `reports/` HTML。
- **不进 Git：** 训练图、权重、Es/Ec cache、整棵 `runs/`、live log / pid。
- 新逻辑放 `code/variants/<id>/`，不要改 `code/official/` 业务逻辑，不要把 `ours/` 叫官方。
- 新实验用**新 run id**，禁止覆盖本机已有的 `F0-…` / `F2-DELTARSI-…` 目录。
- 开训前：`git status` 干净 + `python scripts/pm_preflight.py`。

报告页：clone 后在本机起服务，**不要**用 GitHub 网页当看板（private，且相对路径图片会断）。

```bash
cd /root/projects/hrfont/reports/f03_test16_strat
python3 -m http.server 8767
# http://127.0.0.1:8767/
```

离线汇总：`reports/collab_offline/index.html`。

## 2. 环境

本机训练配置（新机必须对齐，否则不能当同一实验）：

- Python 3.10，`torch 2.7.1+cu126`，accelerate 1.14，diffusers 0.38
- **单卡**、`train_batch_size=8`、`accum=1`、**fp16**（V100 **不要 bf16**）
- seed **3407**，96px，协议 A
- F0：100k，`offset=0`；F 臂：80k，parent=`runs/F0-RSIFREE-FT-A-S3407/best`
- V100 与 3090 **不能比特复现**已训权重；评测靠拷 ckpt，新实验用同一超参做协议对照

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# V100 应为 (7, 0)。8 张卡不要同时开 8 路训练（Ec 94GB + Swap 压力）。
```

## 3. 数据与权重（不进 Git）

打包位置（源机器）：`artifacts/migrate_v100/`（gitignored）。  
两机暂时不通时：填 [`V100_SCP_TRANSFER.md`](V100_SCP_TRANSFER.md) / [`manifests/v100_scp_map.json`](../manifests/v100_scp_map.json)，在跳板上 scp。

| 包 | 约体积 | 必须？ | 内容 |
|----|--------|--------|------|
| Git clone | 小 | 必须 | 代码、docs、精选 reports |
| `01_small.tar` | ~0.7GB | 必须 | Support bank、官方 P1 ckpt |
| rsync 协议 A | 0.7GB | 必须 | `data/fontdiffuser-p253-t295-s338-cn2west-v2/`（16 万 PNG，不打 tar） |
| `03_es_cache.tar` | 1.7GB | 必须 | F0 Es（**不要用 E1 cache**） |
| `04_f0_best.tar` | 1.1GB | 必须 | F 臂父模型 |
| `05_eval_ckpts.tar` | ~4GB | 评旧臂才要 | F1@80k、F2@75k+80k、F3@80k |
| **rsync Ec** | **94GB** | 必须（训/评 F 臂） | `artifacts/f0/ec_multiscale_f0/` |

**不要拷：** 整份 `runs/`（单臂 18–23GB）、旧 `data/fontdiffuser` / `data/font`、E12 cache（除非做 E12）。

源机器：

```bash
# 已打好包则跳过
bash scripts/pack_newhost_migrate.sh

# NEW 换成新机 ssh
bash scripts/rsync_newhost_migrate.sh root@NEW
# 或只拷 tar： scp -r artifacts/migrate_v100 root@NEW:/root/projects/hrfont/artifacts/
```

新机器（先 clone）：

```bash
bash /root/projects/hrfont/scripts/unpack_newhost_migrate.sh \
  /root/projects/hrfont/artifacts/migrate_v100
# Ec 由 rsync_newhost_migrate.sh 直接铺到 artifacts/f0/ec_multiscale_f0/
```

## 4. 文档与报告怎么写

- 结论、状态、通过标准 → 只改 `PROJECT.md`。
- 实验命令、数据指纹、ckpt 来源 → `provenance/runs/<id>.json` + `provenance/REGISTRY.md`。
- 评测 HTML / Glyph Board → `reports/`，相对路径图片，随 Git 或本机 http.server。
- 训练过程 log、`status.json` → 不进仓。
- 计划未跑完不要写成「已完成」。
