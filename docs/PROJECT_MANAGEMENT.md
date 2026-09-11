# HR-Font 项目管理规则

> 目的：进度可查、版本可追、Git 干净、禁止无效重复堆叠。  
> **状态与结论只写 [`PROJECT.md`](../PROJECT.md)**；本文件只定流程，不写实验结果。

## 1. 单一入口

| 文件 | 职责 |
|------|------|
| `PROJECT.md` | 当前阶段、决策、通过标准、实验登记、变更记录 |
| `docs/EXPERIMENTS.md` | 实验 ID ↔ 结果入口 ↔ 是否改官方代码 |
| `docs/OFFICIAL_VS_OURS.md` / `code/README.md` | 官方 vs 补丁防误用 |
| `COLLABORATOR_GUIDE.md` | 相对稳定的技术事实（协议、Loss、历史 FT） |
| `provenance/REGISTRY.md` | 已登记 dataset / run / code-variant ID 清单 |
| `provenance/datasets/*.json` | 数据树指纹 |
| `provenance/runs/*.json` | 单次实验来源清单 |

**禁止**：再新建 plan / status / handoff / summary / NEXT_STEPS 类重复文档。需要更新时改 `PROJECT.md` 或对应 provenance。

## 2. 目录与版本命名

```text
code/
  official/FontDiffuser/     # 只读官方快照，禁止改业务逻辑
  ours/FontDiffuser/         # 历史补丁恢复版（含旧实验线），默认只读
  variants/<id>/             # 新实验专用最小补丁树（从 official 派生）
data/
  <dataset_id>/              # 新数据新目录，禁止覆盖旧盘
runs/
  <experiment_id>/           # 新 run 新目录，禁止覆盖旧 ckpt/metrics
docs/patches/                # 相对 official 的可审查 diff
```

- `dataset_id` 例：`fontdiffuser-p251-ref8-cn2west-v2`
- `experiment_id` 例：`FT-P251-REF8-V2-LR1E5`
- `variant_id` 例：`cn2west_ft_v2`

旧路径 `data/fontdiffuser*`、`runs/ft_*`、`code/ours` **冻结为历史证据**，新工作不原地改。

## 3. Git 规则

1. 唯一协作仓：`git@github.com:du2621281811/hrfont.git`（工作目录 `/root/projects/hrfont`）。网页仍是 `https://github.com/du2621281811/hrfont`。`fetch` / `push` **走 SSH**，不要用 HTTPS（本机 `github.com:22` 会被劫持，SSH 经 `ssh.github.com:443`）。
2. **开训 / 正式评测前**：`git status` 干净；相关改动已 commit；再写 provenance。
3. **进 Git**：脚本、台账、docs、provenance、`code/official`、`code/ours`、`code/variants`（无权重）、精选 reports。
4. **不进 Git**：训练 JPG、`.pt`、大 logs、本机 `data/`/`runs/` symlink。
5. 改官方相关代码时：
   - 新逻辑放 `code/variants/<id>/`；
   - 同步导出 `docs/patches/<id>.diff`（相对 `code/official`）；
   - 在 `provenance/REGISTRY.md` 登记 variant + commit。
6. 禁止 force-push `main`；禁止把「计划」写成「已完成」。
7. **多机**：Git 只同步代码与精选报告；协议 A 数据、F0 `best`、Es/Ec cache 用 rsync（§3.1）。每台机尽量保持 `/root/projects/hrfont` + conda `boogu`。
8. **2h 扫描**：`python scripts/pm_sync_scan.py` 把脏文件分成 `must_sync` / `hold_wip` / `never`。Agent 只自动提交 `must_sync`（评测脚本、报告页、台账）；训练 WIP（如未完成的 F2P）进 `hold_wip`，等人确认。落后 remote 时先 `fetch` + `pull --rebase`（禁止 `-i`）。

## 3.1 新服务器要拷的非 Git 文件

| 路径 | 约体积 | 用途 |
|------|--------|------|
| `data/fontdiffuser-p253-t295-s338-cn2west-v2/` | 0.7GB | 协议 A 训练/评测图 |
| `artifacts/f0/es_spatial_f0/` | 1.7GB | F0 Es cache |
| `artifacts/f0/ec_multiscale_f0/` | **94GB** | F0 Ec cache（优先 rsync，勿轻易重建） |
| `runs/F0-RSIFREE-FT-A-S3407/best/` | 1.1GB | F 臂父模型 |
| `artifacts/f0/support_bank.json` | 小 | 或从 `reports/artifacts_sync/` 拷回 |

不要 rsync 整份 `runs/`（单臂可 20GB+）。官方 ckpt 仅 P1 评测需要。

## 4. 实验生命周期

```text
设计（写入 PROJECT.md）
  → 实现（variants + scripts，commit）
  → preflight（scripts/pm_preflight.py）
  → 登记 dataset 指纹（若新数据）
  → 开训 / 评测
  → 写 provenance/runs/<id>.json
  → 结论回填 PROJECT.md + REGISTRY.md
```

开训前必须锁定并写进 provenance：数据 ID、初始化权重、可训参数、步数、seed、评测协议、通过标准、代码 variant + commit。

控制实验：**同数据、同初始化、同预算、同 seed；一次只改一个目标变量。**

## 5. 追溯等级

| 等级 | 含义 |
|------|------|
| `exact` | 干净 Git + 数据指纹 + 命令 + 产物齐全 |
| `retro_partial` | 历史实验；部分可核、源码或数据当时状态不可逐字节恢复 |

新正式实验默认必须 `exact`。`retro_partial` 只用于归档旧结果。

## 6. Preflight

```bash
python scripts/pm_preflight.py
python scripts/pm_preflight.py --experiment-id FT-P251-REF8-V2 --variant cn2west_ft_v2
```

失败项未清零不得开训。
