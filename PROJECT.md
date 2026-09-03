# HR-Font / ICLR 2027 项目台账

> 唯一内部入口。计划、状态、决策和结果只更新本文件。  
> 工作区：`/root/projects/hrfont` · 远程：`https://github.com/du2621281811/hrfont`  
> 管理流程：[`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) · 登记表：[`provenance/REGISTRY.md`](provenance/REGISTRY.md)  
> 实验速查：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) · 官方 vs 我们：[`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md)

## 当前状态

- **阶段**：新基模微调设计（`cn2west_ft_v2`），**尚未开训**。
- **方向已定**：从 `code/official` 派生最小补丁变体；不在 `code/ours` 上继续堆功能。
- **暂停**：Stage B、RS-Gap、Support、4/8-shot、256、完整 Plan v2。
- **历史**：Stage A MVP = `INCONCLUSIVE`；`FT-CNSTYLE-25K` / `FT-P253` 为旧证据（部分 `retro_partial`）。

## 下一步（待确认后执行）

1. 建立 `code/variants/cn2west_ft_v2/`（StyleImage + 官方 ckpt 加载 + resume/路径；无 SCR）。
2. 重建版本化数据 `fontdiffuser-p251-ref8-cn2west-v2`（去掉 2 个残缺字体；Style 池与字体清单写死指纹）。
3. 登记 dataset provenance；LR 消融 → 主曲线；内部 val + Demo-8 仅终评。
4. 每次开训：`pm_preflight` → 干净 Git → `provenance/runs/<id>.json` → 结论回填本文件。

未确认四项设计选择（251 字体、ref8、内部 val、LR 双轨+步数）前，不开长训练。

## 实现边界

- 新实验用新入口、新 run ID、新数据版本目录；不覆盖历史 metrics/ckpt。
- `code/official` 与 `code/ours` 冻结只读；新逻辑只进 `code/variants/<id>/`。
- 数据/权重不进 Git；本机用 symlink。
- 未收到明确开训指令前，不启动长训练或扩展实验。

## 实验登记（摘要）

| ID | 状态 | 说明 |
|----|------|------|
| `FT-CNSTYLE-25K` | `retro_partial` | 42 字体历史 FT |
| `FT-P253-CNSTYLE-12K` | legacy，provenance 缺失 | 253 字体；Style 池存疑 |
| `A-MVP-CONTROL` / `A-MVP-DELTA` | 完成，`INCONCLUSIVE` | Stage A 因果早筛 |
| `FT-P251-REF8-CN2WEST-V2` | planned | 新基模（待确认） |

完整表见 `provenance/REGISTRY.md`。

## 版本与恢复

- 根 Git 管代码、台账、规则、provenance、精选报告。
- 正式实验要求 `exact`：干净 commit + 数据指纹 + 命令 + 产物。
- 旧实验诚实标 `retro_partial`，不用推测补齐。

---

## 归档：Stage A MVP（2026-09）

- 结论：`INCONCLUSIVE`；ΔL1 改善 0.00155 < 0.002；不进 Stage B。
- Control L1=0.08255，Delta L1=0.08099；CI=[-0.00235,-0.00079]；胜率 54.71%。
- 协议：同数据、同 `ft_cnstyle@25k` 初始化、10k、1-shot「永」；详见历史变更记录与 `provenance/runs/A-MVP-*.json`。

## 精简变更记录

- 2026-09-02：Stage A 收缩为两臂最小验证；建立单一台账与护栏。
- 2026-09-03：Stage A 完成 → `INCONCLUSIVE`；建根 Git 与补丁提交；迁入 `/root/projects/hrfont`。
- 2026-09-03：单仓双目录 `code/official` + `code/ours`；推送 GitHub `hrfont`。
- 2026-09-03：固化项目管理：`docs/PROJECT_MANAGEMENT.md`、`provenance/REGISTRY.md`、`code/variants/`、`scripts/pm_preflight.py`；台账转向新基模设计阶段。
