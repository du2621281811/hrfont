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

## 数据集准备：cn2west v2 渲染协议 A–H

261 个方正字体 × 295 target + 338 style 字符，train/val/test = 237/16/8。  
字符集：`manifests/charset_cn2west_v2_planned.json`；字体拆分：`manifests/pipeline_v2_*_stems_v2.txt`。  
ContentImage 统一使用 Noto Sans CJK Regular，按各协议对应逻辑渲染。

### 协议规格

| 协议 | 画布 | 字号策略 | 居中 | 缩放 | 输出 | 脚本 |
|------|------|----------|------|------|------|------|
| **A** | 96×96 | 逐字体固定：搜最大 fs 使全部字的 textbbox w,h ≤ 84 | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto A` |
| **B** | 96×96 | 逐字体 height-fit：搜最大 fs 使全部字的 textbbox h ≤ 84；宽溢出则逐字缩小 | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto B` |
| **C** | 128→96 | 固定 fsize=128 | textbbox 居中 @ 128×128 | BILINEAR → 96×96 | RGB PNG | `build_cn2west_v2_proto_abc.py --proto C` |
| **D** | 96×96 | 逐字逐字形：二分搜最大 fs 使 textbbox w,h ≤ 88（margin=8px） | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_df.py --proto D` |
| **F** | 96×96 | 逐字逐字形：二分搜最大 fs 使 ink margin ≈ 8px（8% canvas） | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_df.py --proto F` |
| **H** | 96×96 | 同 A（逐字体一号、内框 84），但用**墨迹像素框**搜最大 fs | textbbox 居中 | 无 | RGB PNG | `build_cn2west_v2_proto_h.py` |

**关键区别**：
- A/B/H 是**逐字体**统一字号（所有字符共享一个 font size），D/F 是**逐字符**独立搜索最大字号。
- A 用 `textbbox`（逻辑边界框）约束字号，H 用 `ink bbox`（实际墨迹像素）约束字号，其余与 A 相同。
- D 用 `textbbox` 约束，F 用 `ink bbox` 约束。
- C 是唯一做缩放的方案，字填充率最高但可能有抗锯齿模糊。

### 数据集 ID

| 协议 | 目录名 |
|------|--------|
| A | `fontdiffuser-p253-t295-s338-cn2west-v2` |
| B | `fontdiffuser-p253-t295-s338-cn2west-v2b-hfit` |
| C | `fontdiffuser-p253-t295-s338-cn2west-v2c-official128` |
| D | `fontdiffuser-p253-t295-s338-cn2west-v2d-perglyph-max96` |
| F | `fontdiffuser-p253-t295-s338-cn2west-v2f-perglyph-fit96` |
| H | `fontdiffuser-p253-t295-s338-cn2west-v2h-inkfit` |

### 合作者重建

```bash
# A/B/C（约 15-30 分钟）
python scripts/build_cn2west_v2_proto_abc.py --proto all --workers 16

# D/F（约 5-10 分钟）
python scripts/build_cn2west_v2_proto_df.py --proto both --workers 16

# H（A 的墨迹框版，约 10-20 分钟）
python scripts/build_cn2west_v2_proto_h.py --workers 16

# Review 网页（约 1 分钟）
python scripts/build_cn2west_v2_protocol_review.py
python -m http.server 8777 --directory data/  # 打开 http://127.0.0.1:8777/cn2west_v2_abc_review/
```

### QA Review

- 入口页：`data/render_qa_hub.html`
- 协议对比 Review：`data/cn2west_v2_abc_review/index.html`（A/B/C/D/F/H 切换；筛选基于 B；永字高度按协议分别显示）
- A 墨量分布预览：`data/cn2west_v2_abc_review/proto_A_ink/`（全量 `max(ink_h,ink_w)/96` 分档；字体按 median 墨量排序）
  - 重建：`python scripts/build_cn2west_v2_proto_a_ink_preview.py`
- A vs H 对比：`data/cn2west_v2_abc_review/proto_AH_compare/`（逻辑框 vs 墨迹框定号）

**A 墨量摘要（已扫描 165,213 张）**：median≈75%；主体 70–85%（约 63%）；Style 汉字 median≈78%，ASCII 字母≈63%；极小墨量（&lt;25%）约 0.18%，主要来自个别纤细字体。

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
- 2026-09-04：补齐渲染协议 A–H 规格与重建脚本；修复 D/F 字号搜索上界；新增协议 H（A 的墨迹框版）；A 全量墨量分布预览页。
