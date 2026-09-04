# HR-Font / ICLR 2027 项目台账

> 唯一内部入口。计划、状态、决策和结果只更新本文件。  
> 工作区：`/root/projects/hrfont` · 远程：`https://github.com/du2621281811/hrfont`  
> 管理流程：[`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) · 登记表：[`provenance/REGISTRY.md`](provenance/REGISTRY.md)  
> 实验速查：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) · 官方 vs 我们：[`docs/OFFICIAL_VS_OURS.md`](docs/OFFICIAL_VS_OURS.md)

## 当前状态

- **阶段**：**E1 满训进行中**（`E1-FTV2-A-S3407`，watchdog 托管，约 **92k / 100k**）。
- **训练协议已定**：仅 **A**；池 **260=228/16/16**；drop `FZXianZTJW`。
- **E1 冻结**：seed **3407 only**；**GPU3**；**bs=8 / accum=1**；100k；lr=1e-5；warmup=5k；fp16；SCR off。
- **入口**：`configs/e1_ft_v2_a_s3407.yaml` · `scripts/launch_cn2west_ft_v2_e1.py` · `code/variants/cn2west_ft_v2/`
- **看板**：本机 http://127.0.0.1:8777/e1_ft_v2_dashboard/（train/val loss + 多时间步 Pred 对比；`runs/` 不进 Git）。
- **冒烟**：`runs/smoke-E1-FTV2-*` 20 step @ bs=8 **已通过**。

## 下一步

1. 等 E1 到 100k → 写 `provenance/runs/E1-FTV2-A-S3407.json` + 回填结论
2. （可选）清僵尸显存后再议 `bs=16` 复现实验
3. 正式 p260 manifest/SHA 可并行补登记

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
| `FT-P260-A-CN2WEST-V2` | planned | R0 后正式 A 盘 FT-v2（228/16/16） |

完整表见 `provenance/REGISTRY.md`。

## 版本与恢复

- 根 Git 管代码、台账、规则、provenance、精选报告。
- 正式实验要求 `exact`：干净 commit + 数据指纹 + 命令 + 产物。
- 旧实验诚实标 `retro_partial`，不用推测补齐。

---

## 数据集准备：cn2west v2 渲染协议 A–H

**活跃池 260 字体**（已 drop `FZXianZTJW`）× 295 target + 338 style；**train/val/test = 228/16/16**。  
字符集：`manifests/charset_cn2west_v2_planned.json`；  
拆分真源：`manifests/pipeline_v3_{train,val,test}_stems.txt`（`pipeline_v2_*_stems_v2.txt` 已同步为同一内容）；  
拆分 provenance：`manifests/split_v3_228_16_16.json`（旧 237/16/8 备份为 `*.bak_*.txt`）。  
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
- 协议对比 Review：`data/cn2west_v2_abc_review/index.html`（A/B/C/D/F/H 切换；旧 B 筛查仅对照）
- **R0 主审查**：`data/cn2west_v2_abc_review/proto_A_ink/review.html`（`ink_ratio_rank` · pass/drop/rerender）
  - 重建：`python scripts/build_cn2west_v2_ink_ratio_rank.py`
- A 墨量预览：`data/cn2west_v2_abc_review/proto_A_ink/`（边长 / **框面积** 可切换排序）
  - 重建：`python scripts/build_cn2west_v2_proto_a_ink_preview.py`
- A vs H 对比：`data/cn2west_v2_abc_review/proto_AH_compare/`

**A 墨量摘要（165,213 张，基于重划前全量扫描）**：边长 median≈75%；**框面积** median≈48%。  
**字体门（已冻结）**：`mean_bbox < 20%` → 仅 `FZXianZTJW` drop；其余 B-screen 对照候选 **pass**。  
**Split（已冻结）**：228/16/16；从原 test8 起，用 seed=3407 自 train 增补 8 字进 test。详见 `reports/R0_INK_GATE_PROPOSAL.md`、`manifests/split_v3_228_16_16.json`。

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
- 2026-09-04：台账对齐 R0；正式 `ink_ratio_rank` + pass/drop/rerender；提案字体门 mean_bbox&lt;20%。
- 2026-09-04：PI 冻结 ink 门（drop FZXianZTJW）；重划 **228/16/16**；A–H 盘目录已同步；目标 ID 改为 p260。
- 2026-09-04：落地 `cn2west_ft_v2` + E1 满训（`E1-FTV2-A-S3407`）；训练看板（train/val loss + Pred 对比）。
