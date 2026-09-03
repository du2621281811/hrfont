# 实验一览（合作者速查）

> **不要只 clone 一个仓就动手改官方代码。** 本项目固定为两个 private 仓，职责不同。

## 两个仓

| 仓 | 默认分支 | 内容 |
|----|----------|------|
| [`du2621281811/hrfont`](https://github.com/du2621281811/hrfont) | `main` | **我们的**实验脚本、台账、结果页、provenance、补丁 *说明与 diff 文件* |
| [`du2621281811/fontdiffuser-hrfont`](https://github.com/du2621281811/fontdiffuser-hrfont) | **`main` = 官方干净** | 上游 FontDiffuser；补丁在分支 `hrfont/local-patches-20260903` |

- Compare（官方 vs 我们的 FD 补丁）：  
  https://github.com/du2621281811/fontdiffuser-hrfont/compare/main...hrfont/local-patches-20260903  
- 根仓内也有全文：`docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff`

**误用风险：** `hrfont` 仓 **不包含** `code/FontDiffuser` 源码树（故意 gitignore）。若合作者只打开 `hrfont`，看到的全是自研；要跑训练必须另 clone fork，并分清自己检出的是 `main` 还是补丁分支。

## 实验与结果在哪看

| 实验 | 做了什么 | 结果 / 证据（均在 `hrfont` 仓） | 是否改了官方 FD 内核 |
|------|----------|--------------------------------|----------------------|
| `FT-CNSTYLE-25K` | 42 字体，CN style→西文，25k | `provenance/runs/FT-CNSTYLE-25K.json`；数字见 `reports/retrain_v2/FD_CNSTYLE_RESULTS.json`、`CN2WEST_INDEX.*`；权重本机 `runs/ft_cnstyle/`（不进 Git） | **是（协议补丁）**：需 `StyleImage` 采样等，见 Compare；**不是** Stroke-SCR |
| `FT-P253-CNSTYLE-12K` | 253 字体放大，12k | 同上结果表；权重 `runs/ft_p253_cnstyle/` | 同左（同一套 FD 补丁协议） |
| `A-MVP-CONTROL` / `A-MVP-DELTA` | Stage A：RSI←永 vs RSI←特征Δ，10k，976 对 | `PROJECT.md`；`provenance/runs/A-MVP-*.json`；人眼页 `reports/hrfont_stagea_mvp_visual/` | **否（不改 UNet/MCA/RSI 结构）**：逻辑在 `scripts/hrfont_stagea_mvp_*.py` + `hrfont_delta_feature.py`；仍依赖补丁版数据协议与 `ft_cnstyle@25k` 初始化 |
| Overnight Stage A/B formal | 历史长训 / Mentor 页 | `reports/hrfont_overnight/`；`PROJECT.md` 写明 B 未作主结论 | 自研脚本；B 未推进为主线 |

更细的官方超参 / Loss / 渲染：`COLLABORATOR_GUIDE.md`。  
边界总述：`docs/OFFICIAL_VS_OURS.md`。

## 铁律（避免后续实验用错代码）

1. **要官方原文** → `fontdiffuser-hrfont` 的 **`main`**（或上游 `yeungchenwa/FontDiffuser`），不要默认以为本机 `code/FontDiffuser` 就是官方（本机开发常检出补丁分支）。
2. **要跑我们的中→西 FT / Stage A** → 使用补丁分支 + `hrfont/scripts/*`，不要把 `scripts/` 里的东西当成官方仓库的一部分。
3. **Stroke-SCR / 256 / overnight 消融** → 后加实验线；**不得**说成 `ft_cnstyle@25k` 当天的官方或补丁全集。
4. 权重与训练 JPG **不在 Git**；缺数据时先看 `docs/DATA_AND_WEIGHTS.md`，不要假设 clone 后即可复现数值。
