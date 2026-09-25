# K7-B 消融结果图像（协作者版）

本分支提供可视化看板和已完成推理的生成图，供合作者按字体、字符逐例检查。图像大包由 Git LFS 管理；克隆后请先执行 `git lfs pull`。

## 快速查看

1. 克隆分支 `codex/k7b-ablation-results-20260925`，安装 Git LFS 并执行 `git lfs pull`。
2. 将下方列出的三个 `.tar` 归档都解压到同一个仓库根目录；FZ49 归档会把新文件叠加到 `board/`，不要单独移动其中的图片。
3. 打开 `board/supplemental.html` 浏览 A1/A2/A3 补充测试对照；打开 `board/crp_fz49.html` 浏览 CRP 路由策略对照。若 Safari 直接打开本地文件无法载入数据，在 `board/` 目录启动 `python3 -m http.server 8000`，再访问 `http://127.0.0.1:8000/`。

## 文件内容

| 归档 | 内容 |
|---|---|
| `supplemental-board.tar` | Extra32 6,559 行、Plus17 3,046 行的 A1/A2/A3 补充测试图像看板及其资源。 |
| `completed-test-val-predictions.tar` | 14 组已完成的 8-shot VAL/TEST 预测 PNG，共 50,345 张，按模型和 split 放置。 |
| `crp-fz49-routing-comparison.tar` | FZ49 两种 CRP 路由策略的 19,210 张预测图，以及 `board/crp_fz49.html` 和对应逐例 JSON。与前两个归档解压到一起后，看板可复用其中的 GT 和 K7-B 基线图。 |

FZ49 路由包覆盖 Extra32（6,559 个样本）与 Plus17（3,046 个样本）。看板每行一个样本，并排显示 GT、K7-B learned Top-K 基线、`uniform_topk` 和 `top1`；支持按数据集、字体筛选。逐项文件核验：两数据集共 9,605 行，GT、基线及两种策略均齐全，无缺图。

## 实验协议与解释边界

- 路由对照是**推理期干预分析**，没有重新训练模型；checkpoint、bank 与每个目标/字族排除规则保持不变。
- 使用 K7-B EMA 32k checkpoint；8-shot，seed 3407，CFG 1.0，20 steps，order 2。
- `uniform_topk` 对相同的 learned Top-K 候选等权融合；`top1` 只使用排序最高的候选；K7-B 列为已完成的 learned Top-K 基线。
- 前两个归档是 2026-09-25 17:44 CST 的 20/28 阶段快照，不含当时尚未完成的路由对照；第三个归档是随后完成的 FZ49 路由补充，四个路由/数据集条件均已完成。
- Extra32 和 Plus17 是补充观察池，不应与 V3 VAL/TEST 混称。图像供视觉审阅，不单独构成指标结论或因果结论。
- 本分支只包含图像、看板及说明；不包含训练数据、checkpoint、推理缓存或活动队列文件。

## 分支

`codex/k7b-ablation-results-20260925`
