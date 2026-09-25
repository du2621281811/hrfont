# K7-B 消融结果图像（协作者快照）

本目录提供已完成结果的可视化与原始预测图，便于逐字对照检查。它是 **2026-09-25 17:44 CST 的阶段性快照**，不是最终完整队列；当时后处理推理队列为 20/28 阶段完成，未完成阶段（包括 CRP 路由对照）没有纳入本包。

## 快速查看

1. 用 Git LFS 克隆本分支：`codex/k7b-ablation-results-20260925`。等待 LFS 文件下载完成。
2. 解开 `supplemental-board.tar` 和 `completed-test-val-predictions.tar`。
3. 在浏览器打开 `board/supplemental.html`。Safari 若直接打开本地文件没有加载表格数据，请在 `board/` 目录运行 `python3 -m http.server 8000`，再访问 `http://127.0.0.1:8000/supplemental.html`。
4. 补充测试看板支持按字体/字符筛选，逐行并排查看 GT、K7-B、A1、A2、A3。另一个归档按模型和数据 split 保存已完成 VAL/TEST 的生成 PNG。

## 快照内容与边界

| 文件 | 内容 |
|---|---|
| `supplemental-board.tar` | 可筛选补充测试看板及依赖资源；Extra32 6,559 行、Plus17 3,046 行，逐行对齐 GT、K7-B、A1、A2、A3。 |
| `completed-test-val-predictions.tar` | DONE 确认完成的 14 组 8-shot VAL/TEST 预测 PNG，按 `模型/split/` 放置。每组图片数已与 DONE 记录核对。 |

已归档的 14 组共 **50,345 张**：A1、A2、A3 各自的 `val_0913`（4,123）、`val_0917`（3,208）、`test_0913`（3,880）、`test_0917`（3,208）；K7-B 的 `test_0913`（3,880）、`test_0917`（3,208）。单独预测图归档只含生成结果 PNG，不含 GT、评测指标、日志或模型权重；不要将它误读为逐行 GT 对照看板。

## 实验身份与比较条件

- K7-B：32k checkpoint；A1/A2/A3：各自 20k checkpoint。
- A1：Delta on / TC off；A2：Delta off / TC on；A3：Delta off / TC off。
- 统一推理协议：8-shot、固定参考组、seed 3407、CFG 1.0、DPM++ 20 steps / order 2。
- `0913`、`0917` 分别表示对应版本的 V3 验证/测试数据池；Extra32、Plus17 是独立补充观察池，不与 V3 VAL/TEST 混称。
- 图像用于视觉审阅；本包不声称队列全部完成，也不替代完整指标表、质量判断或因果结论。尚未完成的模型/数据组合应保持缺失，不以计划值填补。

## 文件与版本

- 图像归档采用 Git LFS。克隆时需安装 Git LFS 并执行 `git lfs pull`（若未自动下载）。
- 仓库分支：`codex/k7b-ablation-results-20260925`。
- 此快照只包含图像、看板及本说明；不含训练数据、checkpoint、推理缓存或活动队列文件。
