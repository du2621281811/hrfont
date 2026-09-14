# 人工评分协议 · Cross-Script Style Consistency

对应老师验证第 3 项：与人工分的 **Spearman** 相关。自动分数见 `scores_test16_cosine_*.json`（φ cosine）。

## 任务一句话

看左侧 **中文参考字**（同一字体），给右侧 **西文字形** 打 1–5 分：  
**「西文是否像属于这块中文参考所代表的同一字体风格？」**

## 不是在评什么

- 不是「字清不清楚 / 可不可读」
- 不是「和 GT 像素像不像」
- 不是「好不好看 / 艺术性」

## 量表

| 分 | 含义 |
|---:|------|
| 5 | 明显同一字体风格（粗细、衬线、姿态一致） |
| 4 | 大体一致，小瑕疵 |
| 3 | 中性 / 说不准 |
| 2 | 风格偏离较大 |
| 1 | 明显另一套风格（或错字体） |

## 刺激构成

- 文件：`human_board/items.json` · 看板：`human_board/index.html`
- 约 144 题：16 字体 ×（4 方法 × 2 Latin 字 + 1 注意力题）
- 方法档：`GT` / `F2C_80000` / `F0C_100000` / `F0_100k`（看板 **不显示** 方法名）
- 注意力题 `ATTN_wrong`：中文 refs 与西文来自 **不同字体**；期望低分（≤2）。若标注员在多数注意力题上打 ≥4，该员数据作废重标。

## 流程

1. 两位标注员 **独立** 打分（`R1` / `R2`），互不讨论。
2. 打开看板，按随机顺序逐题填写。
3. 把分数写入 [`human_ratings_template.csv`](human_ratings_template.csv) 的 `score` 列（只填整数 1–5）。
4. 汇总：

```bash
/root/miniforge3/envs/boogu/bin/python scripts/e12_spearman_human.py \
  --ratings reports/e12_paper/human_ratings_template.csv \
  --items reports/e12_paper/human_board/items.json
```

输出：`human_spearman_summary.json`（均值分 vs `style_cosine` 的 Spearman；人际相关；注意力通过率）。

## 看板启动（任选）

在仓库根目录：

```bash
python -m http.server 8793 --directory reports/e12_paper/human_board
# 另需能访问 preds 与 StyleImage：建议从项目根起静态服务，或用看板内嵌的相对路径约定
```

推荐从项目根：

```bash
cd /root/projects/hrfont && python -m http.server 8793
# 打开 http://127.0.0.1:8793/reports/e12_paper/human_board/index.html
```
