# 28 份做完的问卷

只包含正式题 30/30 全部做完的 28 人。练习题不在这里。没做完的人不在这里。同一姓名如果另有一次没做完，那一次也没有放进来。

每人一份自动编号，见 `sessions.jsonl`。`is_font_designer` 只是记录，算指标时不拆开。

## 文件

- `responses.jsonl`：每一行是一个候选的名次。正式题 `phase` 为 `formal`，注意力检查为 `attention`。
- `items.json`：30 道正式题和 2 道注意力检查。
- `sessions.jsonl`：编号、姓名、是否字体设计师、开始时间。
- `analyze_iclr_human_eval.py`：指标脚本。
- `stats.json`：跑完脚本后的结果。

`rank` 从 1 到 6，1 最好。六个方法的内部名字是：

| 内部名字 | 论文里写 |
| --- | --- |
| gt | Ground Truth |
| hrfont | HR-Font |
| fcagan | FCA-GAN |
| ftransgan | FTransGAN |
| garfont | GAR-Font |
| fontdiffuser | FontDiffuser |

## 怎么算

需要 Python 和 numpy。在本目录执行：

```bash
python analyze_iclr_human_eval.py
```

结果写到 `stats.json`，并在终端打印。

脚本只使用 `phase == formal` 的行。同一人同一题如果有多条记录，同一方法保留文件里最后一条。

一道题要至少 3 个人、并且每个人都给满 6 个方法名次，才进入指标。不够的题丢掉。

注意力检查：只有当某人每一次注意力检查都把不匹配的候选排进前 2，才整人剔除。这 28 人里没有人被剔除。

算出的数：

- **Mean Rank**：该方法所有有效「人 × 题」名次的平均。越低越好。
- **Top-2**：名次为 1 或 2 的比例。
- **HR-Font pairwise preference**：HR-Font 的名次严格好于对方的比例。分母是有效的「人 × 题」。
- **Kendall's W**：同一题里各人名次的一致程度，再对题取平均。1 是完全一致。
- **95% CI**：按题做 2000 次 bootstrap，种子 3407，取 2.5% 和 97.5% 分位。

设计师和非设计师合在一起算，不分组。

和 CSC、LPIPS、SSIM、DINO 的 Kendall tau-b 要另放一份 `metric_scores.json`，并且 `samples` 的键必须是这 30 道题的 `sample_id`。没有这份文件时，`evaluator_tau` 为空。当前这 30 题还没有对齐的自动指标分数。
