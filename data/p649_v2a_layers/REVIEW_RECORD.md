# p649 原260 三层审查记录（精简）

日期：2026-09-13  
范围：只完成 **overlap260（原260）**。新389 未三层审完，不进训练。  
磁盘：L1 `data/p649_v2a_review/decisions.json`（只读）· L2/L3 `data/p649_v2a_layers/layer2.json` `layer3.json`  
映射：`data/p649_v2a_layers/training_map/`  
重建：`python3 scripts/build_p649_script_usability_map.py`

审查页（不要发给导师）：http://172.19.45.13:19000/p649_v2a_layers/review.html  
19003 笔记本打不开。首页 / 19001 无此入口。

## 漏斗（已对账，0 条不一致）

| 结果 bucket | 层规则 | n | target 保留 |
|---|---|---|---|
| `all_scripts` | L1 pass：各语种都能用 | 146 | 295 |
| `no_bopomofo` | L2 pass：只有注音不能用 | 69 | 258 |
| `han_latin_digit` | L3 pass：注音+平假+片假不能用 | 40 | 89 |
| `exclude` | L3 drop：仍不可用 | 5 | 0 |

146 + 69 + 40 + 5 = 260。L2 候选 = L1 drop 114；L3 候选 = L2 drop 45。

### split（不改 228/16/16）

| split | all_scripts | no_bopomofo | han_latin_digit | exclude |
|---|---:|---:|---:|---:|
| train | 130 | 58 | 35 | 5 |
| val | 9 | 5 | 2 | 0 |
| test | 7 | 6 | 3 | 0 |

训练 target：**223 / 228**（去掉 5 套 exclude）。val/test 无 exclude。

exclude：`FZBenMWYJW` `FZCHYJW` `FZGuangHTJW-H` `FZHeJYSXZJW` `FZZhuoQHJW`  
不做 GT、不做 donor。

## 训练 / val / test 同一条规则

按 `(字体, 字)` 过滤，不要整套扔掉 val/test 字体。

1. `font_to_bucket.json` → bucket  
2. `exclude` → 跳过该字体  
3. 否则只用该 bucket 的语种；汉字 style 仍 338（exclude 除外）  
4. 不改 split

val 有效对数 4123 / 4720；test 3880 / 4720。  
按语种报分：注音只在 `all_scripts` 上平均；假名只在 `all_scripts` + `no_bopomofo` 上平均。

## 假名口径（审 L2/L3 时）

英文花体连笔（截图那种 A/a/à）**不会出现在平假/片假**。  
平假可有中文行草连笔；片假几乎仍是断笔。不要用拉丁花体标准套假名。

## 文件

| 文件 | 用途 |
|---|---|
| `training_map/script_usability_map.json` | 每套字体 l1/l2/l3、skip_scripts |
| `training_map/font_to_bucket.json` | stem → bucket |
| `training_map/script_usability_map.csv` | 表 |
| `training_map/stems_*.txt` | 名单 |

未接 sampler 前，训练不要指向 p649。
