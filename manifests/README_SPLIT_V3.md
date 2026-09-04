# cn2west split v3（228 / 16 / 16）

- **活跃字体**：260（已排除 `FZXianZTJW`）
- **比例**：train 228 · val 16 · test 16
- **真源 JSON**：[`split_v3_228_16_16.json`](split_v3_228_16_16.json)
- **清单**：`pipeline_v3_{train,val,test}_stems.txt`（与已更新的 `pipeline_v2_*_stems_v2.txt` 内容一致）
- **旧版备份**：`pipeline_v2_*_stems_v2.bak_*.txt`（237/16/8）

### 相对 v2 的变更

1. drop `FZXianZTJW`（ink 门 mean_bbox&lt;20%）
2. 保留原 val16、原 test8
3. seed=`3407` 从 train 增补 8 字到 test：  
   `FZBuGTJW`, `FZDeSHJW_515H`, `FZFeiHHJW-H`, `FZJianLTJW_Te`, `FZJingLTJW-H`, `FZLiuGQKJW-L`, `FZMaWDBSJW`, `FZTJLSJW`

盘上 A–H 协议目录的 `train/val/test/{Target,Style}Image` 已按此搬家；被 drop 字体在 `excluded/`。
