# Eval index / display audit — 2026-09-09

Scope: `scripts/eval_f03_test16_strat.py`, `reports/f03_test16_strat/{index,timeline*,browse_index,metrics_*}`, Compare Portal `:8768`.

## Verdict

**主协议看板与指标索引正确，未见错位。** 显示路径、文件名中的 `font/cp/seed`、metrics 行与 PNG 一一对应；抽检重算 L1 与入库值一致。

## Checks passed

| 检查项 | 结果 |
|---|---|
| `STRATIFIED` ↔ `PROTOCOL.chars` ↔ `browse.chars` | 47 字一致，无重复，均为单码点 NFC |
| `fonts()` ↔ stems 文件 ↔ split.test ↔ PROTOCOL | 16 字序完全一致 |
| `item.cp` ↔ `cp_of(char)` ↔ 文件名 `…__{cp}__s3407.png` | 0 错配 |
| `bucket` ↔ `script_bucket(char)` | 0 错配 |
| browse 752×4 pred / content / style / gt 文件 | **3008/3008 pred 存在，refs 无缺失** |
| `metrics_items` 与 browse 覆盖 | 0 缺失、0 重复；`n_total=3008` |
| summary L1 对 items 重聚合 | 四方法 delta = 0 |
| 单条 L1 对 pred↔GT 重算 | 与 stored 一致 |
| board GT ↔ data `TargetImage`（96 resize） | MAE = 0 |
| pred sidecar `.json` meta 与路径 | 抽检 40 条 OK |
| `index.html` 内嵌 DATA ↔ browse | fonts/chars/items/methods 同步 |
| timeline_f2 / timeline | 16×16=256；TIMELINE_CHARS 与磁盘 cp 集合相等；图路径 0 missing |
| F2/F3 timeline 子集在全量目录中可解析 | 0 missing |
| Portal / 8767 / 8766 HTTP | 200 |

## 非 bug、但需注意的呈现点

1. **方法列顺序**为 `P1 → F0_100k → F3_80k → F2_75000`（字典定义序），不是 F0/F1/F2/F3 叙事序；数字仍对应当列方法，**不是索引错了**。
2. **F2 主表是 `@75k` 全量 752**；训练虽 DONE@80k，主指标尚未刷 `F2_80k`（命名与文件一致，勿当成 80k）。
3. **Timeline 是 n=256 子集**（`TIMELINE_CHARS` 16 字×16 字体）；与主表 752 **不可混读**。
4. **Content 图按字共享、Style 按字体**，符合数据协议（ContentImage 不按字体分叉）。
5. **F3_80k 为 legacy Support**，与在训 F3b 不同协议；门户已标 LEGACY。
6. Compare Portal 总览表来自 `manifest.json` 快照；新 eval 后需重生成 manifest 才会更新数字。

## 代码侧结论

- `pred_path` / `gt_path` / `cp_of` / gallery `items[].preds` 同一公式，HTML 用 `m.id` 取 `it.preds[m.id]`，**列与图绑定正确**。
- `cmd_metrics` 读 pred 旁 meta 再 `gt_path(meta.font, meta.char)`，与生成时写入一致；未发现「图是 A、分是 B」类错绑。

## 未覆盖（本次未报错，但范围外）

- 生成采样器数值正确性（仅验索引/落盘/指标聚合）。
- E12 eye HTML 多为脚本拼图，未做静态 `src=` 扫图（`probe.json` 存在）。
- F1 / F3b 尚无 pred，无索引可验。
