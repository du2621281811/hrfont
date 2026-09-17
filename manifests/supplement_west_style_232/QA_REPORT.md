# v0917 QA 记录（对照用）

> **协作政策（已更新）**：232 **全部同步**；下表「未满」的 7 套**不剔除**，缺字见 `MISSING_CHARS.md` / 各条 `missing_chars`。  
> 本文件保留当时全量 cmap 检查结果，便于核对。

- 已选：**232**
- 勾选语种 target 全满：**225**
- Ext 不全（已写明缺字、仍收录）：**7**

## 训练 target（295）
- `ascii_digits`: 10
- `ascii_letters`: 52
- `latin_ext_letters`: 27
- `hiragana`: 83
- `katakana`: 86
- `bopomofo`: 37

## 预览 vs 全表
审核页每语种约 8 个探针字；QA 按 **全量 target bucket** 查 cmap。探针有墨 ≠ Ext 27 全满。

## 覆盖汇总（勾选语种上）
- `ascii_digits`: 全满 230/230
- `ascii_letters`: 全满 230/230
- `latin_ext_letters`: 全满 199/206（7 套不全）
- `hiragana`: 全满 144/144
- `katakana`: 全满 143/143
- `bopomofo`: 全满 53/53

## Ext 不全的 7 套（仍在正式 232 内）
- `fzsj_1966717` / `fzsj_1966721` / `fzsj_1966722` / `FZSJ-WULXEM` → 缺 `āēěīńňōūǎǐǒǔǖǘǚǜ`
- `fzsj_2923481` / `fzsj_2923966` / `FZXLB` → 缺 `ńň`

正式用法见 [`README.md`](README.md)。
