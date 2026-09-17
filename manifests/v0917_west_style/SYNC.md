# v0917_west_style 同步清单

**版本 id：`v0917_west_style`（v0917 数据）**

## 给合作者什么

| 内容 | 路径 |
|------|------|
| **232 全量清单（正式）** | `fonts_all_232.json` |
| 缺字明细 | `MISSING_CHARS.md` / `MISSING_CHARS.json` |
| 仅全满子集（225） | `fonts_verified.json`（可选） |
| stem | `stems_all_232.txt` |
| TTF 包 | `artifacts/v0917_west_style_ttf_232.zip`（约 1.4G+） |

## 缺字政策

- **不剔除** Ext 不全的 7 套；它们仍在 232 内。
- 每条有 `qa_status`：`ok` 或 `incomplete_selected_target`。
- 不全时看 `missing_chars`（按训练 bucket → 缺哪些字）。
- 训练/渲染：**跳过缺字码点**，不要当成 295 全满。

## UI 语种 → 训练 bucket

| 勾选 scripts | 必须覆盖的 target |
|--------------|-------------------|
| latin | ascii_digits(10) + ascii_letters(52) |
| latin_ext | latin_ext_letters(27) |
| hiragana | hiragana(83) |
| katakana | katakana(86) |
| zhuyin | bopomofo(37) |

## 不全的 7 套摘要

见 `MISSING_CHARS.md`（多为拼音韵母 `āēěī…` 或仅缺 `ńň`）。
