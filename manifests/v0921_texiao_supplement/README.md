# v0921 特效字体补充数据（合作者说明）

**版本**：`v0921_texiao_supplement` · tag `data-v0921` · 冻结日 2026-09-21

## 重要（先读）

1. **正式规模是 315 套**（特效池人工一筛 + 协议 A 二筛）。
2. **每套字体带可用语种**：字段 `scripts`（`latin` / `latin_ext` / `hiragana` / `katakana` / `zhuyin`）。
3. **勾选语种 target 不全仍收录**（15 套）；缺字见 `missing_chars` / [`MISSING_CHARS.md`](MISSING_CHARS.md)。训练时跳过缺字码点。
4. **审核页探针 ≠ 训练 target**（UI 约 8 字预览；训练字表 cn2west **295**）。
5. **本包是字体名单 + 语种/缺字元数据**，尚未并入 `v0913_clean` 的 PNG pair；**TTF 不进 git**。

| 计数 | 值 |
|------|-----|
| 正式字体 | **315** |
| 勾选语种 target 全满 | 300 |
| 不全但已写明缺字 | 15 |

## 正式文件（请用这些）

| 用途 | 路径 |
|------|------|
| **合同** | [`../v0921_texiao_supplement.json`](../v0921_texiao_supplement.json) |
| **315 全量清单** | [`fonts_all_315.json`](fonts_all_315.json) |
| stem 列表 | [`stems_all_315.txt`](stems_all_315.txt) |
| **缺字说明** | [`MISSING_CHARS.md`](MISSING_CHARS.md) |
| 二筛导出（原始 picks） | [`picks_pass2_export.json`](picks_pass2_export.json) |

每条字体关键字段：

- `scripts`：可用语种（勾选）
- `qa_status`：`ok` | `incomplete_selected_target`
- `coverage`：各 target bucket 的 hit/n/full
- `missing_chars`：不全时 bucket → 缺字串
- `font_file` / `font_sha1_12`：对齐本机 TTF

## 勾选语种 → 训练 bucket

| `scripts` | 对应 target |
|-----------|-------------|
| `latin` | `ascii_digits`(10) + `ascii_letters`(52) |
| `latin_ext` | `latin_ext_letters`(27) |
| `hiragana` | `hiragana`(83) |
| `katakana` | `katakana`(86) |
| `zhuyin` | `bopomofo`(37) |

## 与其它波次

- `v0913_clean`：主线可用性 pair。
- `v0917_west_style`：方正西文风格补充（232）。
- `v0921_texiao_supplement`：特效字体池人工二筛补充（本包）。

## TTF

**不在本仓库。** 逻辑目录提示：`texiao_fonts/live/<font_file>`。需要字体包时另传，勿把 TTF 推进 git。
