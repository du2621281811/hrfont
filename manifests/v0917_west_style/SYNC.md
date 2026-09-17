# v0917 西文风格补充数据（合作者说明）

**版本**：`v0917_west_style` · tag `data-v0917` · 冻结日 2026-09-17

## 重要（先读）

1. **正式规模是 232 套，不是 225。**  
   225 只是「勾选语种上 295-target 全满」的子集；**给合作者的正式名单是 232。**
2. **每套字体带可用语种**：字段 `scripts`（审核勾选：`latin` / `latin_ext` / `hiragana` / `katakana` / `zhuyin`）。
3. **7 套 Ext 不全仍收录**，不剔除；缺哪些字写在 `missing_chars` / [`MISSING_CHARS.md`](MISSING_CHARS.md)。  
   训练或渲染时：**跳过缺字码点**，勿当成 Ext 27 全满。
4. **审核页探针 ≠ 训练 target。**  
   UI 每语种约 8 字预览；训练字表是 cn2west **295**（数字10+拉丁52+Ext27+平假83+片假86+注音37）。
5. **本包是字体名单 + 语种/缺字元数据**，尚未并入 `v0913_clean` 的 PNG pair 表；TTF **不进 git**。

| 计数 | 值 |
|------|-----|
| 正式字体 | **232** |
| 其中勾选语种 target 全满 | 225 |
| Ext 不全但已写明缺字 | 7 |

## 正式文件（请用这些）

| 用途 | 路径 |
|------|------|
| **合同** | [`../v0917_west_style.json`](../v0917_west_style.json) |
| **232 全量清单（正式）** | [`fonts_all_232.json`](fonts_all_232.json) |
| stem 列表 | [`stems_all_232.txt`](stems_all_232.txt) |
| **缺字说明** | [`MISSING_CHARS.md`](MISSING_CHARS.md) |
| 全满子集（可选） | `fonts_verified.json`（225，**不是**正式全集） |

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

## 不全的 7 套（仍在 232 内）

| clean | 缺字（`latin_ext_letters`） |
|-------|---------------------------|
| `fzsj_1966717` | `āēěīńňōūǎǐǒǔǖǘǚǜ`（11/27） |
| `fzsj_1966721` | 同上 |
| `fzsj_1966722` | 同上 |
| `FZSJ-WULXEM` | 同上 |
| `fzsj_2923481` | `ńň`（25/27） |
| `fzsj_2923966` | `ńň` |
| `FZXLB` | `ńň` |

## 与 v0913 关系

- `v0913_clean`：既有 228/16/16 可用性 pair 映射（训练主线）。
- `v0917_west_style`：方正补充池人工筛选的**西文风格补充字体**；元数据已冻，**PNG split 未合并**。

## TTF

见 [`DOWNLOAD.md`](DOWNLOAD.md)。逻辑目录：`font_files_0914/founder_fonts_download/<font_file>`。
