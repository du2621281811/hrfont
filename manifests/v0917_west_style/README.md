# v0917_west_style 同步清单

**版本 id：`v0917_west_style`（记作 v0917 数据）**

Git 只传字体清单与 QA。TTF / 审核预览 PNG **不进仓**。

## 用哪份清单

| 文件 | 用途 |
|------|------|
| `fonts_verified.json` | **正式可用：225 套**（295 target 勾选语种全满 + 协议 A 墨迹抽检） |
| `stems_verified.txt` | stem 列表 |
| `QA_REPORT.md` | 校验报告 |
| `fonts_raw_232.json` | 原始人工导出 232（含 7 套 Ext 不全，**勿训练**） |

合同：`manifests/v0917_west_style.json`

## 他机取文件

```bash
git pull
# 按 fonts_verified.json 的 font_file / font_sha1_12 对齐本机：
#   font_files_0914/founder_fonts_download/<font_file>
```

## 剔除的 7 套（假预览风险）

勾了 latin_ext，UI 探针有字，但 Ext 全表 27 不全：
`fzsj_1966717` `fzsj_1966721` `fzsj_1966722` `fzsj_2923481` `fzsj_2923966` `FZSJ-WULXEM` `FZXLB`

## 与 v0913 关系

v0913_clean = 既有 228/16/16 可用性映射。
v0917_west_style = 方正补充池人工筛选后的**西文风格补充字体名单**（metadata），尚未并入 train PNG split。
