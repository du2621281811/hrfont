# v0917 TTF 获取方式（232）

Git **不含** TTF 二进制（约 1.4G，`.gitignore` 排除 `*.zip` / `artifacts/*`）。
清单与缺字说明已在本目录；字体文件从打包机取。

## 包内容
| 文件 | 说明 |
|------|------|
| `artifacts/v0917_west_style_ttf_232.zip` | 整包 232 TTF + manifest |
| `artifacts/v0917_west_style_ttf_232_shards/v0917_ttf_232.part{1..4}of4.zip` | 四分卷（各约 360MB，可并行） |
| 包内 `MISSING_CHARS.md` | 7 套 Ext 不全的缺字（仍收录） |

## 加速下载（推荐顺序）
1. **同机拷贝**：`cp artifacts/v0917_west_style_ttf_232.zip <目标>`
2. **SSH/rsync**（合作者有账号时，通常远快于浏览器）
   ```bash
   rsync -avP user@host:/root/projects/hrfont/artifacts/v0917_west_style_ttf_232.zip .
   # 或并行拉 4 个 part
   ```
3. **四分卷并行**（浏览器）：打包机上打开 `reports/paper_fonts_0914_screen/downloads/`，同时下 part1–4 + meta，解压后合并 `fonts/`。

## 校验
```bash
# 整包
sha256sum -c artifacts/v0917_west_style_ttf_232.zip.sha256
# 或按 fonts_all_232.json 的 font_sha1_12 抽查
```

## 与清单对齐
- 正式清单：`fonts_all_232.json`（232，含 `missing_chars`）
- 全满子集：`fonts_verified.json`（225，可选）
