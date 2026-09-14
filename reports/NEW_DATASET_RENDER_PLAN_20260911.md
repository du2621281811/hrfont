# 新数据集渲染方案（协议 A）· 2026-09-11

**状态：** 规格已对照现有 A 实现、831 cmap 交集、旧并池筛选做过检查；**尚未开渲**。  
**不要**覆盖现盘 `data/fontdiffuser-p253-t295-s338-cn2west-v2/`（F0–F3 仍用它）。

---

## 0. 一句话

从「随体 792 ∪ font_50 独有 39」得到 831 套，去掉 JF 副本、缺字字体、已冻结的极细体，得到 **649** 套；按协议 A 原生 96 渲 Target 295 + Style 338；Content 仍是 Noto 同一套 A；字符表不扩大。

---

## 1. 锁定项（渲什么）

| 项 | 决定 | 依据 |
|----|------|------|
| 协议 | **只渲 A**，不渲 B/C/D/F/H | PI：H 已弃用；A 为唯一训评协议 |
| 画布 | 96×96 RGB PNG，白底黑字，**无缩放** | `build_cn2west_v2_proto_abc.py` `CANVAS_96=96` |
| 字号 | **每套字体一个** `fs`：二分最大，使该字体全部**存在的** 633 字 `textbbox` w,h ≤ **84**（margin 6） | `find_size_A` / `INNER_AB=84` |
| 居中 | `textbbox` 几何居中 | `render_glyph_AB` |
| Target | `charset_cn2west_v2_planned.json` 的 295 | 不扩 ASCII 标点、不收回 `ì` |
| Style | 同文件 338 汉字（含 ref8 `永和书风骨韵天地`） | 评测仍 1-shot「永」或 ref8，与现主线一致 |
| Content | `NotoSansCJK-Regular.ttc`，对 **295 target** 做一次 A 定号后落盘 | 现盘 `content_size=83`；B₀=Noto |
| 文件名 | Target/Style：`{stem}+uXXXX.png`；Content：`uXXXX.png` | 现 loader 依赖此格式 |

**不扩大字符表。** 831 套共同码位有 7568 个，但多出的主要是 GB2312 汉字和 ASCII 标点；额外西文字母几乎没有（`ÀÄÑß` 仅 4/831）。扩 Target 会破坏与当前 F0–F3 的字符对齐。

---

## 2. 字体漏斗（渲哪些）

源：已解压 `font_crosslingual/data/suiti_fonts_probe/随体字体集/`（792 TTF）+ `/root/data/font_50`（50）。zip 不必再解。preview PNG / xlsx 不进盘。

| 步 | 规则 | 剩余 |
|----|------|-----:|
| 并池 | stem 去重；11 个两源同文件只留一份 | 831 |
| 缺字 | drop `FZGoolongWuxiaFontR`（cmap 缺 `ńň`） | 830 |
| JW/JF | 存在对应 JW 则 drop JF（180 对，笔画近重复） | 650 |
| ink 先例 | drop `FZXianZTJW`（现冻结 `mean_bbox<20%`） | **649** |
| 跨家族近重复 | **不删** | 649 |

旧 `stroke≥0.50` **不是协议 A 的门**，本次不作为渲染硬过滤（可在训前另做子集）。字重全部保留：同家族多字重算不同风格，但 **split 必须按 `family_key` 隔离**。

`family_key`：与 `scripts/suiti_rich_preview_and_dedup.py` 相同（去 JW/JF，再剥 `L/R/B/M/H/EB/SB/DB/EL/Te/Da/Cu/Zhong/Zhun/Xi/Italic`）。

当前 p260 的 260 个 stem **全部落在这 649 里**。Demo-8 八款都在。

---

## 3. 缺字与定号（必须改现脚本）

现成 `build_cn2west_v2_proto_abc.py` **不能直接跑**（见 §6）。

1. 渲前 `getBestCmap`；缺 ref8 任一 → 整套 drop。  
2. 缺的码点 **不写 PNG**，记入 `coverage.json`；禁止 tofu / 空白顶替。  
3. `find_size_A` **只用 cmap 存在的字**。  
4. 搜不到合法 `fs`（或保存前 `w,h>84`）→ fail closed，不 silent 回退 `fs=10`。  
5. 渲后：空墨迹、跨字相同 hash（tofu）、触边；不过门不发布。

本池 cmap 上 649 应对 633 全齐；真正风险是 tofu/空墨，不是缺表项。

---

## 4. 目录与规模

工作目录（只写一次，存在即拒绝）：

```text
data/fontdiffuser-p649-t295-s338-cn2west-v2a-r0/
  inventory.json          # path, sha256, source, family_key
  coverage.json
  summary.json
  train|val|test/
    ContentImage/uXXXX.png          # 每 split 各一份 295，内容相同
    TargetImage/<stem>/<stem>+uXXXX.png
    StyleImage/<stem>/<stem>+uXXXX.png
```

规模：649 × 633 ≈ **41.1 万** 张字图 + Content 295。现盘约 16.6 万张 / 0.7GB → 新盘约 **1.7GB** 量级。

---

## 5. Split（渲完后再冻；评测对齐另锁）

**不要**把旧 228/16/16 数量直接套到 649 上。

建议两层：

1. **Matched 评测茎**（和 F0–F3 比）：仍用现在的 val16 + test16。  
2. **家族隔离**：`family_key` 不得跨 train/val/test。  
   - 现 train/val/test 茎保持原 split  
   - 新字体：家族已在 val/test → 进同一 split，**不进 matched 评测表**（除非日后扩评测）  
   - 全新家族 → train  

按此分配：train 577 / val 38 / test 34（多出来的是 val/test 家族的额外字重）。Matched 表仍只打 16+16 茎。

---

## 6. 现成脚本检查（为何不能直接 `--proto A`）

| 检查 | 结果 |
|------|------|
| 字体名单 | **不合格**：从旧 B `summary.json` 读 261 套，不是 649 |
| 输出目录 | **危险**：写死现 A 盘，会覆盖 F0–F3 数据 |
| cmap | **不合格**：不查缺字，缺字仍落盘 |
| 定号 | **风险**：`hi=300, best=10` 无解会 silently 用 10 |
| 保存前 assert ≤84 | **缺** |
| 空图/tofu QA | **不合格**：只数文件个数 |
| Content | Noto 路径存在；现逻辑（295 上定号）可复用，但应写入新目录 |
| 文件名 / PNG / 无 Resize | 与 `cn2west_f0_rsifree` loader 一致，新盘必须保持 |

必须 **新脚本 + 新目录**；可复制 `find_size_A` / `render_glyph_AB`，但自举清单、覆盖、fail-closed、原子发布要重写。

---

## 7. 执行顺序

1. 写 `inventory.json`（649 条 path+sha256+family）。  
2. 10 套试渲：含现 A 同茎、font_50 独有、多字重、曾 ink 尾部。同茎与现 A 盘对图（字号应接近）。  
3. 全量渲到临时目录 → coverage/QA/ink_ratio。  
4. ink 门：沿用 `mean_bbox<20%` 为自动 drop 起点，尾部人工过一遍；drop 后改最终 ID（可能不再是 p649）。  
5. 冻结 split + 树 hash；preflight 后才允许训练指向新盘。

---

## 8. 明确不做

- 不覆盖 `fontdiffuser-p253-t295-s338-cn2west-v2`  
- 不把 ASCII 标点 / `ì` / 额外 6425 汉字进 Target/Style  
- 不把 JF 与 JW 同时当未见风格  
- 不把 val/test 家族的新字重放进 train  
- loader 不 Resize；官方 `font_dataset.py` 的 BILINEAR 不得用于本盘  
