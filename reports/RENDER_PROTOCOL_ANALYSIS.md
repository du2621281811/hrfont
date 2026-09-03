# cn2west v2 渲染协议分析与代码审计

**日期：2026-09-04　结论：有条件选择 B；修复与分布对齐验证完成前，不得冻结正式数据版本。**

## 1. 五协议速览

v2 字符集是 295 个 target（52 ASCII 字母、10 数字、27 Latin Extended、83 平假名、86 片假名、37 注音）和 338 个中文 style，ref8 为“永和书风骨韵天地” [manifests/charset_cn2west_v2_planned.json:5](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:5) [manifests/charset_cn2west_v2_planned.json:13](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:13) [manifests/charset_cn2west_v2_planned.json:23](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:23)。正式 split 为 237/16/8，review 的逐 split 文件数也与 295/338 完全相符 [fonts.json:16](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:16) [fonts.json:22](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:22) [fonts.json:28](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:28)。

| 协议 | 实际实现 | 核心收益 | 核心代价 | 判断 |
|---|---|---|---|---|
| A | 原生 96；每字体一个字号，使全部字符 `textbbox` 宽高 ≤84；无 resize [PROJECT.md:60](/Users/xiaoweiliang/projects/hrfont/PROJECT.md:60) | 字族内字号统一，保留相对尺寸 | 一个宽/高 outlier 压小全族 | 不选 |
| B | 原生 96；每字体按最大 `textbbox` 高度定字号；宽溢出的字逐字缩小 [PROJECT.md:61](/Users/xiaoweiliang/projects/hrfont/PROJECT.md:61) | 多数字符共享字号，无全局宽 outlier，且无二次插值 | 高 outlier 仍压全族；宽字局部改尺度 | **有条件首选** |
| C | Pillow 以 fs=128 画在 128，再 BILINEAR 到 96 [build_cn2west_v2_proto_abc.py:139](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:139) | 最接近“固定 128 后缩放”的表象，occupancy 最大 | 可能裁切/触边、缩放模糊；并非官方 pygame 等价 | 仅作分布敏感性对照 |
| D | 原生 96；逐字体逐字符搜最大字号，`textbbox` 宽高约束 [build_cn2west_v2_proto_df.py:55](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:55) | 每字尽量占满，少受 outlier 影响 | 完全抹平字体内自然尺寸比例；逻辑框非 ink | 不作训练主协议 |
| F | 原生 96；逐字搜索使 ink 最小边距约 8% [build_cn2west_v2_proto_df.py:81](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:81) | 唯一直接控制实际墨迹 | 最强逐字归一化，字体度量语义损失最大；搜索本身有缺陷 | QA 诊断协议，不作主协议 |

## 2. 协议 B 逐项分析

### 2.1 为什么 B 值得选

B 在一个字体内先用全部 target+style 搜共享字号 [build_cn2west_v2_proto_abc.py:188](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:188)，仅宽度溢出的 glyph 才缩小 [build_cn2west_v2_proto_abc.py:117](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:117)。因此它比 D/F 更保留字族内部的 x-height、汉字/拉丁/假名相对尺度，也避开 A 被单个超宽字符拖小全族的问题。输出直接是 96×96 RGB PNG，无一次 128→96 插值 [build_cn2west_v2_proto_abc.py:117](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:117) [build_cn2west_v2_proto_abc.py:136](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:136)，适合把抗锯齿差异与模型效应分开。

### 2.2 风险、训练影响与修复

| 风险 | 证据与机制 | 严重度 / 对训练的影响 | 必做修复或监控 |
|---|---|---|---|
| 带叠置变音符压小全族 | 27 个 Latin Extended 中含 `ǖǘǚǜ` [charset:8](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:8)；B 对 633 个 target+style 取最大 `textbbox` 高度 [build_cn2west_v2_proto_abc.py:66](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:66) [build_cn2west_v2_proto_abc.py:188](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:188)。某个叠音组合的异常竖直 metrics 会降低整个字体字号。 | **高**：所有训练样本系统性变小，模型可能把字体 metric 异常学成 style。 | 输出决定字号的 `tall_ch/max_h` 全分布；单独报告是否为 ǖ/ǘ/ǚ/ǜ。采用稳健 height-fit（例如正常 cmap glyph 的 P99）后，对超高字逐字缩小并打标；与严格 max-B 做 3–5 字体像素/特征敏感性。 |
| 宽字逐字缩小破坏族内尺度 | 循环会对任何 `w>84` 的字逐点减字号 [build_cn2west_v2_proto_abc.py:124](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:124) [build_cn2west_v2_proto_abc.py:128](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:128)。W/M/m/w、宽假名会比同族字符更矮；本来就 condensed 的设计也会被强制归一。 | **高**：身份与 style 纠缠，特别会扭曲 D3 的 a/o/p/q 结构论证。 | 每字体记录 `base_fs`、每字 `actual_fs`、缩小比例；按脚本/字符列出 shrink rate。主训练前门槛：非异常 glyph 的逐字 shrink 比例应很低；否则改为 letterbox/宽度裁决策略并做小规模 A/B。 |
| `textbbox` 不等于 ink | A/B 尺寸搜索只看逻辑框 [build_cn2west_v2_proto_abc.py:73](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:73)，F 才从阈值像素求 ink 边界 [build_cn2west_v2_proto_df.py:96](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:96)。大 bearings/vertical metrics 字体可有合规 bbox 但墨迹系统性偏小。 | **中高**：跨字体 occupancy 偏差进入 style encoder；细长字体尤其明显。 | B 保持逻辑尺度语义，但补全每字 ink bbox、coverage、四边 margin；字体级报告中位数、P5/P95 和脚本间比值。F 只作 QA 参照，不用 F 替换 B。 |
| fs=10 仍溢出会裁切 | shrink 条件是 `cur_fs > 10`；到 10 后不再验证，直接按可能超界 bbox 绘制 [build_cn2west_v2_proto_abc.py:127](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:127) [build_cn2west_v2_proto_abc.py:133](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:133)。 | **高但低频**：静默坏 GT，数量 QA 看不见。 | 循环后 assert 宽高≤inner；失败即记录 `(font,char,bbox,fs)` 并终止该字体，不允许保存。 |
| 缺字/tofu 不会被发现 | 渲染脚本没有调用 cmap 检查；只数 PNG 数量 [build_cn2west_v2_proto_abc.py:257](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:257) [build_cn2west_v2_proto_abc.py:265](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:265)。官方工具其实已有逐 cmap 检查 [utils.py:85](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/utils.py:85)。 | **阻断**：tofu 可能同时污染 target、style、α/Δ bank，而 QA 仍 `ok`。 | 渲染前用 fontTools cmap 全字符覆盖检查；再做 notdef/tofu 图像哈希聚类、空墨迹、跨字符重复率检查。任何 target/style 缺字要剔字体或版本化缺字 mask，不得静默 fallback。 |
| B summary 是隐式 bootstrap | A/B/C 的 font rows 由 B 的 `summary.json` 读取 [build_cn2west_v2_proto_abc.py:29](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:29) [build_cn2west_v2_proto_abc.py:348](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:348)；D/F 同样依赖 B [build_cn2west_v2_proto_df.py:23](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:23) [build_cn2west_v2_proto_df.py:318](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:318)；review 的筛查排序又读 B screen report [build_cn2west_v2_protocol_review.py:12](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:12)。 | **阻断可复现性**：干净 checkout 无法先建 B；旧 B 元数据可把错误 TTF/split 传给全部协议。 | font rows 直接来自 v2 split manifests + 版本化 TTF inventory；review 不依赖单协议筛查，独立生成共同 QA 表。 |
| Content 与目标各用自己的字号 | Content 的 Noto 单独在 target chars 上求 B 字号 [build_cn2west_v2_proto_abc.py:236](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:236) [build_cn2west_v2_proto_abc.py:246](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:246)，目标字体在 target+style 上求另一字号 [build_cn2west_v2_proto_abc.py:188](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:188)。 | **高**：身份支路的 content 尺度和 GT 尺度不再严格对应；若 B₀ 又以另一套范围定字号，`mix−neutral` 包含尺度差而非纯风格变化。 | 明确三类角色的“尺度契约”：Content、库同字、B₀、GT 均从同一 protocol implementation 生成；对同一 c 比较 ink centroid/height。Δ 两项必须同字符、同画布、同拟合规则；另报 Content↔GT 尺度比。 |
| 无输入 hash、输出可覆盖 | 输出目录是常量 [build_cn2west_v2_proto_abc.py:39](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:39)，保存会覆盖同名文件 [build_cn2west_v2_proto_abc.py:204](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:204)；summary 未记 TTF/charset/script Git hash，只记 Pillow 版本 [build_cn2west_v2_proto_abc.py:331](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:331)。 | **阻断 provenance**：无法证明 ckpt 用的是哪批位图，重跑可混入旧文件。 | 新数据 ID 含协议修订号，如 `...-v2b-r2-<manifest8>`；目录存在即拒绝。记录每个 TTF SHA-256、charset/split SHA-256、脚本 commit、Noto hash、整棵输出 Merkle/hash 清单。先写临时目录，QA 通过后原子发布。 |
| Pillow 只记录不锁定 | `summary.software` 仅写运行时版本 [build_cn2west_v2_proto_abc.py:324](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:324) [build_cn2west_v2_proto_abc.py:334](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:334)。 | **中**：FreeType/Pillow 变化会改变 bbox 与抗锯齿，进而改变字号和像素。 | 锁定容器/requirements、Pillow 与 FreeType 版本；保存渲染探针 hash，CI/重建时逐像素或容差比对。 |

## 3. 实际数据对比

review 数据明确列出五个数据目录 [fonts.json:4](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:4)，且五者均是 237/16/8、每字体 295 target 与 338 style [fonts.json:11](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:11) [fonts.json:84](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:84) [fonts.json:108](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:108)。下面由 `fonts[].proto_yong` 对 261 字体重算；这是“永”的 occupancy 代理，不是全字符分布，也没有可靠 touch 字段，不能夸大为全盘统计。

| 协议 | min | P5 | median | P95 | max | `永_h<0.5` | touch 证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| A | .2083 | .5938 | .7708 | .8229 | .8542 | 4 | `proto_yong` 无 touch 字段 |
| B | .2292 | .6563 | .7813 | .8333 | .8646 | 1 | 同上 |
| C | .2500 | .7083 | .9063 | .9688 | 1.0000 | 1 | 至少 1 个高度占满，触边风险 |
| D | .4896 | .6354 | .8333 | .9167 | .9167 | 1 | 无可靠全字统计 |
| F | .4792 | .6458 | .8125 | .8333 | .8333 | 1 | 无可靠全字统计 |

自动筛查本身给出 261 字体中 drop=2、review=12、ok=247 [fonts.json:2](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:2)，但排序/严重度来自 B 的 screen report，而非五协议独立 QA [build_cn2west_v2_protocol_review.py:58](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:58)。例如 FZXianZTJW 在 B 下 `永_h=.229`、跨语种色散 4.11，被判 drop；D/F 的 `永_h` 升至约 .615/.719（review 首条字体记录可见 [fonts.json:131](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:131)），说明逐字放大可掩盖字体本身极端跨脚本比例，而不是修好字体。

`proto_DF_compare/metrics.json` **仅是 3 个字体×11 个探针的 D/F 对比，不是全数据 occupancy**。例如 FZBaiZBYTJW 的 `永`：D 缩后高 .885、最小 margin 6；F 高 .833、margin 7 [metrics.json:3](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/proto_DF_compare/metrics.json:3)，而小写 `w` 分别为 .531/.469 [metrics.json:58](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/proto_DF_compare/metrics.json:58)。这支持“D 更满、F 更接近固定 ink margin”，不支持 D/F 全盘优于 B。

当前 `fonts.json` 没有 `small_stems`、真实 `n_yong_touch_edge` 或每字符 occupancy；build summary 虽定义这些字段 [build_cn2west_v2_proto_abc.py:276](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:276)，review 合并时只留下各字体的 `proto_yong` [build_cn2west_v2_protocol_review.py:98](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:98)。正式选择前必须从原 summary 或重建盘导出完整 `small_stems`、touch、空图、shrink 字符清单；不能凭 review JSON 宣称它们为零。

## 4. 与历史协议的一致性核对

历史 v2 规范是：128×128 灰度、二分搜最大字号、margin 8，PNG 转 RGB JPG q95，再由网络 `Resize(96,BILINEAR)` [HRFONT_TRAIN_PLAN_V2.md:49](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:49)。Content 指定 DejaVu，B₀=FZKTJW 且相同 `render()`@128 [HRFONT_TRAIN_PLAN_V2.md:57](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:57)。

官方历史工具的实际行为不同：`load_ttf` 用 `pygame.freetype.Font(..., size=128)` [utils.py:94](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/utils.py:94)，`ttf2im` 先得到 glyph surface，若高或宽超过 128 才用 OpenCV 等比缩小，再居中到 128 白底 [utils.py:101](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/utils.py:101) [utils.py:113](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/utils.py:113)。Dataset 加载后统一 BILINEAR resize 到训练 resolution [font_dataset.py:9](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/dataset/font_dataset.py:9)。它不是“二分搜 margin=8”，也不是 Pillow `textbbox`。

因此：

- **没有一个 A/B/C/D/F 与历史 v2 spec 完全一致。** D/F 是原生 96；A/B 是原生 96 且 margin 6；C 固定 fs=128 后 resize，但使用 Pillow `textbbox` 居中，代码自己称“pygame-style/equiv” [build_cn2west_v2_proto_abc.py:139](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:139)，实际并未复刻 pygame surface、OpenCV overflow resize、JPG q95。
- **与官方历史 pipeline 最接近的是 C，但仍不能称一致。** 与 `HRFONT_TRAIN_PLAN_V2 §2.1` 最接近的也只是“128→96”部分；真正一致者应新建一个严格复用历史 `build_retrain_v2_dataset.render` 的协议 H。
- `ft_cnstyle@25k` 被文档定义为旧 42 字体、官方接线的历史 ckpt [PROJECT.md:35](/Users/xiaoweiliang/projects/hrfont/PROJECT.md:35)，旧计划则明确说它使用 `build_retrain_v2_dataset.render@128` 和 JPG→Resize96 [HRFONT_TRAIN_PLAN_V2.md:15](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:15)。现有 provenance 是 `retro_partial` [REGISTRY.md:24](/Users/xiaoweiliang/projects/hrfont/provenance/REGISTRY.md:24)，所以应表述为“按历史文档推断”，需对原数据样本/hash 再核实，不能把 B 当已匹配。
- 若直接用 B 热启，改变了 canvas、margin、renderer、编码格式、resize/抗锯齿、Content 字体（B 是 Noto [build_cn2west_v2_proto_abc.py:236](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:236)，旧计划是 DejaVu），模型可能先适应域漂移而非学习 Δ。这样 D1 的 FT-continue 与 Stage A 因果对照只有在两臂都从同一 FT ckpt、同一 B 数据、相同步数时仍成立；但 D3 所称“渲染与 ft 相同”已经被打破，且历史 FT 行不再是完全 matched 的数据域对照。

**决策门：** 若论文优先强调热启动可归因性，首选新建 H（严格历史 render 链）；若 PI 更重视新 v2 的视觉尺度并接受对所有方法做 B-domain 适配，则可选修复后的 B-r2，但必须把 FT-continue 作为新域适配控制，不能用旧 FT 原地分数代替。

## 5. 最终建议与可执行检查单

建议选 **B-r2（有条件）**：它是五协议中“保留字体内相对尺度、避免宽 outlier、无插值”的最好折中；同时保留 **H/C 小样本分布桥接对照**。不建议 D/F 作为训练主盘，因为逐字符最大化会把待学习的真实尺度特征预先归一掉；C 的高 occupancy 与触边风险也过强。

冻结 B-r2 前逐项通过：

- [ ] 从 v2 split manifests 和 TTF inventory 建盘，不读取旧 B summary；核对 237/16/8 且互斥。
- [ ] cmap 覆盖、空墨迹、tofu/notdef hash、跨字符重复图四项均为零；异常必须显式 mask/剔除。
- [ ] 后缩验证 `bbox≤84`，任何 fs=10 仍溢出立即失败。
- [ ] 导出每字体 `base_fs/tall_ch/max_h` 与每字 `actual_fs/shrink_ratio`；重点审查 ǖǘǚǜ、W/M/m/w、宽假名。
- [ ] 导出每字 ink bbox、四边 margin、coverage、touch、脚本内与跨脚本尺度分布；人工复核 drop/review 14 字体及全部 test8。
- [ ] Content/GT/ref8/库同字/B₀ 统一 B-r2 实现；Δ 的 minuend/subtrahend 同字符同预处理。禁止离线 resize 与 loader 重复 resize/letterbox。
- [ ] 做 3–5 字体的 B-r2 vs 历史 H probe：FT ckpt 的 Ec/Es 特征漂移、旧 FT 零步 L1/ID-CLS/coverage；若明显退化，先跑 B-domain FT-continue warmup。
- [ ] 新目录只写一次；登记 TTF、Noto、charset、split、脚本、Pillow/FreeType、输出 hash；QA 后原子发布。
- [ ] 训练前 `pm_preflight`；所有下游 cache/gap/ref/eval 均校验同一 dataset fingerprint。

## 6. 两个 build 脚本代码 review 清单

### 6.1 A/B/C build

1. **bootstrap 死结（阻断）**：脚本无论建 A/B/C 都先读 B summary [build_cn2west_v2_proto_abc.py:348](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:348)，干净环境无法自举，且污染全部协议。
2. **搜索边界与 fallback（高）**：A/B 固定 `lo=8, hi=300, best=10` [build_cn2west_v2_proto_abc.py:82](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:82) [build_cn2west_v2_proto_abc.py:98](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:98)。若 fs=8 都不合格会静默返回 10（反而更大）；若 300 仍合格则不是“最大”。应先验证单调性、自动扩 hi、无解时报错。
3. **溢出静默裁切（高）**：逐字缩小止于 10 后不复验 [build_cn2west_v2_proto_abc.py:128](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:128)。
4. **缺字 QA 无效（阻断）**：只统计 glob 数量 [build_cn2west_v2_proto_abc.py:257](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:257)，无 cmap/tofu/空图验证。
5. **C 伪“官方等价”（高）**：Pillow `textbbox` 渲染被注释称 pygame-equiv [build_cn2west_v2_proto_abc.py:141](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:141)，但官方用 pygame surface 和 OpenCV overflow resize [utils.py:101](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/utils.py:101)。应更名并避免论文误述。
6. **覆写/残留（高）**：固定输出目录、`mkdir(exist_ok=True)`、逐文件覆盖 [build_cn2west_v2_proto_abc.py:183](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:183) [build_cn2west_v2_proto_abc.py:204](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:204)，不会清除旧多余文件；数量恰好也不能证明内容属于本次构建。
7. **并行结果虽排序、异常却不聚合（中）**：`as_completed` 后最终按 split/stem 排序，元数据顺序确定 [build_cn2west_v2_proto_abc.py:311](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:311) [build_cn2west_v2_proto_abc.py:320](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:320)；但首个 `fut.result()` 异常会中断，未汇总全部失败，也无临时盘回滚 [build_cn2west_v2_proto_abc.py:314](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:314)。图片渲染理论上无 RNG，仍应固定 job 清单并汇总异常。
8. **软件 provenance 不足（中）**：只记 Pillow [build_cn2west_v2_proto_abc.py:324](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:324)，未锁 FreeType、numpy、字体 hash、源码 commit。

### 6.2 D/F build

1. **同一 B bootstrap（阻断）**：font list 仍读 B summary [build_cn2west_v2_proto_df.py:318](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:318)。
2. **D 搜索上限/无解 fallback（高）**：`hi=800` 是硬上限；若无合格点则直接取 48（`max(10,canvas//2)`），不保证 bbox 合格 [build_cn2west_v2_proto_df.py:58](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:58) [build_cn2west_v2_proto_df.py:69](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:69)。若 800 仍合格也不是最大。
3. **F 对 margin 的单调性假设不稳（高）**：二分搜索假设 ink margin 随 fs 单调 [build_cn2west_v2_proto_df.py:103](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:103)，hinting/抗锯齿阈值可造成跳变；并同样截断于 hi=800。
4. **F fine-tune 可选到违反约束的字号（高）**：±2 候选按 `abs(m-target)` 排序，不再要求 `m>=target` [build_cn2west_v2_proto_df.py:116](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:116)，可能选 margin 小于目标甚至负 margin 的图。
5. **F `best_im` 路径会静默输出空白（阻断）**：无 ink/无合格点时保存纯白图且 fs=0 [build_cn2west_v2_proto_df.py:127](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:127)；数量 QA 仍通过 [build_cn2west_v2_proto_df.py:203](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:203)。
6. **居中与 ink-fit 不完全一致（中）**：F 仍按 textbbox 居中，再测 ink 四边最小 margin [build_cn2west_v2_proto_df.py:89](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:89) [build_cn2west_v2_proto_df.py:93](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:93)；有不对称 bearing/ink 时并非 ink 居中。
7. **并行/异常问题同 A/B/C（中）**：完成顺序最终排序，结果清单确定 [build_cn2west_v2_proto_df.py:272](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:272) [build_cn2west_v2_proto_df.py:281](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:281)，但异常在 `fut.result()` 处逐个抛出、无聚合 [build_cn2west_v2_proto_df.py:275](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:275)。
8. **覆写与 provenance 不足（高）**：固定目录直接写 [build_cn2west_v2_proto_df.py:32](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:32) [build_cn2west_v2_proto_df.py:159](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:159)，summary 同样只记 Pillow [build_cn2west_v2_proto_df.py:292](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_df.py:292)。
