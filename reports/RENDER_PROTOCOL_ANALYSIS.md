# cn2west v2 渲染协议分析与代码审计

**版本：** v2（2026-09-04）
**结论：** 按 PI 决策 D-A，**A 协议是唯一暂定训练/评测协议，并增加按字体平均 ink 比例排序的人工审查门**；B-r2/H 选择树作废。A 已完成 237/16/8、每字体 295 target + 338 style [fonts.json:12](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:12) [fonts.json:16](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:16)。

## 1. 决策与协议冻结

A 的冻结语义：96×96 原生画布、逐字体统一字号、该字体全部 633 字符的 `textbbox` 最大宽和高均≤84、margin 6、RGB PNG、无缩放。实现注释/常量见 [build_cn2west_v2_proto_abc.py:4](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:4) [build_cn2west_v2_proto_abc.py:34](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:34)，字号由 `find_size_A` 搜索 [build_cn2west_v2_proto_abc.py:82](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:82)，渲染居中并输出 RGB [build_cn2west_v2_proto_abc.py:117](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:117) [build_cn2west_v2_proto_abc.py:133](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:133)。

所有 FT-v2、Stage A/B、SCR 变体必须消费同一 A dataset fingerprint；loader 只做 `ToTensor → Normalize([0.5],[0.5])`。官方 loader 当前含 BILINEAR Resize [train.py:97](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:97) [train.py:107](/Users/xiaoweiliang/projects/hrfont/code/official/FontDiffuser/train.py:107)，variant 必须禁用并由 preflight 验证。

现名 `p253` 与实际 237+16+8=**261** 不一致：manifest 仍写 p253 [charset_cn2west_v2_planned.json:3](/Users/xiaoweiliang/projects/hrfont/manifests/charset_cn2west_v2_planned.json:3)，review 实际为 261 [fonts.json:133](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:133)。新 ID 用 `fontdiffuser-p261-t295-s338-cn2west-v2a-r1-<manifest8>`；旧目录只读，以 parent hash 关联。

## 2. A 协议遗留风险与处置

| 风险 | 代码证据与影响 | ink 门覆盖 | 必须动作 |
|---|---|---|---|
| outlier 全局缩字 | A 在全部字符取最大 bbox [build_cn2west_v2_proto_abc.py:66](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:66) [build_cn2west_v2_proto_abc.py:188](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:188)。 | **部分**：低 mean/p5 会进尾部，不能定位肇事字。 | 必出 `fs_A,tall_ch,max_h,wide_ch,max_w`，审 ǖǘǚǜ、W/M/m/w、宽假名。 |
| `textbbox≠ink` | 搜索只看逻辑框 [build_cn2west_v2_proto_abc.py:73](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:73)，当前 ink 仅测“永” [build_cn2west_v2_proto_abc.py:158](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:158)。 | **主要覆盖**。 | 全字符 ink bbox；另报 ink 像素率区分细体与缩小。 |
| `hi=300/best=10` | 固定 `lo=8,hi=300,best=10` [build_cn2west_v2_proto_abc.py:84](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:84)，无解会错误回 10，上限合规则非最大。 | **部分**。 | 指数扩 hi；验证 fs=8；无解 fail closed；记录终止原因。 |
| fs=10 裁切 | 循环到 10 后不复验 [build_cn2west_v2_proto_abc.py:124](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:124) [build_cn2west_v2_proto_abc.py:128](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:128)。 | **部分**：touch 可发现部分。 | 保存前 assert `w,h≤84`，失败阻断发布。 |
| cmap/tofu/空图 | QA 只数文件 [build_cn2west_v2_proto_abc.py:257](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:257) [build_cn2west_v2_proto_abc.py:265](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:265)。 | **不充分**。 | cmap 全覆盖、notdef/tofu hash、空墨迹、跨字符重复率。 |
| bootstrap | font rows 读旧 B summary [build_cn2west_v2_proto_abc.py:31](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:31) [build_cn2west_v2_proto_abc.py:348](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:348)。 | **不覆盖**。 | 从 split manifests + 版本化 TTF inventory 自举，验证 237/16/8 互斥。 |
| 覆写/残留 | 固定目录 [build_cn2west_v2_proto_abc.py:39](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:39)，直接 `exist_ok` 写入 [build_cn2west_v2_proto_abc.py:183](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:183) [build_cn2west_v2_proto_abc.py:204](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:204)。 | **不覆盖**。 | 新 ID 只写一次；临时目录→QA→原子发布；存在即拒绝。 |
| provenance | summary 只记 Pillow [build_cn2west_v2_proto_abc.py:324](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:324) [build_cn2west_v2_proto_abc.py:331](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:331)。 | **不覆盖**。 | 记录 TTF/Noto/charset/split/script/FreeType/文件树 SHA-256 与审查结论。 |

ink 门是**异常发现和人工批准门**，不是正确性替代品；“部分/不覆盖”项在 p261 发布前必须修复或 fail closed。

## 3. A 协议下的 ink 比例审查门（规格）

### 3.1 精确定义与 probe

固定阈值 `T=250`。对字体 (f)、字符 (c) 的 96×96 灰度图 (I_{f,c})，令 (S_{f,c}=\{(x,y):I_{f,c}(x,y)<250\})。若为空，定义 (r_{f,c}=0) 并标记 empty；否则取最小轴对齐 ink bbox ([x_0,x_1]\times[y_0,y_1])：

\[
r_{f,c}=\frac{(x_1-x_0+1)(y_1-y_0+1)}{96^2},\qquad
\bar r_f=\frac{1}{633}\sum_{c\in C_{295}\cup R_{338}}r_{f,c}.
\]

主排序选 **ink bbox 面积/画布面积**：它直接检测几何占画布量，对合法笔画粗细较不敏感。次级报告 `mean_ink_pixel_ratio=mean(|S|/96²)`；小画布下该量对细体/粗体、hinting 和抗锯齿很敏感，作为主门会误伤合法细体。`T=250` 与现有 `ink_yong` 一致 [build_cn2west_v2_proto_abc.py:159](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_proto_abc.py:159)。

**推荐 probe 为全部 633 字符。** 261×633=165,213 张图已在盘，读取成本可接受；全量使排序和 p5/p50/p95 稳定，避免代表集脚本偏置。`ref8 + 26 大写字母 + 数字 + 平/片假名代表` 只作 HTML 快速首屏；未来未渲染数据可用它预筛，但发布门仍跑全量。

### 3.2 输出与阈值冻结

输出 `ink_ratio_rank.csv/json`，按 `mean_ink_ratio ASC,stem ASC` 稳定排序。字段：`rank,stem,split,mean_ink_ratio,ink_ratio_p5/p50/p95,mean_ink_pixel_ratio,fs_A,tall_ch,max_h,wide_ch,max_w,touch_edge,n_empty,auto_screen_severity,auto_screen_reasons,review_decision,reviewer,reviewed_at`。`touch_edge` 表示任一字符 ink bbox 触四边。旧 severity 来自 B screen [build_cn2west_v2_protocol_review.py:58](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:58)，只作对照。HTML 集成 `render_qa_hub`，默认显示低 ratio 尾部，可展开全类别、bbox/touch/tofu 提示。

阈值只在 train237 预注册 calibration 字体上定，不看 val16/test8 模型效果：

1. 初始人工候选为 `mean_ink_ratio < calibration P5` 或绝对值 `<0.20`，以及任意 touch/empty/tofu；这些是校准起点，不是已冻结事实。
2. 初审后以“确认墨迹过小”为标签，选择召回≥0.95且审查量最小的 review 阈值。直接剔除起点 `<0.12`，只有该区间校准样本零误剔才启用；否则不自动 drop。
3. 将分位数/绝对阈值/T/charset hash/calibration stems/reviewer protocol 写入 `ink_gate_calibration.json` 并取 SHA-256；冻结后 val/test 只应用不重调。

### 3.3 人工审查工作流

- 数据负责人初审全部候选；字体/视觉负责人复核全部 `drop/rerender` 和随机 10% `pass`；分歧由 PI/第三人裁决。
- 检查墨迹过小、触边/裁切、tofu/空图/重复图、跨脚本比例异常，以及 `tall_ch/max_h/fs_A` 是否显示单个 outlier 全局缩字。
- 结论仅三类：`pass`；`drop`；`rerender`（修复后必须新 dataset ID，不能覆盖）。
- 回写 append-only `ink_review_decisions.jsonl`：dataset SHA、stem、决策、reason code、reviewer、时间、证据 hash。provenance 记录审查/校准 SHA；manifest 只列 pass，并列 `excluded_fonts[]/rerender_parent_id`，重算 split 与树 hash。

## 4. 数据现状与旧结论处置

旧自动筛查给出 2 drop、12 review、247 ok [fonts.json:133](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:133)，但阈值基于“永”高度/跨语色散 [fonts.json:139](/Users/xiaoweiliang/projects/hrfont/data/cn2west_v2_abc_review/fonts.json:139)，builder 又读取 B report [build_cn2west_v2_protocol_review.py:14](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:14)。这 14 个标签只作新表对照和优先复核，不是 A 的最终结论。

历史 128→96、DejaVu Content、B₀=FZKTJW 设计 [HRFONT_TRAIN_PLAN_V2.md:53](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:53) [HRFONT_TRAIN_PLAN_V2.md:57](/Users/xiaoweiliang/projects/hrfont/reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md:57) 已被 D-A/D-FT 对新实验覆盖。旧 `ft_cnstyle@25k` 仅 legacy 参考，不作初始化或 matched 证据。

## 5. 最终建议与发布检查单

**执行 A + ink-ratio 人工审查门；B-r2/H 不再作为候选、桥接或训练依赖。**

- [ ] 新 ID 为 p261；split 237/16/8 互斥且 hash 固定。
- [ ] 633 字符主 bbox ratio + 次级 pixel ratio；CSV/JSON/HTML hash 一致。
- [ ] calibration 阈值冻结；尾部、touch、empty、tofu 和旧筛查异常完成双人流程。
- [ ] cmap/tofu/空图/重复图为零或显式 drop；`bbox≤84` 后验 assert 全过。
- [ ] `fs_A,tall_ch,max_h,wide_ch,max_w` 齐全；搜索无 silent fallback。
- [ ] Content/Style/ref8/Target/GT/Δ/B₀ 同一 A fingerprint；网络无 resize。
- [ ] 新目录只写一次；字体、manifest、脚本、软件、文件树及审查 hash 完整。
- [ ] 训练前 preflight。现脚本检查 Git clean、marker、variant 与 run/provenance ID 唯一 [pm_preflight.py:41](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:41) [pm_preflight.py:73](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:73) [pm_preflight.py:84](/Users/xiaoweiliang/projects/hrfont/scripts/pm_preflight.py:84)；计划另强制 config/dataset SHA。

## 6. 实现审计结论

当前 A 盘可执行 ink 排序和人工审查；正式 p261 发布前仍须解决 B-summary bootstrap、搜索边界/fallback、fs=10 后验、cmap/tofu、只计数 QA、覆写和 provenance。review builder 应以 A 的 `ink_ratio_rank.json` 为主数据，旧 B severity 仅对照，并把当前单一“丢弃”复选框 [build_cn2west_v2_protocol_review.py:265](/Users/xiaoweiliang/projects/hrfont/scripts/build_cn2west_v2_protocol_review.py:265) 扩展为 `pass/drop/rerender` 与签名导出。
