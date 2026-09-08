# E12 v5.1 实验计划与执行规格

日期：2026-09-08 ｜ 状态：review 后执行版。本文是 E12 v5.1 的当前执行入口；与旧优化建议冲突时以本文为准。
代码统一放在 `scripts/eval_framework/`。本次已修复现有 membership 的数据配对、校准分集与指标计算；下面的 SupCon、四折评测和新 backbone probe 仍需执行机实现。

## 0. 实验目标与默认选择

E12 回答「候选字形在 CN reference 语境下是否是合理的家族成员」，主输入为 `(R, P)`，GT 不进入主打分。保持单 seed=3407、外部字型池全量利用、family-aware multi-positive SupCon、membership 概率输出与人评终审。先做 ResNet18 主线，其他骨干作为便宜候选；不等待 CoAtNet 权重即可推进。

训练、选模和温度校准只使用外部字体；A/260、生成方法输出及 Es/Ec 不参与这些环节。最终部署使用一个全池重训 encoder。四折中的模型是测量训练流程的临时模型，不是多 seed ensemble。

## 1. 数据与 episode

- 外部池由 26 扩到 ≥60 个 lineage 去重字型；复用 fetch/render 管线。每族记录 lineage、字体 SHA、来源与许可。
- 渲染 target295：10 digits、52 letters、27 latin_ext、83 hiragana、86 katakana、37 bopomofo；字符列表读取 `manifests/charset_cn2west_v2_planned.json`。
- 96×96、L 模式、per_font_height_fit、margin 6、不 resize。逐字符检查 cmap 与实际渲染，覆盖矩阵进入 manifest；各 script 使用实际覆盖子池，报告族数/字符数。CN ref8 必须完整。缺字不作负例或零图代替。
- cache：`artifacts/e12/external_font_cache/`，锁定 manifest、覆盖矩阵、渲染代码和 payload SHA。
- R 从同族 CN 字池抽样，训练 k 从 2 到该族可用 CN 字数；主评测固定 `永和书风骨韵天地`。全池 ref 只作单列探针。
- 正例为同族跨字符/跨脚本字形，负例为不同 lineage。每个 `(ref_family, query_char)` 同时构造正负 episode；负例保持候选字符不变，各族/字符均衡遍历。同族另一字形可作训练正锚，首轮不启用额外 GT 正则。

## 2. 单 seed、四折测量、全池部署

采用 lineage 分组 **4-fold**，seed=3407；≥60 族时每个外层测试折约 ≥15 族。按 script 覆盖/风格类别尽量分层，先保存 split manifest。用四折代替逐族重训，控制计算量。

每个外层折：
1. 外层测试族不进入该折 encoder、聚合器或 head 的训练。冻结通用预训练骨干可直接提特征；所有数据适配参数只在该折训练族学习。
2. 外层训练族再划分 fit/validation/calibration，按 lineage 隔离。validation 做候选筛查和 checkpoint 选择；calibration 只拟合 temperature，不能与外层测试合并。
3. ResNet18 是主候选；B/C/D 若参加，候选选择仅看该折内部 validation。外层测试只用于评价已确定的选择流程，不用其分数反向挑骨干。报告每折选中的配置和 out-of-fold 原始分数。
4. 聚合四折得到 macro-family AUC。每族内正例与不同 lineage 的 wrong-ref 配对；使用全部合法 wrong-ref，先在每个 query 内等权，避免候选数多的字符获得额外权重。再对字体族等权求均值。
5. 以族为 cluster 做 2000 次 bootstrap，seed=3407，报告 percentile 95% CI。这是字体抽样不确定性，不称为训练 seed 方差。

流程通过后，按预注册规则（各折内部 validation 选中次数最多，平局优先 R18）确定部署架构；在全池重训一个模型。部署 temperature 由对应架构的 out-of-fold calibration 预测拟合，明确记录这是 OOF 温度迁移；另报告其在各折独立外层测试上的校准质量，不把全池拟合分数当作泛化分数。

## 3. Backbone 与训练

| 路 | 配置 | 执行顺序 |
|---|---|---|
| A | ResNet18 从零训练 | 主线先实现 |
| B | frozen CLIP-ViT-B/16 + probe | 数据与 episode 相同 |
| C | frozen DINOv2 + probe | 数据与 episode 相同 |
| D | frozen CoAtNet-font + probe | 权重许可与 A/260 overlap 核验完成后加入；未就绪记 unavailable |

96×96 渲染不变。预训练骨干若需要 RGB、归一化、patch padding 或其他尺寸适配，在配置中显式记录；不要静默改变渲染数据。probe 时长先测实际吞吐，不将 15 分钟视为完整训练预算。

- z = L2-normalize(f(image))；R 默认 mean pool 后归一化。attention pool 是内层 validation 候选。
- logit `l(P|R)=z_P^T W z_R / 0.07`，主概率 `s=sigmoid(l/T)`；temperature T 只在 calibration 拟合。
- 首轮两阶段：encoder + bilinear head 用 family-aware multi-positive SupCon 训练；随后冻结 encoder（含 BatchNorm 状态），训练 membership head，最后做温度校准。联合 SupCon+BCE 作为后续候选，不与首轮混用。
- encoder：AdamW lr3e-4/wd.01/bs64/50ep/cosine/5ep warmup。membership：lr1e-4/bs32/30ep。device:auto，seed3407。
- 单模型预估 1–3h；四折加全池重训按约 5 次主模型预算安排。首夜完成数据、首折和吞吐记录；四折按实际吞吐继续，无须增加 seed。

## 4. 指标与验收定义

| 项 | 定义 | 判据/用途 |
|---|---|---|
| T1 跨脚本 embedding 一致性 | CN ref8 平均 embedding 归一化后，与同族 Latin 候选的 cosine 中位数 | ≥0.60，沿用旧公式；0.644 不是旧门限 |
| T2 主 membership 判别 | 外层 OOF logit 的 macro-family ROC-AUC | ≥0.90，族 bootstrap 95% CI 下界 ≥0.85 |
| T3 字符识别 | 独立 ID-CLS 在 held-out 字体上的 top-1 accuracy | ≥0.90；旧 0.977 是 62 类识别准确率，不是 membership AUC |
| T4 wrong-ref | 每个 query 的 `s(P|R_true)-mean_wrong s(P|R_wrong)`，先按族平均 | mean drop≥0.02，族 bootstrap 95% CI 下界>0 |
| C1 概率校准 | 外层测试上的 Brier、ECE(10 bins)、NLL、reliability curve，缩放前后同时报告 | 描述概率质量，不复用 T3 编号或把 AUC 称为校准 |

T1/T2/T4 分 script 报告，同时固定报告 legacy Latin/digits 子集便于对照。T3 的 62 类历史结果单列；295 类模型须重新训练与评测，报告覆盖范围。主门使用上述预先定义的公式，不将 v4 cosine T2 与新 membership T2 当作同一分数的提升。

v4 历史参考：cosine T2=0.791/0.795/0.725；62 类 ID T3=0.977。v5.1 的主要成功标准是新 membership T2 及配套 T1/T3/T4，而非复刻旧分数。

## 5. 输出与主实验使用

- 主输出：8-ref 条件下 membership 概率；同时保存未缩放 logit、font、char、script、fold、checkpoint SHA。
- SC-R/SC-Gap 单列为 embedding 距离诊断，独立报告 cosine 判别效度，不从 bilinear membership 过门推断距离也有效。
- GT 正对照、异族负对照同框；GT 融合分只作附加分析，默认关闭，不能取代主分数。若启用，w 仅在外部 calibration 冻结。
- 方法比较用同一批 font/char/noise，按 font 做 paired bootstrap。生成域的家族匹配、可读性和笔触可用性由盲化人评补充。
- E12 未完成验收时继续训练/看图，自动分数注明诊断；通过后进入正式主表。人评保留最终判断权。
- VGG19 笔触特征和 skeleton 先作诊断；拓扑选字共享的 skeleton 实现不能单独证明 F3b 改善了感知风格。

## 6. 执行交付

依次交付：M1 数据覆盖与 lineage manifest → M2 首折 R18/候选 probe 和实际时长 → M3 四折 OOF 分数与 T1/T2/T3/T4/C1 → M4 全池部署模型及 SHA → M5 F0/F2/F3/F3b 固定样本面板与人评。

本次代码修复：`train_membership.py` 将 val 校准与 test 报告分离，冻结 encoder 的 BN；小池 smoke 明示 `smoke_overlap`，正式运行不借用 test。`data.py` 修复偶数字体池的 family-label 绑定。`train_utils.py` 正确处理 AUC/AP 的同分样本。旧 self_tests.py 仍是 legacy gate，执行机按本文新增 v5.1 入口，保留旧入口用于复现。
