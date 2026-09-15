# 最新 E12-b 代码与证据审核

日期：2026-09-15。审核基线：`fefab546d185a602e740ba40fe1de02c81fefcc1`。

Git 中最新一批确实是今天凌晨（北京时间）：`7156c6d46` 00:43 同步 scorer/cache/teacher 表，`33684b879` 00:59 发布权重，`03aaf1515` 01:02 补交接检查。审查覆盖实际脚本、resolved config、训练曲线、机器结果和人工评分文件，而非只读汇总文档。

## 总体判断

方向成立：独立跨语种编码器衡量字体兼容性，**cosine 为主指标，membership 为辅助**，比把概率当作唯一风格分更清楚。当前实现已经训练结束，不应继续在论文写“正在训练”。

目前更适合把它称为“跨语种字体兼容性指标”，不能由 GT 字体检索成绩直接推导“能可靠评价生成图笔触、特效”。现有实现中有可修复的数据/选模/统计问题，建议修复后复验；本次只审核，没有改写 E12 训练代码、权重或已发布原始报告，也没有占用 I1 的 GPU。

## 按优先级排列的发现

### P1 — 训练字表与缓存字表不同，且选模验证只看排序前128对

位置：`scripts/build_e12_cache_v0913_b.py::main`；`configs/e12_phi_s2_b_s3407.yaml`；`scripts/eval_framework/data.py::CrossScriptPairDataset`；`train_style_encoder.py::evaluate`。

缓存由 `ref8 + style_han_338[:96]` 组成，去重后102汉字；resolved配置另列104字，两者实际交集只有**31字**。Dataset静默丢弃缓存不存在的中文字，而没有assert完整覆盖。

验证 `range(min(len(dataset),128))` 配合按字体排序的 keys，并非均匀抽样。按Git清单重建，覆盖为前4字体各31对，第5字体4对，其余28个验证字体不参与该选模AUC。报告的 best=step500 / valAUC=.97161865 因而是这个有限子集的成绩，不是33字体完整验证集成绩。

影响：风格训练覆盖收缩；早停/选模可能过度偏向少量字体。**这里是Git配置重建证据**：V100执行机没有同步E12缓存，未声称已经读取原E12执行机的manifest。无缺图时上述数量精确成立，应在原执行机用manifest再次assert。

建议：共用一个训练字符清单并在Dataset启动时验证；先利用已保存的250/500/.../2000 checkpoints，在全部33验证字体、固定平衡CN/Latin字符上重新选模。这一步不必立即重训。随后才评估统一字表的短训练。

### P1 — 同字体不同字被对称InfoNCE当成负样本

位置：`train_style_encoder.py` 中 `a,b,_=next(iterator)` 与对角线 cross-entropy。

batch按glyph pair随机抽样，字体标签被丢弃；对角线以外一律为负。当同一字体在batch出现多次，模型会被要求推开本来应共享字体风格的跨字符特征。在31字×156字体、batch64、无放回随机batch假设下，期望每batch约**12.51对无序同字体碰撞**，不是极小概率。

建议：优先每batch每字体只取一个pair，64小于156，改动小且保持原InfoNCE；若希望同字体多字训练，改为以字体标签构建多正样本目标。比较原版与修复版时保持数据、预算和完整验证集。不能保证此一处修复就让指标符合视觉，但它消除了确定的目标冲突。

### P1 — 声称typeface-group split，缓存却强制group=文件stem

位置：`build_e12_cache_v0913_b.py` 的字体行与记录行 `group: stem`；`data.py::group_of` 优先使用该字段。

结果是 `split_by_group: true` 仍按字体文件名拆分，绕过已有字重归并。Git元数据存在明确待审的同前缀集合：`FZYouHK_513B`在train，`508R/509R/510M`在val，`512B`在test。名称足以证明当前程序没有将该系列合并，**是否属于需共同隔离的同一设计谱系还需字体metadata确认**；不能凭前缀把所有字一概判为泄漏。

影响：内部“未见typeface/设计谱系泛化”结论没有实现保证。它不等于主生成模型val/test字体文件被用于E12训练——当前cache来源确实限制在主train223。

建议：根据字体家族metadata维护显式 `lineage_id`，先决定评价的是“具体字重实例匹配”还是“同设计谱系兼容”；二者正负样本定义不同。内部split按lineage隔离，实例匹配可在split内部保留字重区分。先列跨split候选组给PI审，再建立E12-b2新版本，不覆写现有b结果。

### P1 — 人工相关性统计未正确处理并列值，且已有初报不支持强视觉一致性结论

位置：`scripts/e12_spearman_human.py::spearman/kendall_tau`。

`argsort().argsort()` 给相同1–5分分配不同秩；Kendall函数标称τ-b，实际分母仅用concordant+discordant，缺少ties修正。评分有大量并列，结果会受到不应有的排序影响。

机器文件 `human_spearman_summary.json` 已经是 `pending:false`，1位有效评分者、39个非检查样本。用相同CSV/items重算：

| 统计 | 已发布值 | 标准并列处理重算 |
|---|---:|---:|
| Spearman | −0.067409 | **−0.080750**（平均秩） |
| Kendall | −0.068729 | **−0.060910**（τ-b） |

这不是“还没收任何评分”，也不是统计意义上证明负相关：样本少、单评分者，结论是**尚未观察到正向视觉一致性支持**。汇总MD仍写pending，已经落后于JSON。

建议：使用标准带ties实现；先核对评分方向、图片/score/item_id对应、注意力检查，再按生成方法、脚本和装饰风格分层补多评分者。所有方法用同一批items重新相关。保留初报版本，不覆盖或挑掉不理想结果。

统计定义已对照官方文档：[平均秩处理](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)、[Kendall τ-b及并列值分母](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)。

### P1（接入I时）— 旧scorer固定ref8与旧目录，不能直接评分I的few-shot曲线

位置：`scripts/score_preds_e12_b.py`，固定 `REF8` 及旧预测目录；`scripts/i_eval.py` 的 `protocol.json/jobs`。

I 每张图由 `(font,cp,group,k,refs,seed)` 决定；旧scorer不读取这些信息。若直接改目录套用，它会给1-shot输出也用固定8字作评价参照，而且对缺图静默跳过、各方法summary可能不同样本数。

建议：I适配器按完整sample key读取每图真实ref，严格检查unknown字体/字符/缺失图片。主few-shot分数按实际k；如另报固定8字的family评价，明确这是**独立评分参考预算**，不可冒充生成输入预算。对方法排序采用共同有效交集并列出coverage；Latin主结果与kana/注音探索分开。当前52Latin教师训练不能自动验证其他脚本。

### P2 — 缓存指纹不足，构建入口会直接删除旧cache

位置：`build_e12_cache_v0913_b.py`。

`fonts[].sha256`实际是stem字符串的hash；`cache_sha256`只绑定字体id列表和图片总数，无法发现同数量像素变更。`OUT.exists()`直接 `shutil.rmtree(OUT)`，重跑会删除已发布cache，且没有显式overwrite开关。

建议：新版本输出目录，构建临时目录后原子发布；指纹绑定原PNG哈希、字符映射、split/group表、转换规则。图像不是96px时应报错而非静默resize。此处不影响已经导出的同一批分数的算术，但影响复现和后续结果身份。

## 正确实现与可以保留的部分

- φ为随机初始化ResNet18、512维单位向量；主cosine定义为 `(1+cos)/2`，不是校准概率。
- membership训练时φ冻结且维持eval，验证集拟合温度后再测内部test，当前这条边界清楚。
- membership正负样本共享ref与query字符的配对逻辑已修复，未发现旧标签和字体索引绑定问题重现。
- 最新权重有release与SHA；Git记录不是单纯口头“已经训练”。本次未下载/复跑该权重，因此性能数字属于已提交执行产物核对。
- 已提交内部检索：34字体、1768 Latin query，R@1=.364、R@5=.784、MRR=.547；cosine verification AUC≈.947。它们衡量GT跨脚本识别，保留在审核记录，不提前升级成I生成视觉质量结论。
- 表格caption把“Latin query → Chinese prototype gallery”写反了；论文这次按代码修正方向。LPIPS-Alex基线是trunk embedding的cosine，不能标成原生LPIPS距离。

## 最小修复/验证顺序（供批准，不在本轮自动执行）

1. CPU分钟级：修正ties、汇总与JSON同步、I评分键与coverage检查、统一字表assert、列lineage候选。复核脚本及结果见下。
2. 少量GPU验证：现有checkpoints完整val重排；本轮不抢I1八卡。
3. 若选模/目标修复仍必要：独立E12-b2，统一字表+无同字体假负+审核后lineage拆分，保留b作参照；沿用短预算，在完整val早停。
4. 对生成图的人评关联决定E12在主表中的解释力度，而不是GT AUC单独决定。论文主体故事不变，评测体系继续同时覆盖identity、视觉风格与family compatibility。

复核：`python scripts/audit_e12_review_20260915.py`；输出 [e12_audit.json](e12_audit.json)。该脚本纯CPU、读取Git产物，不改原scorer或报告。
