# 论文改稿说明：三贡献统一

## 本轮主线

从少量中文参考扩展一个跨语种字体家族：Mean-Delta 提供目标字的可变化先验，TC 学习参考外观在目标字上的表达，任务与评测回答字对不对、风格像不像、跨语种是否属于同一家族。

## 改了什么、为什么改

| 部分 | 修改 | 原因 |
|---|---|---|
| 标题 | Completing Font Families across Scripts | 先交代读者得到的能力，再在摘要介绍机制 |
| 摘要 | 问题→两个条件→生成框架→评测；数值暂空 | 不以已弃用组件定义工作，不把训练完成写成效果胜出 |
| 引言 | 从设计师扩展字体的真实需求出发；三项贡献 | 让“变化先验”和“目标外观补全”分别解决清晰的问题 |
| 任务与评测 | identity / reference consistency / family compatibility；script-aware清洗 | 从单个GT重建拓展到跨语种兼容性，同时保留可重复的定量协议 |
| Mean-Delta | Ec多尺度同字符加权残差，明确中性锚点和alpha作用 | 对齐当前代码；alpha是邻域先验，不用中文相似度承诺西文精确对应 |
| TC-v2 | 896维VGG统计、707维content、256宽8头、SmoothL1、共享1024维残差 | 使实现和训练目标能够被复现；独立内容预测直接连接第三贡献 |
| 相关工作 | 聚焦跨语种生成、basis-font内容融合、reference attention、外观统计 | 正面说明方法位置，不把已有attention或统计描述器本身包装成新贡献 |
| 实验 | 对应贡献组织问题；两个parent续训控制、目标query/监督控制 | 先验证图像收益，再补清楚收益来自哪里 |
| 数值与空缺 | 去掉旧E12表格；当前主表、消融、shot表保留空值 | 新自动化指标还在训练；旧口径不进入新主张 |
| 图示 | LaTeX可编辑公式结构图与空白qualitative布局 | 保证准确、可编辑；历史AI/绘图资产不继续传达旧方案 |
| 附录 | 单一seed、训练配置、参考字符、选择规则 | 正文聚焦贡献，把工程细节放在易查位置 |

## 学术表达策略

- 介绍工作带来的能力、输入输出、设计理由和验证方式。
- 不以“无support/非Set-Delta/非Es依赖”等否定标签命名方法。
- Mean-Delta和TC通过功能与监督目标区分，不依赖完美解耦或latent可解释性才能成立。
- 使用中性可检验语言；没有结果的地方不写优于、显著或SOTA。
- TC与生成的联系写为方法设计；是否改善以匹配对照的输出为准。
- 模块输入边界保留在公式和实现细节中，不写成贯穿全文的防御段落。

## 新旧材料的关系

本轮 active files：main.tex、figures/method_overview.tex、figures/qualitative_layout.tex、README、RESULTS_TODO、PAPER_ISSUE_TRACKER、EXPERIMENT_EXECUTION_PLAN。
旧 Set-Delta / Graphics-Ref 的图像和生成脚本仍在 Git 中保存，但当前 main.tex 不加载。旧稿可从 Git 历史取回。
本轮不变更训练网络或执行机未提交的推理/矢量代码。

## 编译与审校

已用现有 pdfLaTeX / latexmk 编译为9页PDF（含参考文献和附录），逐页渲染检查公式、表格、标题、占位图和引用；修复了贡献列表引导句跨页孤立的问题。最终日志无undefined citation/reference和overfull box。结果表为空是本轮要求。
接续队列8项单元检查通过；默认dry-run、py_compile通过；执行机launcher/train SHA与本地一致。清洗计数重新核验：train56429、test47合法704。这里只声称脚本与排程检查，不声称新CONT已经完成。
新架构图采用可编辑LaTeX，未调用生成式图片模型；图内没有伪造实验输出。
论文写作skill用于统一叙事和证据顺序；训练skill用于匹配parent/batch/learning rate与短验证阶段。通用训练配方不覆盖项目已经验证的V100设置。

## 文献核验

保留且本轮复核的核心文献：FontDiffuser（arXiv:2312.12142）、CF-Font（arXiv:2303.14017，作者仓库）、FSFont（arXiv:2205.09965，CVF）、FTransGAN（WACV2021/CVF）。
新增基础外观表示引用从作者/官方页面取得，保留检索来源于 references.bib 注释。

## 待 PI review

新主标题和三贡献表述；主ref组；TC最终底座；消融预算；评测器和人评细节。完整实验决策见 reports/G_NEXT_8V100_PLAN_20260914.md。
