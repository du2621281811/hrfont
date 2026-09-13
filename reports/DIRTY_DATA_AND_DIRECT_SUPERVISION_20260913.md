# 脏数据影响与直接监督候选（PI review）

> 2026-09-14：本文数据审计仍保留；§4候选1的 Es-teacher + Delta-input 设计已由 [`G_STYLE_COMPLETION_PLAN_20260914.md`](G_STYLE_COMPLETION_PLAN_20260914.md) 的独立 TC-v2 替代，不再按旧候选开实验。

日期：2026-09-13。审计基点：`baa9b2fc`。Set-Delta 已按 PI 决定弃用；本报告不启动新训练，不修改 Group G 配置。新候选仅供 review，CGE × vector 仍 parked。

## 1. 数据影响的量级

来源：`manifests/v0913_clean.json`、pair TSV、`script_usability_map.json` 与当前 dataset 实现。旧训练枚举目标 PNG 并 shuffle，完整 228×295 下是近似均匀 pair 采样。

| 训练目标 | 原 pair 数 | 保留 | 排除 | 排除比例 |
|---|---:|---:|---:|---:|
| Latin + digits | 20,292 | 19,847 | 445 | 2.19% |
| Kana | 38,532 | 31,772 | 6,760 | 17.54% |
| Bopomofo | 8,436 | 4,810 | 3,626 | 42.98% |
| 全部 | 67,260 | 56,429 | 10,831 | 16.10% |

这是基于语种审查的排除量，不是逐字人工确认的错误率。部分 bucket 整个语种被排除，并不证明其中每个字都错。

判断：脏监督很可能是注音/假名风格不足的重要原因，尤其注音约四成的旧训练候选被标为不可用。它不是小规模随机噪声，而可能是中文 Ref 与目标 glyph 风格的系统性错配。反复训练不会自动消除此矛盾。

三条影响路径：

1. target：中文 Ref 与西文 GT 不一致，削弱学习 Ref→目标风格映射的信号；
2. donor：中文相似 donor 的目标语种若不可用，Mean-Delta 也可能输入错误结构/外观。训练 target 清洗与 donor 按目标语种过滤必须同时生效；
3. parent：F0 在线训练 Ec/Es/UNet；冻结旧 parent encoder 后，即便筛掉 target，特征仍继承旧训练影响。G0 重训后应按新 encoder SHA 重编缓存。

历史预测 metadata 未记录 donor alpha，不能把库中的排除比例直接当成实际 top-K 污染权重。

## 2. 已执行：同一旧预测，重算干净评测 mask

运行：`python3 scripts/audit_v0913_dirty_impact.py --output reports/DIRTY_DATA_IMPACT_20260913.json`。

只读已有 `metrics_items.json` 并按 `pairs_eval47.tsv` 重新聚合；未重新生成、未重训、未修改旧指标。每个方法校验 752 个唯一 key 完整，筛后 704；排除 48（21 Kana、27 Bopomofo）。

| 比较（前者减后者） | 旧752 L1差 | 有效704 L1差 | 旧752 LPIPS差 | 有效704 LPIPS差 |
|---|---:|---:|---:|---:|
| F2-P@40k − F2@40k | +0.001207 | +0.001195 | +0.012572 | +0.014139 |
| F2-VEC@40k − F2@40k | -0.001687 | -0.001886 | +0.002183 | +0.002311 |
| F2@80k − F1@80k | +0.000619 | +0.000528 | -0.004629 | -0.004837 |

L1/LPIPS 越低越好。VEC 行只用于 mask 算术比较，其执行机/训练 lineage 不应被据此称为完全 matched。没有对这些描述统计宣称显著性。

结论：坏 GT 会改变数字，但没有消除上述差距。F2-P 相对 F2 的 L1 差距几乎原样保留，LPIPS 差距更大。可以排除“主要只是坏评测 GT 造成差距”的解释；仍不能由此确定清洗训练数据会提升多少。

补充：Latin 测试的592项一个都未被 mask 排除。该子集 F2-P 相对 F2 的 L1仍差 +0.001054、LPIPS仍差 +0.016218。坏 Kana/Bopomofo GT 不能直接解释此比较；训练期共享参数的间接影响仍可能存在。

## 3. 新 clean 实验同时改变的因素

- 过滤 pair；
- 按目标语种过滤 donor；
- 采样比例从约 Latin/Kana/Bopomofo = 30.17/57.29/12.54 改为 50/38/12；Latin 抽样占比提高约65.7%；
- Group G 的 G0 从 P1 起，global batch=256、lr=3.2e-4、10k horizon，之后 G2/G1/G2-PRL 10k。与旧 F 系的小 batch、parent和步数不同。

因此旧 F2 vs 新 G2 是整套训练方案效果比较，不是纯粹清洗的因果估计。现有预算下优先让 G0→G2→G1/G2-PRL 完成，观察 clean 方法效果；无需为了追溯百分比立刻加多臂实验。

训练的 pair filter、WeightedRandomSampler 与 `_v0913_donor_exclude` 已进入代码。旧 `eval_f03_test16_strat.py` 的既有方法路径仍用 train228 库配置；未来接 G2 时需要显式接新 cache、donor mask 与704评测 mask，不能只改 checkpoint 路径。

缓存规则：同 PNG、同 encoder、只过滤 pair 时可复用；更换 G0 encoder 后必须重编。这两种情况不可混写。

## 4. 新 idea 的筛选标准

上一轮的响应/共享采样重视机制验证；本轮优先真实 GT、已有网络、短训练、可关闭的新模块。没有实测前不给成功概率百分比。

### 候选1：目标字风格补全（主要贡献候选）

叙事：Mean-Delta 提供目标字形的 bank 先验，中文 refs 给出已知风格，小模块把两者转成“当前目标字应有的风格条件”。

实现：

1. 在 clean train pairs 上用冻结的新 G0 Es 编码真实目标字，得到 teacher `z*=pool(Es(y_f,c))`；可先只用 pooled 1024维，后续加入中层 mean/std。teacher 不需要额外训练。
2. 小型共享 MLP/attention head `H` 输入现有 Ref tokens、Content特征与 pooled Mean-Delta，预测 `z_hat_c`。不使用 font-ID lookup。
3. 先冻结 generator，在缓存上训练 `L_code = SmoothL1(z_hat_c,z*)`，使用 train 统计归一化；比较直接 Ref mean、线性预测器和该 head，在 val 字体上判断可预测性。不能把特征误差改善直接当成图像改善。接入 generator 后，再用只接受 L_base、不接受 L_code 的等容量 head 作图像效果对照。
4. 通过可关闭的零初始化残差投影将预测条件加入现有 style tokens，再与 generator 的 style attention 做短微调。保持原 Ref 通道。
5. 总损失 `L_base + lambda_code L_code`，其中 L_base 保持现有 diffusion/VGG/offset。

训练有可信目标字，推理只有 Ref、Content、Mean-Delta。Es(y) 不是“纯风格真值”，但对同一目标字符的直接 feature supervision 避免了拿中文/西文整图强行对齐。先验证 pooled teacher 是否能区分我们关心的字体效果。

代价：约5.6万条 pooled FP16目标特征约0.12GB（十进制，不含索引）。无需新增整库 Ec，也不用新 diffusion teacher。估计1–2天工程准备；缓存训练先给信号，再决定是否使用G2分叉的2k–5k增量pilot。baseline同样续训同样步数，不把增量训练当成免费收益。

贡献候选：跨语种缺失目标的目标字风格条件补全。蒸馏、MLP本身不新；需要证明相对直接Ref conditioning与等容量head的增量。与CGE相比，它直接使用训练GT，不依赖E12奖励或可解释轴。

### 候选2：参考条件的残差精修（直接改善笔触的备选）

冻结clean G2，离线生成训练字体预测，构造 `(base_pred, Ref, Content, GT)`。独立小CNN从预测和Ref局部特征估计图像残差，最终 `y_final = clip(y_base + r_theta(y_base,R,C))`。输出层零初始化，初始即基线。

训练：前景/邻近边界加权L1 + 小系数Sobel或Laplacian多尺度细节损失；需要时复用VGG损失。背景仍保留损失，不能随意抹掉阴影或飞白。监督直接来自同字符GT，不用合成干预或E12。

先用48个train字体×32目标字=1536个合法训练样本；所有donor遵循leave-one-out和语种mask。只在独立val上判断泛化，不能用test预测训练head。样本要来自实际推理输出，不能完全用teacher-forced x0hat替代；后者与真实推理误差有分布差异。

收益范围：轮廓修补、局部粗细和细节；大幅结构错误应由G2修。对照包括等容量不看Ref的refiner，检验是否只是通用锐化。已有多阶段字体细化研究，这条工程成功链短，但独立创新性弱于候选1。

代价：一次离线生成是主要成本，训练小头与推理一次CNN相对便宜；先量实测吞吐，不承诺固定小时数。

### 效果基线：同字符的局部细节监督（最省时间，不凑独立贡献）

当前F2/F2-PRL loss为noise MSE +0.01 VGG feature MSE +0.5 offset；当前dataset以scr=False使用。不能说模型完全没有风格梯度，但没有专门的局部笔触目标。

复用训练已算出的x0hat，加入温和的前景/边缘加权重建与多尺度gradient loss；高噪声x0hat不稳定，对这类辅助loss做SNR门控/限权。只增加少量图像运算，无新encoder、无新监督数据、无离线生成。

先将它作为候选1的强baseline或极短增量pilot。普通style contrastive/精修是既有工作，不单独包装成第三贡献。

## 5. 建议顺序与相关工作

1. 跑完当前clean G2及G1/G2-PRL比较，使用统一704 mask与分语种结果。
2. 若结构已稳定但细节弱，先加局部细节监督；这是最低成本的效果基线。
3. 若要一个能接入三贡献叙事的新机制，优先目标字风格补全：先缓存监督，再短微调。
4. 独立残差精修作为备选，不同时开所有方向。Set-Delta不再进入任何排程。

- [FontDiffuser](https://arxiv.org/abs/2312.12142) 已有style contrastive refinement；恢复类似监督是增强基线，不是首创。
- [Generate Like Experts](https://openaccess.thecvf.com/content/CVPR2024/html/Fu_Generate_Like_Experts_Multi-Stage_Font_Generation_by_Incorporating_Font_Transfer_CVPR_2024_paper.html) 已采用多阶段字体生成与细化；候选2不能泛称首次粗到细。
- [HFH-Font](https://arxiv.org/abs/2410.06488) 已有蒸馏与style-guided super-resolution；候选1关注预测缺失目标字的style condition，不能泛称首次字体蒸馏。

以上是定向相关工作核查后的候选，不是已完成新颖性证明，也不承诺某一模块一定改善图像。
