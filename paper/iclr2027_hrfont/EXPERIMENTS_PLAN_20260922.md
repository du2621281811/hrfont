# Experiments 写作方案（PI review 稿）

日期：2026-09-22
依据：用户定点要求「设计 experiment 怎么写」；方法节以 **K6-B** 为主方法（CRP + TC，无 offset 惩罚、无中文辅助）；评测器归属按 `references/paper-method-audit.md` §7c（E12-C 归实验节）；防御性写作判据按同文 §7b；库/检索叙事按 §7d。
目标读者：PI。**本文只给方案与逻辑，不改 main.tex。**

---

## 0. 结论速览

1. 实验节按「**claim → 证据 → 表/图**」三条主张组织（形式化与评测 / CRP / TC），而不是按「我们跑了哪些臂」组织。
2. 五小节：Setup → Measures（含评测器独立小节）→ Main comparison → Contribution studies → Reference budgets & qualitative；细节全部下沉附录。
3. 受控条件（同参考集、同 bank 预算、同模型接口、同训练日程）**只在 Setup 集中写一次**，消融表每行只写变量——现文 §5 里逐行辩解句按 §7b 类目 6 处理。
4. 评测器进实验节，且「独立性口径」必须与实现一致（见 §4.3）；E12c 的已知限制写成**用途边界**，不写成辩解。
5. 现在能填的只有自动指标列；人评列、主表数字、评测器门、外部基线复现状态见 §6 待决项。

---

## 1. 论证主线（claim → 证据）

| # | 主张（与 intro 贡献对应） | 需要的证据 | 落点 |
|---|---|---|---|
| C1 | 跨书写系统字体生成可拆成结构变化与外观表达两块，且 identity 与 cross-script consistency 是**两个可分开测的轴** | 双轴主表（identity 自动指标 + 风格相容性人评）＋兼容性评分；像素指标只作诊断 | 主表 `tab:main`、§5.2 |
| C2 | CRP：库实例化经验形变空间，参考提供判定哪种实现的证据，路由在生成中确定结构偏移 | ① 结构源对照（bank-supported CRP vs 参考自身结构）② **残差形式守卫**（anchor-relative Δ vs 同库同字绝对聚合 Σα·E_c(B)）③ 组合方式（routed vs 固定先验权重）④ 第二坐标（有/无 style 编码器中间层读取）⑤ **donor 敏感性**（同 ref 换 donor：结构敏感、外观不敏感） | 消融表 `tab:ablations`、附录 donor audit |
| C3 | TC：从目标内容与源参考预测 target-aligned appearance representation | ① 空间教师 vs 仅生成损失 ② content+slot vs slot-only 查询 ③ 可训练 vs 冻结局部读取器 | 消融表 |
| C4 | 两条件互补 | shared-parent 三臂：仅结构 / 仅外观 / 完整模型 | 消融表 |
| C5 | 跨书写系统场景下优于既有方法与强基线 | 主表：条件扩散（FontDiffuser）、跨语系迁移（FTransGAN）、字体库内容融合（CF-Font）、细粒度参考注意力（FSFont）＋ reference budget 曲线 1/2/4/8 | 主表、§5.5 |

论证顺序：先定义协议与度量（5.1/5.2）→ 对外比较（5.3）→ 内部单变量归因（5.4）→ 稳健性与定性（5.5）。**归因在内、比较在外**：主表回答「有没有用」，消融回答「哪一部分有用、为什么不是库」。

---

## 2. 节结构

### 5.1 Setup and protocol
- 数据与协议：p260（228/16/16）、295 目标字符、338 中文参考、协议 A（96×96 原生、逐字体统一字号、无 resize）、clean 训练对数量；评测面板（47 字符分层、有效测试对数量）。
- 参考预算：nested 参考集 1/2/4/8，同一条 ordered manifest；**bank 预算、参考集、模型接口、训练日程在所有对照间共享，每次只改一个变量**（集中一次写完，不逐行重复）。
- 训练设置：按主臂（K6-B）真实解析值写（GPU、步数、batch、lr 分组、schedule、parent init），完整表进附录。
- 行文规则：只写事实与协议；不写「我们确保/我们仔细保证」；不使用内部臂名（Mean-Delta、dynamic combination）。

### 5.2 Measures
- identity：字符识别/ID 分类准确率（自动）；GT 作 positive control，不作天花板。
- cross-script consistency：① 人评 5 点量表（主维度=风格相容性，次维度=补全合理性；不展示 GT/方法/分数；每图 3 份有效评分）② 独立训练的跨书写系统评测器（family compatibility）。
- 诊断：L1 / SSIM / LPIPS，**仅诊断，不承载风格结论**。
- 统计：逐图配对（同参考集、同采样器、同噪声、同 seed）→ sign test / Wilcoxon + Holm 校正；报 CI；按 script 分层与字体级 win/loss 计数一并给出（避免总体均值掩盖异质性）。

### 5.2b Cross-script consistency evaluator（独立小节）
- 输入：中文参考语境（可含随机子集 k∈{2..n}）＋候选字形；输出：校准后的家族相容性概率（0.5=同族随机先验）。
- 构造：共享跨书写系统表征 + 语境条件 multi-positive SupCon + 温度校准的 membership 头；不接触方法编码器与方法输出（独立性口径见 §4.3）。
- 验证门：T1 跨脚本一致性、T2 家族判别 AUC ≥ 0.90、T3 GT 判定、T4 wrong-ref 下降；**未过门的分数不进主结论**；另报与真人评分的逐图相关性（这是「分数可不可信」的门）。
- 用途边界：用于族相容性排序与配对上升判断；不作「越像 GT 越好」的质量尺（真实 GT 的均分低于生成模型的均分这一事实写在此处，作为工具边界陈述）。

### 5.3 Main comparison
主表 + per-script 分层 + 字体级胜负统计；外部方法的参考预算、bank 访问、预训练数据、跨语系适配方式随表给出（同类信息集中一处）。

### 5.4 Contribution studies
消融表 8 行（见 §3）。每行只写「变量」与「结果」，不写「该对照保留了哪些条件」。

### 5.5 Reference budgets and qualitative analysis
- shot 曲线：同 checkpoint 在 1/2/4/8 参考下的三轴变化（现有 `reports/g_v0913_shot_k1248` 协议产物可复用）。
- 定性：普通笔画风格、装饰字体、困难跨书写实现三档；每行所有方法共享目标、参考、采样器、噪声。

---

## 3. 表格设计

**主表**（`tab:main`）

| Method | Identity ↑ | Style pref. ↑ | Family ↑ | LPIPS ↓ |
|---|---|---|---|---|
| P1（官方） / FontDiffuser / FTransGAN / CF-Font / FSFont / CRP（fixed weights） / ours | | | | |

配套：per-script 分解表（Latin / 假名 / 注音 / 变音符号）＋ 字体级 win/loss 计数；单元格为空即待跑，不预填估计值。

**消融表**（`tab:ablations`，Question | Matched comparison | Result）

| Question | Matched comparison | 备注 |
|---|---|---|
| 结构源 | bank-supported CRP vs 参考自身结构 | 对应官方 RSI 源 |
| 残差形式 | anchor-relative Δ vs 同库同字绝对聚合 | **Δ 独立声称的唯一守卫** |
| 组合方式 | routed vs 固定先验权重 | 行名即基线定义，正文不再另设专名 |
| 第二坐标 | 含/不含 style 编码器中间层读取 | 结构头参数、显存、吞吐进附录 |
| 外观监督 | 空间教师 vs 仅生成损失 | |
| 字符条件 | content+slot vs slot-only | |
| 参考读取 | 可训练 vs 冻结局部读取器 | |
| 互补性 | 仅结构 / 仅外观 / 完整 | |

**附录**：完整超参表、逐字体/逐字符分解、donor 敏感性数字、评测器细节与门结果、第二坐标的参数量与吞吐。

---

## 4. 关键写法

### 4.1 主张与证据不要错位
像素指标只回答「重建像不像」，不回答「风格对不对」。风格结论只由人评与过了门的评测器承载；两轴不同源，正文里不得互相替代。

### 4.2 归因纪律
- 「某正则/某模块去掉更好」这类话在没有 matched 对照时只能写成「该臂的设定 + 进消融表」。
- 每个消融行必须能单变量复述（改了什么、其余全部相同）。
- 采样/预算类改动（配额、drop 率）属协议级变更，需登记，不能与结构改动同行。

### 4.3 评测器独立性口径（必须先定）
现状：E12c 的拟合数据是训练域（train228 的 Style/Target 渲染），与 intro 里「independently trained cross-script evaluator」的读法可能冲突。落地前二选一：
- (a) 按实际口径诚实描述为「同域 protocol scorer」，独立性由「不接触方法编码器与输出」保证；
- (b) 补外部字体版本以支撑更强的独立声称。
推荐 (a) 先落，若 T1–T4 全过再考虑 (b) 升级。

### 4.4 非防御性写法（实验节版）
- 受控披露一次性写在 Setup：「所有对照共享参考集、bank 预算、模型接口与训练日程，每行只改一个变量。」
- 删掉现文 §5 的逐行辩解句（"This comparison preserves bank access, anchor-relative features, and the appearance interface." / "These comparisons retain the 144-token interface, reference pixels, and training schedule."）。
- 不出现否认式澄清（does not leak / never sees）；改写成设定陈述。
- 不用内部臂名当专名；基线在实验节自足定义（"the retrieval prior used as fixed weights"）。
- 不写「我们确保」「我们仔细」类自证句。

---

## 5. 与被删内容的关系
现文实验节的 **auxiliary supervision**（训练设置里的 auxiliary 段、§5 开头的三项内部对照之一、消融表行、附录相关段）与「K6-B 为主实验、辅助监督不用」的口径冲突：方案为整段撤出主文（如需要，降为附录中的历史臂记录）。**此项需 PI 确认后执行。**

---

## 6. 待决项（每项给推荐）

| # | 事项 | 推荐 |
|---|---|---|
| D-E1 | 训练设置段以哪套数值为准 | 按执行机 run 的 `config.json` / `DONE.json` 解析值重写（步数、GPU、lr、batch、schedule、parent） |
| D-E2 | 数据版本写 p260 还是 V3（v2 + v0921，315 字体全进 train、不进 bank） | 按主臂实际训练版本写，并用一句话讲清「训练域 ⊃ bank 域」 |
| D-E3 | 外部基线（FontDiffuser / FTransGAN / CF-Font / FSFont）在本协议下是否已有复现结果 | 先向执行机确认；没有则列入待跑，主表留空 |
| D-E4 | 人评是否启动（288 图 / 864 次判断，E12C-LITE 协议已就绪未采集） | 尽快启动，作为风格轴 co-primary |
| D-E5 | 评测器口径 (a)/(b) 与 T1–T4、T2 门 | 先 (a)；门结果决定 Family 列是否进主结论 |
| D-E6 | donor 敏感性 audit 是否进主文 | 进：正文一句结论 + 附录数字（直接回应「增益来自库」） |
| D-E7 | 「同库同字绝对聚合」对照是否已有臂 | 若无，优先补（Δ 的独立声称守卫） |
| D-E8 | shot 曲线是否复用现有 1/2/4/8 协议产物 | 复用，并补主臂与新臂的 shot 扫描 |

---

## 7. 风险预判与应答材料

| 审稿人可能问 | 应答材料 |
|---|---|
| 总体均值是不是掩盖了退化字体？ | per-script 分层 + 字体级 win/loss 计数进主表/附录 |
| 增益是否只是「读到了库中同字渲染」？ | Setup 的 bank 设定披露 + 残差形式对照 + donor 敏感性 + 同库对照 |
| 评测器是不是自证？ | 独立性口径 + T1–T4 门 + 与真人评分的逐图相关性 + 不用于挑候选 |
| 基线是否调到最优？ | 每个基线的参考预算、bank 访问、预训练数据、跨语系适配方式随表给出；shot 扫描 |
| 96×96 光栅的上限？ | Discussion 已写；正文只在 setup 交代分辨率契约 |

---

## 8. 未核实项（不写进正文前的必查清单）
1. 主表各行数字（含主臂 K6-B/K7-B 的固定协议推理：`DONE.json` 的 `inference_complete` 常为 false，「训练完成」≠「可入表」）。
2. 人评采集状态与样本清单。
3. 评测器门（T1–T4、T2 阈值）与独立性口径的最终选择。
4. 外部基线在本协议下的复现状态。
5. 主臂训练配置的权威解析值。
6. V3 数据版本与 bank 域的关系。
