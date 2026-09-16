# K 系列设计 Review 与论文叙事影响

日期：2026-09-16  
关联执行规格：`K_EXPERIMENT_PLAN_20260916.md`  
状态：设计 review。K 系列采用效果优先路线；原 I7 proposal/review 保留历史，不覆盖。

## 0. 总评

K 系列的核心方向合理，而且比原 I7 更适合当前阶段：先让模型在最终图像上明显恢复目标字体的笔端、断笔、空心、装饰和局部连接，再补最少量必要的 contribution study。

现有证据已经足够支持一个主要工作假设：模型并非完全没有 reference 信息，而是当前 objective + injection interface 允许一个低风险 shortcut。neutral content 和 global style 足以把字符身份、粗细、宽窄、整体倾斜等大面积属性做对；TC/local branch 即使学到了局部风格，也可以被生成器通过自由 `local_gain` 压到接近零。因此训练 loss 可以收敛，但最终生成仍缺少真正决定字体身份的高风险局部设计。

K1 的设计不是再造第四个方法，而是堵住这个 shortcut：

1. TC 从“可选残差”改为固定非零、幅度归一的生成条件；
2. 输出监督把优化优先级转向 `content -> GT` 真正需要改变的位置；
3. sampler 显著提高复杂风格与 same-content/different-font 的曝光，降低只靠 content 解题的吸引力。

Dynamic Delta 本轮不改架构，这一点对方法叙事和论文稳定性都很重要。

---

## 1. 为什么不是“监督信号没进 loss”

更准确的说法是：**真正关心的监督在总优化里的有效话语权不足，而且存在绕过路径。**

当前困难包含三个层次：

### 1.1 content shortcut

neutral content 已给出字符身份和基础轮廓。只做字重、宽窄、倾斜、边缘平滑等低频修改，就能改善 diffusion/perceptual/reconstruction objective 的大部分区域。

笔端、断笔、空心、装饰、局部连接往往只占很小面积。做对它们的收益相对有限，做错却可能增加像素误差，因此模型倾向保守。

### 1.2 TC consumption shortcut

I 系当前生成接口为：

`global attention + learned_gain * local TC attention`

TC readout supervision 不经过该 gain。因此 Reader 可以获得梯度、降低自身误差，而生成器同时把 TC contribution 压小。I5 的冻结权重干预已经说明 local TC 对完整生成的实际影响很弱，这比只看 `local_gain` 数值更有说服力。

### 1.3 高噪声阶段 detail pressure 不够

现有 detail supervision 使用 `alpha_bar` 时间权重。高噪声阶段恰恰更需要 reference 来决定字体实现，但细节监督被明显压低。K1 改为 `sqrt(alpha_bar)` 并使用有界距离，是在提高高噪声条件学习压力，同时避免完全取消时间权重造成不稳定。

因此“模型偷懒”可以作为工程语言，但论文/正式报告建议表述为：

> the current objective admits a content-dominant shortcut in which the denoiser can minimize training losses while under-utilizing target-conditioned local appearance evidence.

---

## 2. 为什么 K1 保持 global coefficient=1

原 I7 proposal 曾提出：

`(1-beta) * A_global + beta * A_TC`

当 beta=0.8 时，相当于同时把成熟 global style path 压到20%。这样即使 K1 成功，也无法区分是 TC 被打开，还是旧 global path 被削弱后模型被迫重学；失败时也无法区分 TC 有问题还是 global conditioning 被破坏。

K1 因此固定：

`A = A_global + beta * normalized(A_TC)`

即：强制使用 TC，但不拆掉旧路。

这对效果优先也更安全，因为 K0 的 global style 能力被保留。

---

## 3. 为什么把 add/remove 分开

如果只用一个 `|GT-content|` change mask，大面积加粗/扩张区域可能吞掉少量但更关键的“需要删掉”的区域，例如：

- 断笔；
- 空心；
- 缺口；
- 某些局部连接的消失。

K1 将 `M_add` 与 `M_remove` 分别归一再等权合并，使“该加什么”和“该删什么”在优化中拥有接近对等的话语权。这比单纯增加 edge loss 更贴近当前失败模式。

---

## 4. 为什么本轮允许 sampling / drop 一起改

从严格归因角度，sampling、drop、TC injection、detail loss 同时变化会降低单因素可解释性。但当前优先级已经明确调整为：**先获得明显效果，再补最少量必要对照。**

因此 K1 接受：

- ~60% confirmed complex exposure；
- same-content / different-font pairing；
- joint CFG drop 0.10→0.02；
- source drop 0.25→0.05。

这些都应被定义为 training recipe，而不是独立论文贡献。

K1 之后如果最终提升明显，再从投稿需要出发补 1–2 个信息量最高的 matched controls，不在本轮提前铺矩阵。

---

## 5. 对整体故事的影响

当前论文主线是：

1. cross-script font completion task/evaluation；
2. Dynamic Delta：目标字符的 bank-supported variation prior；
3. TC：从真实 reference 中完成 target-character appearance。

K1 **不改变这个角色分工**。

可以继续使用下面这套一句话故事：

- Neutral Content：告诉模型“这是什么字”；
- Dynamic Delta：告诉模型“这个字在训练字体库中有哪些可支持的结构变化”；
- TC：告诉模型“目标字体真实展示了怎样的视觉语言，以及这些证据应如何作用到目标字符位置”；
- K1 training recipe：防止 denoiser 退化到只依赖 Content/global style 的安全解。

因此 K1 是对原故事的“兑现”，不是换故事。

---

## 6. 对三个贡献的具体影响

### Contribution 1：跨文字补全任务与评测

**基本不变，反而更一致。**

任务本来就要求 family identity，而不只是字符可读。K1 的 change-region/detail diagnostics 只是让训练与评估更贴近该目标，不需要新增贡献条目。

建议最终评估仍保持：identity、blind style preference、family compatibility 为主要叙事；L1/SSIM/D_change 等作为 paired reconstruction diagnostics。

### Contribution 2：Dynamic Delta

**基本不变。**

K1 不修改 donor set、anchor-relative residual、Alpha、空间/时间动态 router 或 structural conditioning 定义。source-drop 降低属于训练 recipe。

这也是本轮不实现 I7 差分 DCN 重构的主要叙事理由：避免在效果主实验里同时改写第二贡献的方法定义。

### Contribution 3：Target-Character Appearance Completion

**需要中等幅度改写，但属于加强，不是推翻。**

当前正文把 TC 写成 target-aligned appearance memory，并通过额外 reference-attention residual 注入 generator；问题在于 learned gain 可被优化到接近零。

K1 后建议把公式从：

`Z = CA(h,G) + eta * r(u) * CA(h,A)`

改写为：

`Z = CA(h,G) + beta(u) * Norm(CA(h,A), CA(h,G))`

其中 beta 是预定义的 non-zero schedule，Norm 是 stop-gradient 的幅度匹配，不是可学习门。

新的叙事可以更强：

> TC not only predicts a target-aligned appearance representation from local reference evidence; the denoiser is explicitly required to consume this representation during synthesis.

TC 的核心定义——online local reader、target-aligned 144-token memory、direct target feature supervision——完全保留。

---

## 7. 不要新增“第四贡献”

这是最重要的论文边界。

以下全部只能写成 training recipe / optimization strategy：

- `D_change` / `D_high`；
- add/remove masks；
- `sqrt(alpha_bar)`；
- complex oversampling；
- same-content pairing；
- lower condition drop；
- fixed non-zero TC consumption schedule。

不要命名为新的“Change-Aware Style Loss”等第四个方法贡献，否则论文会从清楚的 `Task + Delta + TC` 变成多个补丁堆叠。

推荐表述：

> To prevent a content-dominant shortcut, training emphasizes target-specific deviations from neutral content and uses a non-vanishing appearance injection schedule.

这属于如何把现有方法训练出来，而不是新的独立 contribution。

---

## 8. 对正文需要修改的地方

K1 结果确认后，再修改论文；不要在结果出来前把效果写死。

预计只需要改：

1. Method / TC injection 公式：去掉 learned `eta`，换成 fixed schedule + magnitude normalization；
2. Training objective：增加 change-focused detail term，并说明它是 optimization recipe；
3. Training settings：更新 drop、sampling、CFG1 的 K 最终协议；
4. Reproducibility：记录 K0/K1 parent、sampler manifest、complex manifest、loss coefficients；
5. Experiments：主结果以 K1 作为 full model 候选，旧 I 结果保留开发过程 provenance，不必进入主表。

三条 contribution item 本身暂时不用加第四条。

---

## 9. 投稿前最少量的后续证据

当前不阻塞 K1。若 K1 最终有明显提升，投稿前只补最少量、信息量最高的 matched evidence。优先候选：

- K1 full vs K1 去除 forced TC consumption（验证第三贡献确实被使用）；
- K1 full vs K0 或现有 Mean-Delta baseline 的 matched final board；
- Delta contribution 使用已有/最小必要对照，不在 K1 前铺大矩阵。

具体跑哪些，等 K1 成果后再决定。

---

## 10. 最终设计判断

批准 K 系列方向，推荐直接实现 `K_EXPERIMENT_PLAN_20260916.md` 中的 K1。

本轮核心纪律：

- 效果优先；
- K0 保持与 I0/G0b 一致；
- 强制 TC 被生成器消费，但不削弱 global path；
- loss 重点惩罚 content→GT 的必要变化；
- complex/pair sampling 提高高信息量样本密度；
- Dynamic Delta 架构不动；
- 不新增第四贡献；
- 2k 只做安全/质量中检，正常情况跑满10k；
- 最终用 K0/K1 matched board + clean 三语种全协议判断是否真正出现肉眼明显提升。