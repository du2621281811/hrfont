# Bank 风格先验叙事（2026-09-06，PI 观察驱动；已按 REVIEW_BANK_STYLE_20260906 修订）

> 状态：**叙事草案 v2**。机制框架成立，量化证据（P1/P2）未跑；论文强度等级见 §4。
> 上游：`DESIGN_E2E3_FUSION_QKV_20260905.md`、`DECISION_F123_20260906.md`、`REVIEW_BANK_STYLE_20260906.md`。

## 1. 观察

对比 official baseline / F0 / F3 的生成结果，PI 发现生成字形存在**风格偏好**，并提出两点主张：

1. bank 的 style 数量足够多时，生成的 content 更符合**设计师通用原则**；
2. 通过**限定 bank 的风格**，可以为生成文字限定偏好（风格旋钮）。

## 2. 机制（已验证的代码事实 + 待验证的直觉）

- **Δ 实际只混合 top-10**：完整 228/LOO-227 库是**候选池**，每个样本只对 top-10 做 softmax(cos/τ, τ=.07) 加权混合；完整池通过**候选竞争**影响这 10 个，而非 227 路直接平均。`Δ = Σ_s∈top10 ᾶ_s·Ec(B_s, c) − Ec(Content, c)`。
- **「平均即去噪」是直觉假设，非定理**：top-10 加权平均可能抵消特异怪癖、保留共识偏差，但尚无实验或定理支持；策展子集也可能引入系统性偏置（近邻全被拉向策展方向）。**措辞按「假设」写，等 P1/P2。**
- **α 无绝对阈值（D-P4）→ 输出由 bank 邻域构成**：无「不够像就拒识」分支，bank 的组成决定 top-10 的组成。因此可做**推理期 bank intervention**：同一 ref、不同 bank → 不同输出。**「可控性」的强度表述：无需更新参数的 inference-time steering；方向/强度/质量/OOD 稳健性待 P2 验证。**

## 3. 待跑证据（推理期零训练；先过 D-B4 parity 准入）

| Probe | 做法 | 回答 |
|---|---|---|
| **P1 bank 规模扫** | 固定 F2/F3 checkpoint，**固定 K=10**，只变候选池 {10, 30, 100, 228}（多个固定随机种子/嵌套 manifest，避免单子集偶然性）；K∈{10,30} 另作独立消融 | bank 规模对 Δ/输出的 OOD sensitivity |
| **P2 bank 风格限定** | 同一 ref，bank 换成衬线-only / 黑体-only / 书法-only 子集；**F2 与 F3 都要跑**（检验 support 是否缓冲/放大 steering） | inference-time steering 方向可控性 |

指标：风格方向 φ_s2 SC-Gap；质量 coverage/LPIPS；终审 E11 人评 2AFC。**φ_s2 过 T2 门是量化前置**（S3 字型池已入库 `artifacts/e12/external_font_cache`，SHA 0bd0af23）。
**准入（D-B4）**：执行机推理实现必须过 full-bank parity（indices/weights/每尺度 Δ 与训练语义一致）+ bank manifest/SHA 记录；仓库 F123 `sample.py` 尚无 Δ/support 路径，不得拿旧 sample 出图。

## 4. 论文强度等级

- **可写（机制级）**：Δ = 候选池竞争出的 top-10 加权结构先验；bank 可被策展从而改变先验（inference-time intervention）。
- **暂不可写（结果级）**：任何「更符合设计师通用原则」的定量主张；「策展是可靠控制机制」的强 claim 需 **matched retrain（full vs curated）+ 随机同规模控制**，最好 `train {full,curated} × infer {full,curated}` 2×2（D-B2）。
- **禁止**：把目测样本当证据；把推理期 bank 变换写成「训练过的可控性」；把 V2 AUC 换算成污染上界。

## 5. 训练/推理 bank 不一致（Codex 审查结论）

训练 228 / 推理子集**不是「换 bank 重训」**——它是固定 checkpoint 上的 Δ 输入分布干预：新池重排邻居、在新 top-k 内重 softmax（候选 <K 时取全部并归一化，仓库语义如此；执行机必须验证重排/重归一 parity）。shift 的存在确定、大小暂无实测；identity-safe residual + source_drop .25 提供旁路/无源鲁棒性，**但不保证任意策展 Δ 的正确响应**。

## 6. Es 与「非参考风格信息」（Codex 审查结论）

- Es 库特征**从不进入生成器**（只用于排序）；无 test-side 泄漏（α query 只用目标字体自身 ref，leave-one-out）。
- **真正的非参考信息通道 = α 加权的 library Ec(target char)**——这是 bank prior 的定义，不是代码 bug；但「错误邻居」可能是 **top-1**，不能只归因于小权重尾部；弱/错误邻居被强制采用是 D-P4 必须披露的风险。
- **τ=.07 未必接近 one-hot**：top-10 分数跨 0.4 时 top-1 约 47%，跨 0.05 时仅约 14%——没有真实 cosine/V6 输出就不能声称尾部污染小。
- **改用 Ec 排 α 不会消除非参考信息，只改变检索误差分布**——应定位为 retrieval-source 消融，不是「防泄漏修复」。
- 代码门限实际为 **V1≥.80、V2≥.75**（非 0.90/0.80）；正式 V1/V2/V6 结果在仓库未找到。
- F3 own-font 8-char support 与 α library 分流；但 joint 模型可能产生功能交互（P2 双跑检验）。**exec-spec 数据契约「support 图不来自目标字体」与运行代码/D7 冲突**（D-B5）。
