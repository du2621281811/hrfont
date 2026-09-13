# CGE × 可微矢量生成：融合备忘录（待 PI 评估）

日期：2026-09-13  
状态：**proposal / parked；未获 PI 批准，不进入当前冻结主线，不据此启动训练。**  
主线边界：当前可执行基线仍为 clean-data 上的 **F2 mean-Delta**。本文不把 Set-Delta、CGE 或矢量分支写成已验证贡献，也不修改现有 F2/F2-RL128/F2-PRL 的 lineage。

关联材料：

- CGE 原始草案：`reports/IDEA_CGE_20260912.md`
- 矢量多任务设计：远端分支 `origin/v100/f2-vec-mt-40k-eval` 的 `reports/F2_VEC_MULTITASK_DESIGN_20260913.md`
- 当前 mean-Delta：`scripts/hrfont_delta_v2.py`

## 1. 一句话方案

Delta 定义可编辑的目标字符证据空间；CGE 根据 reference compatibility 在该空间中导航；可微矢量 decoder 将笔画、轮廓和拓扑变成可度量的结构坐标与约束，避免导航器只追逐单一风格分数。

```text
reference R + content c
        │
        ├─ evidence weights α(z) ──> mean-Delta ──> shared generator features
        │                                      ├─ raster decoder ──> image
        │                                      └─ vector decoder ──> paths ──> differentiable render
        │
        └─ CGE surrogate <── compatibility / identity / vector geometry / topology labels
```

矢量 decoder 在第一版中是 CGE 的结构接口和辅助任务，不作为替代 raster decoder 的独立主贡献。纹理、飞白、阴影和装饰效果仍由 raster 路径负责。

## 2. CGE 的对象、监督与训练

### 2.1 编辑对象

保留原始检索权重 `α0`，只在 top-L donor 候选池内编辑：

```text
α_j(z) = TopKSoftmax(log(α0_j + eps) + beta * a_j^T z)
```

- `z`：4–8 维控制变量；
- `a_j`：donor 的证据属性向量，可由 Ec/Es 特征与矢量几何描述共同得到；
- 输出仍保持稀疏 top-K，第一版不改变 mean-Delta 聚合公式。

### 2.2 冻结与可训练部分

第一版冻结 generator、Ec、Es、mean-Delta 和 E12。仅训练：

1. `U_theta`：把 donor 特征/矢量描述投影到证据属性轴 `a_j`；
2. `H_psi`：根据 `(z, ref embedding, target-character embedding, α0 summary)` 预测候选配置的相对收益和约束量。

不要求对完整 diffusion sampling 反向传播。训练集通过冻结生成器离线采样得到；推理时通过 `H_psi` 优化 `z`，再用生成器产生少量最终候选。

### 2.3 监督信号

同一 episode 内固定 reference、目标字符和噪声，只改变证据配置，生成候选对 `(y_a, y_b)`。监督由以下部分组成：

- compatibility：E12 raw/calibrated logit 的候选内相对排序；
- identity/readability：独立字符识别或 identity gate；
- raster quality：artifact、render consistency 等；
- vector geometry：字重、倾斜、曲率、端点/角点、字面比例；
- topology validity：闭合、孔洞/连通域、自交、开放 path。

主损失优先采用 episode 内 pairwise ranking，而不是跨字体拟合 E12 绝对分数：

```text
L_CGE = L_rank + 0.25 L_score + 0.5 L_id + 0.25 L_geom/topology
```

系数仅是待验证初值。E12 可以做导航 teacher，但论文最终结果不能只报告同一个 E12；需要独立人评和未参与导航训练的 identity/quality 指标。当前 E12 未完成跨假名/注音域验证前，不作为这些 script 的唯一监督。

## 3. 矢量 decoder 与 CGE 的三个融合点

### F1：共享表示的多任务正则

Raster 与 vector 两个头共享 F2 条件和中间表示。Vector 路径通过可微渲染得到 raster，与 GT 和 raster-head 输出对齐：

```text
L = L_raster
  + λ_render D(Render(v), y_gt)
  + λ_cross D(Render(v), stopgrad(y_raster))
  + λ_topo L_topology
  + λ_smooth L_curve
```

用途是提高轮廓、断笔、粘连和孔洞稳定性。它不能替代数据清洗，也不能单独解决纹理/特效迁移。

### F2：把矢量几何变成 CGE 的约束

CGE 从单目标最大化 compatibility，升级为受约束导航：

```text
maximize    S_compatibility(z)
subject to  S_identity(z) >= τ_id
            E_render(z)    <= τ_render
            E_topology(z)  <= τ_topology
```

这可降低导航器为提高 E12 而破坏字符身份或轮廓合法性的风险。

### F3：用矢量描述定义 evidence axes

把 `a_j` 从纯 latent PCA 方向升级为有结构语义的 donor 描述，例如 stroke width、slant、curvature、contrast、terminal geometry 和字面比例。CGE 因而可以表达“提高粗笔画 donor 权重”或“降低高倾斜 donor 权重”，并可视化每次证据编辑。

## 4. 建议实验顺序（本备忘录不授权执行）

1. `F0-Clean`：冻结清洗数据和共同 parent。
2. `F2-Mean-Clean`：确认数据修复后风格响应和 Delta intervention 有效。
3. `F2-Mean-Clean + VectorAux`：只改变 vector 辅助任务。
4. `F2-Mean-Clean + CGE`：只改变证据导航。
5. `F2-Mean-Clean + VectorAux + CGE`：测试结构约束对 CGE 的增量。
6. `F2-Set-Clean`：独立验证 Set-Delta；不与首次 vector/CGE 实验同时引入。

最低 matched controls：

| ID | 数据/parent | Delta | Vector | CGE | 识别问题 |
|---|---|---|---|---|---|
| M | clean / same | mean | off | off | clean mean-Delta 基线 |
| V | clean / same | mean | on | off | 矢量多任务增量 |
| C | clean / same | mean | off | on | CGE 导航增量 |
| VC | clean / same | mean | on | on | vector 是否改善 CGE |
| S | clean / same | set | off | off | Set 相对 Mean 的增量 |

固定 generator-call budget 比较默认 `α0`、随机搜索、CGE 搜索；否则 CGE 的收益可能只是因为生成了更多候选。

## 5. 进入执行前的 gate

- 数据 gate：清洗后的假名/注音与中文 reference 不再存在系统性风格冲突，并冻结 manifest。
- controllability gate：固定 checkpoint 下 donor swap / α intervention 能稳定推动输出；阴性则 CGE 暂停。
- vector gate：Render(v) 的 IoU、闭合率和自交率达到可用水平，且 raster 指标没有明显退化。
- evaluator gate：E12 导航域和最终独立评测域分离；跨 script 适用性单独验证。
- attribution gate：不在同一首轮实验中同时切 clean data、Set-Delta、Vector 和 CGE。

## 6. 当前状态与待 PI 拍板

截至本备忘录创建时：

- CGE 仍是独立 proposal，未进入冻结实验计划；
- main 已包含 F2-RL128 完成记录和 F2-PRL 启动记录；
- `origin/v100/f2-vec-mt-40k-eval@0aae4d1e` 的矢量 variant、启动/评测脚本、设计文档、test16 few/one-shot 预测与像素指标已于 2026-09-13 选择性迁入 main；该旧分支对论文、主线看板和其他实验的删除/覆盖没有迁入；
- 来源分支报告 F2-VEC-MT 40k 已完成，但权重和完整训练日志未随 Git 提供，因此按 `reported completed / evidence partial` 管理；
- 本文件只保存 CGE 融合设计，不把 F2-VEC 的栅格诊断自动升级为矢量或 CGE 结论，也不触发新训练。

待 PI 决策：

- D-CV1：是否把 vector decoder 定位为 CGE 的结构接口，而不是第四项独立贡献；
- D-CV2：优先采用真实 TTF/OTF outline 监督，还是 raster-only 的 render consistency；
- D-CV3：CGE v1 使用哪些矢量属性轴；
- D-CV4：clean Mean-F2 通过 controllability gate 后，先做 VectorAux 还是 CGE Phase 0；
- D-CV5：是否补传 F2-VEC-MT 的权重、训练日志和矢量指标，以完成独立验收。
