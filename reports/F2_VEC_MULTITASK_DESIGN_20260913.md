# F2-VEC：栅格 + 矢量双 decoder 多任务（设计记录）

日期：2026-09-13
状态：**remote run reported complete / evidence partial**；代码、test16 栅格预测和像素指标已于 2026-09-13 选择性迁入 main，权重与完整训练日志未随分支提供
作者记录：decoder / 损失 / 臂定义由执行侧锁定，便于追溯
主线关系：side study。不覆盖 `F2-DELTARSI-A-S3407`，不改 `code/official/`。
**当前可执行主方法仍是 F2 mean-Δ**（已训完）。Set-Delta + Graphics-Ref 只是设计稿，**尚未实现，不是现行主方法。**

父文档：F2 协议见 `docs/EXPERIMENTS.md`、`scripts/launch_cn2west_f123.py`；上一版「先 stop-grad」讨论作废，以本文为准。

---

## 0. 对「为什么不能一起训」的更正

上一版把 **UNet ← stop-grad(L_vec)** 写成默认，是为了**保护已发表 F2 像素数字不被新损失带偏**，不是因为多任务在这个任务上更差。

**主方案改为联合 multi-task。** stop-grad 降为对照臂，用来回答：「矢量头只是多一个输出，还是真的改了共享表示？」

### 0.1 一起训在干什么

两个头共享同一套 F2 条件：

- style：Es n-shot tokens（协议 A，CN refs）
- content：Ec(目标字符)
- structure：mean-Δ RSI（与现行 F2 相同；本臂不实现 Set-Delta）

栅格头：已有 UNet，出 96px。
矢量头：可微渲染的固定基元轮廓，出可编辑路径，并栅格回 96px 对齐 TargetImage。

一条损失：

```text
L = exp(-s_img) L_img + s_img
  + exp(-s_vec) L_vec + s_vec
```

`s_*` 是 Kendall & Gal (CVPR 2018) 同方差不确定性标量，防止矢量项在中文长轮廓上盖过扩散 MSE。`L_img` 的公式和系数与 F2 **逐项相同**。

### 0.2 联合训练的优势（只谈本任务）

本任务不是「把已见汉字画得更像」，而是：**中文 refs 可见风格，目标拉丁/假名在 refs 里不可观察**。96px 扩散对未见码位容易出断笔、粘连、填充分裂。矢量头提供的是**分辨率无关的闭合几何**，不是另一张更糊的图。

| 机制 | 在 CN→未观察西文上的具体作用 |
|------|------------------------------|
| 闭合 + 绕组占用 | 强迫「一个字形 = 若干闭合填充」，抑制未见字母的碎斑、开放缺口 |
| 笔画是曲线不是纹理 | Δ / style token 被拉向「对比度、字重、起收笔」而不是 96px 墨点纹理；跨语系时纹理本来就不可迁移 |
| 渲染对齐 Target | 矢量不是后处理；和 UNet 抢同一张 GT，共享条件必须同时解释得通像素和轮廓 |
| 推理双输出 | 同一 forward：论文图用栅格，附录/产品用 SVG；不是第二套模型 |

联合相对 stop-grad 的**可证伪增量**：若 MT 只在矢量 IoU 上赢、图像 L1 与 SG 对照相同，说明共享干没有吃到几何；若未见拉丁的断笔率下降且 L1 不坏，才能说多任务改了跨语系表示。

### 0.3 已知风险（用臂隔离，不靠口头）

- 梯度冲突：中文轮廓复杂，`L_vec` 可能主导。→ 不确定性加权 + 按 script 对 `L_vec` 做 batch 内均值（拉丁/假名/汉字各一档，再等权）。
- 96px 可微渲染偏钝。→ 轮廓参数在 **em-square [0,1]** 回归，只在渲染时落到 96；不在像素网格上回归控制点。
- 矢量把 UNet 拉成「剪纸风」。→ `L_consist = ||render(V) - x0_hat.detach()||` 只作弱锚，且 **x0_hat 永远 detach**，避免扩散中段噪声当轮廓 GT。
- 不能和旧 F2@80k 比特比。→ 图像主张只对 **同机同协议的 F2 复跑 / SG 对照**；旧 3090 ckpt 只作参考锚，不写「超越 F2」。

---

## 1. 执行侧已锁定（不要再开设计讨论）

改这些必须改本文版本日期，禁止口头漂移。

### 1.1 代码与 run

| 项 | 锁定 |
|----|------|
| variant | `code/variants/cn2west_f2_vec/`（从 `cn2west_f123_rsi` 派生） |
| 主 run | `F2-VEC-MT-A-S3407` |
| 对照 run | `F2-VEC-SG-A-S3407`（L_vec 不进 F2 模块） |
| 干对照 | `F2-VEC-TRUNK-A-S3407`（L_vec 只进 Δ/style 投影，不进 UNet） |
| parent | `runs/F0-RSIFREE-FT-A-S3407/best` |
| 数据 / split / seed | 协议 A，228/16/16，**3407** |
| 超参 | 单卡、bs=8、accum=1、fp16、lr=1e-5、warmup=5k、source_drop=0.25 |
| cache | **F0** Es + Ec；E1 禁用 |
| RSI | `rsi_source=delta`，support=off |
| 步数 | **默认训满 40k**；10k/20k 只写中期记录，不提前停、不据中期改参 |
| 精度 | V100 只用 fp16 |

### 1.2 矢量 decoder（可微渲染，非自回归）

不采用 DeepVecFont-v2 序列 decoder 作为 v1：自回归 CE 与扩散 MSE 的梯度尺度差、且强依赖 TTF 命令 GT。v1 对齐 **VecFontSDF（CVPR 2023）** 的「图像方法扩到矢量」+ **DualVector（CVPR 2023）** 的解析占用，不引入 DiffVG 运行时依赖。

**表示（VecFontSDF-lite + 有符号占用）**

- 固定 **P = 16** 个基元。每个基元 6 个数：抛物线/二次 Bézier 的 3 个控制点 `(x,y)∈(0,1)`（tanh 后仿射到 em-box）。
- 另输出 16 维 **符号** `s_i ∈ (-1,1)`（tanh）：正=外轮廓实体，负=孔。这是 DualVector dual-part 的最小版，避免上完整布尔网络。
- 另输出 16 维 **存在** `π_i = sigmoid(logit_i)`：软开关，空基元不硬切。
- 闭合：每个基元视为一条二次曲线段；**4 个基元一组** 组成一条 path（3 条外轮廓 + 1 组孔 = 16）。组内第 4 段终点与第 1 段起点用 L2 拉齐（`L_close`）。
- 解析占用：对 96×96 网格点，用二次 Bézier 的解析 winding / 有符号距离近似（VecFontSDF 抛物线→二次 Bézier 的同一转换）。填充 `α = sigmoid(-β · SDF_signed)`，`β=8` 固定。
- 输出 SVG：推理时 `π_i>0.5` 的段写成 `Q` 命令；训练不走 SVG 解析器。

**条件（t 无关）**

矢量头**不吃扩散时刻 t**。输入只做一次：

1. style：F2 现用 n-shot Es，空间均值成 9 token，再线性到 `d=256`
2. content：Ec 五尺度均值拼接，线性到 256
3. Δ：F2 `_structure_features` 的 pooled 向量，线性到 256
4. 三个 256-d 相加 + LayerNorm → 4 层 MLP（256-256-256-out）直接出 `16×(6+1+1)` 参数

参数量小，避免再挂一套 Transformer 抢档期。

**梯度路由（MT 主臂）**

| 损失 | VecHead | Δ / style / content 投影 | UNet |
|------|---------|--------------------------|------|
| `L_img`（F2 原样） | 不进 | 进（与 F2 同） | 进 |
| `L_vec` | 进 | **进** | **5k 后进**，`λ_unet=0.15` 乘在反传前（不是改 `L_img` 系数） |
| `L_consist` | 进 | 进 | **不进**（`x0_hat.detach()`） |

前 5k：只训 VecHead + 不确定性标量，F2 模块只走 `L_img`（热身，避免随机轮廓炸 UNet）。
5k–40k：按上表打开联合。

### 1.3 损失（锁系数，5k 只准看 NaN）

```text
L_img     = L_diff + 0.01 L_percep + 0.5 L_offset     # 与 F2 逐项相同
L_render  = ||α - I_ink||_1                            # I_ink = Target 墨迹 [0,1]
L_iou     = 1 - softIoU(α, I_ink)
L_close   = mean ||end_k - start_k||^2                 # 4 组 path
L_thin    = mean ReLU(π_i - 0.05) * width_proxy         # 防塌成点；width_proxy=控制点跨度
L_vec     = L_render + 0.5 L_iou + 0.2 L_close + 0.05 L_thin
L_consist = ||α - ink(x0_hat.detach())||_1             # 系数 0.1，写入 L_vec 外另加

L = e^{-s_img} L_img + s_img + e^{-s_vec} (L_vec + 0.1 L_consist) + s_vec
```

`L_vec` 在 batch 内按目标字符 script（Han / Latin / Kana）先组内平均再等权，避免汉字项主导。

**为什么这样拆，而不是再加一项「矢量 CE」：**

| 项 | 作用 | 若不加 |
|----|------|--------|
| `L_img` 原样 | 保证栅格任务仍是 F2；系数动了就不是同一实验 | 无法说「在 F2 上加了头」 |
| `L_render` L1 | 每个像素都有梯度，占位从空轮廓拉向墨迹 | 只有 IoU 时早期全零 mask 梯度极弱 |
| `L_iou` | 抗「缩成小黑点也能降 L1」（背景像素占优） | 模型用缩小字形刷 L1 |
| `L_close` | 组内首尾相接，推理才能写成合法 SVG | 开放折线，IoU 仍可能好看 |
| `L_thin` | 惩罚「π 开着但控制点塌成一点」的空基元 | 16 基元名义存在、实际没面积 |
| `L_consist` + detach | 两个头不要画出两个不同字；**不**让 UNet 去拟合矢量渲染 | 栅格/矢量各画各的，无法谈共享表示 |
| `e^{-s} L + s` | 两任务量纲差一个数量级时自动压权重 | 手调 λ 会在中文 batch 上反复炸 |
| script 等权 | 本任务看重未见拉丁，不能让汉字 `L_vec` 主导 | 「idea 有意义」会被 train 汉字刷出来 |

**v1 不用 TTF 命令 CE。** 轮廓 JSON 若后来到了，只允许加一项 `0.2 L_pt`（控制点 L1），不得在 40k 中途改其它系数。

### 1.4 评测（图像 / 矢量拆开）

图像（必须，F2 同脚本、test16）：L1、已有像素协议。
快证伪（同一条 MT run，不另开 SG）：10k 相对 5k，val IoU 上升且 `L_img` 无非有限；这只作 idea readout，**不是**停训点。
正式对照（40k 后若 IoU/断笔像有事再补）：`F2-VEC-SG` > 同机纯 F2 40k > 3090 F2@80k（仅锚点）。

矢量：render IoU、Chamfer（轮廓 64 点）、自交率、开放 path 率、未见拉丁/假名的断笔人工 2 档（通 / 不通）。

**过门（写死，未过不进文、不进 REGISTRY 主表）**

- 图像：相对 SG 对照，test16 L1 **恶化 < 0.002**（与 Stage A 门同量级）
- 矢量：val 上相对「冻 UNet 只训 VecHead」的 5k 检查点，IoU **+0.05**，自交 **< 5%**
- 跨语系：未见拉丁断笔「通」的比例高于 SG；否则只称「多了一个可渲染头」，不称表示增益

### 1.5 算力与「会不会再堵上」（单卡 V100）

**不会复现 3090 上 F2+F3 双进程堵死。** 那次是两条训练抢同一份 94G Ec mmap：GPU util 0–3%、4–6 s/step。本臂是 **一个进程、一张卡、两个 decoder**，Ec 只 mmap 一次，和单独跑 F2 的 I/O 画像相同。

会变慢的是 **同一步里多算可微占用**，不是第二份 cache。锁定实现约束，避免把渲染写成 Python 双层循环（那种会变成另一种 CPU 堵）：

- `α` 必须是 batched tensor：`[B,16,96,96]` 一次算完，禁止 Python 扫像素
- 独占 GPU，**禁止**与另一条 F 臂同时读这份 Ec
- 不 8 卡 DDP

墙钟：纯 F2 40k ≈ 20 h；MT **+25–40% → 约 26–28 h / 40k**。相对「先 MT 再 SG」的 55 h，快路径只开 MT。

历史启动门：V100 上 Ec 补齐后启动 `F2-VEC-MT`。来源分支报告该 40k run 已完成；main 不含大 cache 与权重。

### 1.6 已拍板（2026-09-13）

| 点 | 决定 |
|----|------|
| K1 定位 | **side study**。现行主方法是 **F2 mean-Δ**，不是未实现的 Set-Delta。过门后另议是否进文。 |
| K2 档期 | **尽快出结果**：Ec 齐后只开 MT，不先排 SG/TRUNK；10k 记 idea readout，**默认跑满 40k**。 |
| K3 停则 | **不因 20k L1 停**。图像若变差，40k 后只报矢量/对照，不把坏 L1 写成增益。 |

SG / TRUNK 仅当 40k MT 的 IoU/断笔看起来像「有事」时再补，用来拆干 vs 头。

### 1.7 已迁入的 40k 栅格结果（2026-09-13）

来源：`v100/f2-vec-mt-40k-eval@0aae4d1e`。每个模式 test16 × 47 字符 = 752 项，blank=0。

| 模式 | L1 ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| `F2VEC_40000` few-shot | 0.070522 | 0.647722 | 0.151318 |
| `F2VEC_40000_s1` one-shot | 0.072233 | 0.633893 | 0.158000 |

这些是 raster output 对 GT 的像素诊断。它们证明预测资产完整，不单独证明矢量质量或共享表示增益；当前迁移资产中没有 render IoU、Chamfer、自交率、开放 path 率，也没有 SG/TRUNK matched control。

**迁入时实现核对：** 当前 `src/vec_decoder.py::render_occupancy` 用三个控制点的中心与最大半径构造 signed ellipse/circle occupancy；它没有实现本设计 §1.2 所写的二次 Bézier winding/SDF。控制点可写成 `Q` 命令，但训练 renderer 与最终 Bézier 曲线的几何并不等价。因此现有代码应称为 **fixed-primitive occupancy prototype**，不能称为已经完成的 Bézier 可微矢量渲染器。后续若继续该方向，应先统一 renderer 与 SVG 表示，再补矢量指标。

---

## 2. 关键点已收口

§1.6。再改定位/档期/停则必须改日期。

---

## 3. 明确不做（v1）

- 不改 F2 旧目录、旧 ckpt、旧 provenance
- 不上 DeepVecFont-v2 自回归 decoder、不上 DiffVG
- 不把本臂写成「已经替换 F2 / 已经上 Set-Delta」
- 不把本臂和未实现的 D0–D3 / R1–R3 混编号
- 不用 T1 / 本臂 5k plumbing 当结果
- 不在 V100 上 bf16、不 8 卡 DDP 一条 run
